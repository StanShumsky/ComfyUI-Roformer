"""Cross-session alignment math: align one recording to a reference so two files from
different sessions can be ensembled or A/B'd.

The full alignment pass - auto / forced phase flip of the secondary, optional
leading-silence matching, the multi-position intro cross-correlation pass (unique
candidate shifts), then per candidate either the sliding-window time correction
(Hanning windows, per-window cross-correlation over the full dB x phase-shift grid,
overlap-added residual) or the plain dB grid - with the final selection: the 240-split
lowest-mean ensemble over all candidates. The per-window correlate uses an FFT
(sample-exact, the same argmax as a direct np.correlate) and the dB x phase grid is
batched in chunks, so the same search runs vectorized instead of a Python loop.
"""
from __future__ import annotations

import torch

from . import ensemble

# ------------------------------------------------- option tables (the search grids)

PHASE_OPTIONS = ("automatic", "positive", "negative", "native")

# intro analysis: the positions (fractions of the track length) at which a 1 s segment
# is cross-correlated; "shifts: *" try several fractions and keep the unique shifts
INTRO_OPTIONS = ("default", "1/8", "1/6", "1/4", "1/2", "shifts: low", "shifts: medium", "shifts: high")
INTRO_MAP = {
    "default": [10],
    "1/8": [8], "1/6": [6], "1/4": [4], "1/2": [2],
    "shifts: low": [1, 10],
    "shifts: medium": [1, 10, 8],
    "shifts: high": [1, 10, 8, 6, 4],
}

# sliding-window time correction: the window sizes (seconds) to refine per window;
# off = the global dB grid only
ALIGN_WINDOW_OPTIONS = ("off", "0.0625 s", "0.125 s", "0.25 s", "0.5 s", "0.75 s",
                        "1 s", "2 s", "shifts: low", "shifts: medium", "shifts: high")
ALIGN_WINDOW_MAP = {
    "off": None,
    "0.0625 s": [0.0625], "0.125 s": [0.125], "0.25 s": [0.25], "0.5 s": [0.5],
    "0.75 s": [0.75], "1 s": [1.0], "2 s": [2.0],
    "shifts: low": [0.0625, 0.5],
    "shifts: medium": [0.0625, 0.125, 0.5],
    "shifts: high": [0.0625, 0.125, 0.25, 0.5],
}

# volume analysis of the secondary: (base dB, the dB adjustments to try)
VOLUME_OPTIONS = ("off", "low", "medium", "high", "very high")
VOLUME_MAP = {
    "off": (0.0, [0.0]),
    "low": (-4.0, [float(d) for d in range(0, 8)]),
    "medium": (-6.0, [float(d) for d in range(0, 12)]),
    "high": (-6.0, [x * 0.5 for x in range(0, 25)]),
    "very high": (-10.0, [x * 0.5 for x in range(0, 41)]),
}

# phase-shift grid: the step (degrees); off (190) = no phase variants tried
PHASE_SHIFT_OPTIONS = ("off", "180", "90", "45", "20", "10", "1")
PHASE_SHIFT_MAP = {"off": 190, "180": 180, "90": 90, "45": 45, "20": 20, "10": 10, "1": 1}

# per-window variant-grid chunk size: one window's full dB x phase set can be tens of
# thousands of candidates; the search stays exact (min over all of them), just chunked
_VARIANT_CHUNK = 256


# ------------------------------------------------- elementary steps

def check_if_phase_inverted(ref: torch.Tensor, wav: torch.Tensor) -> bool:
    """The phase check: the Pearson correlation of the first 1000 samples of the
    channel-mono mix is negative -> the secondary starts phase-flipped."""
    n = min(1000, ref.shape[-1], wav.shape[-1])
    a = ref.mean(0)[:n]
    b = wav.mean(0)[:n]
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt())
    if denom == 0:
        return False
    return float((a * b).sum() / denom) < 0


def _silence_end(wav: torch.Tensor, frame: int = 1024, threshold: float = 0.01) -> int:
    """The sample index where content starts: the first `frame`-long block (on the
    channel with the most energy) whose peak exceeds `threshold`."""
    x = wav[int(torch.argmax(torch.abs(wav).sum(1)))]
    for i in range(0, x.numel(), frame):
        if x[i:i + frame].abs().max() > threshold:
            return i
    return x.numel()


def adjust_leading_silence(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Shift `target` so its leading silence matches `reference`'s (zero-pad or trim
    the head)."""
    d = _silence_end(reference) - _silence_end(target)
    if d > 0:
        return torch.cat([torch.zeros(target.shape[0], d), target], dim=-1)
    if d < 0:
        # -d is the positive skip; a raw
        # target[:, d:] would take the LAST |d| samples (python negative slicing)
        return target[:, -d:]
    return target


def _full_lag(a: torch.Tensor, b: torch.Tensor) -> int:
    """The full-correlate lag: the k that maximizes sum_t a[t+k] b[t] over all lags -
    the argmax of numpy's `correlate(a, b, 'full')` minus (len-1), as one zero-padded
    FFT (sample-exact, O(n log n)). Sign convention: k > 0 when b's content starts
    EARLIER than a's (b leads), k < 0 when it lags - b's head is trimmed by -k on
    k < 0 and k zeros are prepended on k > 0."""
    m = min(a.numel(), b.numel())
    a, b = a[:m], b[:m]
    n = 1 << (2 * m - 1).bit_length()
    # conj on b, not a: the product conj(B)*A is the correlate sum_t a[t+k] b[t] (numpy's
    # correlate(a, b, 'full')); conj(A)*B would give the negated lag
    c = torch.fft.irfft(torch.conj(torch.fft.rfft(b, n)) * torch.fft.rfft(a, n), n)
    # c[k] holds lag k for k = 0..m-1 at index k, and lag -(m-1)..-1 at index n-(m-1)..n-1
    lags = torch.cat([c[0:m], c[n - (m - 1):n]])  # lags 0..m-1 then -(m-1)..-1
    j = int(torch.argmax(lags))
    # in the second half, position j = m+i holds lag -(m-1)+i = j - 2m + 1
    return j if j < m else j - 2 * m + 1


def intro_align_variants(ref: torch.Tensor, wav: torch.Tensor, sr: int,
                         intro_divs: list[int]) -> list[torch.Tensor]:
    """The first alignment pass: for each fraction, cross-correlate the 1 s segments
    one position apart and shift `wav` by the winning lag; duplicate shifts collapse,
    so the downstream dB/phase search runs once per unique candidate."""
    seconds = min(ref.shape[-1], wav.shape[-1]) // sr
    variants: list[torch.Tensor] = []
    for sec_len in intro_divs:
        sec_seg = 1 if sec_len == 1 else int(seconds // sec_len)
        idx = sr * sec_seg
        # Score on the left channel (both channels are available; the left's lag wins);
        # the lag has the opposite sign of shift_wave's offset (shift_wave: positive =
        # wav lags), so the shift is -diff
        diff = _full_lag(ref[0, idx:idx + sr], wav[0, idx:idx + sr])
        offset = -diff
        length = wav.shape[-1] + (-offset if offset < 0 else 0)
        aligned = ensemble.shift_wave(wav, offset, length) if diff else wav
        if not any(torch.equal(aligned, v) for v in variants):
            variants.append(aligned)
    return variants or [wav]


def match_length(wave: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Zero-pad / trim `wave` to `reference`'s length."""
    n = reference.shape[-1]
    if wave.shape[-1] == n:
        return wave
    if wave.shape[-1] > n:
        return wave[:, :n]
    return torch.cat([wave, torch.zeros(wave.shape[0], n - wave.shape[-1])], dim=-1)


# ------------------------------------------------- phase shifts (the Hilbert set)

def _hilbert(x: torch.Tensor) -> torch.Tensor:
    """The Hilbert transform of a real signal, as one rfft (scipy.signal.hilbert's output)."""
    n = x.shape[-1]
    h = torch.fft.rfft(x)
    y = torch.zeros_like(h)
    if n % 2 == 0:
        y[1:-1] = -1j * h[1:-1]
    else:
        y[1:] = -1j * h[1:]
    return torch.fft.irfft(y, n)


def phase_shift_hilbert(signal: torch.Tensor, degree: float) -> torch.Tensor:
    """Rotate the analytic signal by `degree` and take the
    real part - cos(d)*x - sin(d)*H{x}."""
    rad = torch.tensor(degree * torch.pi / 180, dtype=torch.float32)
    return torch.cos(rad) * signal - torch.sin(rad) * _hilbert(signal)


def phase_shifted_tracks(track: torch.Tensor, step: int) -> list[torch.Tensor]:
    """The original and negated track plus the Hilbert
    rotations at +/-step, +/-2*step, ... up to (but not over) 180."""
    if step == 180:
        return [track, -track]
    end = 180 - (180 % step) if 180 % step == 0 else 181
    out = [track, -track]
    for i in range(step, end, step):
        out.append(phase_shift_hilbert(track, i))
        out.append(phase_shift_hilbert(track, -i))
    return out


# ------------------------------------------------- selection ensembles

def ensemble_wav(waveforms: list[torch.Tensor], split_size: int = 240) -> torch.Tensor:
    """Cut every candidate into `split_size` pieces and keep, per
    piece, the lowest-mean segment across candidates - the noise-floor-minimizing
    selection."""
    parts = [torch.chunk(w, split_size, dim=-1) for w in waveforms]
    n = len(parts[0])
    out: list[torch.Tensor] = []
    for i in range(n):
        means = torch.stack([parts[c][i].abs().mean() for c in range(len(waveforms))])
        out.append(parts[int(torch.argmin(means))][i])
    return torch.cat(out, dim=-1)


def ensemble_for_align(waveforms: list[torch.Tensor], length: int) -> torch.Tensor:
    """The min-spec blend (the 2048/1024 STFT geometry the
    ensemble module already implements exactly)."""
    return ensemble.blend("min spec", [match_length(w, waveforms[0]) for w in waveforms],
                          2048, length)


# ------------------------------------------------- the sliding-window refinement

def time_correction(mix: torch.Tensor, inst: torch.Tensor, windows: list[float],
                    db_analysis: tuple[float, list[float]], sr: int, phase_step: int,
                    progress=None) -> torch.Tensor:
    """The sliding-window refinement: for each window size, slide a Hanning window across the
    track; in every window, cross-correlate the mix against every (dB x phase-shift)
    variant of the secondary and keep the shift with the lowest mean residual; overlap-
    add the residuals (mix - aligned) and divide by the window sums; the per-window-size
    candidates are reduced with the 12-split selection. A naive per-window search is a
    Python loop over the variant grid with a direct correlate; here the grid is built
    once per window (the phase set, scaled per dB - the Hilbert transform is linear in
    the gain) and searched as batched FFT cross-correlations chunked by
    _VARIANT_CHUNK - the same argmax, the same minimum. `progress(1)` is called once
    per window position per channel."""
    base_db, db_range = db_analysis
    subs: list[torch.Tensor] = []
    for secs in windows:
        window_size = int(sr * secs)
        step_size = int(sr * secs / 2)
        window = torch.hann_window(window_size, periodic=False)
        sub = torch.zeros_like(mix)
        divider = torch.zeros_like(mix)
        n_fft = 1 << (2 * window_size - 1).bit_length()
        for i in range(0, max(mix.shape[-1] - window_size + 1, 1), step_size):
            wm = mix[:, i:i + window_size] * window
            wi = inst[:, i:i + window_size] * window
            # The variant set for this window: the phase set of the instrumental,
            # every member scaled by the base dB and each per-dB adjustment
            phase_set = [wi] if phase_step == 190 else phase_shifted_tracks(wi, phase_step)
            for ch in range(mix.shape[0]):
                wmv = wm[ch]
                cm = torch.fft.rfft(wmv, n_fft)
                best: torch.Tensor | None = None
                variants = [p[ch] * (10 ** ((base_db + db) / 20))
                            for p in phase_set for db in db_range]
                for c0 in range(0, len(variants), _VARIANT_CHUNK):
                    wv = torch.stack(variants[c0:c0 + _VARIANT_CHUNK])  # (V, W)
                    corr = torch.fft.irfft(
                        torch.conj(cm)[None] * torch.fft.rfft(wv, n_fft), n_fft)
                    # corr[k] holds lag k at index k (k = 0..window-1) and lag -k at index
                    # n_fft - k: argmax on abs can land in either half. Our lag is the k
                    # maximizing sum_t a[t+k] b[t]; the np.correlate argmax lag is its
                    # negative, and the track is rolled by that - so roll by -lag here.
                    k = torch.argmax(corr.abs(), dim=1)
                    lag = torch.where(k >= n_fft - window_size + 1, k - n_fft, k)
                    shifted = torch.stack([torch.roll(wv[v], -int(lag[v]), dims=-1)
                                           for v in range(wv.shape[0])])
                    scores = (wmv[None] - shifted).abs().mean(dim=-1)
                    k = int(torch.argmin(scores))
                    cand = wmv - shifted[k]
                    if best is None or cand.abs().mean() < best.abs().mean():
                        best = cand
                # best IS the residual (wm - the aligned variant - the accumulated
                # window_mix - aligned_signal) - do not subtract again
                sub[ch, i:i + window_size] += best
                divider[ch, i:i + window_size] += window
                if progress is not None:
                    progress(1)
        sub = torch.where(divider > 1e-6, sub / divider, sub)
        subs.append(sub)
    if len(subs) == 1:
        return subs[0]
    return ensemble_wav(subs, split_size=12)
