"""UVR-style ensemble math: blend the same stem from N model runs.

The three algorithms reproduce UVR's ensemble exactly - per-STFT-bin magnitude
selection that keeps the winning input's complex value (phase included) for max spec /
min spec, a plain time-domain mean for average - with UVR's STFT geometry (Hann window,
n_fft 2048, hop n_fft // 2, centred). The companions are the levers the mvsep.com
winners use around the blend: a deterministic time-offset alignment search (for inputs
from different processing paths), anti-clip / full-scale normalization, and the exact
(time-domain) residual - mix minus blend, so stem + output == mix exactly.
"""
from __future__ import annotations

import torch

ALGORITHMS = ("max spec", "min spec", "average")
NORMALIZE_MODES = ("off", "anti clip", "full scale")


def _hann(n_fft: int, device) -> torch.Tensor:
    # periodic Hann = scipy.get_window('hann', n_fft, fftbins=True) = what librosa's default
    # 'hann' is (torch.stft's window=None would be a rectangular window - not UVR's geometry)
    return torch.hann_window(n_fft, periodic=True, device=device)


def stft_spec(wave: torch.Tensor, n_fft: int) -> torch.Tensor:
    """(C, T) float32 -> complex (C, F, frames). Hann, hop n_fft // 2, centred - the same
    geometry as UVR's librosa.stft(n_fft=2048, hop_length=1024)."""
    return torch.stft(wave, n_fft, hop_length=n_fft // 2, win_length=n_fft,
                      window=_hann(n_fft, wave.device), return_complex=True)


def istft_spec(spec: torch.Tensor, n_fft: int, length: int) -> torch.Tensor:
    """The inverse of stft_spec, trimmed to exactly `length` samples."""
    return torch.istft(spec, n_fft, hop_length=n_fft // 2, win_length=n_fft,
                       window=_hann(n_fft, spec.device), length=length)


def blend(algorithm: str, waves: list[torch.Tensor], n_fft: int, length: int) -> torch.Tensor:
    """Blend N (C, T) stems (same channel count) to (C, length).

    average: time-domain mean. max spec / min spec: UVR's per-bin magnitude selection in
    the STFT domain - at every bin the complex value of the input with the larger /
    smaller magnitude wins (folded pairwise, exactly UVR's np.where math)."""
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unknown ensemble algorithm {algorithm!r}: use one of {', '.join(ALGORITHMS)} (the algorithm widget)."
        )
    w = [x[:, :length] for x in waves]
    if algorithm == "average":
        return torch.stack(w).sum(0) / len(w)
    acc = stft_spec(w[0], n_fft)
    for x in w[1:]:
        s = stft_spec(x, n_fft)
        if algorithm == "max spec":
            acc = torch.where(s.abs() >= acc.abs(), s, acc)
        else:
            acc = torch.where(s.abs() <= acc.abs(), s, acc)
        del s
    return istft_spec(acc, n_fft, length)


def normalize_wave(wave: torch.Tensor, mode: str) -> torch.Tensor:
    """off: as-is; anti clip: scale only when the peak exceeds 1.0 (never raises a quiet
    result); full scale: always rescale to peak 1.0."""
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"Unknown normalize mode {mode!r}: use one of {', '.join(NORMALIZE_MODES)}.")
    if mode == "off":
        return wave
    peak = wave.abs().max()
    if peak <= 0 or (mode == "anti clip" and peak <= 1.0):
        return wave
    return wave / peak


def exact_residual(mix: torch.Tensor, stem: torch.Tensor) -> torch.Tensor:
    """mix - stem over the common length: the exact (time-domain) residual, so
    stem + output == mix holds sample for sample."""
    n = min(mix.shape[-1], stem.shape[-1])
    return mix[:, :n] - stem[:, :n]


def shift_wave(wave: torch.Tensor, offset: int, length: int) -> torch.Tensor:
    """out[n] = wave[n + offset] with zero fill (offset > 0: wave lags the reference)."""
    out = torch.zeros(wave.shape[0], length, dtype=wave.dtype)
    k = int(offset)
    if k >= 0:
        n = min(length - k, wave.shape[-1] - k)
        if n > 0:
            out[:, :n] = wave[:, k:k + n]
    else:
        n = min(length + k, wave.shape[-1])
        if n > 0:
            out[:, -k:] = wave[:, :n]
    return out


def align_offset(ref: torch.Tensor, wav: torch.Tensor, max_seconds: float, rate: int) -> int:
    """The number of samples by which `wav` lags `ref` (positive: wav's content starts
    later): the full-rate cross-correlation peak inside the +/-max_seconds radius, in one
    FFT per input pair (zero-padded circular correlation, read only inside its linear
    region). Full rate, not a downsampled template, so the result is sample-exact for
    any signal type - a 4x coarse stage would quantise the lag and lose on aperiodic
    audio. Both signals: (C, T) float32 at `rate`."""
    r = ref.mean(0)
    w = wav.mean(0)
    m = min(r.numel(), w.numel())
    t = max(1, min(int(max_seconds * rate), m // 2 - 1))
    if m < 8 or t < 1:
        return 0
    r, w = r[:m], w[:m]
    n = 1 << (2 * m - 1).bit_length()
    a = torch.fft.rfft(r, n)
    b = torch.fft.rfft(w, n)
    c = torch.fft.irfft(torch.conj(a) * b, n)
    # ifft(conj(A)B) is the cross-correlation with wav lagging by k peaking at c[k]
    # (verified against a direct sum; the mirror image at c[n - k] is the -k side), so
    # the lag is the argmax of c over [-t, t] minus t
    scores = torch.cat([c[n - t:n], c[0:1], c[1:t + 1]])
    return int(torch.argmax(scores) - t)
