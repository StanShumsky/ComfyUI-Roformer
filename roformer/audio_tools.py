"""The audio-tool math behind the pitch / tempo / key nodes: librosa for the transforms
(phase-vocoder pitch shift and time stretch) and a Krumhansl-Schmuckler chroma profile
match for the key."""
from __future__ import annotations

import torch

# ------------------------------------------------- pitch & tempo

def pitch_shift(wave: torch.Tensor, sr: int, semitones: float, time_correct: bool) -> torch.Tensor:
    """Shift `wave` (C, N) by `semitones` (0.5 steps allowed), staying at `sr`.

    time_correct: a phase-vocoder shift (librosa.effects.pitch_shift) - the pitch moves,
    the length does not.
    Not time-corrected: the classic resample - the pitch moves
    AND the length changes by 2 ** (-semitones / 12)."""
    import librosa
    if time_correct:
        y = librosa.effects.pitch_shift(wave.numpy(), sr=sr, n_steps=semitones)
        return torch.from_numpy(asfloat32(y))
    # raise the pitch by resampling DOWN (fewer samples, played back at sr): the new
    # length is N * 2 ** (-semitones / 12)
    factor = 2 ** (semitones / 12)
    target = sr / factor
    out = [torch.from_numpy(asfloat32(librosa.resample(
        wave[ch].numpy(), orig_sr=sr, target_sr=target, res_type="soxr_hq")))
        for ch in range(wave.shape[0])]
    return torch.stack(out)


def time_stretch(wave: torch.Tensor, sr: int, rate: float) -> torch.Tensor:
    """Stretch `wave` (C, N) by `rate` (1.0 = as-is, > 1 = shorter), pitch preserved -
    a phase-vocoder pass at `sr`."""
    import librosa
    y = librosa.effects.time_stretch(wave.numpy(), rate=rate)
    return torch.from_numpy(asfloat32(y))


def asfloat32(y):
    import numpy as np
    return np.ascontiguousarray(np.asarray(y, dtype=np.float32))


# ------------------------------------------------- key detection

#: Krumhansl-Schmuckler key profiles (circular, C = index 0)
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key(wave: torch.Tensor, sr: int) -> tuple[str, float]:
    """The (key, confidence) of `wave` (C, N): the mean chroma-CQT pitch-class profile
    correlated against all 24 Krumhansl-Schmuckler key profiles (major + minor at each
    tonic) - the standard key-estimation recipe. Confidence is the best Pearson r
    (typical range 0.5-0.9 on clean vocals)."""
    import librosa
    import numpy as np
    y = wave.mean(0).numpy()
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)  # (12, frames)
    profile = chroma.mean(axis=1)
    p = torch.from_numpy(profile.astype("float32"))
    best_key, best_r = "", 0.0
    for tonic in range(12):
        for label, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            # profile[(pc - tonic) mod 12]: the tonic's pitch class gets the tonic weight
            rotated = torch.tensor(prof[(12 - tonic):] + prof[:12 - tonic], dtype=torch.float32)
            a = p - p.mean()
            b = rotated - rotated.mean()
            denom = a.norm() * b.norm()
            if denom == 0:
                continue
            r = float((a * b).sum() / denom)
            if r > best_r:
                best_r = r
                best_key = f"{PITCH_CLASSES[tonic]} {label}"
    return best_key, best_r
