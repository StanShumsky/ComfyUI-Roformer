"""ComfyUI AUDIO payloads <-> tensors, at the file's own sample rate.

Batched payloads (B, C, N) stay batched end to end: audio_to_tensor keeps the batch,
separation runs per batch element (one at a time on the GPU, staged in RAM), and the
stems payload carries (B, C, N) waves, so To Audio / Save Audio hand the batch through
or write one file per element.
"""
from __future__ import annotations

import torch


def audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    """AUDIO {waveform (B, C, N) or (C, N), sample_rate} -> float32 (B, C, N) cpu tensor
    (B = 1 for an unbatched input) and the rate."""
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    rate = audio.get("sample_rate")
    if rate is None:
        raise ValueError(
            "The audio input has no sample rate - a decoder that could not read it (PyAV leaves "
            "codec_context.sample_rate None on some files). Re-encode the file (e.g. ffmpeg) or use "
            "a different source node."
        )
    rate = int(rate)
    if rate <= 0:
        raise ValueError(f"Audio sample rate must be positive, got {rate}.")
    x = waveform.detach().float()
    if x.dim() == 2:
        x = x[None]
    if x.dim() != 3:
        raise ValueError(
            f"Audio waveform must be (channels, samples) or (batch, channels, samples), got {tuple(x.shape)}."
        )
    if x.shape[-1] == 0:
        raise ValueError("The audio is empty.")
    return x.cpu().contiguous(), rate


def resample(x: torch.Tensor, src_rate: int, dst_rate: int) -> torch.Tensor:
    """float32 (B, C, N) resampled with soxr HQ (one element at a time), staying on its
    device."""
    if src_rate == dst_rate:
        return x
    import soxr
    out = []
    for e in x:  # soxr takes (samples, channels), not (channels, samples)
        y = soxr.resample(e.cpu().numpy().T, src_rate, dst_rate, quality="HQ").astype("float32").T
        out.append(torch.from_numpy(y).to(x.device))
    return torch.stack(out)


def to_audio_payload(wave: torch.Tensor, rate: int) -> dict:
    """(C, N) or (B, C, N) -> ComfyUI AUDIO payload (batched, channels-first,
    LoadAudio-compatible)."""
    wave = wave.float().cpu()
    if wave.dim() == 2:
        wave = wave[None]
    return {"waveform": wave.contiguous(), "sample_rate": int(rate)}
