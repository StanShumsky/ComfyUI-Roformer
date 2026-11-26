"""ComfyUI AUDIO payloads <-> tensors, at the file's own sample rate.

Batched payloads (B, C, N) stay batched end to end: audio_to_tensor keeps the batch,
separation runs whole device batches with automatic sub-batch fallback, and the stems
payload carries (B, C, N) waves, so To Audio / Save Audio hand the batch through or write
one file per element.
"""
from __future__ import annotations

import torch

import comfy.model_management as mm


def audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    """AUDIO {waveform (B, C, N) or (C, N), sample_rate} -> float32 (B, C, N) cpu tensor
    (B = 1 for an unbatched input) and the rate."""
    mm.throw_exception_if_processing_interrupted()
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
    if x.shape[0] == 0 or x.shape[1] == 0 or x.shape[-1] == 0:
        raise ValueError(
            f"The audio has an empty batch, channel, or time dimension: {tuple(x.shape)}."
        )
    x = x.cpu().contiguous()
    mm.throw_exception_if_processing_interrupted()
    return x, rate


def resample(x: torch.Tensor, src_rate: int, dst_rate: int,
             length: int | None = None) -> torch.Tensor:
    """float32 (B, C, N) resampled with soxr HQ (one batch element at a time), staying on its
    device. ``length`` pads or trims the result to an exact sample count."""
    if src_rate == dst_rate:
        out = x
    else:
        import soxr
        out = []
        for e in x:  # soxr takes (samples, channels), not (channels, samples)
            mm.throw_exception_if_processing_interrupted()
            y = soxr.resample(e.cpu().numpy().T, src_rate, dst_rate, quality="HQ").astype("float32").T
            out.append(torch.from_numpy(y).to(x.device))
        mm.throw_exception_if_processing_interrupted()
        out = torch.stack(out)
        if out.shape[-1] == 0:
            out = torch.nn.functional.pad(out, (0, 1))
    if length is not None:
        if out.shape[-1] < length:
            out = torch.nn.functional.pad(out, (0, length - out.shape[-1]))
        else:
            out = out[..., :length]
    return out


def to_audio_payload(wave: torch.Tensor, rate: int) -> dict:
    """(C, N) or (B, C, N) -> ComfyUI AUDIO payload (batched, channels-first,
    LoadAudio-compatible)."""
    wave = wave.float().cpu()
    if wave.dim() == 2:
        wave = wave[None]
    return {"waveform": wave.contiguous(), "sample_rate": int(rate)}
