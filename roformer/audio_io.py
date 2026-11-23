"""ComfyUI AUDIO payloads <-> tensors, at the file's own sample rate."""
from __future__ import annotations

import torch


def audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    """AUDIO {waveform (B, C, N) or (C, N), sample_rate} -> float32 (C, N) cpu tensor and the rate."""
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
    if x.dim() == 3:
        x = x[0] if x.shape[0] == 1 else x.mean(0)
    if x.dim() != 2:
        raise ValueError(
            f"Audio waveform must be (channels, samples) or (1, channels, samples), got {tuple(x.shape)}."
        )
    if x.shape[-1] == 0:
        raise ValueError("The audio is empty.")
    return x.cpu().contiguous(), rate


def resample(x: torch.Tensor, src_rate: int, dst_rate: int) -> torch.Tensor:
    """float32 (C, N) resampled with soxr HQ, staying on its device."""
    if src_rate == dst_rate:
        return x
    import soxr
    # soxr takes (samples, channels), not (channels, samples)
    y = soxr.resample(x.cpu().numpy().T, src_rate, dst_rate, quality="HQ").astype("float32").T
    return torch.from_numpy(y).to(x.device)


def to_audio_payload(wave: torch.Tensor, rate: int) -> dict:
    """(C, N) -> ComfyUI AUDIO payload (batched, channels-first, LoadAudio-compatible)."""
    wave = wave.float().cpu()
    return {"waveform": wave[None].contiguous(), "sample_rate": int(rate)}
