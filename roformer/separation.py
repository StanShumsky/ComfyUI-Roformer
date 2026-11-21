"""One Roformer checkpoint -> stems, following the checkpoint's own YAML geometry.

The model forward takes (1, C, T) at 44.1 kHz and returns (1, C, T) for a single stem or
(1, N, C, T) for N stems. Audio longer than the training chunk is processed in overlapping
windows (roformer.chunking) and the unmodelled instruments come back as the
mixture-minus-separation residual.
"""
from __future__ import annotations

import torch

from comfy.utils import ProgressBar

from .chunking import run_chunked


class Separator:
    def __init__(self, payload: dict):
        self.p = payload

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (C, T) at the model rate -> (stems, C, T) (stems >= 1)."""
        p = self.p
        y = p["net"](x[None])
        y = torch.as_tensor(y, dtype=torch.float32)
        if y.dim() == 3:      # (1, C, T) single stem: keep the leading stem dim
            return y
        if y.dim() == 4:      # (1, N, C, T)
            return y[0]
        raise ValueError(f"Unexpected model output shape {tuple(y.shape)}")

    @torch.inference_mode()
    def run(self, x44: torch.Tensor, chunk_seconds: int, progress: ProgressBar | None = None) -> dict[str, torch.Tensor]:
        """x44: (C, T) float32 at the model's sample rate -> {stem name: (C, T) cpu float32}."""
        p = self.p
        chunk = p["chunk_size"] if chunk_seconds <= 0 else int(chunk_seconds * p["sample_rate"])
        chunk = max(chunk, 1024)
        hop = max(1, int(chunk * 0.75))
        device = p["device"]
        xdev = x44.to(device=device, dtype=torch.float32)

        n_chunks = 1 if x44.shape[-1] <= chunk else max(1, (x44.shape[-1] - chunk) // hop + 1)
        if progress is None:
            progress = ProgressBar(n_chunks + 1)

        def work(seg: torch.Tensor) -> torch.Tensor:
            y = self._forward(seg)
            progress.update(1)
            return y

        tail_pad = int(p["cfg"].get("model", {}).get("stft_win_length", 2048))
        y = run_chunked(work, xdev, chunk, hop, tail_pad=tail_pad)

        stems: dict[str, torch.Tensor] = {}
        names = p["stems"]
        for i, name in enumerate(names):
            if i < y.shape[0]:
                stems[name] = y[i]
            elif y.shape[0] == 1 and len(stems) == 0:
                stems[name] = y[0]
        for j in range(len(names), y.shape[0]):  # more model outputs than named instruments
            stems[f"stem_{j + 1}"] = y[j]
        if len(stems) < len(names):
            # mixture-minus-separation residual for the instruments the model does not predict
            rest = [s for s in names if s not in stems]
            if rest:
                stems[rest[0]] = xdev - sum(stems.values())
        del xdev
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        progress.update(1)
        return {k: v.cpu().contiguous() for k, v in stems.items()}
