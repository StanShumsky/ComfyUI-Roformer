"""One Roformer checkpoint -> stems, following the checkpoint's own YAML geometry.

The model forward takes (1, C, T) at 44.1 kHz and returns (1, C, T) for a single stem or
(1, N, C, T) for N stems. Audio longer than the training chunk is processed in overlapping
windows (roformer.chunking) and the unmodelled instruments come back as the
mixture-minus-separation residual.
"""
from __future__ import annotations

import torch

import comfy.model_management as mm
from comfy.utils import ProgressBar

from .chunking import run_chunked


class Separator:
    def __init__(self, payload: dict):
        self.p = payload

    @staticmethod
    def chunk_plan(total: int, chunk: int) -> tuple[int, int, int]:
        """(chunk, hop, n_chunks) for a total of `total` samples at a `chunk`-sample window;
        the node uses it to size the progress bars before the run."""
        chunk = max(chunk, 1024)
        hop = max(1, int(chunk * 0.75))
        n_chunks = 1 if total <= chunk else max(1, (total - chunk) // hop + 1)
        return chunk, hop, n_chunks

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
    def run(self, x44: torch.Tensor, chunk_seconds: int, progress: ProgressBar | None = None,
            tqdm_bar=None, chunk: int | None = None) -> dict[str, torch.Tensor]:
        """x44: (C, T) float32 at the model's sample rate -> {stem name: (C, T) cpu float32}.
        `chunk` overrides the window (the OOM retry path halves it); `chunk_seconds` sets it
        when no override is given (0 = the model's own training window)."""
        p = self.p
        if chunk is None:
            chunk = p["chunk_size"] if chunk_seconds <= 0 else int(chunk_seconds * p["sample_rate"])
        chunk, hop, n_chunks = self.chunk_plan(x44.shape[-1], chunk)
        device = p["device"]
        xdev = x44.to(device=device, dtype=torch.float32)

        if progress is None:
            progress = ProgressBar(n_chunks + 1)

        def work(seg: torch.Tensor) -> torch.Tensor:
            y = self._forward(seg)
            progress.update(1)
            if tqdm_bar is not None:
                tqdm_bar.update(1)
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
        mm.soft_empty_cache()
        progress.update(1)
        if tqdm_bar is not None:
            tqdm_bar.update(1)
        return {k: v.cpu().contiguous() for k, v in stems.items()}
