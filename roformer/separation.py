"""One Roformer checkpoint -> stems, following the checkpoint's own YAML geometry.

The model forward takes (B, C, T) at 44.1 kHz and returns (B, C, T) for a single stem or
(B, N, C, T) for N stems. Audio longer than the training chunk is processed in overlapping
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
        """x: (B, C, T) at the model rate -> (stems, B, C, T) (stems >= 1)."""
        p = self.p
        if p.get("precision", "float32") == "float16" and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = torch.as_tensor(p["net"](x), dtype=torch.float32)
        else:
            y = torch.as_tensor(p["net"](x), dtype=torch.float32)
        if y.dim() == 3:      # (B, C, T) single stem
            if y.shape[0] != x.shape[0] or y.shape[1] != x.shape[1] or y.shape[-1] > x.shape[-1]:
                raise ValueError(
                    f"Model changed the batch or audio shape: input {tuple(x.shape)}, "
                    f"output {tuple(y.shape)}"
                )
            return y.unsqueeze(0)
        if y.dim() == 4:      # (B, N, C, T) -> (N, B, C, T)
            if (y.shape[0] != x.shape[0] or y.shape[2] != x.shape[1]
                    or y.shape[-1] > x.shape[-1]):
                raise ValueError(
                    f"Model changed the batch or audio shape: input {tuple(x.shape)}, "
                    f"output {tuple(y.shape)}"
                )
            return y.permute(1, 0, 2, 3)
        raise ValueError(f"Unexpected model output shape {tuple(y.shape)}")

    @torch.inference_mode()
    def run(self, x44: torch.Tensor, chunk_seconds: int, progress: ProgressBar | None = None,
            tqdm_bar=None, chunk: int | None = None) -> dict[str, torch.Tensor]:
        """x44: (B, C, T) float32 at the model's sample rate -> {stem name: (B, C, T) cpu float32}.
        A legacy (C, T) input keeps its unbatched output shape. `chunk` overrides the window
        (the OOM retry path halves it); `chunk_seconds` sets it when no override is given
        (0 = the model's own training window)."""
        mm.throw_exception_if_processing_interrupted()
        p = self.p
        batched = x44.ndim == 3
        if not batched and x44.ndim != 2:
            raise ValueError(f"Expected (B, C, T) or (C, T) audio, got {tuple(x44.shape)}")
        if chunk is None:
            chunk = p["chunk_size"] if chunk_seconds <= 0 else int(chunk_seconds * p["sample_rate"])
        chunk, hop, n_chunks = self.chunk_plan(x44.shape[-1], chunk)
        device = p["device"]
        xdev = x44.to(device=device, dtype=torch.float32)
        if not batched:
            xdev = xdev.unsqueeze(0)

        if progress is None:
            progress = ProgressBar(n_chunks + 1)

        def work(seg: torch.Tensor) -> torch.Tensor:
            mm.throw_exception_if_processing_interrupted()
            y = self._forward(seg)
            mm.throw_exception_if_processing_interrupted()
            progress.update(1)
            if tqdm_bar is not None:
                tqdm_bar.update(1)
            return y

        tail_pad = int(p["cfg"].get("model", {}).get("stft_win_length", 2048))
        try:
            y = run_chunked(work, xdev, chunk, hop, tail_pad=tail_pad).permute(1, 0, 2, 3)

            stems: dict[str, torch.Tensor] = {}
            names = p["stems"]
            for i, name in enumerate(names):
                if i < y.shape[1]:
                    stems[name] = y[:, i]
                elif y.shape[1] == 1 and len(stems) == 0:
                    stems[name] = y[:, 0]
            for j in range(len(names), y.shape[1]):
                stems[f"stem_{j + 1}"] = y[:, j]
            if len(stems) < len(names):
                # Unpredicted instruments are the exact mixture-minus-separation residual.
                rest = [s for s in names if s not in stems]
                if rest:
                    stems[rest[0]] = xdev - sum(stems.values())
            result = {k: v.cpu().contiguous() for k, v in stems.items()}
            if not batched:
                result = {k: v[0] for k, v in result.items()}
        finally:
            del xdev
            mm.soft_empty_cache()
        progress.update(1)
        if tqdm_bar is not None:
            tqdm_bar.update(1)
        return result
