"""Overlap-add chunking for audio longer than the model's training window.

The weights sum to one everywhere (normalised overlap-add with a linear crossfade over the
overlap), so there is no gain drift and the output length is exactly the input length. The
model's istft can drop a tail of each chunk's output; those samples are simply not written,
and the next window (whose fade-in covers the same region) takes over for them. tail_pad
zeros at the end of the input keep even the final window's dropped tail from landing at the
end of the song.
"""
from __future__ import annotations

import torch


def run_chunked(fn, x: torch.Tensor, chunk: int, hop: int, tail_pad: int = 0) -> torch.Tensor:
    """x: (..., T). fn maps a window x[..., s:s+chunk] -> (..., chunk) shaped output.

    Returns (..., T). For T + tail_pad <= chunk this is a single unwindowed pass.
    tail_pad: zero samples appended to the input before chunking so that tail samples fn
    drops (e.g. the model's istft losing a few per chunk) fall in the padding or in the
    crossfades instead of at the end of the song.
    """
    t = x.shape[-1]
    if tail_pad > 0:
        x = torch.cat([x, torch.zeros(*x.shape[:-1], tail_pad, device=x.device)], dim=-1)
    t_pad = x.shape[-1]
    if t_pad <= chunk:
        return fn(x)[..., :t]
    ov = max(1, chunk - hop)
    starts = list(range(0, t_pad - chunk + 1, hop))
    if starts[-1] != t_pad - chunk:
        starts.append(t_pad - chunk)

    y0 = fn(x[..., :chunk]).float()
    out = torch.zeros(*y0.shape[:-1], t_pad, device=x.device, dtype=torch.float32)
    wsum = torch.zeros((1,) * (y0.dim() - 1) + (t_pad,), device=x.device, dtype=torch.float32)
    for i, s in enumerate(starts):
        y = y0 if i == 0 else fn(x[..., s:s + chunk]).float()
        n = min(y.shape[-1], chunk)  # the model may drop a tail; the next window covers it
        w = torch.ones((1,) * (y0.dim() - 1) + (n,), device=x.device, dtype=torch.float32)
        if i > 0:
            w[..., :ov] = torch.linspace(0.0, 1.0, ov, device=x.device, dtype=torch.float32)
        if i < len(starts) - 1 and n == chunk:
            w[..., -ov:] = torch.linspace(1.0, 0.0, ov, device=x.device, dtype=torch.float32)
        out[..., s:s + n] += y[..., :n] * w
        wsum[..., s:s + n] += w
    return (out / wsum.clamp_min(1e-8))[..., :t]
