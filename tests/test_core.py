"""Chunking / overlap-add properties: no model involved."""
import torch

from roformernode.roformer.chunking import run_chunked


def test_short_input_single_pass():
    x = torch.randn(2, 500)
    out = run_chunked(lambda s: s + 1.0, x, 10000, 7500)
    assert torch.equal(out, x + 1.0)


def test_length_and_normalised_reconstruction():
    x = torch.randn(2, 30000)
    out = run_chunked(lambda s: s * 2.0, x, 10000, 7500)
    assert out.shape == x.shape
    assert torch.allclose(out, x * 2.0, atol=1e-4)  # weights sum to one everywhere


def test_multi_stem_leading_dims():
    x = torch.randn(3, 2, 40000)  # (stems, C, T)
    out = run_chunked(lambda s: s.flip(-1), x, 8000, 6000)
    assert out.shape == x.shape


def test_truncated_chunk_output_is_padded():
    x = torch.full((2, 30000), 2.0)

    def fn(s):
        return s[..., : s.shape[-1] - 37]  # simulate istft tail loss

    # without tail padding the last 37 samples of the final chunk are lost at the end of
    # the signal (finite, but zero); with padding the loss falls in the pad
    raw = run_chunked(fn, x, 10000, 7500)
    assert raw.shape == x.shape and torch.isfinite(raw).all()
    assert torch.allclose(raw[..., :-37], x[..., :-37], atol=0.05)
    out = run_chunked(fn, x, 10000, 7500, tail_pad=400)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-4)
