"""Chunking, separator and model-construction properties."""
import warnings

import torch

from roformernode.roformer.model.attend import Attend
from roformernode.roformer.chunking import run_chunked
from roformernode.roformer.model import BSRoformer
from roformernode.roformer.separation import Separator


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


def test_separator_forward_normalizes_batch_and_stems():
    payload = {"net": lambda x: x * 0.5, "precision": "float16"}
    x = torch.randn(3, 2, 4096)
    assert Separator(payload)._forward(x).shape == (1, 3, 2, 4096)

    payload["net"] = lambda x: torch.stack((x * 0.5, x * 0.25), dim=1)
    out = Separator(payload)._forward(x)
    assert out.shape == (2, 3, 2, 4096)
    assert torch.equal(out[0], x * 0.5)
    assert torch.equal(out[1], x * 0.25)

    payload["net"] = lambda x: torch.stack((x[..., :-37] * 0.5, x[..., :-37] * 0.25), dim=1)
    out = Separator(payload)._forward(x)
    assert out.shape == (2, 3, 2, 4059)


def test_separator_run_preserves_batch_and_legacy_unbatched_shape():
    payload = {
        "net": lambda x: x * 0.5,
        "device": torch.device("cpu"),
        "sample_rate": 44100,
        "chunk_size": 88200,
        "cfg": {"model": {}},
        "stems": ["vocals", "other"],
    }
    separator = Separator(payload)
    x = torch.randn(3, 2, 4096)
    batched = separator.run(x, 0)
    assert batched["vocals"].shape == x.shape
    assert torch.equal(batched["vocals"], x * 0.5)
    assert torch.equal(batched["vocals"] + batched["other"], x)

    unbatched = separator.run(x[0], 0)
    assert unbatched["vocals"].shape == x[0].shape
    assert torch.equal(unbatched["vocals"] + unbatched["other"], x[0])


def test_constructor_counts_stft_bins_without_a_probe(monkeypatch):
    def unexpected_stft(*args, **kwargs):
        raise AssertionError("model construction must not run a throwaway STFT")

    monkeypatch.setattr(torch, "stft", unexpected_stft)
    BSRoformer(
        dim=32, depth=1, stereo=True, num_stems=1,
        time_transformer_depth=1, freq_transformer_depth=1,
        freqs_per_bands=(3, 2), dim_head=16, heads=2,
        stft_n_fft=8, stft_hop_length=4, stft_win_length=8,
        mask_estimator_depth=1, mlp_expansion_factor=1, flash_attn=False,
    )


def test_attend_uses_current_sdpa_context():
    q = torch.randn(2, 2, 7, 8)
    attend = Attend(flash=True)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        out = attend(q, q, q)
    expected = torch.softmax(torch.einsum("bhid,bhjd->bhij", q, q) * 8 ** -0.5, dim=-1)
    expected = torch.einsum("bhij,bhjd->bhid", expected, q)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)
    assert not any("sdp_kernel" in str(item.message) for item in seen)
