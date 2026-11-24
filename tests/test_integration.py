"""Integration tests: download the real weights and run the nodes end to end.

Opt in with RUN_RF_WEIGHTS=1; weights download once into the models dir (default: a temp
dir) on first use, on ComfyUI's own device selection (auto - CUDA/MPS/CPU).
"""

import json
import os

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_RF_WEIGHTS") != "1",
        reason="set RUN_RF_WEIGHTS=1 to download weights and run integration tests",
    ),
]


def synth_mix(t=4.0, rate=44100):
    """A stereo mix at the model rate: a harmonic chord plus a high lead - enough content
    for a separator to pull apart."""
    n = int(t * rate)
    x = np.zeros(n, np.float32)
    for f, amp in ((261.63, 0.2), (329.63, 0.15), (392.0, 0.12), (440.0, 0.18), (880.0, 0.08)):
        seg = np.arange(n)
        env = np.clip(np.minimum(seg / 120.0, (n - seg) / 2400.0), 0.0, 1.0)
        x += amp * env * sum(np.sin(2 * np.pi * f * h * seg / rate) / h for h in (1, 2, 3))
    x *= 0.7
    wave = torch.from_numpy(x)[None, None].repeat(1, 2, 1)
    return {"waveform": wave, "sample_rate": rate}


def _separate(nodes, model_id: str):
    """Loader + Separate on the synthetic mix, on ComfyUI's own device selection (auto)."""
    (payload, stems_text) = nodes.RoformerLoader.execute(model_id, "auto").args
    (stems, target, other, info_json) = nodes.RoformerSeparate.execute(payload, synth_mix(), 0).args
    return payload, stems, target, other, json.loads(info_json)


@pytest.fixture(scope="module")
def sw6(nodes):
    return _separate(nodes, "sw-6stem")  # the default: 6 stems (bass/drums/other/voc/gtr/pno)


@pytest.fixture(scope="module")
def melband(nodes):
    return _separate(nodes, "melband-small")  # one mel-band family: the target plus the rest


def test_sw6_structure(sw6):
    payload, stems, target, other, info = sw6
    assert stems["sample_rate"] == 44100  # the synth is already at the model rate
    names = list(stems["stems"].keys())
    assert len(names) == 6 and {n.lower() for n in names} >= {"bass", "drums"}
    for name, w in stems["stems"].items():
        assert w.shape == (1, 2, int(4.0 * 44100)), name  # (B, C, N) in the payload; To Audio passes it through
        assert torch.isfinite(w).all() and w.dtype == torch.float32
    # the mix has no low content, so the bass stem must come back quiet - and everything is not silent
    assert float(stems["stems"]["bass"].abs().max()) < 0.01
    assert max(float(w.abs().max()) for w in stems["stems"].values()) > 0.01
    assert target["sample_rate"] == 44100 and other["sample_rate"] == 44100
    assert info["target"] in names and info["model"].startswith("SW 6-stem")
    assert info["seconds"] == pytest.approx(4.0, abs=0.05)
    assert payload["stems"] == names


def test_melband_structure(melband):
    payload, stems, target, other, info = melband
    names = list(stems["stems"].keys())
    assert len(names) >= 2  # mel-band models: the target plus the rest
    assert info["family"] == "melband"
    for name, w in stems["stems"].items():
        assert torch.isfinite(w).all() and float(w.abs().max()) > 1e-4, name


def test_stems_reconstruct_the_mix(sw6):
    """The stems decompose the input: their sum is the input (the model's residual
    guarantee), within resampling tolerance."""
    payload, stems, target, other, info = sw6
    total = sum(w for w in stems["stems"].values())
    src = synth_mix()["waveform"]  # (B, C, N) - the payload is batched
    assert total.shape == src.shape
    err = float((total - src).abs().max())
    assert err < 0.05, f"stems do not sum back to the input (max abs {err:.4f})"


def test_save_audio_both_formats(nodes, sw6, output_dir):
    payload, stems, target, other, info = sw6
    nodes.RoformerSaveAudio.execute(stems, "integration/wav", "wav")
    wav = list((output_dir / "integration").glob("wav_*.wav"))
    assert len(wav) == len(stems["stems"]), [str(p) for p in wav]
    nodes.RoformerSaveAudio.execute(stems, "integration/flac", "flac")
    # the native FLAC helper nests under the prefix subfolder: <out>/integration/flac/<stem>_000.flac
    flac = list((output_dir / "integration" / "flac").glob("*.flac"))
    assert len(flac) == len(stems["stems"]), [str(p) for p in flac]


def test_to_audio_nodes(nodes, sw6):
    payload, stems, target, other, info = sw6
    (audio,) = nodes.RoformerToAudio.execute(stems, "(auto)").args
    assert audio["waveform"].shape == (1, 2, int(4.0 * 44100))
    assert audio["sample_rate"] == 44100
    (audio,) = nodes.CustomRoformerToAudio.execute(stems, info["target"]).args
    assert audio["sample_rate"] == 44100
