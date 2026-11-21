"""Unit tests: no weights download, no ComfyUI install, no network."""
import json

import numpy as np
import pytest
import torch


# --------------------------------------------------------------------- schemas


def test_node_schemas(nodes):
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    assert set(nodes.NODE_CLASS_MAPPINGS) == {
        "RoformerLoader", "RoformerSeparate", "RoformerToAudio", "RoformerSaveAudio", "RoformerInfo",
    }
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert isinstance(cls.DESCRIPTION, str) and cls.DESCRIPTION
        assert cls.CATEGORY == "audio/Roformer"
        assert callable(getattr(cls, cls.FUNCTION))
        returns = cls.RETURN_TYPES
        assert len(returns) == len(getattr(cls, "RETURN_NAMES", returns))
        required = cls.INPUT_TYPES()["required"]
        assert required
        for spec in required.values():
            assert isinstance(spec, tuple) and len(spec) in (1, 2)
            if isinstance(spec[0], list):
                assert spec[0]


def test_loader_choices(nodes, models_dir, tiny_model):
    required = nodes.RoformerLoader.INPUT_TYPES()["required"]
    assert set(required) == {"model", "device"}
    choices = required["model"][0]
    assert "leap-voc" in choices and "melband-small" in choices and "melband-big-beta7" in choices
    # a local collection folder appears in the dropdown
    assert "tiny" in choices
    assert required["device"][0][0] == "auto"


# --------------------------------------------------------------------- audio


def test_audio_to_tensor(nodes, make_audio):
    x, rate = nodes.audio_to_tensor(make_audio(t=0.5, rate=48000, freq=220.0, channels=1))
    assert rate == 48000 and x.shape == (1, 24000) and x.dtype == torch.float32
    stereo = {"waveform": torch.zeros(1, 2, 100), "sample_rate": 16000}
    stereo["waveform"][0, 0] = 1.0
    x, _ = nodes.audio_to_tensor(stereo)
    assert x.shape == (2, 100) and np.allclose(x.numpy(), [[1.0] * 100, [0.0] * 100])
    with pytest.raises(ValueError):
        nodes.audio_to_tensor({"waveform": torch.zeros(1, 1, 0), "sample_rate": 44100})
    with pytest.raises(ValueError):
        nodes.audio_to_tensor({"waveform": torch.zeros(2, 10), "sample_rate": 0})


# --------------------------------------------------------------------- tiny model: the full real path


def test_load_and_separate(nodes, models_dir, tiny_model, make_audio):
    wrapper, = nodes.RoformerLoader().load("tiny", "cpu")
    assert wrapper["family"] == "bs" and wrapper["target"] == "vocals"
    assert wrapper["stems"] == ["vocals", "other"]
    assert wrapper["num_channels"] == 2 and wrapper["sample_rate"] == 44100

    audio = make_audio(t=2.0, rate=44100, channels=2)
    payload = nodes.RoformerSeparate().separate(wrapper, audio, 0)
    stems_payload, target_audio, other_audio, info_json = payload
    assert set(stems_payload["stems"]) == {"vocals", "other"}
    for name, wave in stems_payload["stems"].items():
        assert wave.shape == (2, 88200), name
    # the residual is mixture-minus-target, so target + other is the input exactly
    total = stems_payload["stems"]["vocals"] + stems_payload["stems"]["other"]
    assert torch.allclose(total, audio["waveform"][0], atol=1e-5)
    assert target_audio["sample_rate"] == 44100 and target_audio["waveform"].shape[0] == 1
    info = json.loads(info_json)
    assert info["target"] == "vocals" and info["seconds"] == pytest.approx(2.0, abs=0.01)
    assert info["model"] == "tiny (local)"


def test_separate_resamples_to_source_rate(nodes, models_dir, tiny_model, make_audio):
    wrapper, = nodes.RoformerLoader().load("tiny", "cpu")
    audio = make_audio(t=1.0, rate=48000, channels=2)
    stems_payload, _, _, _ = nodes.RoformerSeparate().separate(wrapper, audio, 0)
    assert stems_payload["sample_rate"] == 48000
    for wave in stems_payload["stems"].values():
        assert wave.shape[1] == pytest.approx(48000, rel=0.005)  # soxr resample length


def test_separate_chunk_override(nodes, models_dir, tiny_model, make_audio):
    wrapper, = nodes.RoformerLoader().load("tiny", "cpu")
    audio = make_audio(t=6.0, rate=44100, channels=2)  # 3x the tiny 2 s chunk
    stems_payload, _, _, info_json = nodes.RoformerSeparate().separate(wrapper, audio, 2)
    assert json.loads(info_json)["chunk_seconds"] == 2
    for wave in stems_payload["stems"].values():
        assert wave.shape[1] == 264600  # exact length in and out
    total = stems_payload["stems"]["vocals"] + stems_payload["stems"]["other"]
    assert torch.allclose(total, audio["waveform"][0], atol=2e-3)


def test_to_audio_and_info(nodes, models_dir, tiny_model, make_audio):
    wrapper, = nodes.RoformerLoader().load("tiny", "cpu")
    stems_payload, _, _, _ = nodes.RoformerSeparate().separate(wrapper, make_audio(t=1.0), 0)
    (audio,) = nodes.RoformerToAudio().to_audio(stems_payload, "other")
    assert audio["waveform"].shape[0] == 1 and audio["sample_rate"] == 44100
    (fallback,) = nodes.RoformerToAudio().to_audio(stems_payload, "no-such-stem")
    assert torch.equal(fallback["waveform"][0], stems_payload["stems"]["vocals"])
    (text,) = nodes.RoformerInfo().info(stems_payload)
    assert json.loads(text)["stems"] == ["vocals", "other"]


def test_save_audio(nodes, output_dir, models_dir, tiny_model, make_audio):
    wrapper, = nodes.RoformerLoader().load("tiny", "cpu")
    stems_payload, _, _, _ = nodes.RoformerSeparate().separate(wrapper, make_audio(t=1.0), 0)
    result = nodes.RoformerSaveAudio().save(stems_payload, "unit/stems", "", "wav")
    assert result == ()  # no outputs, no preview: the node just writes the files
    files = sorted((output_dir / "unit").glob("stems_*.wav"))
    names = {f.stem.split("_")[-1] for f in files}
    assert names == {"vocals", "other"}
    import soundfile as sf
    data, rate = sf.read(str(files[0]))
    assert rate == 44100 and data.shape == (44100, 2)
    # a single named stem writes one more file; saves never overwrite each other
    nodes.RoformerSaveAudio().save(stems_payload, "unit/stems", "vocals", "wav")
    assert len(list((output_dir / "unit").glob("stems_*.wav"))) == 3
    with pytest.raises(ValueError):
        nodes.RoformerSaveAudio().save(stems_payload, "unit/stems", "nope", "wav")


def test_unknown_model_raises(nodes, models_dir):
    with pytest.raises(FileNotFoundError, match="not found"):
        nodes.RoformerLoader().load("no-such-model", "cpu")


def test_unsupported_model_refused(nodes, models_dir):
    # the refusal path: an entry flagged supported=False must be rejected before any download.
    # (all shipped ids are currently supported; this exercises the mechanism with a fake entry)
    import importlib
    manifest_mod = importlib.import_module("roformernode.roformer.manifest")
    loading_mod = importlib.import_module("roformernode.roformer.loading")
    Checkpoint = manifest_mod.Checkpoint
    fake = Checkpoint("fake-refused", "Fake", "pcunwa/fake-repo", "x.ckpt", None,
                      "bs", 1.0, supported=False, note="test refusal")
    orig = loading_mod.BY_ID
    loading_mod.BY_ID = {**orig, "fake-refused": fake}
    try:
        with pytest.raises(ValueError, match="not supported"):
            nodes.RoformerLoader().load("fake-refused", "cpu")
    finally:
        loading_mod.BY_ID = orig


def test_manifest_families(nodes):
    import importlib
    manifest_mod = importlib.import_module("roformernode.roformer.manifest")
    by_id, models = manifest_mod.BY_ID, manifest_mod.MODELS
    assert len(models) == 41  # 36 pcunwa + 5 community (2 multi-stem, 3 specialized)
    assert all(m.supported for m in models)  # every checkpoint is loadable
    # each release lineage maps to the model code it was trained with
    expected_family = {
        "leap-voc": "bs", "revive": "bs", "resurrection-voc": "bs",
        "hyperace": "bs-hyperace", "hyperace-v2-voc": "bs-hyperace-v2",
        "large-inst": "bs-large-inst", "exp-siamese": "siamese",
        "inst-fno": "bs-fno",
        "melband-small": "melband", "kim-ft": "melband",
        "sw-6stem": "bs", "ft-4stems": "bs",  # community multi-stem, standard BSRoformer
        "karaoke-bs": "bs", "dereverb": "bs", "guitar-mel": "melband",
    }
    for mid, fam in expected_family.items():
        assert by_id[mid].family == fam, mid


# --------------------------------------------------------------------- stems payload contract


def test_stems_payload_contract(nodes, tiny_model, make_audio):
    """The stems payload shape other packs rely on - ComfyUI-Dettato's Transcribe accepts it
    directly: {'stems': {name: (channels, N) tensor at the file's own rate}, 'sample_rate',
    'info'}."""
    wrapper, = nodes.RoformerLoader().load("tiny", "cpu")
    payload, target, other, info_json = nodes.RoformerSeparate().separate(
        wrapper, make_audio(t=0.2, rate=48000, freq=440.0), 0
    )
    assert set(payload) == {"stems", "sample_rate", "info"}
    assert payload["sample_rate"] == 48000  # the file's own rate, not the model's
    assert list(payload["stems"]) == ["vocals", "other"]
    n = int(0.2 * 48000)
    for name, wave in payload["stems"].items():
        assert torch.is_tensor(wave) and wave.dim() == 2 and wave.shape[0] == 2
        assert abs(wave.shape[1] - n) <= 2  # exact to within a soxr round-trip sample
    json.loads(info_json)
