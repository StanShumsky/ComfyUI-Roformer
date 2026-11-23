"""Unit tests: no weights download, no ComfyUI install, no network."""
import json

import numpy as np
import pytest
import torch


# --------------------------------------------------------------------- schemas


def test_node_schemas(nodes):
    """V3: every node's schema (node_id / display name / category / description / io ids) is consistent."""
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    assert set(nodes.NODE_CLASS_MAPPINGS) == {
        "RoformerLoader", "RoformerSeparate", "RoformerToAudio", "CustomRoformerToAudio",
        "RoformerSaveAudio", "RoformerInfo",
    }
    categories = set()
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        schema = cls.define_schema()
        assert schema.node_id == name
        assert schema.display_name == nodes.NODE_DISPLAY_NAME_MAPPINGS[name]
        categories.add(schema.category)
        assert isinstance(schema.description, str) and schema.description
        assert schema.inputs
        in_ids = [i.id for i in schema.inputs]
        assert len(in_ids) == len(set(in_ids))
        out_ids = [o.id for o in schema.outputs]
        assert len(out_ids) == len(set(out_ids))
    assert len(categories) == 1  # one menu category for the whole pack


def test_tooltips_and_aliases(nodes):
    """V3 UX baseline (the native geometry packs): hover text on every widget input, search
    aliases on every node but the loader, is_output_node on exactly the save node."""
    widget_types = {"FLOAT", "RANGEFLOAT", "INT", "BOOLEAN", "COMBO", "STRING", "AUDIO",
                    "IMAGE", "MASK", "AUTOGROW"}
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        schema = cls.define_schema()
        for inp in schema.inputs:
            io_type = getattr(type(inp), "io_type", None)
            if io_type is not None and io_type not in widget_types:
                continue  # a link input: the hover text travels with the producer's output
            assert inp.tooltip, (name, inp.id)
        if name.endswith("Loader"):
            continue
        assert schema.search_aliases, name
    assert {n for n, c in nodes.NODE_CLASS_MAPPINGS.items()
            if c.define_schema().is_output_node} == {"RoformerSaveAudio"}


def test_v3_entrypoint_lists_all_nodes(nodes):
    import asyncio

    import roformernode

    ext = asyncio.run(roformernode.comfy_entrypoint())
    node_list = asyncio.run(ext.get_node_list())
    assert [c.__name__ for c in node_list] == list(nodes.NODE_CLASS_MAPPINGS)


def test_loader_choices(nodes, models_dir, tiny_model):
    schema = nodes.RoformerLoader.define_schema()
    inputs = {i.id: i for i in schema.inputs}
    assert set(inputs) == {"model", "device"}
    choices = inputs["model"].options
    assert "leap-voc" in choices and "melband-small" in choices and "melband-big-beta7" in choices
    # a local collection folder appears in the dropdown
    assert "tiny" in choices
    assert inputs["device"].default == "auto"


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
    wrapper, stems_text = nodes.RoformerLoader.execute("tiny", "cpu").args
    assert wrapper["family"] == "bs" and wrapper["target"] == "vocals"
    assert wrapper["stems"] == ["vocals", "other"]
    assert stems_text == "vocals, other"  # the loader's text output, for previewing
    assert wrapper["num_channels"] == 2 and wrapper["sample_rate"] == 44100

    audio = make_audio(t=2.0, rate=44100, channels=2)
    out = nodes.RoformerSeparate.execute(wrapper, audio, 0)
    stems_payload, target_audio, other_audio, info_json = out.args
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
    wrapper, stems_text = nodes.RoformerLoader.execute("tiny", "cpu").args
    audio = make_audio(t=1.0, rate=48000, channels=2)
    stems_payload, _, _, _ = nodes.RoformerSeparate.execute(wrapper, audio, 0).args
    assert stems_payload["sample_rate"] == 48000
    for wave in stems_payload["stems"].values():
        assert wave.shape[1] == pytest.approx(48000, rel=0.005)  # soxr resample length


def test_separate_chunk_override(nodes, models_dir, tiny_model, make_audio):
    wrapper, stems_text = nodes.RoformerLoader.execute("tiny", "cpu").args
    audio = make_audio(t=6.0, rate=44100, channels=2)  # 3x the tiny 2 s chunk
    stems_payload, _, _, info_json = nodes.RoformerSeparate.execute(wrapper, audio, 2).args
    assert json.loads(info_json)["chunk_seconds"] == 2
    for wave in stems_payload["stems"].values():
        assert wave.shape[1] == 264600  # exact length in and out
    total = stems_payload["stems"]["vocals"] + stems_payload["stems"]["other"]
    assert torch.allclose(total, audio["waveform"][0], atol=2e-3)


def test_to_audio_and_info(nodes, models_dir, tiny_model, make_audio):
    wrapper, stems_text = nodes.RoformerLoader.execute("tiny", "cpu").args
    stems_payload, _, _, _ = nodes.RoformerSeparate.execute(wrapper, make_audio(t=1.0), 0).args
    (audio,) = nodes.RoformerToAudio.execute(stems_payload, "other").args
    assert audio["waveform"].shape[0] == 1 and audio["sample_rate"] == 44100
    # the dropdown default and a name the payload does not have both fall back to the first
    # stem, so a stale choice can never fail a run; matching is case-insensitive
    (auto,) = nodes.RoformerToAudio.execute(stems_payload, nodes.AUTO_STEM).args
    (fallback,) = nodes.RoformerToAudio.execute(stems_payload, "no-such-stem").args
    (upper,) = nodes.RoformerToAudio.execute(stems_payload, "OTHER").args
    assert torch.equal(upper["waveform"][0], stems_payload["stems"]["other"])
    assert torch.equal(auto["waveform"][0], stems_payload["stems"]["vocals"])
    assert torch.equal(fallback["waveform"][0], stems_payload["stems"]["vocals"])
    (text,) = nodes.RoformerInfo.execute(stems_payload).args
    assert json.loads(text)["stems"] == ["vocals", "other"]


def test_to_audio_dropdown(nodes):
    """Roformer To Audio: a dropdown seeded with every known stem name; Custom: free text."""
    stem = next(i for i in nodes.RoformerToAudio.define_schema().inputs if i.id == "stem")
    assert isinstance(stem, nodes.io.Combo.Input)
    assert stem.default == nodes.AUTO_STEM == stem.options[0]
    assert set(nodes.ALL_STEMS) <= set(stem.options) and len(stem.options) == len(set(stem.options))

    custom = next(i for i in nodes.CustomRoformerToAudio.define_schema().inputs if i.id == "stem")
    assert isinstance(custom, nodes.io.String.Input) and custom.default == ""


def test_stem_names_cover_the_manifest(manifest, stem_names):
    """The static map: one entry per curated checkpoint, canonical (lower case, aliases
    merged) unique names in dropdown order - priority first, vocals at the top of every
    list that has it, and no case-duplicates anywhere (vocals, not vocals and Vocals)."""
    stems = stem_names.MODEL_STEMS
    assert set(stems) == {m.id for m in manifest}
    for mid, names in stems.items():
        assert isinstance(names, tuple) and len(names) >= 2, mid
        assert len(names) == len(set(names)) and all(isinstance(n, str) and n for n in names), mid
        assert all(n == stem_names.canonical(n) for n in names), mid
        assert list(names) == stem_names.sort_stems(names), mid  # stored in dropdown order
        if "vocals" in names:
            assert names[0] == "vocals", mid
    all_names = {n for names in stems.values() for n in names}
    assert set(stem_names.ALL_STEMS) == all_names
    assert list(stem_names.ALL_STEMS) == stem_names.sort_stems(all_names)


def test_pick_stem_case_and_alias(nodes):
    """The dropdown is canonical lower case; a payload may use any spelling or capitalisation."""
    a = torch.arange(320).reshape(2, 160).float()
    b = a + 1000.0
    payload = {"stems": {"Vocals": a, "Instrumental": b}, "sample_rate": 44100}
    (vocals,) = nodes.RoformerToAudio.execute(payload, "vocals").args             # lower-case option
    (inst,) = nodes.CustomRoformerToAudio.execute(payload, "instrument").args     # alias for Instrumental
    (inst2,) = nodes.RoformerToAudio.execute(payload, "INSTRUMENTAL").args        # case-insensitive too
    assert torch.equal(vocals["waveform"][0], a)
    assert torch.equal(inst["waveform"][0], b) and torch.equal(inst2["waveform"][0], b)


def test_save_audio(nodes, output_dir, models_dir, tiny_model, make_audio):
    wrapper, stems_text = nodes.RoformerLoader.execute("tiny", "cpu").args
    stems_payload, _, _, _ = nodes.RoformerSeparate.execute(wrapper, make_audio(t=1.0), 0).args
    result = nodes.RoformerSaveAudio.execute(stems_payload, "unit/stems", "wav")
    assert result.args == ()  # no outputs, no preview: the node just writes the files
    files = sorted((output_dir / "unit").glob("stems_*.wav"))
    names = {f.stem.split("_")[-1] for f in files}
    assert names == {"vocals", "other"}  # every stem in the payload is saved
    import soundfile as sf
    data, rate = sf.read(str(files[0]))
    assert rate == 44100 and data.shape == (44100, 2)
    # a second run writes more files; saves never overwrite each other
    nodes.RoformerSaveAudio.execute(stems_payload, "unit/stems", "wav")
    assert len(list((output_dir / "unit").glob("stems_*.wav"))) == 4


def test_unknown_model_raises(nodes, models_dir):
    with pytest.raises(FileNotFoundError, match="not found"):
        nodes.RoformerLoader.execute("no-such-model", "cpu")


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
            nodes.RoformerLoader.execute("fake-refused", "cpu")
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
    """The stems payload shape the To Audio / Save Audio nodes consume:
    {'stems': {name: (channels, N) tensor at the file's own rate}, 'sample_rate', 'info'}."""
    wrapper, stems_text = nodes.RoformerLoader.execute("tiny", "cpu").args
    payload, target, other, info_json = nodes.RoformerSeparate.execute(
        wrapper, make_audio(t=0.2, rate=48000, freq=440.0), 0
    ).args
    assert set(payload) == {"stems", "sample_rate", "info"}
    assert payload["sample_rate"] == 48000  # the file's own rate, not the model's
    assert list(payload["stems"]) == ["vocals", "other"]
    n = int(0.2 * 48000)
    for name, wave in payload["stems"].items():
        assert torch.is_tensor(wave) and wave.dim() == 2 and wave.shape[0] == 2
        assert abs(wave.shape[1] - n) <= 2  # exact to within a soxr round-trip sample
    json.loads(info_json)
