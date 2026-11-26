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
        "RoformerLoader", "RoformerSeparate", "RoformerEnsemble", "RoformerAlign",
        "RoformerPitchShift", "RoformerTimeStretch", "RoformerKeyDetect",
        "RoformerToAudio", "CustomRoformerToAudio", "RoformerSaveAudio", "RoformerInfo",
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
            # dynamic inputs (autogrow): the generated slots are copies of the template input,
            # so its tooltip is what the user hovers - it counts as the input's
            template = getattr(inp, "template", None)
            if template is not None:
                assert getattr(template.input, "tooltip", None), (name, inp.id)
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
    assert set(inputs) == {"model", "device", "precision"}
    choices = inputs["model"].options
    assert "leap-voc" in choices and "melband-small" in choices and "melband-big-beta7" in choices
    # a local collection folder appears in the dropdown
    assert "tiny" in choices
    assert inputs["device"].default == "auto"
    assert inputs["precision"].options == ["float32", "float16"]
    assert inputs["precision"].default == "float32"


def test_known_model_reuses_complete_local_files(nodes, models_dir, monkeypatch):
    import importlib
    import sys
    import types

    manifest_mod = importlib.import_module("roformernode.roformer.manifest")
    loading = importlib.import_module("roformernode.roformer.loading")
    model = manifest_mod.Checkpoint(
        "local-known", "Local known", "test/repo", "weights/model.ckpt", "weights/config.yaml", "bs", 1.0
    )
    monkeypatch.setitem(loading.BY_ID, model.id, model)
    target = models_dir / model.id / "weights"
    target.mkdir(parents=True)
    ckpt, config = target / "model.ckpt", target / "config.yaml"
    ckpt.write_bytes(b"weights")
    config.write_text("audio: {}\nmodel: {}\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace())

    found_ckpt, found_config, found_model = loading._locate(model.id)

    assert (found_ckpt, found_config, found_model) == (ckpt, config, model)


def test_known_model_reuses_renamed_local_checkpoint(nodes, models_dir, monkeypatch):
    import importlib

    manifest_mod = importlib.import_module("roformernode.roformer.manifest")
    loading = importlib.import_module("roformernode.roformer.loading")
    model = manifest_mod.Checkpoint(
        "renamed-known", "Renamed", "test/repo", "published.ckpt", "published.yaml", "bs", 1.0
    )
    monkeypatch.setitem(loading.BY_ID, model.id, model)
    target = models_dir / model.id
    target.mkdir()
    ckpt, config = target / "community.ckpt", target / "config.yaml"
    ckpt.write_bytes(b"weights")
    config.write_text("audio: {}\nmodel: {}\n", encoding="utf-8")

    def unexpected_download(*args, **kwargs):
        pytest.fail("a complete local model must not contact Hugging Face")

    monkeypatch.setattr(loading, "download_model", unexpected_download)

    found_ckpt, found_config, found_model = loading._locate(model.id)

    assert (found_ckpt, found_config, found_model) == (ckpt, config, model)


def test_incomplete_known_model_downloads_missing_files(nodes, models_dir, monkeypatch):
    import importlib

    manifest_mod = importlib.import_module("roformernode.roformer.manifest")
    loading = importlib.import_module("roformernode.roformer.loading")
    model = manifest_mod.Checkpoint(
        "incomplete-known", "Incomplete", "test/repo", "model.ckpt", "config.yaml", "bs", 1.0
    )
    monkeypatch.setitem(loading.BY_ID, model.id, model)
    target = models_dir / model.id
    target.mkdir()
    (target / model.ckpt_file).write_bytes(b"partial weights")
    calls = []

    def download(found_model, root):
        calls.append((found_model, root))
        return target

    monkeypatch.setattr(loading, "download_model", download)

    ckpt, config, found_model = loading._locate(model.id)

    assert calls == [(model, models_dir)]
    assert ckpt == target / model.ckpt_file
    assert config == target / model.config_file
    assert found_model is model


def test_download_failure_explains_offline_install(models_dir, monkeypatch):
    import importlib
    import sys
    import types

    loading = importlib.import_module("roformernode.roformer.loading")
    model = loading.BY_ID["sw-6stem"]

    def fail_download(**kwargs):
        raise OSError("network unavailable")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fail_download))

    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE") as error:
        loading.download_model(model, models_dir)

    assert str(models_dir / model.id) in str(error.value)
    assert model.ckpt_file in str(error.value)


def test_loader_honors_interrupt_before_model_lookup(nodes, monkeypatch):
    import importlib
    import comfy.model_management as mm

    loading = importlib.import_module("roformernode.roformer.loading")

    def interrupted():
        raise mm.InterruptProcessingException()

    def unexpected_lookup(*args):
        pytest.fail("an interrupted loader must not locate or download weights")

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupted)
    monkeypatch.setattr(loading, "_locate", unexpected_lookup)

    with pytest.raises(mm.InterruptProcessingException):
        nodes.RoformerLoader.execute("sw-6stem", "cpu")


# --------------------------------------------------------------------- audio


def test_audio_to_tensor(nodes, make_audio):
    x, rate = nodes.audio_to_tensor(make_audio(t=0.5, rate=48000, freq=220.0, channels=1))
    assert rate == 48000 and x.shape == (1, 1, 24000) and x.dtype == torch.float32
    stereo = {"waveform": torch.zeros(1, 2, 100), "sample_rate": 16000}
    stereo["waveform"][0, 0] = 1.0
    x, _ = nodes.audio_to_tensor(stereo)
    assert x.shape == (1, 2, 100) and np.allclose(x.numpy()[0], [[1.0] * 100, [0.0] * 100])
    batched = {"waveform": torch.zeros(2, 2, 50), "sample_rate": 44100}
    x, _ = nodes.audio_to_tensor(batched)
    assert x.shape == (2, 2, 50)  # the batch survives, for batched separation
    with pytest.raises(ValueError):
        nodes.audio_to_tensor({"waveform": torch.zeros(1, 1, 0), "sample_rate": 44100})
    with pytest.raises(ValueError, match="empty batch"):
        nodes.audio_to_tensor({"waveform": torch.zeros(0, 2, 100), "sample_rate": 44100})
    with pytest.raises(ValueError, match="empty.*channel"):
        nodes.audio_to_tensor({"waveform": torch.zeros(1, 0, 100), "sample_rate": 44100})
    with pytest.raises(ValueError):
        nodes.audio_to_tensor({"waveform": torch.zeros(2, 10), "sample_rate": 0})


def test_resample_preserves_exact_length_and_nonempty_output():
    from roformernode.roformer.audio_io import resample

    for rate in (88200, 96000):
        for n in (1, 1025, 48001):
            x = torch.randn(2, 2, n)
            y = resample(resample(x, rate, 44100), 44100, rate, length=n)
            assert y.shape == x.shape and torch.isfinite(y).all()
    assert resample(torch.zeros(1, 2, 1), 96000, 44100).shape[-1] == 1


def test_resample_honors_interrupt_between_batch_elements(monkeypatch):
    import comfy.model_management as mm
    from roformernode.roformer.audio_io import resample

    calls = 0

    def interrupted():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupted)

    with pytest.raises(mm.InterruptProcessingException):
        resample(torch.zeros(2, 1, 1000), 44100, 48000)

    assert calls == 2


# --------------------------------------------------------------------- tiny model: the full real path


def test_load_and_separate(nodes, models_dir, tiny_model, make_audio):
    wrapper, stems_text = nodes.RoformerLoader.execute("tiny", "cpu").args
    assert wrapper["family"] == "bs" and wrapper["target"] == "vocals"
    assert wrapper["stems"] == ["vocals", "other"]
    assert stems_text == "vocals, other"  # the loader's text output, for previewing
    assert wrapper["num_channels"] == 2 and wrapper["sample_rate"] == 44100
    assert wrapper["precision"] == "float32"
    with pytest.raises(ValueError, match="precision"):
        nodes.RoformerLoader.execute("tiny", "cpu", "bfloat16")

    audio = make_audio(t=2.0, rate=44100, channels=2)
    out = nodes.RoformerSeparate.execute(wrapper, audio, 0)
    stems_payload, target_audio, other_audio, info_json = out.args
    assert set(stems_payload["stems"]) == {"vocals", "other"}
    for name, wave in stems_payload["stems"].items():
        assert wave.shape == (1, 2, 88200), name  # (B, C, N), B = 1 for a single song
    # the residual is mixture-minus-target, so target + other is the input exactly
    total = stems_payload["stems"]["vocals"] + stems_payload["stems"]["other"]
    assert torch.allclose(total, audio["waveform"], atol=1e-5)
    assert target_audio["sample_rate"] == 44100 and target_audio["waveform"].shape[0] == 1
    info = json.loads(info_json)
    assert info["target"] == "vocals" and info["seconds"] == pytest.approx(2.0, abs=0.01)
    assert info["model"] == "tiny (local)"
    assert info["tta"] is False and info["batch"] == 1


def test_float16_precision_keeps_cpu_path_float32(nodes, models_dir, tiny_model, make_audio):
    audio = make_audio(t=0.25)
    default, _ = nodes.RoformerLoader.execute("tiny", "cpu", "float32").args
    fp16, _ = nodes.RoformerLoader.execute("tiny", "cpu", "float16").args
    expected = nodes.RoformerSeparate.execute(default, audio, 0).args[0]["stems"]
    actual = nodes.RoformerSeparate.execute(fp16, audio, 0).args[0]["stems"]
    assert fp16["precision"] == "float16"
    assert all(torch.equal(actual[name], wave) for name, wave in expected.items())


def test_separate_resamples_to_source_rate(nodes, models_dir, tiny_model, make_audio):
    wrapper, stems_text = nodes.RoformerLoader.execute("tiny", "cpu").args
    audio = make_audio(t=1.0, rate=48000, channels=2)
    stems_payload, _, _, _ = nodes.RoformerSeparate.execute(wrapper, audio, 0).args
    assert stems_payload["sample_rate"] == 48000
    for wave in stems_payload["stems"].values():
        assert wave.shape[2] == pytest.approx(48000, rel=0.005)  # soxr resample length


def test_separate_preserves_single_sample_at_96k(nodes, models_dir, tiny_model):
    wrapper, _ = nodes.RoformerLoader.execute("tiny", "cpu").args
    audio = {"waveform": torch.ones(1, 2, 1), "sample_rate": 96000}
    payload, _, _, _ = nodes.RoformerSeparate.execute(wrapper, audio, 0).args
    assert payload["sample_rate"] == 96000
    for wave in payload["stems"].values():
        assert wave.shape == (1, 2, 1) and torch.isfinite(wave).all()


def test_separate_downmixes_multichannel_to_stereo(nodes, models_dir, tiny_model):
    wrapper, _ = nodes.RoformerLoader.execute("tiny", "cpu").args

    class HalfNet(torch.nn.Module):
        def forward(self, x):
            return x * 0.5

    wrapper["net"] = HalfNet()
    wave = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1).repeat(1, 1, 4096) / 10
    payload, _, _, info = nodes.RoformerSeparate.execute(
        wrapper, {"waveform": wave, "sample_rate": 44100}, 0
    ).args
    expected = wave.mean(1, keepdim=True).repeat(1, 2, 1)
    assert payload["stems"]["vocals"].shape == (1, 2, 4096)
    assert torch.allclose(payload["stems"]["vocals"], expected * 0.5)
    assert torch.allclose(payload["stems"]["vocals"] + payload["stems"]["other"], expected)
    assert json.loads(info)["channels_in"] == 6


def test_separate_chunk_override(nodes, models_dir, tiny_model, make_audio):
    wrapper, stems_text = nodes.RoformerLoader.execute("tiny", "cpu").args
    audio = make_audio(t=6.0, rate=44100, channels=2)  # 3x the tiny 2 s chunk
    stems_payload, _, _, info_json = nodes.RoformerSeparate.execute(wrapper, audio, 2).args
    assert json.loads(info_json)["chunk_seconds"] == 2
    for wave in stems_payload["stems"].values():
        assert wave.shape == (1, 2, 264600)  # exact length in and out
    total = stems_payload["stems"]["vocals"] + stems_payload["stems"]["other"]
    assert torch.allclose(total, audio["waveform"], atol=2e-3)


def test_separate_interrupts_after_forward_and_cleans_cache(nodes, models_dir, tiny_model,
                                                           make_audio, monkeypatch):
    import comfy.model_management as mm

    wrapper, _ = nodes.RoformerLoader.execute("tiny", "cpu").args
    interrupted = False
    forwards = 0
    cache_clears = 0

    def forward_hook(module, args, output):
        nonlocal interrupted, forwards
        forwards += 1
        interrupted = True

    def check_interrupt():
        if interrupted:
            raise mm.InterruptProcessingException()

    def clear_cache():
        nonlocal cache_clears
        cache_clears += 1

    hook = wrapper["net"].register_forward_hook(forward_hook)
    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", check_interrupt)
    monkeypatch.setattr(mm, "soft_empty_cache", clear_cache)
    try:
        with pytest.raises(mm.InterruptProcessingException):
            nodes.RoformerSeparate.execute(wrapper, make_audio(t=6.0), 2)
    finally:
        hook.remove()

    assert forwards == 1
    assert cache_clears == 1


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
    assert torch.equal(upper["waveform"][0], stems_payload["stems"]["other"][0])
    assert torch.equal(auto["waveform"][0], stems_payload["stems"]["vocals"][0])
    assert torch.equal(fallback["waveform"][0], stems_payload["stems"]["vocals"][0])
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
        assert torch.is_tensor(wave) and wave.dim() == 3
        assert wave.shape[0] == 1 and wave.shape[1] == 2
        assert abs(wave.shape[2] - n) <= 2  # exact to within a soxr round-trip sample
    json.loads(info_json)


def test_separate_batched_audio(nodes, models_dir, tiny_model):
    """A batched (B, C, N) input separates every element; the payload comes back batched."""
    wrapper, _ = nodes.RoformerLoader.execute("tiny", "cpu").args
    a1 = torch.sin(2 * 3.141592653589793 * 440 * torch.arange(44100) / 44100)
    a2 = torch.sin(2 * 3.141592653589793 * 880 * torch.arange(44100) / 44100)
    audio = {"waveform": torch.stack([a1, a2]).unsqueeze(1).repeat(1, 2, 1), "sample_rate": 44100}
    reference = []
    for wave in (a1, a2):
        single = {"waveform": wave[None, None].repeat(1, 2, 1), "sample_rate": 44100}
        reference.append(nodes.RoformerSeparate.execute(wrapper, single, 0).args[0]["stems"])
    calls = 0

    def count_forward(*args):
        nonlocal calls
        calls += 1

    hook = wrapper["net"].register_forward_pre_hook(count_forward)
    try:
        payload, target, other, info_json = nodes.RoformerSeparate.execute(wrapper, audio, 0).args
    finally:
        hook.remove()
    info = json.loads(info_json)
    assert info["batch"] == 2
    assert calls == 1  # the whole batch reaches the model in one forward
    for name, wave in payload["stems"].items():
        assert wave.shape == (2, 2, 44100), name  # (B, C, N)
        for b in range(2):
            assert torch.allclose(wave[b], reference[b][name], atol=1e-5, rtol=1e-5)
    assert target["waveform"].shape == (2, 2, 44100)
    # element 1's residual property still holds inside the batch
    total = payload["stems"]["vocals"][1] + payload["stems"]["other"][1]
    assert torch.allclose(total, audio["waveform"][1], atol=1e-4)
    # and To Audio hands the batch straight through
    (vocals,) = nodes.RoformerToAudio.execute(payload, "vocals").args
    assert vocals["waveform"].shape == (2, 2, 44100)


def test_separate_splits_batch_after_oom(nodes, models_dir, tiny_model, make_audio):
    wrapper, _ = nodes.RoformerLoader.execute("tiny", "cpu").args
    wrapper["chunk_size"] = 44100
    calls = []

    class LimitedNet(torch.nn.Module):
        def forward(self, x):
            calls.append(tuple(x.shape))
            if x.shape[0] > 2:
                raise torch.OutOfMemoryError("test batch limit")
            return x * 0.5

    wrapper["net"] = LimitedNet()
    audio = make_audio(t=0.5, rate=44100, channels=2)
    audio["waveform"] = audio["waveform"].repeat(8, 1, 1)
    payload, _, _, info_json = nodes.RoformerSeparate.execute(wrapper, audio, 0).args

    assert json.loads(info_json)["batch"] == 8
    assert [shape[0] for shape in calls] == [4, 2, 2, 2, 2]
    assert payload["stems"]["vocals"].shape == (8, 2, 22050)
    assert torch.allclose(
        payload["stems"]["vocals"] + payload["stems"]["other"], audio["waveform"], atol=1e-5
    )


def test_separate_halves_window_before_splitting_batch(nodes, models_dir, tiny_model, make_audio):
    wrapper, _ = nodes.RoformerLoader.execute("tiny", "cpu").args
    wrapper["chunk_size"] = 4 * 44100
    lengths = []

    class LimitedNet(torch.nn.Module):
        def forward(self, x):
            lengths.append(x.shape[-1])
            if x.shape[-1] > 2 * 44100:
                raise torch.OutOfMemoryError("test window limit")
            return x * 0.5

    wrapper["net"] = LimitedNet()
    audio = make_audio(t=8.0, rate=44100, channels=2)
    audio["waveform"] = audio["waveform"].repeat(3, 1, 1)
    payload, _, _, _ = nodes.RoformerSeparate.execute(wrapper, audio, 0).args

    assert lengths[0] == 4 * 44100
    assert all(length <= 2 * 44100 for length in lengths[1:])
    assert payload["stems"]["vocals"].shape == (3, 2, 8 * 44100)


def test_separate_tta(nodes, models_dir, tiny_model, make_audio):
    """TTA: the model also runs on the time-flipped mix and the runs are averaged per stem."""
    wrapper, _ = nodes.RoformerLoader.execute("tiny", "cpu").args
    audio = make_audio(t=1.0, rate=44100, channels=2)
    payload, _, _, info_json = nodes.RoformerSeparate.execute(wrapper, audio, 0, True).args
    info = json.loads(info_json)
    assert info["tta"] is True and info["batch"] == 1
    for name, wave in payload["stems"].items():
        assert wave.shape == (1, 2, 44100) and torch.isfinite(wave).all(), name
    # TTA keeps the mixture-minus-separation property: both runs are flipped back, the
    # residual is exact per run, so the averaged stems still decompose the input
    total = payload["stems"]["vocals"] + payload["stems"]["other"]
    assert torch.allclose(total, audio["waveform"], atol=1e-4)
