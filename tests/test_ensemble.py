"""Ensemble math (UVR-exact) and the Roformer Ensemble node: no weights, no network."""
import numpy as np
import pytest
import torch


def _tone(hz=440.0, t=1.0, amp=1.0, rate=44100):
    """A (1, N) tone at the model rate."""
    n = int(t * rate)
    return amp * torch.sin(2 * 3.141592653589793 * hz * torch.arange(n) / rate).view(1, -1)


def _noise(n=44100, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n, generator=g)


# --------------------------------------------------------------------- math: UVR exactness

def test_average_is_plain_mean():
    from roformernode.roformer import ensemble as E
    a, b, c = _noise(), _noise(44100, 1), _noise(44100, 2)
    out = E.blend("average", [a, b, c], 2048, 44100)
    assert torch.allclose(out, (a + b + c) / 3, atol=1e-6)
    # truncates to the requested length
    assert E.blend("average", [a, b], 2048, 1000).shape == (1, 1000)


def test_max_min_spec_keeps_the_winner_verbatim():
    """UVR's fold: per STFT bin the complex value of the larger- / smaller-magnitude input
    wins (magnitude AND phase), and the blend is the istft of that chimeric spec. Two tones
    at different frequencies so both inputs win somewhere."""
    from roformernode.roformer import ensemble as E
    a = _tone(440.0)   # energy in the 440 Hz bin region
    b = _tone(880.0)   # energy in the 880 Hz bin region
    n = a.shape[-1]
    sa, sb = E.stft_spec(a, 2048), E.stft_spec(b, 2048)
    a_wins, b_wins = sa.abs() > sb.abs(), sb.abs() >= sa.abs()
    assert a_wins.any() and b_wins.any()  # both inputs contribute somewhere
    chimera_max = torch.where(b_wins, sb, sa)  # the fold for two inputs, independently
    chimera_min = torch.where(sb.abs() <= sa.abs(), sb, sa)
    # the output is exactly the istft of the chimeric spec: a magnitude-only or
    # phase-averaging blend would differ
    out = E.blend("max spec", [a, b], 2048, n)
    assert torch.allclose(out, E.istft_spec(chimera_max, 2048, n), atol=1e-5)
    out = E.blend("min spec", [a, b], 2048, n)
    assert torch.allclose(out, E.istft_spec(chimera_min, 2048, n), atol=1e-5)


def test_blend_geometry_round_trip():
    """Identical inputs: the chimera is one input's own spec, and the Hann 2048/1024
    centred geometry is COLA, so the blend comes back exactly - pinning the STFT/ISTFT
    pair (a rectangular window or a wrong hop would leak)."""
    from roformernode.roformer import ensemble as E
    a = _tone(440.0)
    assert torch.allclose(E.blend("max spec", [a, a], 2048, a.shape[-1]), a, atol=1e-5)
    assert torch.allclose(E.blend("min spec", [a, a], 2048, a.shape[-1]), a, atol=1e-5)


def test_blend_three_models_folds_pairwise():
    """N models: UVR folds pairwise (acc = where(|x_i| >= |acc|, x_i, acc)) - exactly."""
    from roformernode.roformer import ensemble as E
    torch.manual_seed(0)
    waves = [_noise(30000, i) for i in range(3)]
    out = E.blend("max spec", waves, 2048, 30000)
    acc = E.stft_spec(waves[0], 2048)
    for w in waves[1:]:
        s = E.stft_spec(w, 2048)
        acc = torch.where(s.abs() >= acc.abs(), s, acc)
    assert torch.allclose(out, E.istft_spec(acc, 2048, 30000), atol=1e-4)


def test_blend_unknown_algorithm():
    from roformernode.roformer import ensemble as E
    with pytest.raises(ValueError, match="algorithm"):
        E.blend("median", [_noise(1000)], 2048, 1000)


def test_normalize_modes():
    from roformernode.roformer import ensemble as E
    x = _noise()
    assert torch.equal(E.normalize_wave(x, "off"), x)
    quiet = x * 0.01
    assert torch.equal(E.normalize_wave(quiet, "anti clip"), quiet)  # never raises
    loud = x * 3.0
    out = E.normalize_wave(loud, "anti clip")
    assert float(out.abs().max()) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(out, x / x.abs().max(), atol=1e-6)
    full = E.normalize_wave(quiet, "full scale")
    assert float(full.abs().max()) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="normalize"):
        E.normalize_wave(x, "loud")


def test_exact_residual_is_exact():
    from roformernode.roformer import ensemble as E
    mix = _noise(40000)
    stem = _noise(40000, 1)
    out = E.exact_residual(mix, stem)
    assert torch.equal(out, mix - stem)  # sample for sample
    # length mismatch: the common length, still exact there
    out = E.exact_residual(mix, stem[:, :39000])
    assert out.shape == (1, 39000) and torch.equal(out, mix[:, :39000] - stem[:, :39000])


def test_shift_wave():
    from roformernode.roformer import ensemble as E
    x = torch.arange(20.0).view(1, 20)
    assert torch.equal(E.shift_wave(x, 0, 20), x)
    # offset > 0: the input lags, it is pulled earlier, zero tail
    assert E.shift_wave(x, 3, 20).view(-1).tolist() == [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 0, 0]
    # offset < 0: the input leads, zero head, tail clipped
    assert E.shift_wave(x, -3, 20).view(-1).tolist() == [
        0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def test_align_offset_recovers_known_shifts():
    """Noise (a unique cross-correlation peak, no periodic ambiguity) shifted by known
    amounts: the search recovers each one, within the 1 s radius."""
    from roformernode.roformer import ensemble as E
    rate = 44100
    ref = _noise(4 * rate, 7).view(1, -1)
    for lag in (0, 37, -1234, 8820):  # 0, ~1 ms, ~28 ms, 200 ms
        n = ref.numel()
        wav = torch.zeros(1, n)
        if lag >= 0:
            wav[:, lag:] = ref[:, :n - lag]      # wav lags by `lag` samples
        else:
            wav[:, :n + lag] = ref[:, -lag:]
        found = E.align_offset(ref, wav, 1.0, rate)
        assert found == lag, (lag, found)


def test_align_offset_is_zero_for_identical():
    from roformernode.roformer import ensemble as E
    x = _noise(2 * 44100, 3)
    assert E.align_offset(x, x.clone(), 1.0, 44100) == 0


# --------------------------------------------------------------------- node level

def _tiny_runs(nodes, models_dir):
    """Two 'model runs' over two near-identical songs through the tiny model, so the
    ensemble inputs are genuinely different (finite, non-silent) signals."""
    wrapper, _ = nodes.RoformerLoader.execute("tiny", "cpu").args
    torch.manual_seed(1)
    x = torch.randn(2, 44100) * 0.2
    audio = {"waveform": x[None], "sample_rate": 44100}
    p1, _, _, _ = nodes.RoformerSeparate.execute(wrapper, audio, 0).args
    wrapper2, _ = nodes.RoformerLoader.execute("tiny", "cpu").args
    audio2 = {"waveform": (x * 0.98)[None] + 0.02 * torch.roll(x, 5, -1)[None],
              "sample_rate": 44100}
    p2, _, _, _ = nodes.RoformerSeparate.execute(wrapper2, audio2, 0).args
    return p1, p2, audio


def _payload_audio(nodes, payload, stem="vocals"):
    return nodes.to_audio_payload(payload["stems"][stem][0], payload["sample_rate"])


def test_ensemble_node_average(nodes, models_dir, tiny_model):
    p1, p2, _ = _tiny_runs(nodes, models_dir)
    a1, a2 = _payload_audio(nodes, p1), _payload_audio(nodes, p2)
    (out,) = nodes.RoformerEnsemble.execute(
        {"audio0": a1, "audio1": a2}, "average", "off", False, 1.0, False, None, 2048
    ).args
    n = min(a1["waveform"].shape[-1], a2["waveform"].shape[-1])
    expected = (a1["waveform"][0, :, :n] + a2["waveform"][0, :, :n]) / 2
    assert torch.allclose(out["waveform"][0][:, :n], expected, atol=1e-6)
    assert out["sample_rate"] == 44100


def test_ensemble_node_max_spec_math(nodes, models_dir, tiny_model):
    """Through the node: the output is the istft of the per-bin chimeric spec - the
    complex value of the larger-magnitude input wins at every bin (UVR's exact math)."""
    from roformernode.roformer import ensemble as E
    p1, p2, _ = _tiny_runs(nodes, models_dir)
    a1 = _payload_audio(nodes, p1)
    a2 = _payload_audio(nodes, p2, "other")  # genuinely different content
    (out,) = nodes.RoformerEnsemble.execute(
        {"audio0": a1, "audio1": a2}, "max spec", "off", False, 1.0, False, None, 2048
    ).args
    w1, w2 = a1["waveform"][0], a2["waveform"][0]
    n = min(w1.shape[-1], w2.shape[-1])
    s1, s2 = (E.stft_spec(x[:, :n], 2048) for x in (w1, w2))
    chimera = torch.where(s2.abs() >= s1.abs(), s2, s1)
    assert torch.allclose(out["waveform"][0][:, :n], E.istft_spec(chimera, 2048, n), atol=1e-4)


def test_ensemble_node_invert_is_exact(nodes, models_dir, tiny_model):
    """invert: mix - blend, so blend + output == mix exactly (the leaderboard move)."""
    p1, p2, audio = _tiny_runs(nodes, models_dir)
    a1, a2 = _payload_audio(nodes, p1), _payload_audio(nodes, p2)
    mix = {"waveform": audio["waveform"], "sample_rate": 44100}
    (out,) = nodes.RoformerEnsemble.execute(
        {"audio0": a1, "audio1": a2}, "average", "off", False, 1.0, True, mix, 2048
    ).args
    n = min(w.shape[-1] for w in (out["waveform"], audio["waveform"]))
    blend = (a1["waveform"][0, :, :n] + a2["waveform"][0, :, :n]) / 2
    assert torch.equal(out["waveform"][0][:, :n], audio["waveform"][0, :, :n] - blend)
    # without the mix connected: an actionable error naming the widget
    with pytest.raises(ValueError, match="mix"):
        nodes.RoformerEnsemble.execute(
            {"audio0": a1, "audio1": a2}, "average", "off", False, 1.0, True, None, 2048)


def test_ensemble_node_aligns_shifted_input(nodes, models_dir, tiny_model):
    """align: the second input lags by 200 samples (its content starts later); with align
    on the blend is the unshifted average, without it the blend is smeared."""
    from roformernode.roformer import ensemble as E
    p1, p2, _ = _tiny_runs(nodes, models_dir)
    a1 = _payload_audio(nodes, p1)
    w2 = p2["stems"]["vocals"][0]
    lagged = E.shift_wave(w2, -200, w2.shape[-1])  # content 200 later: lags a1 by 200
    a2 = nodes.to_audio_payload(lagged, 44100)
    (unaligned,) = nodes.RoformerEnsemble.execute(
        {"audio0": a1, "audio1": a2}, "average", "off", False, 1.0, False, None, 2048
    ).args
    (aligned,) = nodes.RoformerEnsemble.execute(
        {"audio0": a1, "audio1": a2}, "average", "off", True, 1.0, False, None, 2048
    ).args
    l = min(a1["waveform"].shape[-1], a2["waveform"].shape[-1])
    # Advancing the lagging input by the detected 200 samples restores the unshifted stem:
    # the expected average is (a1 + w2) over the first l-200 samples, and a1/2 over the
    # shift's zero-filled tail (UVR's head-trim + zero-pad). A wrong detected offset would
    # misalign the body and fail the allclose.
    expected = (a1["waveform"][0][:, :l] + E.shift_wave(lagged, 200, l)) / 2
    assert torch.allclose(aligned["waveform"][0][:, :l], expected, atol=1e-5)
    assert not torch.allclose(unaligned["waveform"][0][:, :l], expected, atol=1e-5)


def test_ensemble_node_guardrails(nodes, models_dir, tiny_model):
    p1, _, _ = _tiny_runs(nodes, models_dir)
    a1 = _payload_audio(nodes, p1)
    a2 = _payload_audio(nodes, p1, "other")
    with pytest.raises(ValueError, match="two audio inputs"):
        nodes.RoformerEnsemble.execute(
            {"audio0": a1}, "average", "off", False, 1.0, False, None, 2048)
    batched = dict(a1)
    batched["waveform"] = a1["waveform"].repeat(2, 1, 1)
    with pytest.raises(ValueError, match="batched"):
        nodes.RoformerEnsemble.execute(
            {"audio0": batched, "audio1": a2}, "average", "off", False, 1.0, False, None, 2048)
    other_rate = {"waveform": a2["waveform"], "sample_rate": 48000}
    with pytest.raises(ValueError, match="sample rates"):
        nodes.RoformerEnsemble.execute(
            {"audio0": a1, "audio1": other_rate}, "average", "off", False, 1.0, False, None, 2048)
    mono = {"waveform": a1["waveform"][:, :1, :], "sample_rate": 44100}
    with pytest.raises(ValueError, match="channel counts"):
        nodes.RoformerEnsemble.execute(
            {"audio0": mono, "audio1": a2}, "average", "off", False, 1.0, False, None, 2048)


def test_ensemble_schema_autogrow_and_widget_order(nodes):
    """The autogrow template: two required slots out of eight; inputs in schema order -
    the order widgets_values is positional against."""
    schema = nodes.RoformerEnsemble.define_schema()
    grp = next(i for i in schema.inputs if i.id == "audios")
    assert isinstance(grp, nodes.io.Autogrow.Input)
    assert grp.template.prefix == "audio" and grp.template.min == 2 and grp.template.max == 8
    assert grp.template.names == [f"audio{i}" for i in range(8)]
    assert [i.id for i in schema.inputs] == [
        "audios", "algorithm", "normalize", "align", "align_max_seconds",
        "invert", "mix", "n_fft",
    ]
    widgets = [i for i in schema.inputs
               if not isinstance(i, nodes.io.DynamicInput) and not i.optional]
    assert [i.id for i in widgets] == ["algorithm", "normalize", "align",
                                       "align_max_seconds", "invert", "n_fft"]
    assert next(i for i in widgets if i.id == "n_fft").default == 2048  # UVR's geometry


def test_save_audio_normalize_and_batch(nodes, output_dir, models_dir, tiny_model):
    import soundfile as sf
    wrapper, _ = nodes.RoformerLoader.execute("tiny", "cpu").args
    a = torch.sin(2 * 3.141592653589793 * 440 * torch.arange(44100) / 44100)
    b = torch.sin(2 * 3.141592653589793 * 660 * torch.arange(44100) / 44100)
    audio = {"waveform": torch.stack([a, b]).unsqueeze(1).repeat(1, 2, 1), "sample_rate": 44100}
    payload, _, _, _ = nodes.RoformerSeparate.execute(wrapper, audio, 0).args
    # batch: one file per element per stem, suffixed _bXXX
    nodes.RoformerSaveAudio.execute(payload, "unit/batch", "wav", "off")
    files = sorted((output_dir / "unit").glob("batch_*.wav"))
    assert len(files) == 4  # 2 stems x 2 batch elements
    assert any(f.name.endswith("_b001.wav") for f in files)
    assert any(f.name.endswith("_b002.wav") for f in files)
    for f in files:
        data, rate = sf.read(str(f))
        assert rate == 44100 and data.shape == (44100, 2)
    # anti clip: a 3x-loud stem comes back at peak 1.0; off: untouched
    loud = {"stems": {"vocals": torch.full((1, 2, 1000), 1.5)}, "sample_rate": 44100, "info": {}}
    nodes.RoformerSaveAudio.execute(loud, "unit/clip_off", "wav", "off")
    nodes.RoformerSaveAudio.execute(loud, "unit/clip_on", "wav", "anti clip")
    off = list((output_dir / "unit").glob("clip_off_*.wav"))[0]
    on = list((output_dir / "unit").glob("clip_on_*.wav"))[0]
    d_off, _ = sf.read(str(off))
    d_on, _ = sf.read(str(on))
    assert float(np.abs(d_off).max()) == pytest.approx(1.5, abs=1e-3)  # as-is, still clipping
    assert float(np.abs(d_on).max()) == pytest.approx(1.0, abs=1e-3)   # scaled back
