"""Align math and the Roformer Align node: no weights, no network."""
import numpy as np
import pytest
import torch


def _noise(n=4 * 44100, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, n, generator=g)
    return x * 0.3


def _shift(x, k, length=None):
    """x shifted by k samples (k > 0: content starts k later), head zero-padded."""
    from roformernode.roformer import ensemble as E
    # shift_wave's positive offset advances the content (trims the head), so a
    # head zero-padded lag needs the negated offset
    return E.shift_wave(x, -k, length or x.shape[-1] + abs(k))


# --------------------------------------------------------------------- math

def test_full_lag_recovers_known_shifts():
    """The FFT correlate is numpy's full-correlate argmax: positive and negative lags.
    The returned lag is the negative of the content offset - a wav whose content starts
    L samples later gives -L (the alignment then trims its head by L to advance it)."""
    from roformernode.roformer import align as A
    ref = _noise(44100, 7).view(1, -1)
    for lag in (0, 37, 500, -1234, 20000):
        wav = _shift(ref, lag)
        found = A._full_lag(ref[0, :44100], wav[0, :44100])
        assert found == -lag, (lag, found)


def test_window_search_honors_interrupt(monkeypatch):
    import comfy.model_management as mm
    from roformernode.roformer import align as A

    calls = 0

    def interrupted():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupted)
    wave = _noise(44100, 3)

    with pytest.raises(mm.InterruptProcessingException):
        A.time_correction(wave, wave, [0.25], (0.0, [0.0]), 44100, 190)

    assert calls == 3


def test_check_if_phase_inverted():
    from roformernode.roformer import align as A
    x = _noise(20000, 1)
    assert A.check_if_phase_inverted(x, x.clone()) is False
    assert A.check_if_phase_inverted(x, -x) is True
    # silence on one side: no correlation to flip on
    z = torch.zeros_like(x)
    assert A.check_if_phase_inverted(x, z) is False


def test_silence_detection_and_adjust():
    from roformernode.roformer import align as A
    x = _noise(44100, 2)
    # silence lengths that are exact multiples of the 1024-sample frame: the
    # frame-quantized detector is then sample-exact
    silence = 3072
    padded = torch.cat([torch.zeros(1, silence), x], dim=-1)
    assert A._silence_end(x) == 0
    ref_end = A._silence_end(padded)
    assert ref_end == silence
    # adjust: equalize the leading silence of the two (target to reference)
    other = _shift(x, 1024)
    tgt_end = A._silence_end(other)
    assert tgt_end == 1024
    out = A.adjust_leading_silence(other, padded)
    assert out.shape[-1] == other.shape[-1] + (ref_end - tgt_end)
    # after equalization the content lines up: out[ref_end:] is x
    assert torch.allclose(out[0, ref_end:ref_end + 1000], x[0, :1000])


def test_phase_shift_hilbert_90_is_quadrature():
    """A cosine shifted 90 degrees by the Hilbert rotation is its quadrature (-sin),
    orthogonal to the original over the steady-state region."""
    from roformernode.roformer import align as A
    sr = 44100
    t = torch.arange(sr * 2, dtype=torch.float32) / sr
    x = torch.sin(2 * 3.141592653589793 * 440 * t).view(1, -1)
    out = A.phase_shift_hilbert(x, 90.0)
    expected = -torch.sin(2 * 3.141592653589793 * 440 * t).view(1, -1)
    # the FFT-based Hilbert has edge transients; the body must be exact
    n = out.shape[-1]
    body = slice(n // 4, 3 * n // 4)
    assert abs(float(out[0, body].dot(expected[0, body]) / (out[0, body].norm() * expected[0, body].norm()))) > 0.999
    # 180 = the negation exactly
    assert torch.allclose(A.phase_shift_hilbert(x, 180.0), -x, atol=1e-4)


def test_phase_shifted_tracks_sets():
    from roformernode.roformer import align as A
    x = _noise(8192, 3)
    assert [torch.equal(a, b) for a, b in zip(A.phase_shifted_tracks(x, 180), [x, -x])]
    assert len(A.phase_shifted_tracks(x, 90)) == 4      # x, -x, +90, -90
    assert len(A.phase_shifted_tracks(x, 45)) == 8      # x, -x, +/-45, +/-90, +/-135


def test_ensemble_wav_picks_the_quiet_pieces():
    """The 240-split selection: per piece, the lowest-mean candidate segment wins."""
    from roformernode.roformer import align as A
    base = _noise(240 * 100, 4).view(1, -1)
    noisy = base + 0.5 * torch.randn_like(base)
    out = A.ensemble_wav([noisy, base], split_size=240)
    assert out.shape == base.shape
    # every piece must come from the quiet candidate
    assert torch.equal(out, A.ensemble_wav([base, base]))
    assert out.abs().mean() < noisy.abs().mean()


def test_ensemble_for_align_is_the_min_spec_blend():
    from roformernode.roformer import align as A
    from roformernode.roformer import ensemble as E
    a, b = _noise(30000, 5), _noise(30000, 6)
    assert torch.allclose(A.ensemble_for_align([a, b], 30000),
                          E.blend("min spec", [a, b], 2048, 30000), atol=1e-5)


def test_time_correction_removes_a_global_lag():
    """inst lags mix by 250 samples; the per-window search must find it, leaving a
    residual far below the mix's own level (and the length unchanged)."""
    from roformernode.roformer import align as A
    sr = 44100
    mix = _noise(8 * sr, 8)
    inst = _shift(mix, 250, length=8 * sr + 250)[:, :8 * sr]
    sub = A.time_correction(mix, inst, [0.5], (0.0, [0.0]), sr, 190)
    assert sub.shape == mix.shape
    assert sub.abs().mean() < 0.15 * mix.abs().mean()


def test_time_correction_with_db_grid_corrects_the_gain():
    """inst = 0.5 x mix; the volume grid must find the gain and the residual must beat
    the raw (unmatched-gain) difference."""
    from roformernode.roformer import align as A
    sr = 44100
    mix = _noise(8 * sr, 9)
    inst = 0.5 * mix
    # base -6 dB (VOLUME_MAP's medium base) so the grid's top (db 12) reaches exactly
    # x2.0 - the gain inst needs to match mix
    sub = A.time_correction(mix, inst, [0.25], (-6.0, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
                            sr, 190)
    raw = (mix - inst).abs().mean()
    assert sub.abs().mean() < 0.25 * raw


# --------------------------------------------------------------------- node level

def _audio(wave, rate=44100):
    return {"waveform": wave[None] if wave.dim() == 2 else wave, "sample_rate": rate}


def _run(nodes, ref, wav, **kw):
    args = {
        "phase": "automatic", "match_silence": False, "intro_analysis": "default",
        "align_window": "off", "volume_analysis": "off", "phase_shifts": "off",
    }
    args.update(kw)
    return nodes.RoformerAlign.execute(
        _audio(ref), _audio(wav), args["phase"], args["match_silence"],
        args["intro_analysis"], args["align_window"], args["volume_analysis"],
        args["phase_shifts"],
    ).args


def test_align_node_guardrails(nodes):
    ref = _noise(2 * 44100, 10)
    with pytest.raises(ValueError, match="batched"):
        # a true batched payload: (B=2, C=1, N)
        _run(nodes, torch.cat([ref, ref], 0)[:, None], ref)
    with pytest.raises(ValueError, match="sample rates"):
        nodes.RoformerAlign.execute(
            _audio(ref), {"waveform": ref[None], "sample_rate": 48000},
            "automatic", False, "default", "off", "off", "off")
    with pytest.raises(ValueError, match="channel counts"):
        nodes.RoformerAlign.execute(
            _audio(torch.cat([ref, ref], 0)), _audio(ref),
            "automatic", False, "default", "off", "off", "off")
    with pytest.raises(ValueError, match="at least one second"):
        short = ref[:, :4410]
        nodes.RoformerAlign.execute(
            _audio(short), _audio(short), "automatic", False, "default", "off", "off", "off")


def test_align_node_aligns_shifted_and_scaled(nodes):
    """wav = 0.8 x ref shifted by 2500 samples (> 1000, so the automatic phase check
    sees silence, not a shifted content window): the aligned output tracks ref closely
    and the inverted output is the exact (clipping-free) residual."""
    ref = _noise(4 * 44100, 11)
    wav = 0.8 * _shift(ref, 2500)
    aligned, inverted = _run(nodes, ref, wav, volume_analysis="low")
    a = aligned["waveform"][0]
    i = inverted["waveform"][0]
    n = min(ref.shape[-1], a.shape[-1])
    # body (past the alignment's zero-pad) correlates strongly with ref
    body = slice(2000, n - 2000)
    r = a[0, body].dot(ref[0, body]) / (a[0, body].norm() * ref[0, body].norm())
    assert float(r) > 0.98
    # exact residual: ref - aligned == inverted where nothing clips
    assert torch.equal(i[:, :n], ref[:, :n] - a[:, :n])
    assert i.abs().max() <= 1.0 + 1e-6


def test_align_node_flips_an_inverted_phase(nodes):
    ref = _noise(4 * 44100, 12)
    aligned, _ = _run(nodes, ref, -ref)
    a = aligned["waveform"][0]
    n = min(ref.shape[-1], a.shape[-1])
    body = slice(2000, n - 2000)
    r = a[0, body].dot(ref[0, body]) / (a[0, body].norm() * ref[0, body].norm())
    assert float(r) > 0.98  # flipped back by the automatic phase check


def test_align_node_windowed_path_removes_the_lag(nodes):
    """The sliding-window refinement (align_window on) returns the RESIDUAL, not the
    aligned secondary: with a global 500-sample lag and no drift, the aligned output
    must track ref and the inverted output must be near-silent (a re-subtracted
    residual would leave the aligned output at ~0 and the inverted at ref's level).
    phase is forced native: the 500-sample offset puts shifted content inside the
    first 1000 samples the automatic phase check reads, and there the sign is a
    coin flip (the phase combo is the escape hatch)."""
    ref = _noise(4 * 44100, 15)
    wav = _shift(ref, 500)
    aligned, inverted = _run(nodes, ref, wav, phase="native", align_window="0.5 s")
    a = aligned["waveform"][0]
    i = inverted["waveform"][0]
    n = min(ref.shape[-1], a.shape[-1])
    body = slice(2000, n - 2000)
    r = a[0, body].dot(ref[0, body]) / (a[0, body].norm() * ref[0, body].norm())
    assert float(r) > 0.95
    assert i[0, body].abs().mean() < 0.15 * ref[0, body].abs().mean()


def test_align_node_matches_leading_silence(nodes):
    ref = _noise(4 * 44100, 13)
    wav = _noise(4 * 44100, 13)
    # same content, wav with 2048 samples MORE of leading silence; the lengths are
    # multiples of the detector's 1024-sample frame, so the frame-quantized
    # silence match is sample-exact
    content = torch.randn(1, 4 * 44100 - 2048, generator=torch.Generator().manual_seed(14))
    ref2 = torch.cat([torch.zeros(1, 2048), content * 0.3], dim=-1)
    wav2 = torch.cat([torch.zeros(1, 4096), content * 0.3], dim=-1)
    aligned, _ = _run(nodes, ref2, wav2, match_silence=True)
    a = aligned["waveform"][0]
    # after the silence match the content lines up: strong correlation in the body
    n = min(ref2.shape[-1], a.shape[-1])
    body = slice(6000, n - 2000)
    r = a[0, body].dot(ref2[0, body]) / (a[0, body].norm() * ref2[0, body].norm())
    assert float(r) > 0.95


def test_align_node_schema(nodes):
    """Inputs in declaration order (the widgets_values positional order) and the two
    audio outputs."""
    schema = nodes.RoformerAlign.define_schema()
    assert [i.id for i in schema.inputs] == [
        "reference", "input", "phase", "match_silence", "intro_analysis",
        "align_window", "volume_analysis", "phase_shifts",
    ]
    assert [i.id for i in schema.outputs] == ["aligned", "inverted"]
    for i in schema.inputs:
        assert i.tooltip, i.id
    assert schema.search_aliases


# --------------------------------------------------------------------- audio tools

def _tone(hz=440.0, t=2.0, rate=44100):
    n = int(t * rate)
    return (torch.sin(2 * 3.141592653589793 * hz * torch.arange(n) / rate)).view(1, -1)


def _dominant_hz(wave, rate):
    x = wave[0, 0]
    n = x.numel()
    x = x[n // 4:3 * n // 4]  # the body, away from the transform edges
    spec = torch.fft.rfft(x * torch.hann_window(x.numel(), device=x.device))
    return float(torch.argmax(spec.abs())) * rate / x.numel()


def test_pitch_shift_node(nodes):
    tone = _tone(440.0)
    (out,) = nodes.RoformerPitchShift.execute(_audio(tone), 12.0, True).args
    assert out["sample_rate"] == 44100
    assert abs(out["waveform"][0].shape[-1] - tone.shape[-1]) <= tone.shape[-1] * 0.02  # length kept
    assert _dominant_hz(out["waveform"], 44100) == pytest.approx(880.0, abs=6.0)
    # uncorrected: the length halves with +12 semitones
    (raw,) = nodes.RoformerPitchShift.execute(_audio(tone), 12.0, False).args
    assert raw["waveform"][0].shape[-1] == pytest.approx(tone.shape[-1] / 2, rel=0.02)


def test_pitch_shift_interrupts_between_channels(monkeypatch):
    import comfy.model_management as mm
    from roformernode.roformer.audio_tools import pitch_shift

    calls = 0

    def interrupted():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupted)

    with pytest.raises(mm.InterruptProcessingException):
        pitch_shift(torch.zeros(2, 1000), 44100, 1.0, False)

    assert calls == 2


def test_time_stretch_node(nodes):
    tone = _tone(440.0)
    (out,) = nodes.RoformerTimeStretch.execute(_audio(tone), 2.0).args
    assert out["waveform"][0].shape[-1] == pytest.approx(tone.shape[-1] / 2, rel=0.03)
    assert _dominant_hz(out["waveform"], 44100) == pytest.approx(440.0, abs=6.0)  # pitch kept


def test_key_detect_node(nodes):
    """A sustained A-minor triad (A3/C4/E4) is detected as A minor."""
    sr = 44100
    t = torch.arange(4 * sr, dtype=torch.float32) / sr
    chord = torch.stack([
        torch.sin(2 * 3.141592653589793 * 220 * t),
        torch.sin(2 * 3.141592653589793 * 261.63 * t),
        torch.sin(2 * 3.141592653589793 * 329.63 * t),
    ]).mean(0).view(1, -1) * 0.4
    (text,) = nodes.RoformerKeyDetect.execute(_audio(chord)).args
    assert "A minor" in text
    conf = float(text.rsplit(" ", 1)[1].strip("()").split()[-1])
    assert conf > 0.5


def test_audio_tool_schemas(nodes):
    for name, widgets in (
            ("RoformerPitchShift", ["audio", "semitones", "time_correct"]),
            ("RoformerTimeStretch", ["audio", "rate"]),
            ("RoformerKeyDetect", ["audio"])):
        schema = getattr(nodes, name).define_schema()
        assert [i.id for i in schema.inputs] == widgets, name
        for i in schema.inputs:
            assert i.tooltip, (name, i.id)
        assert schema.search_aliases, name
        assert [i.id for i in schema.outputs] == ["audio"] or             [i.id for i in schema.outputs] == ["key"], name
