"""ComfyUI nodes for the popular Roformer music source separation checkpoints.

The 13 pcunwa releases (36 checkpoints) plus the popular community releases (BS-Roformer-SW
6-stem, 4-stem FT, karaoke, vocal de-reverb, guitar) are all Roformer-family: band-split or
mel-band RoFormer separators trained at 44.1 kHz stereo. The checkpoints are bare state dicts and each
one is configured by the YAML that ships next to it, so this pack vendors the matching model
code (roformer/model, the BS-RoFormer 0.4.1 line) and builds every model from its own config.
The Loader downloads a checkpoint from Hugging Face into models/roformer on first use (or reads
a local folder from the same directory); the Separate node runs the model in overlapping
windows with normalised overlap-add and returns the target stem, the other stem, and a stems
payload for arbitrary extraction or saving.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import soundfile as sf
from tqdm.auto import tqdm

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from comfy_api.latest import io, UI

from .roformer import align, audio_tools, ensemble
from .roformer.audio_io import audio_to_tensor, resample, to_audio_payload
from .roformer.loading import load_roformer, local_models
from .roformer.manifest import MODELS
from .roformer.separation import Separator
from .roformer.stem_names import ALL_STEMS, canonical

LOGGER = logging.getLogger(__name__)

CATEGORY = "audio/Roformer"
ROFORMER_MODEL = io.Custom("ROFORMER_MODEL")
ROFORMER_STEMS = io.Custom("ROFORMER_STEMS")


def _all_on_device(module, device) -> bool:
    """True when every parameter and buffer of ``module`` sits on ``device``."""
    device = torch.device(device)
    for t in list(module.parameters()) + list(module.buffers()):
        if t.device != device:
            return False
    return True


def _is_oom(exc: BaseException) -> bool:
    """A CUDA OOM, or the RuntimeError wording MPS/XPU use for the same event."""
    return isinstance(exc, torch.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _model_choices() -> list[str]:
    choices = [m.id for m in MODELS]
    for name in sorted(local_models()):
        if name not in choices:
            choices.append(name)
    return choices


class RoformerLoader(io.ComfyNode):
    """Load one Roformer checkpoint as the ROFORMER_MODEL payload the Separate node consumes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerLoader",
            display_name="Roformer Loader",
            category=CATEGORY,
            description=(
                "Load one Roformer checkpoint (BS-Roformer, Mel-Band-Roformer or a community "
                "release: multi-stem, karaoke, de-reverb, guitar). model: a checkpoint id, "
                "auto-downloaded from Hugging Face into models/roformer on first use (0.2-1.7 GB), or "
                "a folder name in that directory holding a .ckpt/.pth plus its config YAML. All models "
                "run at their released float32 precision at 44.1 kHz; GPU strongly recommended. The big "
                "mel-band models (1.5-1.7 GB) want at least 10 GB of VRAM. All 41 checkpoints are "
                "supported: the repos that ship their own "
                "model code (HyperACE, Large-Inst, Siamese) are loaded with exactly that code, and the "
                "FNO checkpoint with the MaskEstimator patch its README publishes (neuraloperator is "
                "installed with the pack). "
                "The Kim FT ids are in a gated repo and need an HF token (HF_TOKEN or huggingface-cli login). "
                "Outputs the model payload and stems: the model's stem names as text, in the order "
                "the model trained them in - preview with any text display."
            ),
            inputs=[
                io.Combo.Input("model", options=_model_choices(), default="sw-6stem",
                               tooltip="A checkpoint id (auto-downloaded into models/roformer on first use) "
                                       "or a folder name in that directory holding the .ckpt/.pth plus its "
                                       "config YAML."),
                io.Combo.Input("device", options=["auto", "cuda", "mps", "cpu"], default="auto",
                               tooltip="Where the model runs: auto = ComfyUI's session device policy for "
                                       "auxiliary models (honors --gpu-only and a no-VRAM session); the "
                                       "others pin the run device. Weights wait in system RAM until a "
                                       "separation runs, then move to the device and are evicted again when "
                                       "other models need the VRAM."),
            ],
            outputs=[
                ROFORMER_MODEL.Output(id="model", display_name="model"),
                io.String.Output(id="stems", display_name="stems",
                                 tooltip="The model's stem names as text, in the order the model "
                                         "trained them in."),
            ],
        )

    @classmethod
    def execute(cls, model: str, device: str) -> io.NodeOutput:
        payload = load_roformer(model, device)
        LOGGER.info("Roformer: loaded %s from %s (load %s, offload %s).",
                    payload["label"], payload["ckpt_path"], payload["device"], payload["offload_device"])
        return io.NodeOutput(payload, ", ".join(payload["stems"]))


class RoformerSeparate(io.ComfyNode):
    """Separate the audio into stems: the stems payload plus the target/other stems as AUDIO."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerSeparate",
            display_name="Roformer Separate",
            category=CATEGORY,
            search_aliases=["separate", "separation", "stems", "source separation"],
            description=(
                "Separate the audio with the loaded model. The input is resampled to the model's "
                "44.1 kHz (stereo, or downmixed / upmixed to the model's trained channel count) and "
                "the stems come back at the file's own sample rate. A batched audio input (B, C, N) "
                "separates every element - the stems payload comes back batched too. chunk_seconds: "
                "0 = the model's own training window from its config (recommended); a positive value "
                "caps the window for low-VRAM machines (overlapping windows are overlap-added with a "
                "linear crossfade). tta: test-time augmentation - the model also runs on the "
                "time-flipped mix and the two runs are averaged (doubles the time, smooths "
                "flaky transients). Outputs: the stems payload (Roformer To Audio / Save Audio "
                "below), the target stem and the other stem as ready AUDIO (for multi-stem models: "
                "the 'other' stem when the model has one, else the first non-target stem - use the "
                "stems payload for the full set), and info_json (model, stems, lengths, chunking, "
                "tta, batch)."
            ),
            inputs=[
                ROFORMER_MODEL.Input("model"),
                io.Audio.Input("audio",
                               tooltip="The mix to separate, any sample rate (resampled internally to the "
                                       "model's 44.1 kHz) and any channel count (downmixed / upmixed to the "
                                       "model's trained count). A batched (B, C, N) audio separates "
                                       "every element; the stems payload comes back batched."),
                io.Int.Input("chunk_seconds", default=0, min=0, max=240,
                             tooltip="0 = the model's own training window from its config (recommended); a "
                                     "positive value caps the window for low-VRAM machines. If a window "
                                     "runs out of GPU memory the node halves it and retries (up to three "
                                     "times, logged)."),
                io.Boolean.Input("tta", default=False,
                                 tooltip="Test-time augmentation: the model also runs on the time-flipped "
                                         "mix and the two runs are averaged per stem. Doubles the run "
                                         "time; smooths flaky transients and edge artifacts."),
            ],
            outputs=[
                ROFORMER_STEMS.Output(id="stems", display_name="stems"),
                io.Audio.Output(id="target", display_name="target"),
                io.Audio.Output(id="other", display_name="other"),
                io.String.Output(id="info_json", display_name="info_json"),
            ],
        )

    @classmethod
    def execute(cls, model: dict, audio: dict, chunk_seconds: int = 0, tta: bool = False) -> io.NodeOutput:
        x, rate = audio_to_tensor(audio)
        p = model
        if p["num_channels"] == 1 and x.shape[1] > 1:
            x = x.mean(1, keepdim=True)
        elif p["num_channels"] == 2 and x.shape[1] == 1:
            x = x.repeat(1, 2, 1)
        x44 = resample(x, rate, p["sample_rate"])
        B = x44.shape[0]
        # Load the model onto the run device (ComfyUI's model management evicts it again when
        # other nodes need the VRAM) and complete a partial load.
        patcher = p["patcher"]
        mm.load_model_gpu(patcher)
        device = patcher.load_device
        net = p["net"]
        if device.type != "cpu" and not _all_on_device(net, device):
            # The load was partial: model management's low-VRAM streaming only covers
            # ComfyUI's own ops, so under pressure the first forward would see weights
            # split across the offload device and the GPU and crash. Complete the move;
            # if the device cannot take the whole model, run on the CPU instead of
            # OOMing the workflow.
            try:
                net.to(device)
                patcher.model.model_loaded_weight_memory = patcher.model_size()
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                LOGGER.warning(
                    "Roformer: the model does not fit in the free VRAM on %s; "
                    "running this separation on the CPU instead.", device,
                )
                net.to(torch.device("cpu"), dtype=torch.float32)
                net.device = torch.device("cpu")  # the patcher's bookkeeping for the model's device
                patcher.model.model_loaded_weight_memory = 0
                patcher.size = 0
                device = torch.device("cpu")
        p["device"] = device
        # The window: the model's training window (or the cap), halved on each OOM retry.
        chunk = p["chunk_size"] if chunk_seconds <= 0 else int(chunk_seconds * p["sample_rate"])
        chunk = max(chunk, 1024)
        min_chunk = max(1024, int(p["sample_rate"]))  # one second: below this, separation degrades
        rate44 = p["sample_rate"]
        runs = 2 if tta else 1  # TTA: the flipped run doubles the work per song
        # Progress is sized for the whole batch (per-song chunk plans at the current window;
        # an OOM retry only overruns the tail, which the bars clamp).
        total = 0
        for b in range(B):
            _, _, n_chunks = Separator.chunk_plan(x44[b].shape[-1], chunk)
            total += n_chunks * runs + runs
        progress = ProgressBar(total)
        stems_per_song = []
        attempts = 0
        with tqdm(total=total, desc="Roformer separation") as tq:
            for b in range(B):
                while True:
                    try:
                        s = Separator(p).run(x44[b], 0, progress=progress, tqdm_bar=tq, chunk=chunk)
                        if tta:
                            sf_ = Separator(p).run(x44[b].flip(-1), 0, progress=progress,
                                                   tqdm_bar=tq, chunk=chunk)
                            # flipping both the run and its residual stem back keeps the
                            # mixture-minus-separation property: avg(voc) + avg(other) == mix
                            s = {k: (v + sf_[k].flip(-1)) * 0.5 for k, v in s.items()}
                        break
                    except (torch.OutOfMemoryError, RuntimeError) as exc:
                        if not _is_oom(exc):  # non-OOM errors pass through untouched
                            raise
                        if attempts >= 3 or chunk <= min_chunk:
                            raise ValueError(
                                f"Roformer: out of GPU memory separating even at a "
                                f"{chunk / rate44:.1f} s window on {device}. Cap chunk_seconds lower, "
                                "use a smaller model, or set the loader's device to cpu."
                            ) from exc
                        attempts += 1
                        LOGGER.warning(
                            "Roformer: out of memory at a %.1f s window; retrying the rest of "
                            "the batch at %.1f s.",
                            chunk / rate44, chunk / 2 / rate44,
                        )
                        chunk = max(min_chunk, chunk // 2)
                        mm.soft_empty_cache()
                stems_per_song.append(s)
        stems44 = {k: torch.stack([s[k] for s in stems_per_song]) for k in stems_per_song[0]}
        stems = {k: resample(v, p["sample_rate"], rate) for k, v in stems44.items()}
        mm.soft_empty_cache()

        names = list(stems.keys())
        target_name = p["target"] if p["target"] in names else names[0]
        # for multi-stem models prefer the native 'other' stem, else the first non-target stem
        other_name = next((n for n in names if n.lower() == "other" and n != target_name),
                          next((n for n in names if n != target_name), target_name))
        info = {
            "model": p["label"], "family": p["family"], "target": target_name,
            "stems": names, "seconds": round(x.shape[-1] / rate, 3),
            "source_rate": rate, "model_rate": p["sample_rate"],
            "channels_in": int(audio["waveform"].shape[-2]),
            "chunk_seconds": 0 if chunk_seconds <= 0 else chunk_seconds,
            "tta": tta, "batch": B,
        }
        LOGGER.info("Roformer: separated %d song(s), %d stems (%s), %.2f s @ %d Hz%s.",
                    B, len(names), ", ".join(names), x.shape[-1] / rate, rate,
                    ", TTA on" if tta else "")
        return io.NodeOutput(
            {"stems": stems, "sample_rate": rate, "info": info},
            to_audio_payload(stems[target_name], rate),
            to_audio_payload(stems[other_name], rate),
            json.dumps(info, indent=2, sort_keys=True),
        )


class RoformerEnsemble(io.ComfyNode):
    """Blend the same stem from 2-8 model runs (max spec / min spec / average, exact residual)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerEnsemble",
            display_name="Roformer Ensemble",
            category=CATEGORY,
            search_aliases=["ensemble", "blend", "combine stems", "max spec", "min spec",
                            "average", "exact residual", "mvsep ensemble"],
            description=(
                "Blend the same stem from two or more model runs into one better stem - the ensemble "
                "behind the top mvsep.com leaderboard entries (several of the best results are two- "
                "model blends, not single models). Connect the stem's AUDIO from several Separate runs "
                "(e.g. the vocals from two different models) to the audio slots: two are required, and "
                "up to eight appear as you connect them. One song per input, one sample rate, the same "
                "channel count. algorithm: 'max spec' keeps, per STFT bin, the value of the input with "
                "the larger magnitude (phase included - the leaderboard choice for vocals); 'min spec' "
                "the smaller one (suppresses leakage - the leaderboard choice for the instrumental); "
                "'average' is a plain time-domain mean (Hann 2048/1024 STFT). "
                "invert: exact residual - the output becomes mix minus the blend (connect the "
                "original audio to mix): ensemble the vocals, subtract for the instrumental, and "
                "blend + output == mix holds exactly. normalize: 'anti clip' scales a clipping blend "
                "back to full scale (never raises a quiet result); 'full scale' always rescales. "
                "align: search a small time offset (up to the advanced align_max_seconds, default "
                "1 s) that best aligns each input with the first - for inputs from different "
                "processing paths; two runs of this pack are sample-accurate, so leave it off there. "
                "The graph you build is the saved ensemble: every slot, algorithm and option is in "
                "the workflow file."
            ),
            inputs=[
                io.Autogrow.Input(
                    "audios",
                    template=io.Autogrow.TemplatePrefix(
                        io.Audio.Input("audio", tooltip="The same stem from one model run (one song, "
                                                        "one sample rate, same channel count as the "
                                                        "other inputs)."),
                        prefix="audio", min=2, max=8),
                    tooltip="The same stem from each model run - connect the stem's AUDIO from several "
                            "Separate runs (e.g. the vocals from two different models). Two slots are "
                            "required, up to eight appear as you connect."),
                io.Combo.Input("algorithm", options=list(ensemble.ALGORITHMS), default="average",
                               tooltip="max spec: per STFT bin, the value of the input with the larger "
                                       "magnitude (keeps each model's most confident content - the "
                                       "leaderboard choice for vocals). min spec: the smaller "
                                       "magnitude (suppresses leakage - the leaderboard choice for the "
                                       "instrumental). average: plain time-domain mean. Phase is "
                                       "preserved throughout (Hann 2048/1024 STFT)."),
                io.Combo.Input("normalize", options=list(ensemble.NORMALIZE_MODES), default="off",
                               tooltip="off: as-is. anti clip: if the blend clips (peak above 1.0) scale "
                                       "it back to full scale - never raises a quiet result. full "
                                       "scale: always rescale to peak 1.0."),
                io.Boolean.Input("align", default=False,
                                 tooltip="Search a small time offset (up to the advanced "
                                         "align_max_seconds, default 1 s) that best aligns every input "
                                         "with the first before blending - for inputs from different "
                                         "processing paths (different sample rates, trims, "
                                         "resamplers). Off: the inputs are assumed sample-accurate "
                                         "(two runs of this pack always are)."),
                io.Float.Input("align_max_seconds", default=1.0, min=0.0625, max=2.0, step=0.0625,
                               advanced=True,
                               tooltip="Search radius for align, in seconds (0.0625-2). Only read "
                                       "when align is on."),
                io.Boolean.Input("invert", default=False,
                                 tooltip="Exact residual: the output becomes mix minus the blend "
                                         "(connect the original audio to the mix input). The "
                                         "leaderboard move: ensemble the vocals, subtract for the "
                                         "instrumental - blend + output == mix exactly."),
                io.Audio.Input("mix", optional=True,
                               tooltip="The original mix (from Load Audio). Required when invert is "
                                       "on; ignored otherwise."),
                io.Int.Input("n_fft", default=2048, min=512, max=8192, step=512, advanced=True,
                             tooltip="STFT window for the spectral algorithms (hop = half of it). "
                                     "2048/1024 is what the mvsep winners use; leave it."),
            ],
            outputs=[
                io.Audio.Output(id="audio", display_name="audio",
                                tooltip="The blended stem at the inputs' own sample rate."),
            ],
        )

    @classmethod
    def execute(cls, audios: io.Autogrow.Type, algorithm: str, normalize: str, align: bool,
                align_max_seconds: float, invert: bool, mix: dict = None,
                n_fft: int = 2048) -> io.NodeOutput:
        names = sorted((k for k in (audios or {}) if audios[k] is not None),
                       key=lambda n: int(n.removeprefix("audio")))
        if len(names) < 2:
            raise ValueError(
                "Roformer Ensemble: connect at least two audio inputs - the same stem from two model "
                "runs (e.g. the vocals from two different Separates)."
            )
        waves, rates = [], set()
        for k in names:
            w, rate = audio_to_tensor(audios[k])
            if w.shape[0] != 1:
                raise ValueError(
                    f"Roformer Ensemble: input {k} is batched (B={w.shape[0]}) - the ensemble blends "
                    "one song per input; separate the batch first (one song per Separate run)."
                )
            if waves and w.shape[1] != waves[0].shape[0]:
                raise ValueError(
                    f"Roformer Ensemble: the inputs have different channel counts "
                    f"({waves[0].shape[0]} vs {w.shape[1]}) - connect stems from the same mix, at the "
                    "same channel count."
                )
            waves.append(w[0])
            rates.add(rate)
        if len(rates) > 1:
            raise ValueError(
                f"Roformer Ensemble: the inputs have different sample rates {sorted(rates)} - resample "
                "them to one rate first (e.g. save and re-load one of them, or feed both runs from the "
                "same source)."
            )
        rate = rates.pop()
        length = min(w.shape[-1] for w in waves)
        waves = [w[:, :length] for w in waves]
        if align:
            for i, w in enumerate(waves[1:], start=1):
                off = ensemble.align_offset(waves[0], w, align_max_seconds, rate)
                if off:
                    LOGGER.info("Roformer Ensemble: aligned input %d by %d samples (%.4f s).",
                                i + 1, off, off / rate)
                    waves[i] = ensemble.shift_wave(w, off, length)
        y = ensemble.blend(algorithm, waves, n_fft, length)
        if normalize != "off":
            y = ensemble.normalize_wave(y, normalize)
        if invert:
            if mix is None:
                raise ValueError(
                    "Roformer Ensemble: invert (exact residual) needs the mix - connect the original "
                    "audio (e.g. from Load Audio) to the mix input."
                )
            mx, mrate = audio_to_tensor(mix)
            if mx.shape[0] != 1:
                raise ValueError(
                    "Roformer Ensemble: the mix input is batched (B=%d) - connect one song." % mx.shape[0]
                )
            if mrate != rate:
                raise ValueError(
                    f"Roformer Ensemble: the mix sample rate ({mrate} Hz) differs from the inputs "
                    f"({rate} Hz) - resample the mix first."
                )
            y = ensemble.exact_residual(mx[0], y)
        LOGGER.info("Roformer Ensemble: %s of %d input(s), %d samples @ %d Hz%s.",
                    algorithm, len(waves), y.shape[-1], rate,
                    ", exact residual" if invert else "", )
        return io.NodeOutput(to_audio_payload(y, rate))


class RoformerAlign(io.ComfyNode):
    """Align one recording to a reference - for cross-session ensembling and A/B."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerAlign",
            display_name="Roformer Align",
            category=CATEGORY,
            search_aliases=["align", "alignment", "offset", "sync", "time shift",
                            "phase inversion", "leading silence", "manual ensemble"],
            description=(
                "Align one recording to a reference so two files from different sessions can be "
                "ensembled or A/B'd. Connect "
                "the reference (the track to align to, e.g. the original mix) and the input (the "
                "track to be aligned): same sample rate, same channel count, one song each. phase: "
                "'automatic' flips the input when it starts phase-inverted (a correlation check on "
                "the first 1000 samples); 'positive' / 'negative' force a sign; 'native' leaves it. "
                "match_silence: equalize the leading silence of the two first (for files trimmed "
                "differently). intro_analysis: where the first cross-correlation pass samples its "
                "1 s segments (default: one position at 1/10 of the length; 'shifts: *' try several "
                "positions and keep every unique shift as a candidate). The advanced levers refine "
                "the search: align_window slides a Hanning window and re-aligns per window (the "
                "heavy path - the finer the window and the volume/phase grids, the slower); "
                "volume_analysis tries gain corrections on the input (0-40 dB); phase_shifts tries "
                "Hilbert phase rotations of the input. Outputs: aligned (the input aligned to the "
                "reference - feed it to an Ensemble slot) and inverted (reference minus the "
                "aligned input, the exact residual - the inverted track)."
            ),
            inputs=[
                io.Audio.Input("reference", tooltip="The track to align to (e.g. the original mix, "
                                                    "or the first model's run). One song."),
                io.Audio.Input("input", tooltip="The track to be aligned (one song, the same sample "
                                                "rate and channel count as the reference). Its "
                                                "aligned form comes back on the aligned output."),
                io.Combo.Input("phase", options=list(align.PHASE_OPTIONS), default="automatic",
                               tooltip="automatic: flip the input if it starts phase-inverted (the "
                                       "correlation check on the first 1000 samples). positive / "
                                       "negative: force the sign. native: leave it as-is."),
                io.Boolean.Input("match_silence", default=False,
                                 tooltip="Equalize the leading silence of the two tracks first - "
                                         "for files that were trimmed or exported differently. "
                                         "Off: the cross-correlation search handles the offset."),
                io.Combo.Input("intro_analysis", options=list(align.INTRO_OPTIONS), default="default",
                               tooltip="Where the first alignment pass samples its 1 s segments: "
                                       "default = one position (1/10 of the length); 1/8..1/2 = "
                                       "that single fraction; shifts: low/medium/high = several "
                                       "fractions - every unique shift becomes a candidate."),
                io.Combo.Input("align_window", options=list(align.ALIGN_WINDOW_OPTIONS), default="off",
                               advanced=True,
                               tooltip="Per-window refinement: slide a Hanning window of this size "
                                       "across the track and re-align inside every window (for "
                                       "pairs with local timing drift). off = the global search "
                                       "only (fast). The finer the window - and the volume / "
                                       "phase grids below - the slower the run."),
                io.Combo.Input("volume_analysis", options=list(align.VOLUME_OPTIONS), default="off",
                               advanced=True,
                               tooltip="Gain corrections to try on the input during the search: "
                                       "off (none), low (0-7 dB in 1 dB steps), medium (0-11 dB "
                                       "in 1 dB), high (0-12 dB in 0.5 dB), very high (0-20 dB in "
                                       "0.5 dB - the largest grid, the slowest)."),
                io.Combo.Input("phase_shifts", options=list(align.PHASE_SHIFT_OPTIONS), default="off",
                               advanced=True,
                               tooltip="Hilbert phase rotations of the input to try (in degrees): "
                                       "off (none), or 180/90/45/20/10/1 = the step - smaller "
                                       "steps try more rotations (1 tries all of them; the "
                                       "largest grid, the slowest)."),
            ],
            outputs=[
                io.Audio.Output(id="aligned", display_name="aligned",
                                tooltip="The input aligned to the reference (gain- and phase- "
                                        "corrected, offset-removed) - feed it to an Ensemble "
                                        "slot next to the reference."),
                io.Audio.Output(id="inverted", display_name="inverted",
                                tooltip="Reference minus the aligned input (the inverted track) - "
                                        "what the alignment removed, sample-exact."),
            ],
        )

    @classmethod
    def execute(cls, reference: dict, audio: dict, phase: str, match_silence: bool,
                intro_analysis: str, align_window: str, volume_analysis: str,
                phase_shifts: str) -> io.NodeOutput:
        ref, rate = audio_to_tensor(reference)
        wav, wrate = audio_to_tensor(audio)
        if ref.shape[0] != 1:
            raise ValueError(
                f"Roformer Align: the reference is batched (B={ref.shape[0]}) - connect one song.")
        if wav.shape[0] != 1:
            raise ValueError(
                f"Roformer Align: the input is batched (B={wav.shape[0]}) - connect one song.")
        if wrate != rate:
            raise ValueError(
                f"Roformer Align: the sample rates differ ({rate} Hz vs {wrate} Hz) - resample one "
                "of them first.")
        if ref.shape[1] != wav.shape[1]:
            raise ValueError(
                f"Roformer Align: the channel counts differ ({ref.shape[1]} vs {wav.shape[1]}) - "
                "connect stereo to stereo (or mono to mono).")
        if min(ref.shape[-1], wav.shape[-1]) < rate:
            raise ValueError("Roformer Align: both tracks need at least one second.")
        ref, wav = ref[0], wav[0]
        # phase handling
        if phase == "automatic" and align.check_if_phase_inverted(ref, wav):
            LOGGER.info("Roformer Align: the input starts phase-inverted; flipping it.")
            wav = -wav
        elif phase == "negative":
            wav = -wav
        if match_silence:
            wav = align.adjust_leading_silence(wav, ref)
        intro_divs = align.INTRO_MAP[intro_analysis]
        windows = align.ALIGN_WINDOW_MAP[align_window]
        base_db, db_range = align.VOLUME_MAP[volume_analysis]
        phase_step = align.PHASE_SHIFT_MAP[phase_shifts]
        variants = align.intro_align_variants(ref, wav, rate, intro_divs)
        # progress: per window position (per channel) on the windowed path, per candidate on the global one
        if windows:
            total = sum(len(range(0, max(ref.shape[-1] - int(rate * secs) + 1, 1), int(rate * secs / 2)))
                        * ref.shape[0] for secs in windows)
        else:
            total = len(variants) * len(db_range)
        progress = ProgressBar(total)
        with tqdm(total=total, desc="Roformer align") as tq:
            def tick(n=1):
                progress.update(n)
                tq.update(n)
            candidates: list[torch.Tensor] = []
            for s in variants:
                s = align.match_length(s, ref)
                if windows:
                    candidates.append(align.time_correction(ref, s, windows, (base_db, db_range),
                                                            rate, phase_step, progress=tick))
                else:
                    subs = [ref - s * (10 ** ((base_db + db) / 20)) for db in db_range]
                    candidates.extend(subs)
                    tick(len(db_range))
        final = (align.ensemble_wav(candidates, 240) if len(candidates) > 1
                 else candidates[0])
        sub = final.clamp(-1, 1)
        aligned = ref - sub
        LOGGER.info("Roformer Align: %d candidate shift(s), %d samples @ %d Hz -> aligned + "
                    "inverted (residual RMS %.4f).", len(candidates), ref.shape[-1], rate,
                    float(sub.pow(2).mean().sqrt()))
        return io.NodeOutput(to_audio_payload(aligned, rate), to_audio_payload(sub, rate))


class RoformerPitchShift(io.ComfyNode):
    """Shift the pitch by semitones (phase vocoder or classic resample)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerPitchShift",
            display_name="Roformer Pitch Shift",
            category=CATEGORY,
            search_aliases=["pitch", "semitones", "transpose", "key", "rubber band"],
            description=(
                "Shift the audio's pitch by semitones (0.5 steps, -24 to +24) at its own sample "
                "rate. time_correct (on): the pitch moves and the length stays - a phase-vocoder "
                "pass; off: the classic resample - the pitch moves AND the length changes by "
                "2 ** (-semitones / 12). A batched input shifts every element."),
            inputs=[
                io.Audio.Input("audio", tooltip="The audio to shift, any sample rate (stays at "
                                                "its own rate). A batched (B, C, N) input "
                                                "shifts every element."),
                io.Float.Input("semitones", default=0.0, min=-24.0, max=24.0, step=0.5,
                               tooltip="The pitch shift in semitones, 0.5 steps (12 = one octave "
                                       "up; -12 = one octave down)."),
                io.Boolean.Input("time_correct", default=True,
                                 tooltip="on: phase-vocoder shift - the pitch moves, the length "
                                         "stays. off: the resample path - the pitch moves and the "
                                         "length changes too."),
            ],
            outputs=[
                io.Audio.Output(id="audio", display_name="audio",
                                tooltip="The shifted audio at the input's own sample rate."),
            ],
        )

    @classmethod
    def execute(cls, audio: dict, semitones: float, time_correct: bool = True) -> io.NodeOutput:
        wave, rate = audio_to_tensor(audio)
        out = [audio_tools.pitch_shift(wave[b], rate, semitones, time_correct)
               for b in range(wave.shape[0])]
        y = torch.stack(out)
        LOGGER.info("Roformer Pitch Shift: %.1f semitone(s) over %d song(s), %d samples @ %d Hz%s.",
                    semitones, wave.shape[0], y.shape[-1], rate,
                    ", time-corrected" if time_correct else ", length changed")
        return io.NodeOutput(to_audio_payload(y, rate))


class RoformerTimeStretch(io.ComfyNode):
    """Stretch the tempo without changing the pitch (phase vocoder)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerTimeStretch",
            display_name="Roformer Time Stretch",
            category=CATEGORY,
            search_aliases=["tempo", "time", "stretch", "speed", "slow down", "speed up"],
            description=(
                "Stretch the audio's tempo without changing its pitch at its own sample rate (a "
                "phase-vocoder pass). rate: 1.0 = as-is, "
                "> 1 = faster / shorter (2.0 = half the length), < 1 = slower / longer. A batched "
                "input stretches every element."),
            inputs=[
                io.Audio.Input("audio", tooltip="The audio to stretch, any sample rate (stays at "
                                                "its own rate). A batched (B, C, N) input "
                                                "stretches every element."),
                io.Float.Input("rate", default=1.0, min=0.25, max=4.0, step=0.05,
                               tooltip="The tempo factor: 2.0 = twice as fast (half the length), "
                                       "0.5 = half speed (double the length). The pitch is "
                                       "preserved."),
            ],
            outputs=[
                io.Audio.Output(id="audio", display_name="audio",
                                tooltip="The stretched audio at the input's own sample rate."),
            ],
        )

    @classmethod
    def execute(cls, audio: dict, rate: float = 1.0) -> io.NodeOutput:
        wave, sr = audio_to_tensor(audio)
        out = [audio_tools.time_stretch(wave[b], sr, rate) for b in range(wave.shape[0])]
        y = torch.stack(out)
        LOGGER.info("Roformer Time Stretch: x%.2f over %d song(s), %d -> %d samples @ %d Hz.",
                    rate, wave.shape[0], wave.shape[-1], y.shape[-1], sr)
        return io.NodeOutput(to_audio_payload(y, sr))


class RoformerKeyDetect(io.ComfyNode):
    """The musical key of the audio (chroma + Krumhansl-Schmuckler profile match)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerKeyDetect",
            display_name="Roformer Key Detect",
            category=CATEGORY,
            search_aliases=["key", "key detection", "tonality", "major", "minor"],
            description=(
                "Detect the musical key of the audio: the mean chroma pitch-class profile "
                "correlated against all 24 Krumhansl-Schmuckler key profiles (major and minor at "
                "every tonic). Feed it the separated "
                "vocals (the clearest tonal source) rather than the full mix when you have them. "
                "Output: a string like 'A minor (confidence 0.72)' - the confidence is the best "
                "correlation (0.5-0.9 is typical on clean vocals; below about 0.5 the audio is "
                "tonally ambiguous). One song per input; a batched input is detected on its "
                "first element."),
            inputs=[
                io.Audio.Input("audio", tooltip="The audio to analyse (ideally the separated "
                                                "vocals). One song."),
            ],
            outputs=[
                io.String.Output(id="key", display_name="key",
                                 tooltip="The detected key and its confidence, e.g. 'A minor "
                                         "(confidence 0.72)'."),
            ],
        )

    @classmethod
    def execute(cls, audio: dict) -> io.NodeOutput:
        wave, rate = audio_to_tensor(audio)
        if wave.shape[0] > 1:
            LOGGER.info("Roformer Key Detect: %d-song batch - detecting the first element.",
                        wave.shape[0])
        key, conf = audio_tools.detect_key(wave[0], rate)
        LOGGER.info("Roformer Key Detect: %s (r=%.3f) over %.1f s @ %d Hz.",
                    key, conf, wave[0].shape[-1] / rate, rate)
        return io.NodeOutput(f"{key} (confidence {conf:.2f})")


AUTO_STEM = "(auto)"


def _pick_stem(stems: dict, stem: str) -> dict:
    """The selected stem as an AUDIO payload. Matching is case-insensitive and alias-
    tolerant ("instrument" picks a payload's "Instrumental"), with a prefix fallback; an
    unknown/empty name falls back to the first stem (the model's target), so a stale choice
    can never fail a run."""
    s = stems["stems"]
    if stem in s:
        return to_audio_payload(s[stem], stems["sample_rate"])
    wanted = canonical(stem)
    if wanted:
        for name in s:  # canonical: "Vocals" == "vocals"
            if canonical(name) == wanted:
                return to_audio_payload(s[name], stems["sample_rate"])
        for name in s:  # prefix: "instrument" -> "Instrumental", without the alias too
            cn = canonical(name)
            if cn.startswith(wanted) or wanted.startswith(cn):
                return to_audio_payload(s[name], stems["sample_rate"])
    return to_audio_payload(next(iter(s.values())), stems["sample_rate"])


class RoformerToAudio(io.ComfyNode):
    """Extract one stem from a stems payload as an AUDIO you can feed anywhere else."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerToAudio",
            display_name="Roformer To Audio",
            category=CATEGORY,
            search_aliases=["stem", "extract stem", "to audio"],
            description=(
                "Extract one stem from a stems payload as an AUDIO you can feed anywhere else "
                "(preview, save, or another separator). stem: pick from the dropdown - every stem "
                "name the pack knows about, matched case-insensitively (the model's own list is on "
                "the Loader's stems output); "
                f"{(AUTO_STEM)} = the first stem, i.e. the model's target. For a name outside the "
                "list (a local model, a custom checkpoint) use Custom Roformer To Audio."
            ),
            inputs=[
                ROFORMER_STEMS.Input("stems"),
                io.Combo.Input("stem", options=[AUTO_STEM] + list(ALL_STEMS), default=AUTO_STEM,
                               tooltip="The stem to extract (case-insensitive; the model's own list is "
                                       "on the Loader's stems output). (auto) = the first stem, i.e. the "
                                       "model's target. For a name outside the list (a local model, a "
                                       "custom checkpoint) use Custom Roformer To Audio."),
            ],
            outputs=[io.Audio.Output(id="audio", display_name="audio")]
        )

    @classmethod
    def execute(cls, stems: dict, stem: str) -> io.NodeOutput:
        return io.NodeOutput(_pick_stem(stems, stem))


class CustomRoformerToAudio(RoformerToAudio):
    """The same extraction with a free-text name: any stem name from info_json, e.g. from a
    local model or a checkpoint the pack does not curate (empty = the first stem)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CustomRoformerToAudio",
            display_name="Custom Roformer To Audio",
            category=CATEGORY,
            search_aliases=["custom stem", "extract custom stem"],
            description=(
                "Like Roformer To Audio, but the stem is typed by hand instead of picked from the "
                "dropdown: the name from info_json, matched case-insensitively (empty = the first "
                "stem, i.e. the model's target). Use it for local models and custom checkpoints whose "
                "stem names are not in the standard list."
            ),
            inputs=[
                ROFORMER_STEMS.Input("stems"),
                io.String.Input("stem", default="",
                                tooltip="The stem name typed by hand (case-insensitive); empty = the "
                                        "first stem, i.e. the model's target."),
            ],
            outputs=[io.Audio.Output(id="audio", display_name="audio")]
        )


class RoformerSaveAudio(io.ComfyNode):
    """Write the stems to ComfyUI's output folder as <prefix>_XXXXX_<stem>.<format>."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerSaveAudio",
            display_name="Roformer Save Audio",
            category=CATEGORY,
            search_aliases=["save stems", "export stems", "save audio"],
            is_output_node=True,
            description=(
                "Write the stems to ComfyUI's output folder, numbered so runs never overwrite "
                "each other: WAV as <prefix>_XXXXX_<stem>.wav (lossless float32), FLAC through the "
                "native ComfyUI audio saver as <prefix>/<stem>_XXXXX.flac (lossless, with the "
                "workflow prompt embedded in the file metadata). A batched stems payload (from a "
                "batched Separate) writes one file per element: WAV suffixed _b001.., FLAC numbered "
                "sequentially. normalize: 'anti clip' scales a clipping stem back to full scale "
                "before writing (never raises a quiet stem); 'full scale' always rescales. The "
                "saved files are linked on the node (the audio icon)."
            ),
            inputs=[
                ROFORMER_STEMS.Input("stems"),
                io.String.Input("filename_prefix", default="roformer/stem",
                                tooltip="The output file name (without extension, subfolders allowed); "
                                        "runs are numbered so they never overwrite each other."),
                io.Combo.Input("format", options=["wav", "flac"], default="wav",
                               tooltip="wav = lossless float32 (what the audio nodes work with); flac = "
                                       "lossless 16-bit, smaller, and carries the workflow prompt in the "
                                       "file metadata."),
                io.Combo.Input("normalize", options=list(ensemble.NORMALIZE_MODES), default="off",
                               tooltip="off: as-is. anti clip: if a stem clips (peak above 1.0) scale it "
                                       "back to full scale before writing - never raises a quiet stem. "
                                       "full scale: always rescale every stem to peak 1.0."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[],
        )

    @classmethod
    def execute(cls, stems: dict, filename_prefix: str, format: str, normalize: str = "off") -> io.NodeOutput:
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 1, 1
        )
        # get_save_image_path counts PNGs only; count our own format to avoid overwrites.
        counter = len(list(Path(folder).glob(f"{filename}_*.wav")))
        rate = stems["sample_rate"]
        results = []
        for name, wave in stems["stems"].items():
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            wave = ensemble.normalize_wave(wave, normalize)  # (B, C, N)
            n = wave.shape[0]
            if format == "flac":
                # the native FLAC path: the helper embeds the workflow prompt in the file
                # metadata (from the hidden inputs the engine injects into cls.hidden) and
                # hands back the saved-file descriptors the node UI links to; it writes one
                # file per batch element
                saved = UI.AudioSaveHelper.save_audio(
                    {"waveform": wave.float().cpu(), "sample_rate": rate},
                    filename_prefix=f"{filename_prefix}/{safe}",
                    folder_type=io.FolderType.output, cls=cls, format="flac",
                )
                results.extend(saved)
                for s in saved:
                    LOGGER.info("Roformer: wrote %s/%s",
                                s.get("subfolder", ""), s.get("filename", ""))
            else:
                for b in range(n):
                    suffix = f"_b{b + 1:03d}" if n > 1 else ""
                    path = Path(folder) / f"{filename}_{counter:05d}_{safe}{suffix}.wav"
                    sf.write(str(path), wave[b].cpu().numpy().T, rate,  # (N, C) for soundfile
                             subtype="FLOAT")
                    results.append(UI.SavedResult(
                        filename=path.name, subfolder=subfolder, type=io.FolderType.output))
                    LOGGER.info("Roformer: wrote %s", path)
                    counter += 1
        return io.NodeOutput(ui=UI.SavedAudios(results))


class RoformerInfo(io.ComfyNode):
    """The separation summary as a JSON string."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerInfo",
            display_name="Roformer Info",
            category=CATEGORY,
            search_aliases=["info", "summary"],
            description="The separation summary as JSON (model, stems, lengths, rates, chunking).",
            inputs=[ROFORMER_STEMS.Input("stems")],
            outputs=[io.String.Output(id="info_json", display_name="info_json")],
        )

    @classmethod
    def execute(cls, stems: dict) -> io.NodeOutput:
        return io.NodeOutput(json.dumps(stems["info"], indent=2, sort_keys=True))


#: id -> class registry. The ComfyUI loader does not read this dict: the pack registers through
#: the V3 entrypoint in __init__.py (comfy_entrypoint -> get_node_list). The mapping stays as the
#: single reference list and the test seam; it must match get_node_list.
NODE_CLASS_MAPPINGS = {
    "RoformerLoader": RoformerLoader,
    "RoformerSeparate": RoformerSeparate,
    "RoformerEnsemble": RoformerEnsemble,
    "RoformerAlign": RoformerAlign,
    "RoformerPitchShift": RoformerPitchShift,
    "RoformerTimeStretch": RoformerTimeStretch,
    "RoformerKeyDetect": RoformerKeyDetect,
    "RoformerToAudio": RoformerToAudio,
    "CustomRoformerToAudio": CustomRoformerToAudio,
    "RoformerSaveAudio": RoformerSaveAudio,
    "RoformerInfo": RoformerInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RoformerLoader": "Roformer Loader",
    "RoformerSeparate": "Roformer Separate",
    "RoformerEnsemble": "Roformer Ensemble",
    "RoformerAlign": "Roformer Align",
    "RoformerPitchShift": "Roformer Pitch Shift",
    "RoformerTimeStretch": "Roformer Time Stretch",
    "RoformerKeyDetect": "Roformer Key Detect",
    "RoformerToAudio": "Roformer To Audio",
    "CustomRoformerToAudio": "Custom Roformer To Audio",
    "RoformerSaveAudio": "Roformer Save Audio",
    "RoformerInfo": "Roformer Info",
}
