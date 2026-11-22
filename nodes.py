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

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from comfy_api.latest import io

from .roformer.audio_io import audio_to_tensor, resample, to_audio_payload
from .roformer.loading import load_roformer, local_models
from .roformer.manifest import MODELS
from .roformer.separation import Separator
from .roformer.stem_names import ALL_STEMS, canonical

LOGGER = logging.getLogger(__name__)

CATEGORY = "audio/Roformer"
ROFORMER_MODEL = io.Custom("ROFORMER_MODEL")
ROFORMER_STEMS = io.Custom("ROFORMER_STEMS")


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
                io.Combo.Input("device", options=["auto", "cuda", "mps", "cpu"], default="auto"),
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
        torch_device = mm.get_torch_device() if device == "auto" else torch.device(device)
        payload = load_roformer(model, torch_device)
        LOGGER.info("Roformer: loaded %s from %s in fp32 on %s.",
                    payload["label"], payload["ckpt_path"], torch_device)
        return io.NodeOutput(payload, ", ".join(payload["stems"]))


class RoformerSeparate(io.ComfyNode):
    """Separate the audio into stems: the stems payload plus the target/other stems as AUDIO."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerSeparate",
            display_name="Roformer Separate",
            category=CATEGORY,
            description=(
                "Separate the audio with the loaded model. The input is resampled to the model's "
                "44.1 kHz (stereo, or downmixed / upmixed to the model's trained channel count) and "
                "the stems come back at the file's own sample rate. chunk_seconds: 0 = the model's own "
                "training window from its config (recommended); a positive value caps the window for "
                "low-VRAM machines (overlapping windows are overlap-added with a linear crossfade). "
                "Outputs: the stems payload (Roformer To Audio / Save Audio below), the target stem and "
                "the other stem as ready AUDIO (for multi-stem models: the 'other' stem when the model "
                "has one, else the first non-target stem - use the stems payload for the full set), "
                "and info_json (model, stems, lengths, chunking)."
            ),
            inputs=[
                ROFORMER_MODEL.Input("model"),
                io.Audio.Input("audio"),
                io.Int.Input("chunk_seconds", default=0, min=0, max=240,
                             tooltip="0 = the model's own training window from its config (recommended); a "
                                     "positive value caps the window for low-VRAM machines."),
            ],
            outputs=[
                ROFORMER_STEMS.Output(id="stems", display_name="stems"),
                io.Audio.Output(id="target", display_name="target"),
                io.Audio.Output(id="other", display_name="other"),
                io.String.Output(id="info_json", display_name="info_json"),
            ],
        )

    @classmethod
    def execute(cls, model: dict, audio: dict, chunk_seconds: int = 0) -> io.NodeOutput:
        x, rate = audio_to_tensor(audio)
        p = model
        if p["num_channels"] == 1 and x.shape[0] > 1:
            x = x.mean(0, keepdim=True)
        elif p["num_channels"] == 2 and x.shape[0] == 1:
            x = x.repeat(2, 1)
        x44 = resample(x, rate, p["sample_rate"])
        progress = ProgressBar(1)
        stems44 = Separator(p).run(x44, chunk_seconds, progress)
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
        }
        return io.NodeOutput(
            {"stems": stems, "sample_rate": rate, "info": info},
            to_audio_payload(stems[target_name], rate),
            to_audio_payload(stems[other_name], rate),
            json.dumps(info, indent=2, sort_keys=True),
        )


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
                io.Combo.Input("stem", options=[AUTO_STEM] + list(ALL_STEMS), default=AUTO_STEM),
            ],
            outputs=[io.Audio.Output(id="audio", display_name="audio")],
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
            description=(
                "Like Roformer To Audio, but the stem is typed by hand instead of picked from the "
                "dropdown: the name from info_json, matched case-insensitively (empty = the first "
                "stem, i.e. the model's target). Use it for local models and custom checkpoints whose "
                "stem names are not in the standard list."
            ),
            inputs=[
                ROFORMER_STEMS.Input("stems"),
                io.String.Input("stem", default=""),
            ],
            outputs=[io.Audio.Output(id="audio", display_name="audio")],
        )


class RoformerSaveAudio(io.ComfyNode):
    """Write the stems to ComfyUI's output folder as <prefix>_XXXXX_<stem>.<format>."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerSaveAudio",
            display_name="Roformer Save Audio",
            category=CATEGORY,
            is_output_node=True,
            description=(
                "Write the stems to ComfyUI's output folder as <prefix>_XXXXX_<stem>.<format> "
                "(float32 WAV or FLAC, both lossless), numbered so runs never overwrite each other."
            ),
            inputs=[
                ROFORMER_STEMS.Input("stems"),
                io.String.Input("filename_prefix", default="roformer/stem"),
                io.Combo.Input("format", options=["wav", "flac"], default="wav"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, stems: dict, filename_prefix: str, format: str) -> io.NodeOutput:
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 1, 1
        )
        # get_save_image_path counts PNGs only; count our own format to avoid overwrites.
        counter = len(list(Path(folder).glob(f"{filename}_*.{format}")))
        rate = stems["sample_rate"]
        for name, wave in stems["stems"].items():
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            path = Path(folder) / f"{filename}_{counter:05d}_{safe}.{format}"
            sf.write(str(path), wave.cpu().numpy().T, rate,  # (N, C) for soundfile
                     subtype="FLOAT" if format == "wav" else None)
            counter += 1
            LOGGER.info("Roformer: wrote %s", path)
        return io.NodeOutput()  # an output node returns no values


class RoformerInfo(io.ComfyNode):
    """The separation summary as a JSON string."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RoformerInfo",
            display_name="Roformer Info",
            category=CATEGORY,
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
    "RoformerToAudio": RoformerToAudio,
    "CustomRoformerToAudio": CustomRoformerToAudio,
    "RoformerSaveAudio": RoformerSaveAudio,
    "RoformerInfo": RoformerInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RoformerLoader": "Roformer Loader",
    "RoformerSeparate": "Roformer Separate",
    "RoformerToAudio": "Roformer To Audio",
    "CustomRoformerToAudio": "Custom Roformer To Audio",
    "RoformerSaveAudio": "Roformer Save Audio",
    "RoformerInfo": "Roformer Info",
}
