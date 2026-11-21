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

from .roformer.audio_io import audio_to_tensor, resample, to_audio_payload
from .roformer.loading import load_roformer, local_models
from .roformer.manifest import MODELS
from .roformer.separation import Separator

LOGGER = logging.getLogger(__name__)


def _model_choices() -> list[str]:
    choices = [m.id for m in MODELS]
    for name in sorted(local_models()):
        if name not in choices:
            choices.append(name)
    return choices


class RoformerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (_model_choices(), {"default": "sw-6stem"}),
            "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
        }}

    RETURN_TYPES = ("ROFORMER_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "audio/Roformer"
    DESCRIPTION = (
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
        "The Kim FT ids are in a gated repo and need an HF token (HF_TOKEN or huggingface-cli login)."
    )

    def load(self, model: str, device: str):
        torch_device = mm.get_torch_device() if device == "auto" else torch.device(device)
        payload = load_roformer(model, torch_device)
        LOGGER.info("Roformer: loaded %s from %s in fp32 on %s.",
                    payload["label"], payload["ckpt_path"], torch_device)
        return (payload,)


class RoformerSeparate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("ROFORMER_MODEL",),
            "audio": ("AUDIO",),
            "chunk_seconds": ("INT", {"default": 0, "min": 0, "max": 240}),
        }}

    RETURN_TYPES = ("ROFORMER_STEMS", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("stems", "target", "other", "info_json")
    FUNCTION = "separate"
    CATEGORY = "audio/Roformer"
    DESCRIPTION = (
        "Separate the audio with the loaded model. The input is resampled to the model's "
        "44.1 kHz (stereo, or downmixed / upmixed to the model's trained channel count) and "
        "the stems come back at the file's own sample rate. chunk_seconds: 0 = the model's own "
        "training window from its config (recommended); a positive value caps the window for "
        "low-VRAM machines (overlapping windows are overlap-added with a linear crossfade). "
        "Outputs: the stems payload (Roformer To Audio / Save Audio below), the target stem and "
        "the other stem as ready AUDIO (for multi-stem models: the 'other' stem when the model "
        "has one, else the first non-target stem - use the stems payload for the full set), "
        "and info_json (model, stems, lengths, chunking)."
    )

    def separate(self, model: dict, audio: dict, chunk_seconds: int):
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
        return (
            {"stems": stems, "sample_rate": rate, "info": info},
            to_audio_payload(stems[target_name], rate),
            to_audio_payload(stems[other_name], rate),
            json.dumps(info, indent=2, sort_keys=True),
        )


class RoformerToAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "stems": ("ROFORMER_STEMS",),
            "stem": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "to_audio"
    CATEGORY = "audio/Roformer"
    DESCRIPTION = (
        "Extract one named stem from a stems payload as an AUDIO you can feed anywhere else "
        "(preview, save, or another separator). stem: the exact name from info_json (empty = "
        "the first stem, i.e. the model's target)."
    )

    def to_audio(self, stems: dict, stem: str):
        s = stems["stems"]
        name = stem if stem in s else next(iter(s))
        return (to_audio_payload(s[name], stems["sample_rate"]),)


class RoformerSaveAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "stems": ("ROFORMER_STEMS",),
            "filename_prefix": ("STRING", {"default": "roformer/stem"}),
            "stem": ("STRING", {"default": ""}),
            "format": (["wav", "flac"], {"default": "wav"}),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "audio/Roformer"
    DESCRIPTION = (
        "Write the stems to ComfyUI's output folder as <prefix>_XXXXX_<stem>.<format> "
        "(float32 WAV or FLAC, both lossless), numbered so runs never overwrite each other. "
        "stem: empty = save every stem in the payload."
    )

    def save(self, stems: dict, filename_prefix: str, stem: str, format: str):
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 1, 1
        )
        # get_save_image_path counts PNGs only; count our own format to avoid overwrites.
        counter = len(list(Path(folder).glob(f"{filename}_*.{format}")))
        names = [stem] if stem else list(stems["stems"])
        rate = stems["sample_rate"]
        for name in names:
            if name not in stems["stems"]:
                raise ValueError(
                    f"Stem {name!r} not in the payload; available: {list(stems['stems'])}."
                )
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            path = Path(folder) / f"{filename}_{counter:05d}_{safe}.{format}"
            wave = stems["stems"][name].cpu().numpy().T  # (N, C) for soundfile
            sf.write(str(path), wave, rate, subtype="FLOAT" if format == "wav" else None)
            counter += 1
            LOGGER.info("Roformer: wrote %s", path)
        return ()  # an output node must return a tuple


class RoformerInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"stems": ("ROFORMER_STEMS",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_json",)
    FUNCTION = "info"
    CATEGORY = "audio/Roformer"
    DESCRIPTION = "The separation summary as JSON (model, stems, lengths, rates, chunking)."

    def info(self, stems: dict):
        return (json.dumps(stems["info"], indent=2, sort_keys=True),)


NODE_CLASS_MAPPINGS = {
    "RoformerLoader": RoformerLoader,
    "RoformerSeparate": RoformerSeparate,
    "RoformerToAudio": RoformerToAudio,
    "RoformerSaveAudio": RoformerSaveAudio,
    "RoformerInfo": RoformerInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RoformerLoader": "Roformer Loader",
    "RoformerSeparate": "Roformer Separate",
    "RoformerToAudio": "Roformer To Audio",
    "RoformerSaveAudio": "Roformer Save Audio",
    "RoformerInfo": "Roformer Info",
}
