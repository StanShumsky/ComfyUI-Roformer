"""Checkpoint download, local discovery and verified model building.

The .ckpt files are bare state dicts; each one is configured by the YAML that ships next to it
(audio / model / training / inference sections). Loading is strict by design: a checkpoint that
does not match its config is a loud error, never a silently half-initialised model.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

import torch
import yaml

import folder_paths
import comfy.model_management as mm
import comfy.model_patcher

from .manifest import BY_ID, Checkpoint, is_known_id
from .model import (
    BSRoformer,
    BSRoformerFNO,
    BSRoformerHyperACE,
    BSRoformerHyperACEv2,
    BSRoformerLargeInst,
    BSRoformerSiamese,
    MelBandRoformer,
)

LOGGER = logging.getLogger(__name__)

MODELS_DIR = Path(folder_paths.models_dir) / "roformer"
folder_paths.add_model_folder_path("roformer", str(MODELS_DIR))

# family -> model class. The "bs-*" and "siamese" families use the model code each repo ships
# next to its weights (see roformer/model/__init__.py); "bs" and "melband" use the 0.4.1 line.
CLASS_BY_FAMILY = {
    "bs": BSRoformer,
    "melband": MelBandRoformer,
    "bs-fno": BSRoformerFNO,
    "bs-hyperace": BSRoformerHyperACE,
    "bs-hyperace-v2": BSRoformerHyperACEv2,
    "bs-large-inst": BSRoformerLargeInst,
    "siamese": BSRoformerSiamese,
}


def model_dirs() -> list[Path]:
    dirs = [Path(p) for p in folder_paths.get_folder_paths("roformer")]
    if MODELS_DIR not in dirs:
        dirs.append(MODELS_DIR)
    return list(dict.fromkeys(dirs))


def local_models() -> dict[str, dict]:
    """Folders in the model dirs holding at least one checkpoint (id -> info).

    info: {ckpt, config, label, n_ckpts}. With several checkpoints in one folder the first
    (alphabetical) is used; keep one model per folder for unambiguous loading.
    Both .ckpt and .pth files count (community releases ship either extension).
    """
    found: dict[str, dict] = {}
    for base in model_dirs():
        if not base.exists():
            continue
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            if sub.name.startswith(".") or ".cache" in sub.parts:
                continue
            ckpts = sorted({*sub.rglob("*.ckpt"), *sub.rglob("*.pth")})
            if not ckpts:
                continue
            ckpt = ckpts[0]
            # a roformer collection is a checkpoint with its config YAML beside it -
            # the requirement also keeps unrelated .pth folders (faced-rest, SAMs, ...)
            # out of the dropdown when a shared models dir is on the path
            configs = sorted(ckpt.parent.glob("*.yaml"))
            if not configs:
                continue
            found.setdefault(sub.name, {
                "ckpt": ckpt,
                "config": configs[0] if len(configs) == 1 else None,
                "label": f"{sub.name} (local)",
                "n_ckpts": len(ckpts),
            })
    return found


def load_yaml_config(path: Path) -> dict:
    # the community configs use !!python/tuple tags, which need FullLoader (not safe_load)
    cfg = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    for section in ("audio", "model"):
        if section not in cfg:
            raise ValueError(f"{path.name}: missing the {section!r} section.")
    return cfg


def check_weights(state: dict, name: str) -> None:
    """Corrupt-download guard: a truncated or zero-filled file still strict-loads but
    predicts garbage, so it is caught here instead."""
    checked = 0
    nonfinite, zeros = [], []
    for index, (key, value) in enumerate(state.items()):
        if index % 64 == 0:
            mm.throw_exception_if_processing_interrupted()
        if not torch.is_tensor(value) or not value.is_floating_point() or value.numel() <= 100:
            continue
        checked += 1
        if not torch.isfinite(value).all():
            nonfinite.append(key)
        elif not value.any():
            zeros.append(key)
    if nonfinite or (checked and len(zeros) > checked / 2):
        raise RuntimeError(
            f"{name}: weights look corrupt ({len(nonfinite)} tensors with NaN/inf, "
            f"{len(zeros)} of {checked} tensors all zero, e.g. {nonfinite[:2] + zeros[:2]}). "
            "The file is probably an incomplete download."
        )


def download_model(model: Checkpoint, root: Path) -> Path:
    """Fetch missing files; Hugging Face Hub honors HF_HUB_OFFLINE and its own cache/retry policy."""
    target = root / model.id
    patterns = [model.ckpt_file]
    if model.config_file:
        patterns.append(model.config_file)
    else:
        patterns.append(f"{Path(model.ckpt_file).parent}/*.yaml")
    LOGGER.info("Downloading %s/%s to %s", model.repo_id, model.ckpt_file, target)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model.repo_id, allow_patterns=patterns, local_dir=str(target))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {model.repo_id} ({model.ckpt_file}) into {target}. "
            "Check the network connection and Hugging Face token for gated repositories, or "
            f"place {model.ckpt_file} and its YAML config there manually. Complete local files "
            "are reused without network access; HF_HUB_OFFLINE=1 is appropriate only when they "
            "are already installed."
        ) from exc
    required = [model.ckpt_file]
    if model.config_file:
        required.append(model.config_file)
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Download of {model.repo_id} did not produce {missing} in {target}; "
            "remove the incomplete folder and retry."
        )
    return target


def _checkpoint_config(ckpt: Path, model: Checkpoint) -> Path | None:
    if model.config_file:
        preferred = ckpt.parent / Path(model.config_file).name
        if preferred.is_file():
            return preferred
    siblings = sorted(ckpt.parent.glob("*.yaml"))
    if len(siblings) == 1:
        return siblings[0]
    return preferred if model.config_file else None


def _locate(ckpt_ref: str) -> tuple[Path, Path | None, Checkpoint | None]:
    """A manifest id or a local folder name -> (ckpt path, config path, manifest entry)."""
    if ckpt_ref in BY_ID:  # dict lookup, not is_known_id: tests patch loading.BY_ID
        model = BY_ID[ckpt_ref]
        if not model.supported:
            raise ValueError(
                f"Model {ckpt_ref!r} is not supported by this pack: {model.note}"
            )
        for base in model_dirs():
            target = base / model.id
            ckpt = target / model.ckpt_file
            if ckpt.is_file():
                config = _checkpoint_config(ckpt, model)
                if config is not None and config.is_file():
                    return ckpt, config, model
        local = local_models().get(model.id)
        if local is not None and local["config"] is not None:
            return local["ckpt"], local["config"], model
        target = download_model(model, model_dirs()[0])
        ckpt = target / model.ckpt_file
        return ckpt, _checkpoint_config(ckpt, model), model
    local = local_models().get(ckpt_ref)
    if local is not None:
        return local["ckpt"], local["config"], None
    raise FileNotFoundError(
        f"Model {ckpt_ref!r} not found: unknown id and no local folder in "
        f"{[str(d) for d in model_dirs()]}. Known ids: {list(BY_ID)}"
    )


def _constructor_kwargs(cls, mcfg: dict) -> dict:
    # inspect.signature, not __code__: the __init__ is wrapped by beartype, whose generated
    # code object exposes the wrapper's internal names, not the model's parameters
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for key, value in mcfg.items():
        if key not in sig.parameters:
            continue
        ann = sig.parameters[key].annotation
        # YAML flow sequences (freqs_per_bands: [..]) parse as lists; the model wants tuples
        if isinstance(value, list) and ann is not inspect.Parameter.empty and "tuple" in str(ann).lower():
            value = tuple(value)
        kwargs[key] = value
    dropped = sorted(set(mcfg) - set(sig.parameters))
    if dropped:
        LOGGER.debug("Ignoring config keys the model does not take: %s", dropped)
    return kwargs


def apply_family_defaults(cls, mcfg: dict, state: dict, ckpt_name: str) -> dict:
    """Family-specific constructor defaults the released configs leave implicit."""
    kwargs = _constructor_kwargs(cls, mcfg)
    if cls is MelBandRoformer:
        # The mel-band configs omit this and the released checkpoints were trained with the linear
        # (band) transformer disabled, so the per-depth block is [time, freq] (slots 0 and 1).
        if "linear_transformer_depth" not in mcfg:
            kwargs["linear_transformer_depth"] = 0
        if any(k.startswith("layers.0.0.") and k.endswith("temperature") for k in state):
            raise RuntimeError(
                f"{ckpt_name}: the weights contain a linear (band) transformer, but this pack builds "
                "the released [time, freq] layout. Add linear_transformer_depth to the config."
            )
    return kwargs


def _resolve_load_device(device: str) -> torch.device:
    """An explicit run device, always with an index: a bare 'cuda'/'mps' has .index == None,
    and ComfyUI's model management (the VBAR pageout path, used when a model loads under
    VRAM pressure) does int(device.index) - a None index is the "int() argument must be a
    string, a bytes-like object or a real number, not 'NoneType'" crash."""
    if device == "cuda":
        return torch.device(f"cuda:{mm.get_torch_device().index or 0}")
    if device == "mps":
        return torch.device("mps:0")
    return torch.device(device)


def load_roformer(ckpt_ref: str, device: str, precision: str = "float32") -> dict:
    """Build one model payload: verified weights in system RAM, its config, stem names and
    chunk geometry. The weights ride on the offload device until the Separate node's run
    loads them onto the GPU (ComfyUI's model management evicts them again on demand)."""
    if precision not in ("float32", "float16"):
        raise ValueError("Roformer precision must be 'float32' or 'float16'.")
    mm.throw_exception_if_processing_interrupted()
    if device == "auto":
        # The native ComfyUI device policy for auxiliary models: honors --gpu-only and the
        # session VRAM state (a NO_VRAM session runs the model on the CPU).
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()
    else:
        load_device = _resolve_load_device(device)
        offload_device = torch.device("cpu")
    ckpt_path, config_path, model = _locate(ckpt_ref)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(
            f"No config YAML found next to {ckpt_path.name}. Copy the config YAML from the "
            f"Hugging Face repo beside the .ckpt (or check the manifest entry for it)."
        )
    cfg = load_yaml_config(config_path)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mm.throw_exception_if_processing_interrupted()
    state.pop("_metadata", None)  # torch.save state-dict artifact, not weights (some ckpts, e.g. inst-fno)
    check_weights(state, ckpt_path.name)

    mcfg = cfg["model"]
    if model is not None:
        cls = CLASS_BY_FAMILY[model.family]
    else:
        cls = MelBandRoformer if "num_bands" in mcfg else BSRoformer

    kwargs = apply_family_defaults(cls, mcfg, state, ckpt_path.name)
    # Build the weights in system RAM (the state dict is already CPU-resident); no GPU copy
    # exists until a run asks for it.
    net = cls(**kwargs)
    net.load_state_dict(state, strict=True)  # hard fail: no silent partial loads
    net.eval()
    mm.throw_exception_if_processing_interrupted()
    # Hand the weights to ComfyUI's model management (the native pattern for auxiliary
    # models): the model rides on the offload device until a run loads it onto the GPU,
    # and is evicted again when other nodes need the VRAM.
    patcher = comfy.model_patcher.CoreModelPatcher(net, load_device, offload_device)

    instruments = [str(s) for s in cfg.get("training", {}).get("instruments", ["target", "other"])]
    target = cfg.get("training", {}).get("target_instrument")
    if target is not None:
        target = str(target)
        if target not in instruments:
            instruments.append(target)
    audio = cfg["audio"]
    return {
        "net": net,
        "patcher": patcher,
        "device": load_device,
        "offload_device": offload_device,
        "precision": precision,
        "id": model.id if model is not None else ckpt_ref,
        "label": model.label if model is not None else f"{ckpt_ref} (local)",
        "family": model.family if model is not None else ("melband" if "num_bands" in mcfg else "bs"),
        "stems": instruments,
        "target": target if target in instruments else instruments[0],
        "num_channels": int(audio.get("num_channels", 2)),
        "sample_rate": int(audio.get("sample_rate", 44100)),
        "chunk_size": int(audio.get("chunk_size", 882000)),
        "num_stems": int(mcfg.get("num_stems", 1)),
        "ckpt_path": str(ckpt_path),
        "cfg": cfg,
    }
