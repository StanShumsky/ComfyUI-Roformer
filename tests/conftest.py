"""Stubs for the ComfyUI modules the nodes import, so tests run without a ComfyUI install."""
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
STATE = {
    "output_dir": tempfile.gettempdir(),
    "temp_dir": str(Path(tempfile.gettempdir()) / "roformer-temp"),
    "models_dir": os.environ.get("ROFORMER_MODELS_DIR", str(Path(tempfile.gettempdir()) / "roformer-models")),
}


def _install_stubs() -> None:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = STATE["models_dir"]
    folder_paths.add_model_folder_path = lambda *a, **k: None
    folder_paths.get_folder_paths = lambda name: [STATE["models_dir"]]
    folder_paths.get_output_directory = lambda: STATE["output_dir"]

    def get_temp_directory():
        Path(STATE["temp_dir"]).mkdir(parents=True, exist_ok=True)
        return STATE["temp_dir"]

    folder_paths.get_temp_directory = get_temp_directory

    def get_save_image_path(prefix, output_dir, width, height):
        sub = Path(prefix).parent
        subfolder = "" if str(sub) == "." else str(sub)
        filename = Path(prefix).name
        folder = Path(output_dir) / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        counter = len(list(folder.glob(filename + "_*.png")))
        return str(folder), filename, counter, subfolder, prefix

    folder_paths.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    mm.soft_empty_cache = lambda: None
    cu = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.done = 0

        def update(self, n):
            self.done += n

    cu.ProgressBar = ProgressBar
    comfy.model_management = mm
    comfy.utils = cu
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.utils"] = cu


_install_stubs()
_spec = importlib.util.spec_from_file_location(
    "roformernode", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["roformernode"] = _pkg
_spec.loader.exec_module(_pkg)
# pytest's Package.setup imports the repo-root __init__.py as top-level "__init__"
# (the hyphenated directory name is not a valid package identifier); reuse our
# correctly-loaded package instead of letting it re-import without a parent package.
sys.modules.setdefault("__init__", _pkg)


@pytest.fixture(scope="session")
def nodes():
    return sys.modules["roformernode.nodes"]


@pytest.fixture
def output_dir(tmp_path):
    STATE["output_dir"] = str(tmp_path)
    STATE["temp_dir"] = str(tmp_path / "temp")
    return tmp_path


@pytest.fixture
def models_dir(tmp_path):
    """A throwaway models dir, also pointed at by the folder_paths stub."""
    STATE["models_dir"] = str(tmp_path / "models")
    Path(STATE["models_dir"]).mkdir(parents=True, exist_ok=True)
    return Path(STATE["models_dir"])


def _make_audio(t=2.0, rate=44100, freq=440.0, channels=2):
    """A (1, C, N) ComfyUI AUDIO payload: a sine at `freq` Hz on every channel."""
    n = int(t * rate)
    x = torch.sin(2 * 3.141592653589793 * freq * torch.arange(n) / rate).float()
    return {"waveform": x.repeat(channels, 1)[None], "sample_rate": rate}


@pytest.fixture
def make_audio():
    return _make_audio


@pytest.fixture
def tiny_model(models_dir):
    """A tiny BSRoformer (4 bands, dim 32) saved as a local collection, loaded through the
    real loader path: models dir -> .ckpt + config.yaml -> strict build."""
    from roformernode.roformer.model import BSRoformer

    cfg = {
        "dim": 32, "depth": 1, "stereo": True, "num_stems": 1,
        "time_transformer_depth": 1, "freq_transformer_depth": 1,
        "freqs_per_bands": (256, 256, 256, 257),  # must sum to n_fft // 2 + 1
        "dim_head": 16, "heads": 2,
        "stft_n_fft": 2048, "stft_hop_length": 1024, "stft_win_length": 2048,
        "stft_normalized": False, "mask_estimator_depth": 1,
        "mlp_expansion_factor": 1, "flash_attn": False,
    }
    torch.manual_seed(0)
    net = BSRoformer(**cfg)
    d = models_dir / "tiny"
    d.mkdir(parents=True)
    torch.save(net.state_dict(), str(d / "tiny.ckpt"))
    yaml_text = (
        "audio:\n"
        "  chunk_size: 88200\n"
        "  num_channels: 2\n"
        "  sample_rate: 44100\n"
        "model:\n"
        "  dim: 32\n"
        "  depth: 1\n"
        "  stereo: true\n"
        "  num_stems: 1\n"
        "  time_transformer_depth: 1\n"
        "  freq_transformer_depth: 1\n"
        "  freqs_per_bands: [256, 256, 256, 257]\n"
        "  dim_head: 16\n"
        "  heads: 2\n"
        "  stft_n_fft: 2048\n"
        "  stft_hop_length: 1024\n"
        "  stft_win_length: 2048\n"
        "  stft_normalized: false\n"
        "  mask_estimator_depth: 1\n"
        "  mlp_expansion_factor: 1\n"
        "  flash_attn: false\n"
        "training:\n"
        "  instruments:\n"
        "  - vocals\n"
        "  - other\n"
        "  target_instrument: vocals\n"
        "inference:\n"
        "  batch_size: 1\n"
    )
    (d / "config.yaml").write_text(yaml_text, encoding="utf-8")
    return d
