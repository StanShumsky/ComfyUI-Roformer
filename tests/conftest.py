"""Stubs for the ComfyUI modules the nodes import, so tests run without a ComfyUI install."""
import enum
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

    def get_save_image_path(prefix, output_dir, *args):
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
    # RF_TEST_DEVICE forces the suite onto a specific backend (e.g. cpu on a machine whose
    # CUDA driver is busy) without touching the ComfyUI device policy the nodes honour.
    forced = os.environ.get("RF_TEST_DEVICE", "")

    def get_torch_device():
        if forced:
            return torch.device(forced)
        return torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    mm.get_torch_device = get_torch_device
    mm.text_encoder_device = mm.get_torch_device
    mm.text_encoder_offload_device = lambda: torch.device("cpu")
    mm.text_encoder_dtype = lambda device: torch.float16

    def load_model_gpu(patcher):
        # Mirror the real model management's full load: everything the patcher owns ends
        # up on the load device. The real one only partial-loads when the device cannot
        # hold the model - the nodes detect that themselves and complete the move.
        patcher.model.to(patcher.load_device)
        patcher.model.model_loaded_weight_memory = patcher.model_size()

    mm.load_model_gpu = load_model_gpu
    mm.soft_empty_cache = lambda: None
    mp = types.ModuleType("comfy.model_patcher")

    class CoreModelPatcher:
        """The surface the nodes use, with the same bookkeeping the real patcher keeps."""

        def __init__(self, model, load_device, offload_device):
            self.model = model
            self.load_device = load_device
            self.offload_device = offload_device
            self.size = 0
            if not hasattr(model, "model_loaded_weight_memory"):
                model.model_loaded_weight_memory = 0

        def model_size(self):
            if self.size > 0:
                return self.size
            self.size = sum(
                t.numel() * t.element_size()
                for t in list(self.model.parameters()) + list(self.model.buffers())
            )
            return self.size

    mp.CoreModelPatcher = CoreModelPatcher
    cu = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.done = 0

        def update(self, n):
            self.done += n

    cu.ProgressBar = ProgressBar
    comfy.model_management = mm
    comfy.model_patcher = mp
    comfy.utils = cu
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.model_patcher"] = mp
    sys.modules["comfy.utils"] = cu


def _install_comfy_api_stub() -> None:
    """The real comfy_api.latest drags in the whole ComfyUI runtime; the tests only need the io
    surface the nodes use, so install a minimal stand-in when the real one is not importable."""
    try:
        import comfy_api.latest  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass

    class _Base:
        def __init__(self, id=None, display_name=None, tooltip=None, optional=False, default=None, **kwargs):
            self.id, self.display_name, self.tooltip, self.optional = id, display_name, tooltip, optional
            self.default, self.kwargs = default, kwargs

    def _custom(io_type):
        t = io_type  # class bodies do not close over function locals; bind under a plain name

        class _CustomInput(_Base):
            io_type = t

        class _CustomOutput(_Base):
            io_type = t

        outer = type("Custom", (), {"io_type": t})
        outer.Input = _CustomInput
        outer.Output = _CustomOutput
        return outer

    class _Audio:
        Input = _Base
        Output = _Base

    class _Combo:
        class Input(_Base):
            def __init__(self, id=None, options=None, display_name=None, tooltip=None,
                         optional=False, default=None, **kwargs):
                super().__init__(id, display_name, tooltip, optional, **kwargs)
                self.options = options or []
                self.default = default

    class _Int:
        Input = _Base

    class _String:
        Input = _Base
        Output = _Base

    class _Hidden(enum.Enum):
        # mirror the real comfy_api Hidden enum (the schema's hidden= entries)
        prompt = "PROMPT"
        extra_pnginfo = "EXTRA_PNGINFO"

    class _Schema:
        def __init__(self, node_id=None, display_name=None, category=None, description=None,
                     inputs=None, outputs=None, search_aliases=None, hidden=None, **kwargs):
            self.node_id, self.display_name, self.category = node_id, display_name, category
            self.description = description
            self.inputs, self.outputs = inputs or [], outputs or []
            self.search_aliases = search_aliases or []
            self.hidden = hidden or []
            self.is_output_node = kwargs.get("is_output_node", False)
            # mirror the real Schema finalization: outputs without ids get default ids
            for i, output in enumerate(self.outputs):
                if getattr(output, "id", None) is None:
                    output.id = f"_{i}_{getattr(output, 'io_type', 'X')}_"

    class _ComfyNode:
        pass

    class _NodeOutput:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.ui = kwargs.get("ui")

        @property
        def result(self):
            return self.args if len(self.args) > 0 else None

    class _FolderType:
        output = "output"
        temp = "temp"

    class _SavedResult(dict):
        def __init__(self, filename, subfolder, type):
            super().__init__(filename=filename, subfolder=subfolder, type=getattr(type, "value", type))

    class _SavedAudios:
        def __init__(self, results):
            self.results = results

        def as_dict(self):
            return {"audio": self.results}

    class _AudioSaveHelper:
        """The surface AudioSaveHelper.save_audio uses, with the same on-disk behaviour:"""

        @staticmethod
        def save_audio(audio, filename_prefix, folder_type, cls, format="flac", quality="128k"):
            import soundfile as sf_stub
            fp = sys.modules["folder_paths"]  # installed by _install_stubs
            full_output_folder, filename, counter, subfolder, _ = fp.get_save_image_path(
                filename_prefix, fp.get_output_directory()
            )
            results = []
            for batch_number, waveform in enumerate(audio["waveform"].cpu()):
                file = f"{filename}_{counter:05}.{format}"
                sf_stub.write(os.path.join(full_output_folder, file),
                              waveform.T.numpy().astype("float32"), audio["sample_rate"],
                              format="FLAC")
                results.append(_SavedResult(filename=file, subfolder=subfolder, type=folder_type))
            return results

    ui = types.SimpleNamespace(
        SavedResult=_SavedResult, SavedAudios=_SavedAudios, AudioSaveHelper=_AudioSaveHelper,
    )
    io = types.SimpleNamespace(
        Custom=_custom, Audio=_Audio, Combo=_Combo, Int=_Int, String=_String,
        Schema=_Schema, ComfyNode=_ComfyNode, NodeOutput=_NodeOutput,
        Hidden=_Hidden,
        FolderType=_FolderType,
        Autogrow=types.SimpleNamespace(Input=type("_AutogrowInput", (), {})),
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.UI = ui
    latest.ComfyExtension = type("ComfyExtension", (), {})
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest


_install_stubs()
_install_comfy_api_stub()
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


@pytest.fixture(scope="session")
def manifest():
    return sys.modules["roformernode.roformer.manifest"].MODELS


@pytest.fixture(scope="session")
def stem_names():
    return sys.modules["roformernode.roformer.stem_names"]


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
