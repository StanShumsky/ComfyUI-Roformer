"""Vendored Roformer model code for the pcunwa checkpoints.

Three sources of model code, one per architecture family. The checkpoints are bare state dicts
and each was trained against a specific revision of the Roformer code, so loading any of them
strictly requires building the model from the *same* code it was trained with.

Base (bs_roformer.py + mel_band_roformer.py): the BS-RoFormer 0.4.1 wheel (MIT, lucidrains). It
is the last PyPI release whose state-dict layout matches the base BS-Roformer and the Mel-Band
line. Two deliberate changes from upstream:

1. mlp_expansion_factor is plumbed through from the model config (both classes), because some
   checkpoints (e.g. the mel-band line) were trained with factor 1 while upstream hardcodes 4.
2. MelBandRoformer appends the optional linear (band) transformer to each per-depth block only
   when enabled, instead of leaving a None placeholder at slot 0. The released checkpoints were
   trained with it disabled, so their blocks are [time, freq] at slots 0 and 1; the placeholder
   layout would shift the keys to slots 1 and 2 and fail a strict load.

Repo-shipped (bs_roformer_*.py): three pcunwa repos ship their own model code next to the
weights, and their checkpoints do NOT strict-load against 0.4.1 — the MaskEstimator (and, for
the Siamese release, the per-depth transformer) is extended. Each file below is that repo's
bs_roformer.py, taken verbatim with two mechanical changes: the training-repo import
``from models.bs_roformer.attend import Attend`` is repointed to the vendored ``.attend`` (the
Attention/Attend classes are byte-identical across all revisions), and, for the HyperACE v1 file
only, ``import torchaudio`` is guarded (it is present in their training env but unused at
inference, and it is not a dependency of this pack).

- bs_roformer_hyperace.py      : pcunwa/BS-Roformer-HyperACE (SegmModel mask estimator)
- bs_roformer_hyperace_v2.py   : pcunwa/BS-Roformer-HyperACE/v2_* (DSConv/TFC-TDF variant)
- bs_roformer_large_inst.py    : pcunwa/BS-Roformer-Large-Inst (transformer mask estimator)
- bs_roformer_siamese.py       : pcunwa/BS-EXP-SiameseRoformer (SiameseTransformer)
- bs_roformer_fno.py           : pcunwa/BS-Roformer-Inst-FNO — the 0.4.1 base with the
  MaskEstimator replaced by the FNO1d version the repo's README publishes (needs the
  neuraloperator package, pinned to 1.0.2)
"""
from .bs_roformer import BSRoformer
from .mel_band_roformer import MelBandRoformer
from .bs_roformer_fno import BSRoformer as BSRoformerFNO
from .bs_roformer_hyperace import BSRoformer as BSRoformerHyperACE
from .bs_roformer_hyperace_v2 import BSRoformer as BSRoformerHyperACEv2
from .bs_roformer_large_inst import BSRoformer as BSRoformerLargeInst
from .bs_roformer_siamese import BSRoformer as BSRoformerSiamese

__all__ = [
    "BSRoformer",
    "MelBandRoformer",
    "BSRoformerFNO",
    "BSRoformerHyperACE",
    "BSRoformerHyperACEv2",
    "BSRoformerLargeInst",
    "BSRoformerSiamese",
]
