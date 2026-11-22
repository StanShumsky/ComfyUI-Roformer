"""Per-model stem names for the curated checkpoints (static data).

Generated on 2026-11-21 from each checkpoint's config YAML (training.instruments /
target_instrument - exactly what roformer/loading.py resolves at load time). For the
dropdown every name is canonical (lower case, aliases merged: instrumental ->
instrument) and the lists are sorted by sort_stems: the stems users pick most often
first (vocals, the instrumental, drums, bass, guitar, piano, other), the rest
alphabetical. Matching in nodes.py is the mirror image: case-insensitive, alias and
prefix tolerant, so one dropdown covers every spelling a payload may use.
Consumer: nodes.py, the static option list of the Roformer To Audio dropdown
(all models, curated or local: the full superset of stem names).
"""
from __future__ import annotations

#: Dropdown priority per canonical stem name; names without an entry sort last.
PRIORITY: dict[str, int] = {
    "vocals": 0,
    "instrument": 1,
    "drums": 2,
    "bass": 3,
    "guitar": 4,
    "piano": 5,
    "other": 6,
    "noreverb": 7,
    "reverb": 8,
}

#: Different models spell the same stem differently; merge to one option.
ALIASES: dict[str, str] = {
    "instrumental": "instrument",
}

def canonical(name: str) -> str:
    """A stem name in its dropdown form: lower case, alias applied."""
    n = name.strip().lower()
    return ALIASES.get(n, n)

def sort_stems(names: list[str] | tuple[str, ...]) -> list[str]:
    """Stem names for a dropdown: canonical, de-duplicated, priority first, then a-z."""
    out, seen = [], set()
    for n in sorted((canonical(n) for n in names), key=lambda n: (PRIORITY.get(n, 100), n)):
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

MODEL_STEMS: dict[str, tuple[str, ...]] = {
    "sw-6stem": ("vocals", "drums", "bass", "guitar", "piano", "other"),
    "ft-4stems": ("vocals", "drums", "bass", "other"),
    "dereverb": ("noreverb", "reverb"),
    "karaoke-bs": ("vocals", "instrument"),
    "guitar-mel": ("guitar", "other"),
    "hyperace-v2-voc": ("vocals", "instrument"),
    "kim-ft": ("vocals", "other"),
    "leap-voc": ("vocals", "other"),
    "resurrection-voc": ("vocals", "other"),
    "revive": ("vocals", "other"),
    "revive2": ("vocals", "other"),
    "revive3e": ("vocals", "other"),
    "large-inst": ("vocals", "instrument"),
    "inst-fno": ("vocals", "other"),
    "hyperace": ("vocals", "instrument"),
    "melband-inst-v1": ("vocals", "other"),
    "melband-inst-v1e": ("vocals", "other"),
    "leap-inst": ("vocals", "other"),
    "melband-small": ("vocals", "other"),
    "melband-duality-v1": ("vocals", "instrument"),
    "melband-duality-v2": ("vocals", "instrument"),
    "melband-inst-v1e-plus": ("vocals", "other"),
    "melband-inst-v1-plus-test": ("vocals", "other"),
    "kim-ft2": ("vocals", "other"),
    "kim-ft2-bleedless": ("vocals", "other"),
    "kim-ft3-prev": ("vocals", "other"),
    "melband-big-beta1": ("vocals", "other"),
    "inst-exp-vrl": ("vocals", "other"),
    "exp-siamese": ("vocals", "instrument"),
    "leap-xe-voc": ("vocals", "other"),
    "leap-xe-inst": ("vocals", "other"),
    "resurrection-inst": ("vocals", "other"),
    "melband-inst-v2": ("vocals", "other"),
    "hyperace-v2-inst": ("vocals", "instrument"),
    "melband-big-beta2": ("vocals", "other"),
    "melband-big-beta3": ("vocals", "other"),
    "melband-big-beta5": ("vocals", "other"),
    "melband-big-beta4": ("vocals", "other"),
    "melband-big-beta6": ("vocals", "other"),
    "melband-big-beta6x": ("vocals", "other"),
    "melband-big-beta7": ("vocals", "other"),
}

#: Every stem name, in dropdown order (priority, then a-z).
ALL_STEMS: tuple[str, ...] = (
    "vocals",
    "instrument",
    "drums",
    "bass",
    "guitar",
    "piano",
    "other",
    "noreverb",
    "reverb",
)
