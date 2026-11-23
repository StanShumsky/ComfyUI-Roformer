# ComfyUI-Roformer

ComfyUI nodes for 41 of the popular RoFormer music source-separation models: 6-stem and
4-stem separators, specialists (karaoke, vocal de-reverb, guitar), and a full two-way
(vocals / instrumental) lineup. Point it at any recording and get the stems back as
lossless WAV/FLAC files or as live `AUDIO` you can feed into any other node. All models
run at 44.1 kHz stereo. The **Roformer Ensemble** node blends the same stem from several
model runs into one better stem — the recipe behind the top [mvsep.com](https://mvsep.com/quality_checker/multisong_leaderboard)
leaderboard results — and a batched audio input separates every element at once.

## Install

From the ComfyUI-Manager: *Manager → Install Custom Nodes* → search for **ComfyUI-Roformer**
→ Install; Manager installs its requirements as well.

Or clone it manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StanShumsky/ComfyUI-Roformer
pip install -r ComfyUI-Roformer/requirements.txt
```

Restart ComfyUI. That's it — models download automatically on first use (45 MB – 1.7 GB)
into `ComfyUI/models/roformer/`. You can also use your own copies: a `.ckpt`/`.pth`
checkpoint with its config YAML in any subfolder of `models/roformer` shows up in the
loader.

## Quick start

`Load Audio → Roformer Loader → Roformer Separate`, then:

- **Roformer Save Audio** writes one file per stem into the output folder, or
- **Roformer To Audio** picks a single stem from a dropdown as live audio for the rest of the
  graph (the model's own stem list is on the Loader's `stems` output).

The loader defaults to `sw-6stem` — six stems at once (bass, drums, other, vocals,
guitar, piano). For vocals / instrumental, pick one of the two-way models
([Models](#models)).

## Nodes

All nodes are in the `audio/Roformer` category.

### Roformer Loader
- `model` — a checkpoint (see [Models](#models)) or the name of a folder in
  `models/roformer`. The list is ranked for professional use: multi-stem first (by stem
  count), then the specialists, then the two-way models by leaderboard quality (vocals-target
  before instrumental-target), experimental weights last; your local folders follow the
  curated list. The first entry, `sw-6stem`, is the default. Checkpoints download
  automatically on first use; a corrupt download is detected and the error tells you to
  delete the model folder and retry.
- `device` — `auto` (ComfyUI's session device policy for auxiliary models: honors
  `--gpu-only` and falls back to the CPU on machines without usable VRAM), `cuda`, `mps`,
  `cpu`.

The weights ride in system RAM and occupy the GPU only while a separation is running —
ComfyUI's model management evicts them when other nodes need the VRAM, so the loader
composes with any other model in the graph.

- Outputs: the model payload, and `stems` — the model's stem names as text, in the order the
  model trained them in; preview with any text display or wire it where you like.

### Roformer Separate
Runs the loaded model on the audio input — any sample rate (resampled internally to 44.1
kHz) and any channel count (downmixed / upmixed to the model's trained count).
- `model` — from the Loader.
- `audio` — the mix to separate (from **Load Audio**, or the audio track of a loaded
  video). A batched `(B, C, N)` audio separates every element at once (one at a time on
  the GPU, staged in RAM) — the stems payload comes back batched.
- `chunk_seconds` — `0` (default) uses the model's own training window (about 11–22 s);
  set a positive value to use less VRAM. Longer audio is processed in overlapping windows
  with a crossfade, so the output is exactly the length of the input. If a window runs out
  of GPU memory the node halves it and retries (up to three times, logged) before raising
  an actionable error.
- `tta` — test-time augmentation: the model also runs on the time-flipped mix and the two
  runs are averaged per stem. Doubles the run time; smooths flaky transients and edge
  artifacts. The exact-decomposition property survives — the averaged stems still sum to
  the input.
- Outputs (at the file's own sample rate):
  - `stems` — all stems (batched if the input was), for To Audio / Save Audio below.
  - `target` / `other` — the model's target stem and the other stem as ready `AUDIO`
    (for multi-stem models, `other` is the model's own `other` stem when it has one),
    ready to preview or wire on.
  - `info_json` — model, stem names, lengths, rates.

### Roformer Ensemble
Blends the same stem from two or more model runs into one better stem — the ensemble
behind the top [mvsep.com](https://mvsep.com/quality_checker/multisong_leaderboard) leaderboard entries
(several of the best results are two-model blends, not single models). Wire the stem's
`AUDIO` from several Separate runs (e.g. the vocals from two different models) into the
audio slots: two are required, and up to eight appear as you connect them. One song per
input, one sample rate, the same channel count. The graph you build **is** the saved
ensemble: every slot, algorithm and option is in the workflow file.
- `audio` slots (2–8) — the same stem from one model run each.
- `algorithm`
  - `average` (default) — a plain time-domain mean;
  - `max spec` — per STFT bin, keep the value of the input with the larger magnitude
    (phase included) — each model's most confident content; the leaderboard choice for
    vocals;
  - `min spec` — the smaller magnitude — suppresses leakage; the leaderboard choice for
    the instrumental.

  The spectral algorithms are the exact per-bin magnitude-selection math of The Ultimate
  Vocal Remover's ensemble (Hann 2048/1024 STFT, the winner's complex value kept).
- `normalize` — `off` (default); `anti clip` scales a clipping blend (peak above 1.0) back
  to full scale — never raises a quiet result; `full scale` always rescales to peak 1.0.
- `align` — search a small time offset (up to the advanced `align_max_seconds`, default
  1 s) that best aligns each input with the first, before blending. For inputs from
  different processing paths (other tools' resamplers, trims); two runs of this pack are
  sample-accurate, so leave it off there.
- `invert` — exact residual: the output becomes the connected `mix` minus the blend
  (connect the original audio to `mix`). The leaderboard move: ensemble the vocals,
  subtract for the instrumental — blend + output == mix holds exactly.
- `mix` — the original mix (from **Load Audio**); required when `invert` is on, ignored
  otherwise.
- Advanced: `align_max_seconds` (0.0625–2 s search radius) and `n_fft` (STFT window for
  the spectral algorithms, 512–8192; 2048 is what the mvsep winners use).

`example_workflows/roformer_ensemble.json` is the two-model recipe: the mix runs through
`sw-6stem` and a two-way vocal model, **max spec** blends the two vocals, and a second
ensemble (**average** + `invert`, mix connected) subtracts them from the mix for the
exact-residual instrumental.

### Roformer To Audio
Extracts one stem from a stems payload as `AUDIO` for feeding any other node. A batched
payload comes back batched. `stem` is a
dropdown of every stem name the pack knows about, sorted by priority (vocals first, then the
instrumental, drums, bass, guitar, piano, other) and merged case-insensitively — one
`vocals`, one `instrument`, one `other`, regardless of how the model spells them (the model's
own list is on the Loader's `stems` output). Matching is the mirror image — case-insensitive
and alias-tolerant (`instrument` picks a payload's `Instrumental`) — so `(auto)` (the default,
the model's target) or a stale choice always falls back to the first stem and can never fail
a run.

### Custom Roformer To Audio
The same extraction with the stem **typed by hand** instead of picked from the dropdown —
the name from `info_json`, matched case-insensitively (empty = the target stem). Use it for
local models and custom checkpoints whose stem names are not in the standard list.

### Roformer Save Audio
Writes every stem in the payload to ComfyUI's output folder, numbered so runs never
overwrite each other; the saved files are linked on the node (the audio icon). A batched
stems payload (from a batched Separate) writes one file per element — WAV suffixed
`_b001..`, FLAC numbered sequentially.
- `stems` — the stems payload from Separate.
- `filename_prefix` — the output file name (without extension; subfolders allowed).
- `format` — `wav` (lossless float32, what the audio nodes work with) or `flac` (lossless,
  smaller, written through the native ComfyUI audio saver so the workflow prompt is embedded
  in the file metadata).
- `normalize` — `off` (default); `anti clip` scales a clipping stem back to full scale
  before writing — never raises a quiet stem; `full scale` always rescales every stem to
  peak 1.0.

### Roformer Info
The separation summary as a JSON string, for branching or saving run metadata.

## Models

All 41 checkpoints, in the same order as the loader dropdown: multi-stem first (by stem
count), then the specialists, then the two-way models by
[mvsep.com](https://mvsep.com/quality_checker/multisong_leaderboard) leaderboard quality
(vocals-target before instrumental-target), experimental weights last. "vocals + residual"
means the model predicts vocals and the rest of the mix comes back as the other stem
(`-inst` models work the other way round); multi-stem models emit every stem natively.
SDR is the mvsep.com quality score (higher = better). Sizes are the download sizes.

### Multi-stem

BS-Roformer-SW is the top single multi-stem model on the mvsep.com boards.

| Loader id | Stems | Family | Size | Release |
|---|---|---|---|---|
| `sw-6stem` | bass, drums, other, vocals, guitar, piano | BS-Roformer | 699 MB | [BS-RoFormer-SW-6stem](https://huggingface.co/FractalSurfer/BS-RoFormer-SW-6stem) — re-upload of jarredou's SW (the original repo is gone) |
| `ft-4stems` | drums, bass, other, vocals | BS-Roformer | 527 MB | [bs_roformer_4stems_ft](https://huggingface.co/SYH99999/bs_roformer_4stems_ft) |

### Specialized

Single-task fine-tunes the generic separators don't cover.

| Loader id | Stems | Family | Size | Release |
|---|---|---|---|---|
| `dereverb` | noreverb + reverb — **de-reverberation**: feed a roomy vocal, take the `noreverb` stem | BS-Roformer | 204 MB | [dereverb_bs_roformer](https://huggingface.co/anvuew/dereverb_bs_roformer) |
| `karaoke-bs` | Vocals + Instrumental — tuned for backing tracks | BS-Roformer | 204 MB | [bs-roformer-karaoke](https://huggingface.co/becruily/bs-roformer-karaoke) |
| `guitar-mel` | Guitar + Other — dedicated guitar extractor (22 M params) | Mel-Band-Roformer | 45 MB | [mel-band-roformer-guitar](https://huggingface.co/becruily/mel-band-roformer-guitar) |

### Two-way — vocals target (best first)

| Loader id | Stems | Family | Size | Release |
|---|---|---|---|---|
| `hyperace-v2-voc` | vocals + residual · SDR 11.40 | BS-Roformer | 289 MB | [BS-Roformer-HyperACE](https://huggingface.co/pcunwa/BS-Roformer-HyperACE) |
| `kim-ft` | vocals + residual · SDR 11.01 — **gated**: needs a free Hugging Face token (`HF_TOKEN` or `huggingface-cli login`) | Mel-Band-Roformer | 913 MB | [Kim-Mel-Band-Roformer-FT](https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT) |
| `leap-voc` | vocals + residual | BS-Roformer | 238 MB | [BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) |
| `leap-xe-voc` | vocals + residual | BS-Roformer | 268 MB | [BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) |
| `resurrection-voc` | vocals + residual | BS-Roformer | 205 MB | [BS-Roformer-Resurrection](https://huggingface.co/pcunwa/BS-Roformer-Resurrection) |
| `melband-big-beta5` | vocals + residual (max fullness variant) | Mel-Band-Roformer | 1.5 GB | [Mel-Band-Roformer-big](https://huggingface.co/pcunwa/Mel-Band-Roformer-big) |
| `revive` | vocals + residual | BS-Roformer | 639 MB | [BS-Roformer-Revive](https://huggingface.co/pcunwa/BS-Roformer-Revive) |
| `revive2` | vocals + residual | BS-Roformer | 639 MB | [BS-Roformer-Revive](https://huggingface.co/pcunwa/BS-Roformer-Revive) |
| `revive3e` | vocals + residual | BS-Roformer | 639 MB | [BS-Roformer-Revive](https://huggingface.co/pcunwa/BS-Roformer-Revive) |

### Two-way — instrumental target (best first)

| Loader id | Stems | Family | Size | Release |
|---|---|---|---|---|
| `large-inst` | instrumental + residual · SDR 17.61 | BS-Roformer | 238 MB | [BS-Roformer-Large-Inst](https://huggingface.co/pcunwa/BS-Roformer-Large-Inst) |
| `inst-fno` | other + residual · SDR 17.60 | BS-Roformer | 332 MB | [BS-Roformer-Inst-FNO](https://huggingface.co/pcunwa/BS-Roformer-Inst-FNO) |
| `hyperace-v2-inst` | instrumental + residual · SDR 17.40 | BS-Roformer | 289 MB | [BS-Roformer-HyperACE](https://huggingface.co/pcunwa/BS-Roformer-HyperACE) |
| `hyperace` | instrumental + residual · SDR 17.27 | BS-Roformer | 275 MB | [BS-Roformer-HyperACE](https://huggingface.co/pcunwa/BS-Roformer-HyperACE) |
| `melband-inst-v1` | other + residual | Mel-Band-Roformer | 913 MB | [Mel-Band-Roformer-Inst](https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst) |
| `melband-inst-v1e` | other + residual (max instrum fullness) | Mel-Band-Roformer | 913 MB | [Mel-Band-Roformer-Inst](https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst) |

### Two-way — other variants

| Loader id | Stems | Family | Size | Release |
|---|---|---|---|---|
| `leap-inst` | other + residual | BS-Roformer | 238 MB | [BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) |
| `leap-xe-inst` | other + residual | BS-Roformer | 268 MB | [BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) |
| `resurrection-inst` | other + residual | BS-Roformer | 205 MB | [BS-Roformer-Resurrection](https://huggingface.co/pcunwa/BS-Roformer-Resurrection) |
| `melband-small` | vocals + residual | Mel-Band-Roformer | 203 MB | [Mel-Band-Roformer-small](https://huggingface.co/pcunwa/Mel-Band-Roformer-small) |
| `melband-duality-v1` | Vocals + Instrumental (native 2-stem, no residual) | Mel-Band-Roformer | 1.7 GB | [Mel-Band-Roformer-InstVoc-Duality](https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality) |
| `melband-duality-v2` | Vocals + Instrumental (native 2-stem, no residual) | Mel-Band-Roformer | 1.7 GB | [Mel-Band-Roformer-InstVoc-Duality](https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality) |
| `melband-inst-v1e-plus` | other + residual | Mel-Band-Roformer | 913 MB | [Mel-Band-Roformer-Inst](https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst) |
| `melband-inst-v1-plus-test` | other + residual | Mel-Band-Roformer | 913 MB | [Mel-Band-Roformer-Inst](https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst) |
| `melband-inst-v2` | other + residual | Mel-Band-Roformer | 1.5 GB | [Mel-Band-Roformer-Inst](https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst) |
| `kim-ft2` | vocals + residual | Mel-Band-Roformer | 913 MB | [Kim-Mel-Band-Roformer-FT](https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT) (gated) |
| `kim-ft2-bleedless` | vocals + residual | Mel-Band-Roformer | 913 MB | [Kim-Mel-Band-Roformer-FT](https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT) (gated) |
| `kim-ft3-prev` | vocals + residual | Mel-Band-Roformer | 913 MB | [Kim-Mel-Band-Roformer-FT](https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT) (gated) |
| `melband-big-beta1` … `beta4`, `beta6`, `beta6x`, `beta7` (`beta5` in the vocals-target table) | vocals + residual | Mel-Band-Roformer | 0.9-1.7 GB | [Mel-Band-Roformer-big](https://huggingface.co/pcunwa/Mel-Band-Roformer-big) |

### Experimental

| Loader id | Stems | Family | Size | Release |
|---|---|---|---|---|
| `inst-exp-vrl` | other + residual | BS-Roformer | 393 MB | [BS-Roformer-Inst-EXP-Value-Residual](https://huggingface.co/pcunwa/BS-Roformer-Inst-EXP-Value-Residual) — repo README: "may not have practical performance" |
| `exp-siamese` | vocals + residual | Siamese Roformer | 205 MB | [BS-EXP-SiameseRoformer](https://huggingface.co/pcunwa/BS-EXP-SiameseRoformer) — experimental dual-stream release |

Stem names come from each model's training config, so the list above is exactly what you
get — for the multi-stem models pick any of the stems with Roformer To Audio.

## Tests

- `pytest tests` — the unit suite: no weights, no ComfyUI install, no network (CPU, seconds).
- `RUN_RF_WEIGHTS=1 pytest tests` — the integration suite: downloads the real `sw-6stem`
  and `melband-small` checkpoints once and runs the nodes end to end (separation, the
  stems-decompose-the-mix guarantee, both save formats, stem extraction) on ComfyUI's own
  device selection.

**Licensing.** Code: see `LICENSE`. Checkpoints: each Hugging Face repo carries its own license/terms —
at least two (Revive, Resurrection) have open commercial-use permission threads, so check
the repo you use before redistributing or shipping the outputs commercially.
