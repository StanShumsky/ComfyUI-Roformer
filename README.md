# ComfyUI-Roformer

ComfyUI nodes for 41 of the popular RoFormer music source-separation models: 6-stem and
4-stem separators, specialists (karaoke, vocal de-reverb, guitar), and a full two-way
(vocals / instrumental) lineup. Point it at any recording and get the stems back as
lossless WAV/FLAC files or as live `AUDIO` you can feed into any other node. All models
run at 44.1 kHz stereo. The **Roformer Ensemble** node blends the same stem from several
model runs into one better stem — the recipe behind the top [mvsep.com](https://mvsep.com/quality_checker/multisong_leaderboard)
leaderboard results — and a batched audio input separates every element at once. Beyond
the models, four audio-tool nodes round out the workflow: **Align** (cross-session
alignment with the exact residual), **Pitch Shift**, **Time Stretch**, and **Key Detect**.

## Install

From the ComfyUI-Manager: *Manager → Install Custom Nodes* → search for **ComfyUI-Roformer**
→ Install; Manager installs its requirements as well.

Or clone it manually (requires Python 3.10+ and ComfyUI 0.11.0+):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/StanShumsky/ComfyUI-Roformer
python -m pip install ./ComfyUI-Roformer
```

`pip install -r ComfyUI-Roformer/requirements.txt` remains available for an existing
checkout. Restart ComfyUI after installing or updating.

Models download automatically on first use (45 MB – 1.7 GB) into
`ComfyUI/models/roformer/`. Complete local checkpoints and configs are reused without a
network request. Set `HF_HUB_OFFLINE=1` only after the required files are installed. You
can also use your own copies: a `.ckpt`/`.pth` checkpoint with its config YAML in any
subfolder of `models/roformer` shows up in the loader. Keep one checkpoint per folder;
if a folder contains several, the first one (alphabetical) is used.

## Quick start

`Load Audio → Roformer Loader → Roformer Separate`, then:

- **Roformer Save Audio** writes one file per stem into the output folder, or
- **Roformer To Audio** picks a single stem from a dropdown as live audio for the rest of the
  graph (the model's own stem list is on the Loader's `stems` output).

The loader defaults to `sw-6stem` — six stems at once (bass, drums, other, vocals,
guitar, piano). For vocals / instrumental, pick one of the two-way models
([Models](#models)).

## Performance and VRAM

GPU is strongly recommended — the models are designed for CUDA, and slow-but-working CPU
runs are a fallback. The keys that matter most:

- **`precision`** (Loader) — `float32` (default) reproduces the released inference path.
  `float16` runs eligible layers under CUDA autocast: noticeably faster and lower on GPU
  memory, with small numerical changes. The weights stay float32; CPU and MPS runs stay
  float32 even when `float16` is selected.
- **`chunk_seconds`** (Separate) — `0` uses the model's own training window (about 11–22 s).
  A positive value processes shorter windows in a few extra passes, which lowers GPU memory
  at the cost of some time. That is usually the cheapest fix on a small GPU.
- **Batch size** — a batched input is processed in groups of at most four songs per model
  pass, and the node automatically shrinks the group on out-of-memory errors (logged).
  Lowering the window is usually more helpful than raising the batch size.
- **Model size** — the large mel-band models (1.5–1.7 GB checkpoints) need the most headroom;
  start with a smaller or two-way model if you are tight on memory.

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
- `precision` — `float32` (default, the released inference path) or `float16` (opt-in CUDA
  autocast for eligible layers). Weights remain float32; CPU and MPS runs stay float32 even
  when `float16` is selected. See [Performance and VRAM](#performance-and-vram).

The weights start in system RAM. ComfyUI's managed model loader keeps a model on the GPU
while it fits and evicts it under memory pressure or when another model needs the VRAM —
fast sequential runs, without fighting the rest of the graph for memory.

- Outputs: the model payload, and `stems` — the model's stem names as text, in the order the
  model trained them in; preview with any text display or wire it where you like.

### Roformer Separate
Runs the loaded model on the audio input — any sample rate and any channel count (multichannel
input is averaged to mono, then copied to both channels for a stereo model; a mono model receives
the mono downmix). Every output stem is padded or trimmed to the source's exact length at the file's
own sample rate.
- `model` — from the Loader.
- `audio` — the mix to separate (from **Load Audio**, or the audio track of a loaded
  video). A batched `(B, C, N)` audio separates every element, up to four per model pass,
  and automatically splits into smaller groups on OOM — the stems payload comes back
  batched.
- `chunk_seconds` — `0` (default) uses the model's own training window (about 11–22 s);
  set a positive value to use less VRAM. Longer audio is processed in overlapping windows
  with a crossfade, so the output is exactly the length of the input. If a window runs out
  of memory the node halves it and retries (up to three times, logged), then splits the
  batch into smaller groups before raising an actionable error.
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
  - `max spec` — per frequency bin, keep the value of the input with the larger magnitude
    (each model's most confident content); the leaderboard choice for vocals;
  - `min spec` — the smaller magnitude — suppresses leakage; the leaderboard choice for
    the instrumental.
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
- Advanced: `align_max_seconds` (how far the align search can reach, 0.0625–2 s) and
  `n_fft` (the STFT window for the spectral algorithms, 512–8192 — leave the default unless you know why not).

`example_workflows/roformer_ensemble.json` is the two-model recipe: the mix runs through
`sw-6stem` and a two-way vocal model, **max spec** blends the two vocals, and a second
ensemble (**average** + `invert`, mix connected) subtracts them from the mix for the
exact-residual instrumental.

### Roformer Align
Aligns one recording to a reference so two files from different sessions (different
takes, trims, or tools) can be ensembled or A/B'd. Connect the **reference** (the track
to align to, e.g. the original mix or the first model's run) and the **input** (the
track to be aligned): one song each, the same sample rate and channel count, at least
one second. The node searches candidate shifts, scores the residual, and keeps the
quietest result. Outputs: **aligned** (the input with the offset removed and the gain /
phase corrected — feed it to an Ensemble slot next to the reference) and **inverted**
(reference minus the aligned input — what the alignment removed, sample-exact).
- `reference` — the track to align to (e.g. the original mix, or the first model's run).
- `input` — the track to be aligned (same sample rate and channel count as the
  reference).
- `phase` — `automatic` (default): flips the input if it starts phase-inverted;
  `positive` / `negative` force the sign; `native` leaves it as-is.
- `match_silence` — equalizes the leading silence of the two tracks first (for files
  trimmed or exported differently). Off: the search handles the offset directly.
- `intro_analysis` — where the search samples its segments: `default` (one position near
  the start), `1/8`…`1/2` (that single fraction), or `shifts: low / medium / high` (several
  positions — more candidates, slower).
- Advanced (each one refines the search — and slows it down):
  - `align_window` — `off` (the global search only — fast) or a window size
    (`0.0625 s`…`2 s`, or `shifts: low / medium / high` for several sizes): the track is
    re-aligned inside every sliding window — for pairs with local timing drift.
  - `volume_analysis` — gain corrections tried on the input (`off`, `low`, `medium`,
    `high`, `very high`): more steps = more candidates, slower.
  - `phase_shifts` — phase rotations tried on the input, in degrees (`off` or `180`…`1` as
    the step): smaller step = more rotations, slower.

### Roformer Pitch Shift
Shifts the pitch by semitones at the audio's own sample rate. A batched input shifts
every element.
- `audio` — the audio to shift, any sample rate (stays at its own rate).
- `semitones` — the shift in semitones, 0.5 steps, −24…+24 (default 0; 12 = one octave
  up, −12 = one octave down).
- `time_correct` — on (default): the pitch moves and the length stays; off: the classic
  resample — the pitch moves AND the length changes by 2^(−semitones/12).

### Roformer Time Stretch
Stretches the tempo without changing the pitch, at the audio's own sample rate. A batched
input stretches every element.
- `audio` — the audio to stretch, any sample rate (stays at its own rate).
- `rate` — the tempo factor (default 1.0, 0.25–4.0): 2.0 = twice as fast (half the
  length), 0.5 = half speed (double the length). The pitch is preserved.

### Roformer Key Detect
Detects the musical key by comparing the mean chroma profile against all major and minor
profiles (24 keys).
- `audio` — the audio to analyse; feed it the separated vocals (the clearest tonal
  source) when you have them rather than the full mix. A batched input is detected on
  its first element (logged).
- Output `key` — a string like `A minor (confidence 0.72)`: a higher confidence is a
  clearer tonal signal.

### Roformer To Audio
Extracts one stem from a stems payload as `AUDIO` for feeding any other node. A batched
payload comes back batched. `stem` is a dropdown of every stem name the pack knows about,
sorted by priority (vocals first, then the instrumental, drums, bass, guitar, piano, other),
matched case-insensitively — `(auto)` (the default) selects the model's target stem, and a
stale or misspelled choice falls back to the first stem, so a run never fails on a stem
name. The model's own list is on the Loader's `stems` output.

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
- `format` — `wav` or `flac` (both lossless; `flac` is smaller and stores the workflow
  prompt in the file's metadata).
- `normalize` — `off` (default); `anti clip` scales a clipping stem back to full scale
  before writing — never raises a quiet stem; `full scale` always rescales every stem to
  peak 1.0.

### Roformer Info
The separation summary as a JSON string, for branching or saving run metadata.

## Models

All 41 checkpoints, in the same order as the loader dropdown. "vocals + residual" means
the model predicts vocals and the rest of the mix comes back as the other stem (`-inst`
models work the other way round); multi-stem models emit every stem natively. SDR is the
quality score from the [mvsep.com](https://mvsep.com/quality_checker/multisong_leaderboard)
tables (higher = better). Sizes are the download sizes.

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

## Blueprints

### Two-model ensemble (the leaderboard recipe)
`example_workflows/roformer_ensemble.json` — the mix runs through `sw-6stem` and
`hyperace-v2-voc`, **max spec** blends the two vocals, and a second ensemble (**average**
+ `invert`, the mix connected) subtracts them from the mix for the exact-residual
instrumental. Two-model blends are what the mvsep top entries are.

### Manual ensemble (mvsep.com submissions)
The board-climb pipeline: the same track through several models, the same stem blended,
the blend resubmitted. Save every run's output with the mvsep submission naming —
`{n}_{track}_{stem}` (`n` = the track number: `1_mysong_vocals.wav`,
`2_mysong_instrum.wav`) — as **Roformer Save Audio**'s `filename_prefix`. Then blend
either here or in an external manual-ensemble tool:
- **In this graph:** **Load Audio** (one model's file per stem) → **Roformer Align**
  (only when the files come from different sessions or trims — two runs of this pack are
  sample-accurate) → **Roformer Ensemble** (`max spec` for the target stem, `min spec`
  for the residual, `invert` + the mix for the exact complement) → **Roformer Save
  Audio** with the same `{n}_{track}_{stem}` prefix.
- **In an external tool:** the same `{n}_{track}_{stem}` WAVs drop straight into a
  manual-ensemble bulk mode — it groups files by the leading track number and blends
  each track (one stem per track). The naming is the contract: the same files work in
  both.

### Reverb removal
Load **`dereverb`** (anvuew's de-reverberation model, [Models](#models)) → **Roformer
Separate** on the roomy vocal → **Roformer To Audio** (the model's target, `noreverb`) —
or save it. The model splits the input into the dry `noreverb` component and the
`reverb` component; the wet part is the other stem, if you want it.

## Troubleshooting

- **CUDA out of memory:** the node first halves the window (down to one second, at most
  three times), then splits a batch. For a persistent OOM, set `chunk_seconds` to `5`, select
  `precision = float16`, reduce the batch, or choose `device = cpu`. The large mel-band
  models need substantially more activation memory than their checkpoint size suggests.
- **Slow repeated runs:** a model that was already loaded is reused. If another model in the
  graph evicts it, the next separation pays a reload cost — that is expected behavior, not a
  bug.
- **CPU/MPS precision:** `float16` is CUDA-only. CPU and MPS keep float32 — converting the
  weights to half would break the model's internal math.
- **Download or gated-model errors:** sign in with `HF_TOKEN` or `huggingface-cli login` for
  gated repositories. Complete local files are reused offline. If a download leaves an
  incomplete model folder, remove that incomplete folder and retry; the loader requires both
  the checkpoint and its matching YAML.
- **Local model not listed:** use a subfolder of `models/roformer` containing a `.ckpt` or
  `.pth` and a YAML config. Multiple YAMLs make the selection ambiguous; keep the model's
  matching config beside its checkpoint.
- **Malformed/empty audio:** make sure the audio has a non-empty batch, channels and time
  axis, and a positive sample rate. Outputs keep the source rate and exact length.
- **Ensemble errors:** connect one song per input, with matching sample rate and channel
  count. `invert` additionally requires the original mix at that same rate/channel count.
- **Interruption:** ComfyUI's stop/cancel signal is checked during loading, processing and
  other long loops, and cancellation is never mistaken for an out-of-memory error.

Only load checkpoints and YAML configs from sources you trust. Roformer files use PyTorch
checkpoint deserialization and community YAML tuple tags; they are data files, not a safe
format for untrusted uploads. The pack checks strict state-dict compatibility and obvious
NaN/Inf/zero corruption, but does not pin cryptographic hashes for third-party releases.

## Tests

- `pip install -r requirements-dev.txt`, then `pytest tests` — the unit suite: no weights,
  no ComfyUI install, no network, runs on CPU in seconds.
- `RUN_RF_WEIGHTS=1 pytest tests` — the integration suite: downloads the real `sw-6stem`
  and `melband-small` checkpoints once and runs the nodes end to end.

**Licensing.** Code: see `LICENSE`. Checkpoints: each Hugging Face repo carries its own license/terms —
at least two (Revive, Resurrection) have open commercial-use permission threads, so check
the repo you use before redistributing or shipping the outputs commercially.
