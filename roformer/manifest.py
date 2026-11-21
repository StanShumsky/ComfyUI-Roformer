"""The model manifest: the 13 pcunwa releases (36 checkpoints) plus the popular community
releases (5 checkpoints: two multi-stem, three specialized), all as static entries.

Stem names are resolved at load time from each checkpoint's YAML (training.instruments /
target_instrument); the manifest carries labels, file names, the model family and sizes
(MB on the Hub, fetched 2026-11-21). A config_file of None means: take the single .yaml
sibling of the checkpoint.

The tuple order IS the loader dropdown order. It is ranked by: (1) number of stems, (2)
mvsep.com leaderboard position (SDR figures are cited in the tier comments, from the repos'
own READMEs / the mvsep.com quality tables, fetched 2026-11-21), (3) task uniqueness.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    id: str            # stable id == local folder name under models/roformer
    label: str         # loader dropdown label
    repo_id: str
    ckpt_file: str
    config_file: str | None
    family: str        # "bs" -> BSRoformer, "melband" -> MelBandRoformer
    size_mb: float
    supported: bool = True   # False -> the loader refuses it before downloading (see note)
    note: str = ""           # human reason, surfaced in the loader error and the README


MODELS: tuple[Checkpoint, ...] = (
    # ------------------------------------------------------------------ tier 1: multi-stem
    # Native N-stem output, ranked by stem count; BS-Roformer-SW is the top single multi-stem
    # model on the mvsep.com multisong board (bass 14.62 / drums 14.11 / guitar 9.05 /
    # piano 7.83 / other 8.71).
    # BS-Roformer-SW (jarredou): six stems - bass / drums / other / vocals / guitar / piano.
    # The original jarredou/BS-ROFO-SW-Fixed repo is gone from the Hub; this is the
    # most-downloaded public re-upload of the same .ckpt + .yaml pair.
    Checkpoint("sw-6stem", "SW 6-stem (bass/drums/other/voc/gtr/pno)", "FractalSurfer/BS-RoFormer-SW-6stem",
               "BS-Rofo-SW-Fixed.ckpt", "BS-Rofo-SW-Fixed.yaml", "bs", 699.4),
    # SYH99999's 4-stem fine-tune (drums / bass / other / vocals); .pth weights, the standard
    # BSRoformer config.
    Checkpoint("ft-4stems", "4-stems FT (drums/bass/other/voc)", "SYH99999/bs_roformer_4stems_ft",
               "bs_roformer_4stems_ft.pth", "config.yaml", "bs", 527.2),
    # ------------------------------------------------------------------ tier 2: specialized
    # Unique single-task fine-tunes the generic separators don't cover.
    # anvuew's vocal de-reverberation model: not a stem separator - it splits the input into a
    # dry (noreverb) and a reverb component; feed it a roomy vocal, take the 'noreverb' stem.
    Checkpoint("dereverb", "DeReverb (vocals, anvuew)", "anvuew/dereverb_bs_roformer",
               "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt", "config.yaml", "bs", 204.5),
    # becruily's karaoke split, trained for backing-track quality (vocals / instrumental).
    Checkpoint("karaoke-bs", "Karaoke (bs, voc/inst)", "becruily/bs-roformer-karaoke",
               "bs_roformer_karaoke_frazer_becruily.ckpt", "config_karaoke_frazer_becruily.yaml", "bs", 204.4),
    # becruily's tiny dedicated guitar extractor (22 M params, 45 MB).
    Checkpoint("guitar-mel", "Guitar (mel-band, becruily)", "becruily/mel-band-roformer-guitar",
               "becruily_guitar.ckpt", "config_guitar_becruily.yaml", "melband", 45.1),
    # ------------------------------------------------------------------ tier 3: two-way, ranked by leaderboard
    # Vocals-target first (the most common task), then instrumental-target, each best-first
    # where the repo's README publishes an mvsep SDR; unmeasured lines follow in release order.
    # pcunwa/BS-Roformer-HyperACE (ships its own model code: SegmModel mask estimator / DSConv v2)
    Checkpoint("hyperace-v2-voc", "HyperACE v2 (voc)", "pcunwa/BS-Roformer-HyperACE",
               "v2_voc/bs_roformer_voc_hyperacev2.ckpt", "v2_voc/config.yaml", "bs-hyperace-v2", 288.7),  # vocals SDR 11.40
    # pcunwa/Kim-Mel-Band-Roformer-FT - gated repo: downloads need an HF token (Hf token /
    # huggingface-cli login). The reference Mel-Band vocal model (mvsep: vocals SDR 11.01).
    Checkpoint("kim-ft", "Kim FT", "pcunwa/Kim-Mel-Band-Roformer-FT",
               "kimmel_unwa_ft.ckpt", "config_kimmel_unwa_ft.yaml", "melband", 913.1),
    # pcunwa/BS-Roformer-Leap
    Checkpoint("leap-voc", "Leap (voc)", "pcunwa/BS-Roformer-Leap",
               "bs_roformer_leap_voc.ckpt", "bs_leap_voc_conf.yaml", "bs", 238.2),
    Checkpoint("leap-xe-voc", "Leap Xe (voc)", "pcunwa/BS-Roformer-Leap",
               "Xe/bs_leap_xe_voc.ckpt", "Xe/leap_xe_config_voc.yaml", "bs", 267.8),
    # pcunwa/BS-Roformer-Resurrection
    Checkpoint("resurrection-voc", "Resurrection (voc)", "pcunwa/BS-Roformer-Resurrection",
               "BS-Roformer-Resurrection.ckpt", "BS-Roformer-Resurrection-Config.yaml", "bs", 204.5),
    # the max-vocals-fullness variant of the big line (mvsep: vocals SDR 10.59)
    Checkpoint("melband-big-beta5", "Mel-Band Big beta5", "pcunwa/Mel-Band-Roformer-big",
               "big_beta5e.ckpt", "big_beta5e.yaml", "melband", 1479.7),
    # pcunwa/BS-Roformer-Revive
    Checkpoint("revive", "Revive", "pcunwa/BS-Roformer-Revive",
               "bs_roformer_revive.ckpt", "config.yaml", "bs", 639.3),
    Checkpoint("revive2", "Revive 2", "pcunwa/BS-Roformer-Revive",
               "bs_roformer_revive2.ckpt", "config.yaml", "bs", 639.3),
    Checkpoint("revive3e", "Revive 3e", "pcunwa/BS-Roformer-Revive",
               "bs_roformer_revive3e.ckpt", "config.yaml", "bs", 639.3),
    # instrumental-target, best-first (instrum SDR from the repos' own READMEs)
    # pcunwa/BS-Roformer-Large-Inst (ships its own model code: transformer mask estimator)
    Checkpoint("large-inst", "Large Inst", "pcunwa/BS-Roformer-Large-Inst",
               "bs_large_v2_inst.ckpt", "config.yaml", "bs-large-inst", 238.2),  # instrum SDR 17.61
    # pcunwa/BS-Roformer-Inst-FNO (FNO1d mask estimator: the repo's README publishes the
    # MaskEstimator replacement; vendored in roformer/model, needs neuraloperator==1.0.2)
    Checkpoint("inst-fno", "Inst FNO", "pcunwa/BS-Roformer-Inst-FNO",
               "bs_roformer_fno.ckpt", "bsrofo_fno.yaml", "bs-fno", 332.0),  # instrum SDR 17.60
    Checkpoint("hyperace-v2-inst", "HyperACE v2 (inst)", "pcunwa/BS-Roformer-HyperACE",
               "v2_inst/bs_roformer_inst_hyperacev2.ckpt", "v2_inst/config.yaml", "bs-hyperace-v2", 288.7),  # instrum SDR 17.40
    Checkpoint("hyperace", "HyperACE", "pcunwa/BS-Roformer-HyperACE",
               "bs_hyperace.ckpt", "config.yaml", "bs-hyperace", 274.9),  # instrum SDR 17.27
    # pcunwa/Mel-Band-Roformer-Inst
    Checkpoint("melband-inst-v1", "Mel-Band Inst v1", "pcunwa/Mel-Band-Roformer-Inst",
               "melband_roformer_inst_v1.ckpt", "config_melbandroformer_inst.yaml", "melband", 913.1),
    Checkpoint("melband-inst-v1e", "Mel-Band Inst v1e", "pcunwa/Mel-Band-Roformer-Inst",
               "inst_v1e.ckpt", "config_melbandroformer_inst.yaml", "melband", 913.1),
    # the remaining two-way lines without published SDRs, in release order
    Checkpoint("leap-inst", "Leap (inst)", "pcunwa/BS-Roformer-Leap",
               "bs_roformer_leap_inst.ckpt", "bs_leap_inst_conf.yaml", "bs", 238.2),
    Checkpoint("leap-xe-inst", "Leap Xe (inst)", "pcunwa/BS-Roformer-Leap",
               "Xe/bs_leap_xe_inst.ckpt", "Xe/leap_xe_config_inst.yaml", "bs", 267.8),
    Checkpoint("resurrection-inst", "Resurrection (inst)", "pcunwa/BS-Roformer-Resurrection",
               "BS-Roformer-Resurrection-Inst.ckpt", "BS-Roformer-Resurrection-Inst-Config.yaml", "bs", 204.5),
    # pcunwa/Mel-Band-Roformer-small
    Checkpoint("melband-small", "Mel-Band Small", "pcunwa/Mel-Band-Roformer-small",
               "melband_roformer_small_v1.ckpt", "config_melbandroformer_small.yaml", "melband", 202.6),
    # pcunwa/Mel-Band-Roformer-InstVoc-Duality (native 2-stem: Vocals + Instrumental)
    Checkpoint("melband-duality-v1", "InstVoc Duality v1", "pcunwa/Mel-Band-Roformer-InstVoc-Duality",
               "melband_roformer_instvoc_duality_v1.ckpt", "config_melbandroformer_instvoc_duality.yaml", "melband", 1719.1),
    Checkpoint("melband-duality-v2", "InstVoc Duality v2", "pcunwa/Mel-Band-Roformer-InstVoc-Duality",
               "melband_roformer_instvox_duality_v2.ckpt", "config_melbandroformer_instvoc_duality.yaml", "melband", 1719.1),
    Checkpoint("melband-inst-v1e-plus", "Mel-Band Inst v1e+", "pcunwa/Mel-Band-Roformer-Inst",
               "inst_v1e_plus.ckpt", "config_melbandroformer_inst.yaml", "melband", 913.1),
    Checkpoint("melband-inst-v1-plus-test", "Mel-Band Inst v1+ (test)", "pcunwa/Mel-Band-Roformer-Inst",
               "inst_v1_plus_test.ckpt", "config_melbandroformer_inst.yaml", "melband", 913.1),
    Checkpoint("melband-inst-v2", "Mel-Band Inst v2", "pcunwa/Mel-Band-Roformer-Inst",
               "melband_roformer_inst_v2.ckpt", "config_melbandroformer_inst_v2.yaml", "melband", 1574.5),
    Checkpoint("kim-ft2", "Kim FT2", "pcunwa/Kim-Mel-Band-Roformer-FT",
               "kimmel_unwa_ft2.ckpt", "config_kimmel_unwa_ft.yaml", "melband", 913.1),
    Checkpoint("kim-ft2-bleedless", "Kim FT2 bleedless", "pcunwa/Kim-Mel-Band-Roformer-FT",
               "kimmel_unwa_ft2_bleedless.ckpt", "config_kimmel_unwa_ft.yaml", "melband", 913.1),
    Checkpoint("kim-ft3-prev", "Kim FT3 (prev)", "pcunwa/Kim-Mel-Band-Roformer-FT",
               "kimmel_unwa_ft3_prev.ckpt", "config_kimmel_unwa_ft.yaml", "melband", 913.1),
    Checkpoint("melband-big-beta1", "Mel-Band Big beta1", "pcunwa/Mel-Band-Roformer-big",
               "melband_roformer_big_beta1.ckpt", "config_melbandroformer_big.yaml", "melband", 1479.7),
    Checkpoint("melband-big-beta2", "Mel-Band Big beta2", "pcunwa/Mel-Band-Roformer-big",
               "melband_roformer_big_beta2.ckpt", "big_beta5e.yaml", "melband", 1479.7),
    Checkpoint("melband-big-beta3", "Mel-Band Big beta3", "pcunwa/Mel-Band-Roformer-big",
               "melband_roformer_big_beta3.ckpt", "big_beta5e.yaml", "melband", 1479.7),
    Checkpoint("melband-big-beta4", "Mel-Band Big beta4", "pcunwa/Mel-Band-Roformer-big",
               "melband_roformer_big_beta4.ckpt", "config_melbandroformer_big_beta4.yaml", "melband", 1574.5),
    Checkpoint("melband-big-beta6", "Mel-Band Big beta6", "pcunwa/Mel-Band-Roformer-big",
               "big_beta6.ckpt", "big_beta6.yaml", "melband", 1557.1),
    Checkpoint("melband-big-beta6x", "Mel-Band Big beta6x", "pcunwa/Mel-Band-Roformer-big",
               "big_beta6x.ckpt", "big_beta6x.yaml", "melband", 1708.5),
    Checkpoint("melband-big-beta7", "Mel-Band Big beta7", "pcunwa/Mel-Band-Roformer-big",
               "big_beta7.ckpt", "big_beta7.yaml", "melband", 944.7),
    # ------------------------------------------------------------------ tier 4: experimental
    # pcunwa/BS-Roformer-Inst-EXP-Value-Residual (the repo's own README: "this is an
    # experimental weight. It may not have practical performance.")
    Checkpoint("inst-exp-vrl", "Inst EXP Value-Residual", "pcunwa/BS-Roformer-Inst-EXP-Value-Residual",
               "BS_Inst_EXP_VRL.ckpt", "BS_Inst_EXP_VRL.yaml", "bs", 393.4),
    # pcunwa/BS-EXP-SiameseRoformer (experimental release) - dual-stream (siamese) variant;
    # the repo ships the SiameseTransformer model code, vendored in roformer/model.
    Checkpoint("exp-siamese", "EXP SiameseRoformer", "pcunwa/BS-EXP-SiameseRoformer",
               "bs_exp_siameseroformer.ckpt", "config.yaml", "siamese", 204.7),
)

BY_ID: dict[str, Checkpoint] = {m.id: m for m in MODELS}


def is_known_id(value: str) -> bool:
    return value in BY_ID
