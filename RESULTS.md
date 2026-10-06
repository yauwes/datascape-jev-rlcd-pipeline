# Datascape results log (append-only)

| date (UTC) | id | hypothesis | CV | LB public | verdict |
|---|---|---|---|---|---|
| 10-03 | team 56757520 | teammate SAM-SLCA ConvNeXt-S 384 (cluster-aware fold) | val 0.90 (their split) | 0.82086 | team best |
| 10-03 | 56788381 | 0.5·5NN(DINOv2-B) + 0.5·LR(DINOv2-L+CLIP-L) | random-fold OOF 0.8973 | 0.78627 | calibration: CV-LB gap −0.11 |
| 10-03 | 56788438 | same recipe, confident-learning cleaned labels (700 relabelled) | group OOF vs orig 0.8225 (from 0.8053) | 0.76761 | worse by ~2σ of the paired delta (96 Aman→Rusak flips, ~33 public rows): relabelling Aman→Rusak likely wrong for test; keep only as ablation |

## Measured facts
- Test→nearest-train DINOv2 sim median 0.918; 57% of test > 0.9 (random-fold val only 45%). 253 test files are byte-identical to train files.
- Label-correlated corruption: S&P noise on 19.7% Aman, 6.4% Rusak, 3.8% test.
- Probe/kNN OOF: random folds 0.845 (LR) / 0.868 (5NN) / 0.897 (blend); group90 folds LR 0.805 (closest to LB).
- Prior shift estimate on test (LR unweighted): predicted share Aman .58 / Lainnya .017 / Rusak .40; EM (Saerens 2002) .46/.014/.53. Train prior .72/.035/.246.
- Visually confirmed: some train rows are mislabelled (crushed boxes labelled Aman, intact labelled Rusak), but relabelling them hurt LB.

## Offline ablations (group90 folds, LR on DINOv2-L+CLIP-L, vs original labels)
- Public-LB single-sub SE ≈ 0.0165 (bootstrap n=917). Treat LB deltas < 0.02 as noise unless paired-change analysis says otherwise.
- Submitted calib recipe on group90 folds: 0.8445 (Aman .915 / Lainnya .878 / Rusak .741); importance-weighted (adv AUC 0.883) 0.887; none reproduce LB 0.786.
- A7 hierarchical (Lainnya-vs-rest → Aman-vs-Rusak): 0.7686 vs 3-way 0.8053 → drop.
- A11 fold-honest Nelder-Mead class bias: 3-way 0.8053 → 0.8477 (+0.042) → keep.
- Clean vs noisy train rows: Rusak share 31% clean vs 9% noisy; group-OOF prediction mix calibrated within clean rows.
- Team best (0.82086) = Hierarchical SAM-SLCA (frozen Segment-Anything encoder + spatial channel attention, Gu et al. 2025 arXiv:2505.06217) + ConvNeXt-S 384, 4-way TTA, Nelder-Mead. kNN "twin lock" ensemble fell to 0.7896 (same failure as our kNN blend).
- A13a test-time cluster/nn smoothing of predictions (no train labels): +0.002..+0.004 group-OOF (noise level) → low priority.

## Pre-registered candidates for 10-04 (one submission; ours)
- **Backup**: `work/sub_calib_priormatch.csv`, i.e. 56788381 recipe + A14 per-stratum prior matching (clean_other Rusak 48.1% → 31.3%, 414 Rusak→Aman; overall Aman .723 / Lainnya .019 / Rusak .258). Expected LB = OOF at 25.8% share (0.846) − 0.015 ≈ **0.83** (plausible 0.81–0.85). Refutation: LB < 0.80 means the prior-matching hypothesis is wrong.
- **Primary (if ready)**: fine-tuned ensemble (group90 OOF, honest hill-climb) + the same matching rule. Submit instead of the backup only if OOF at the matched share beats 0.846 by ≥ 0.01. Expected LB = that OOF − 0.015.
- Gate (replaces "OOF ≥ 0.885"): expected LB > 0.821 (team best).

## 10-03 06:40 UTC — first fine-tune landed
| tag | group-OOF argmax | honest hill+bias | OOF @ matched Rusak 25.8% | natural test Rusak share | notes |
|---|---|---|---|---|---|
| clipL224_g90 (CLIP-L/14 openai, 6 ep fixed) | 0.8490 | 0.8559 | 0.8516 | 0.366 (clean_other 0.446) | per-fold val F1 still rising at ep5 → ES rerun queued (clipL224_es) |
| clipL224_g90 + calib_g90 (logit mean) | 0.8703 | 0.8771 | **0.8720** | 0.41 (honest-bias fit) | corr 0.717 → diversity gain +0.02 |
- No class weighting in fine-tune, yet test Rusak share 37% vs OOF 20%. Within clean_other: OOF pred 25.6% vs true 31.3% (calibrated), test 44.6%. The test shift is in the inputs; the LB decides between covariate shift (match) and real prior shift (don't).
- Primary candidate for 10-04 = fine-tune ensemble + A14 matching. Expected LB = 0.872 − 0.015 ≈ 0.857, assuming A14 holds.
| cnxB384_g90 (ConvNeXt-B CLIP laion2b 384, 6 ep) | 0.8157 | — | ens(clipL,cnxB,calib)@25.8% 0.8673 < 0.8720 without it | 0.388 | corr 0.918 with clipL, hill weight 0 → no ensemble value |
- Primary component choice: CLIP-L + LR probe (no kNN) OOF@25.8% 0.8667 vs with kNN 0.8720. Chose kNN-free: kNN copies near-dup train labels on test but cannot on group-OOF (OOF/test behaviour mismatch); the 0.005 gain is not worth that risk.
| cnxL320_g90 (ConvNeXt-L CLIP laion2b 320, 6 ep) | 0.8396 | — | ens(clipL,cnxL,lrprobe)@25.8% 0.8631 < 0.8667 without | 0.373 | corr 0.933 with clipL → fine-tunes of CLIP-pretrained nets are near-duplicates; diversity comes from other encoders (DINOv2 probe) |

## 10-03 07:50 UTC — VLM tower probes (emb2), group90, LR C=0.5 unweighted
| tower (family) | OOF argmax | OOF@25.8% |
|---|---|---|
| **PE-Core-L/14@336 (Meta Perception Encoder)** | **0.8431** | **0.8497** |
| SigLIP2-so400m/16@384 (Google; Gemma/PaliGemma family) | 0.8382 | 0.8415 |
| SigLIP2-B/16@384 (SmolVLM/IDEFICS3 family) | 0.8132 | 0.8179 |
| CLIP-L/14@224 (OpenAI) | 0.8083 | 0.8068 |
| MobileCLIP2 FastViT-MCi2 (Apple) | 0.7981 | 0.7991 |
| DINOv2-L | 0.7597 | 0.7611 |
| Florence-2 DaViT-B / Gemma-4 ViT-167M | 0.40 / 0.34 | broken features (pooling/fp16?) — excluded |
| DINOv3-L | NaN | fp16 overflow on T4 → rerun in fp32 |
| Qwen3-VL ViT / Qwen3.5 ViT | — | Kaggle timm too old → rerun with `pip -U timm` |
- Concat probe PE-L+SigLIP2-so400m+DINOv2-L+CLIP-L (LR C=2): **0.8732** alone (beats fine-tuned CLIP-L 0.849); + CLIP-L ft logit mean: **0.8784 @25.8%** → registered `d04_clipft_cat4_pm` (expected LB 0.863 [0.843, 0.883]).
- Next fine-tune by probe rank (Kornblith 2019): PE-Core-L@224 ES (`peL224_es`, yoursbae) running; CLIP-L ES rerun running (asuudane).
| eva02B448_g90 (EVA02-B 448, 6 ep) | 0.8210 | — | lowers ens 0.8784→0.8718 | — | corr 0.925 with clipL → closed |
- emb3 towers: DFN5B CLIP-H 0.8407, EVA02-CLIP-L 0.8271, PE-Core-B 0.8171, CLIP-L@336 0.8051 (no gain over 224), PaliGemma2 tower 0.7853. InternViT-300M 0.28 / AIMv2-L 0.67 / PE-Spatial-B 0.69 (pooled features unsuitable without their own pooling), MobileNetV5 NaN (fp16).
- cat6 (cat4 + DFN5B-H + EVA02-CLIP-L) LR C=2: probe 0.8804, + CLIP-L ft 0.8812 @25.8% → registered `d04_clipft_cat6_pm` (exp 0.866). Gains from extra towers are now ≤0.003 (noise level).
- MLP head on cat6 (pre-set hparams, 3 seeds): **0.8889 OOF, 0.8894 @25.8%**, best single. +CLIP-L ft 50/50 lowers it (0.8819); fold-honest hill = 0.8894. Primary for 10-04 = `d04_mlp6_pm` (exp 0.874).

## 10-03 ~10:00 UTC — matching target corrected (advisor check)
- Test copies by stratum: clean_other MD5 copies (n=252) **38.1% Rusak**, twins ≥0.97 (n=902) 44.2%, ≥0.95 (n=1159) 41.9%, vs train clean_other 31.3%. So test clean_other really is more Rusak-heavy than train; 31.3% matching (A14 as first stated) would over-correct.
- Final rule A14': match clean_other Rusak to the byte-identical-copy rate (38.1%) → overall Rusak 31.0% (team-best LB point was 30.2%). The MLP's natural share was 48.5% in clean_other, 39.5% overall.
- **Chosen for 10-04: `d04_mlp6_pm38`**: expected LB ≈ 0.858 (0.846 if the true test Rusak share is ~26%, ~0.87 if ~31%). Refutation: LB < 0.82 means neither shift story holds. Selection-bias shading −0.005 included.

## 10-03 11:30 UTC — emb4/emb5 + CLIP-L ES (patience-3) results
Local interpreter: `/usr/local/bin/python3.11` (the conda path in .zshrc no longer exists). Disk was full: fine-tune weights stay in the Kaggle kernel outputs only, never download `w_*.pt`.

**Resolution study (frozen, LR probe, group90).** Native resolution is best or tied; upscaling never helps.
| tower | 224 | native | high |
|---|---|---|---|
| PE-Core-L/14 (native 336) | 0.8442 | 0.8431 | 448: 0.8351 |
| SigLIP2-so400m/16 (native 384) | 0.7850 | 0.8382 | 512: 0.8359 |

- DINOv3-L in fp32: clean (no NaN), 0.7999, ensemble gain −0.003.
- PE-Lang-L@448 scores 0.7743 and PE-Spatial-L@448 0.7399, both below PE-Core.
- Qwen3-ViT-306M/88M collapse to all-Aman (test Rusak share 0.000). See closed_ideas.
- **CLIP-L ES** (`clipL224_es_g90`, OneCycle over 50 ep, ES patience 3 on val NLL) scores **0.8012**, against 0.849 for the fixed 6-epoch run.
  - Best epochs per fold: 3/12/1/0/1.
  - The run lowers the MLP blend (0.8889 → 0.8848 at w = 0.2).
  - Cause: discoveries #15.
- **MLP-head check against the current champion `mlp_cat6`** (0.8889 / 0.8894 at 25.8% Rusak):
  - cat6 + PE-L r224 + r448 (multi-resolution views): 0.8825 / 0.8869. Worse; redundant views add noise.
  - cat6 + DINOv3-L: 0.8900 / 0.8906, i.e. +0.001. Noise level, below the 0.005 switch rule.
  - **10-04 candidate unchanged: `d04_mlp6_pm38`.**

## 10-03 13:10 UTC — Colab A100 session 1 (0.77 h, auto-released) + twin study
- **Canary (protocol v2):** CLIP-L fold 3 scores 0.8074 against the 0.8207 bar → FAIL, so the fine-tune ablations were skipped (discoveries #19).
- **Giant frozen towers** (A100 bf16, cv2 median; LR probe C=0.5, group90):

  | tower | OOF argmax | OOF@25.8% |
  |---|---|---|
  | **PE-Core-G/14@448 (Meta, 1.9B)** | **0.8512** | **0.8531** (best single tower; PE-L 0.8431) |
  | EVA02-CLIP-E/14@224 | 0.8432 | 0.8379 |
  | SigLIP2-g-opt/16@384 | 0.8399 | 0.8328 (≈ so400m 0.8382) |
  | MetaCLIP2-H/14@378 | 0.8370 | 0.8496 |
  | MetaCLIP2-G/14@378 | 0.8360 | 0.8353 |

  - As 1:1 logit averages with `mlp_cat6`, every probe lowers the score (−0.008 to −0.015), which is expected when averaging a single probe into a stronger model. The real test is the MLP head with the new features (`mlp_cat9g`, running).
- **Twin LOO table** (`work/twin.py loo dinob mlp_cat6`; clean_other only; similarity-weighted vote over all train twins ≥ thr):
  - thr 0.99: twin acc 0.956 vs group-OOF model 0.833.
  - thr 0.95: 0.943 vs 0.858. Where they disagree, the twin is right 0.787 and the model 0.210. Test rows locked: 1159.
  - The decisive check against a model that has *seen* the twin (random-fold MLP) is running.

## 10-03 14:00 UTC — A100 session 2 (12.9 min, auto-released): GPU MLP heads + SSCD twin study
- **Heads** (5 seeds, `colab/heads.py`, preds in `asuudane/datascape-colab-out2` and `work/preds/`):

  | heads | folds | OOF | @25.8% |
  |---|---|---|---|
  | g_cat6 | group90 | 0.8883 | 0.8884 |
  | g_cat6_peg | group90 | 0.8918 | 0.8934 |
  | g_cat11 | group90 | 0.8894 | 0.8953 |
  | **mean(g_cat6_peg, g_cat11) = `h2_peg_c11`** | group90 | **0.8926** | **0.8967** (best of 15 combos; shade −0.003) |
  | r_cat6 | random folds | 0.9114 | 0.9012 |
  | r_cat11 | random folds | 0.9225 | 0.9125 |
- **SSCD copy detector** (`work/sscd.py`, disc_mixup, 5 variants; `work/twin.py`):
  - It covers far more twins than DINOv2 at the same precision. Test clean_other with a twin: 1834 at SSCD 0.55, against 1159 at DINOv2 0.95.
  - SSCD-only twins (train): 0.974 accurate, against 0.925 for the model.
- **Decisive check, against the twin-seen model `r_cat6`, clean_other:**

  | detector | twin acc | model acc | when they disagree: twin right / model right |
  |---|---|---|---|
  | SSCD 0.55 | 0.955 | 0.924 | **0.743 / 0.257** |
  | DINOv2 0.95 | 0.943 | 0.920 | 0.669 / 0.331 |

  Twin locking wins even against a model that saw the twin.
- **Hybrid on train** (SSCD 0.55 vote on clean_other, else group-OOF `h2_peg_c11` matched per stratum): **macro-F1 0.929** vs 0.896 for the base alone; accuracy 0.937 vs 0.904.
- **224/noisy strata:** SSCD twins of test 224px rows are all Aman (335/378 noisy_224 at 0.6, many of them clean_other train images). The model also predicts Aman there, so nothing changes.
- **Tonight's single submission (10-04 00:00 UTC): `d04_sscd55_h2`.** Expected LB 0.905 [0.87, 0.94].
  - **Refutation rule:** LB ≥ 0.88 confirms that twin labels transfer to test. LB < 0.86 refutes it, and 10-05 falls back to the plain model.
  - **Attribution vs `d04_mlp6_pm38`:**
    - 94 rows come from the base swap (mlp_cat6 → h2, see `work/sub_d04_h2_pm38_nolock.csv`).
    - 273 rows come from the twin lock (136 A→R, 137 R→A).
    - 295 rows differ in total; the Rusak share is identical (31.1%).
- **SSCD-large rule (pre-registered before its results):** adopt `sscd_disc_large`, or its union with mixup, only if both hold:
  - it adds test coverage at equal LOO twin accuracy;
  - it raises the train hybrid F1 by ≥ +0.003 (same procedure: clean_other only, checked against r_cat6).
  - If adopted, it gets a new ledger row; the registered candidate is not edited.
- **Lainnya check (for 10-04/05, one change only):**
  - The h2 group-OOF is calibrated for Lainnya: true 3.46%, predicted 3.29%, F1 0.955, recall 0.931.
  - On test it predicts only 1.8% (55 rows). Only 17 test rows have Lainnya-signature sizes (150×200 and similar).
  - Rusak is the weakest class in OOF (recall 0.703).
  - Open question: is test Lainnya genuinely rarer, or is the model under-predicting it under shift?
- **SSCD-large (`sscd_disc_large`, Kaggle T4):** best hybrid F1 0.9322 vs 0.9318 for mixup, a gain of +0.0004 against the +0.003 needed. **Not adopted** (pre-registered rule).
- **Vote-purity gate** (new, discoveries #21): SSCD 0.55 with purity ≥0.9 gives locked-row LOO accuracy 0.981 (vs 0.955 ungated) and hybrid F1 **0.9318** (vs 0.9289). Test rows locked: 1534.
- **FINAL PICK for 10-04 00:00 UTC (the only submission): `d04_sscd55p90_h2`**, expected LB **0.912 [0.875, 0.945]**.
  - Refutation rule: LB ≥ 0.88 confirms that twin labels transfer; LB < 0.86 refutes it.
  - Attribution vs pm38: 225 rows differ (94 from the base swap, 217 from the gated twin lock, overlapping). Mix identical (Rusak 31.1%).
  - `d04_sscd55_h2` (ungated, exp 0.905) stays registered as an alternative but is not submitted.
- **A100 session 3 (3.8 min, auto-released): `colab/twin2.py`**, rotation/denoise-robust SSCD (database rotations 0/±15/±30/±45/90/180/270 × flip × 2 denoise levels; queries × 2 denoise levels).
  - At 0.55 with purity ≥0.9: hybrid F1 0.9313 vs 0.9318 for plain SSCD; test rows with a twin 1841 vs 1834; test top-sim quantiles barely move.
  - **Not adopted.** The visually confirmed rotated+noised copies still escape global descriptors. Next would be local-feature verification (SIFT/RANSAC) on candidate pairs.
- **FINAL PICK (replaces `d04_sscd55p90_h2`): `d04_sscd55p90_h2_bt2`**, expected LB **0.92 [0.88, 0.955]**. Overall Rusak 28.5%; clean_other 35.0%.
  - Differs from `d04_sscd55p90_h2` in 79 rows, all R→A (`submissions/d04_bt_vs_pm38lock_changed_rows.csv` lists the 133 rows of the first bt variant).
  - Refutation: LB < 0.88 means the twin hypothesis is wrong; drop both the lock and the bias, and fall back to the plain model at a low Rusak share.
  - Rationale: discoveries #22 plus the advisor check (the mixed-vote twin share of 0.612 triggered the split rule).
- **SIFT+RANSAC twin verification** (`work/sift_verify.py`; 2560 no-twin clean_other queries, 48k pairs; top-10 SSCD + top-10 DINOv2 candidates):
  - ≥30 inliers: twin acc 0.981 vs the twin-seen model 0.958 (train n=214), but only 60 test rows.
  - ≥20 inliers: 0.959 vs 0.962, so no gain.
  - Expected value ≈ +1 test row. **Not adopted.** The no-twin rows are genuinely novel images, and the model is the bottleneck there.
- **Lainnya:** no lever (closed_ideas).

## 10-03 15:35 UTC — A100 session 4 (3.7 min, auto-released; Drive mounted, outputs in MyDrive/datascape_2026/session4 + asuudane/datascape-colab-out4)
- **Domain-adapted heads `da_h2`** (`work/da_heads.py`): MLP cat6+PEG and cat11, 5 seeds each, trained on train + 1534 twin-locked test rows; weights saved in `models_da/`.
- **Candidate `d04_da_bt2`**: the same twin lock as bt2; the unlocked rows use da_h2 with b*=−0.8 (no-twin 0.299 Rusak, mixed 0.555). Overall Rusak 27.4%.
  - Differs from bt2 in 97 rows (65 R→A, 32 A→R).
  - Expected LB **0.93 [0.89, 0.96]**. The Colab and local rebuilds are identical.
- **Plan for 10-04 (2 submissions for us, 1 for the teammate):** submit both `d04_sscd55p90_h2_bt2` (exp 0.92) and `d04_da_bt2` (exp 0.93).
  - They form a paired test of the DA lever on 97 rows; the shared twin lock is tested by both.

## 10-04 00:05 UTC — LB: `d04_da_bt2` = **0.84521** (team best; previous 0.82086)
- Expected 0.93 [0.89, 0.96]: **missed by −0.085** (~5 SE of public LB).
- Visual check: test 224px images look like intact boxes, so the 224→Aman prediction is probably not the cause.
- Hypothesis (NOT a finding yet): train label noise is group-consistent, i.e. flipped before copies were made. Then twin labels agree with each other but not with test labels.
- Slot 2 = paired test `d05_h2_nolock_s337`: same per-stratum shares, no twin-derived pieces.
  - The 253 differing rows are in `submissions/d05_plain_vs_dabt2_diff.csv`.
  - Decision rule: plain ≥ da+0.01 → drop the lock, DA heads and twin-calibrated bias; plain ≤ da−0.01 → keep the twins and investigate the unlocked rows; otherwise inconclusive.
- **Paired test: `d05_h2_nolock_s337` = 0.83004** vs da_bt2 0.84521 (same per-stratum shares, ~76 differing rows in public).
  - **Twins + DA help, by +0.015** (rule: plain ≤ da−0.01), far below the LOO-based +0.05–0.07.
  - The plain h2 model at 27.4% Rusak scores 0.830 on LB against group-OOF@25.8% of 0.8967, so the **CV→LB gap is ≈ −0.065**, not −0.015. Retire the old calibration.
  - The share trend continues: 42%→0.768, 39%→0.786, 35%→0.79, 30%→0.821, 27.4%→0.830 (plain) / 0.845 (twins+DA). The test Rusak share may be lower still (train prior 24.5%).
- **Next (10-05, 2 slots):** test the Rusak share below 27% on the da_bt2 recipe, and find what drives the −0.065 base gap (test-domain shift).
- **CL audit:** confident learning on h2 group-OOF flags only 302 rows (2.5%: 178 R→A, 107 A→R). Detectable label noise is too small to explain a −0.065 gap.
  - Working hypothesis: test applies a stricter "Rusak" definition. This fits the monotone share trend.
- **Share dial** (`work/dial.py`): DA heads + soft twin lock (λ=3) + one clean_other Rusak bias. At 0.337 it reproduces da_bt2 (30 rows differ, 0 twin overrides).
- **Registered for 10-05 (2 slots):** `d05_dial_s30` (overall Rusak 24.4%) and `d05_dial_s26` (21.2%), both paired with da_bt2 (27.4% → 0.845).
  - If s26 > s30 > 0.845, keep going lower on 10-06.
  - If s30 is best, the optimum is near 24%.

## 10-04 01:00 UTC — A100 session 5 (6.7 min, released; Drive session5 + asuudane/datascape-colab-out5): label cleaning
- Montage evidence:
  - Rusak-labelled rows the model calls Aman are clean new boxes, opened boxes and product adverts (mislabelled).
  - Aman-labelled rows called Rusak are crushed or torn boxes (mislabelled).
- `colab/clean_heads.py`: 2-round confident learning (Northcutt 2021) plus DINOv2-cluster majority.
  - 433 labels changed (264 R→A, 148 A→R, 15 L→A, 6 A→L); stored in `work/labels_clean_v2.csv`.
  - Heads `c2_h2`: group-OOF 0.9313 vs cleaned labels (circular) and 0.8954 vs original labels.
  - Twin votes with cleaned labels lock 1659 rows (34.4% Rusak).
  - DA heads `c2_da`: raw test Rusak share 0.326.
- **10-05 slots (paired design):**
  1. `d05_dial_s30`: share effect vs da_bt2 (91 rows R→A).
  2. `d05_clean_s30`: cleaning effect at the same share as s30 (118 rows swapped 59/59).

## 10-04 01:40 UTC — A100 session 6 (9.1 min, released) sweep + ensembles; session 7 (fine-tunes) running
- **Sweep** (`colab/sweep.py`, cat6+PE-G, 3 seeds, group 5-fold): columns are OOF vs original labels / OOF vs cleaned labels / F1 vs cleaned twin labels on locked test rows.
  - MLP+CE (original labels): 0.8922 / 0.9271 / 0.9299.
  - GCE, SCE, bootstrap (original labels): +0.0015, noise level.
  - **MLP + bootstrap on cleaned labels: 0.8983 / 0.9340 / 0.9287.** Training on cleaned labels also raises the score against the original labels, so cleaning generalises.
  - Wide MLP on cleaned labels: 0.8944 / 0.9298 / **0.9315**.
  - LR 0.886, kNN 0.860 (vs original labels).
- **Ensembles over heads** (`work/ens2.py`, fold-honest, cleaned labels):
  - hill climbing: 0.9342;
  - log-mean blend: 0.9259;
  - **LR stacking: 0.9418 cleaned / 0.9070 original**, but its twin-locked F1 is 0.924 (lower) and its test Rusak share 0.411 (higher). Possible overfit to the train domain; not chosen yet.
- **Session 7** (`colab/driver7.py`): fine-tunes PE-L@224, EVA02-L@224 (strong aug), DINOv2-L@224 (cutmix) and SigLIP2-so400m@384 on cleaned labels plus 1659 twin pseudo-labels, protocol v2, 4-view TTA, MAX_H 7.
  - Weights go to Drive (`session7/weights`); preds go to asuudane/datascape-colab-out7.
  - PE-L fold 0: val F1 0.891 vs cleaned labels after 2 epochs, ~80 s/epoch.

## 10-04 03:30 UTC — fine-tune PE-L done; source / pixel-copy analysis; Oct 5 slots redesigned
- **ftc_peL224** (PE-L@224 ft, cleaned labels + 1659 twin pseudo rows, hv4 TTA):
  - OOF 0.9049 cleaned / 0.8770 original; natural test Rusak 0.316.
  - Its OOF is **leaky**: `ft.py` adds every pseudo row to every fold, including test copies of val-fold rows.
  - Its twin-locked F1 (0.996) is in-sample.
- **ens2 over ftc_peL224 + 3 heads:** blend 0.9418, hill 0.9413, stack 0.9494 (cleaned OOF). Not used: the leaky fine-tune OOF makes the selection circular, and the stack pushes test Rusak to 0.409, the wrong direction for LB.
- **Test ID order:** no leak (adjacent same-class rate 0.584 vs 0.577 random; flat class mix across ID blocks).
- **Lainnya:** no hidden test Lainnya. Train Lainnya is grocery packaging; 0 test rows have ≥30% Lainnya kNN neighbours. Closed again.
- **Image size = data source; train and test mixes differ a lot.**
  - 512×512: 20% of train, 1.5% of test.
  - 224: 2%, 17%.
  - 416 synthetic conveyor box: 2.4%, 6.3%.
  - 675×900 Vietnamese parcel photos: 4%, 9%.
- **Pixel-exact twins** (`work/pixall.py`: dihedral NCC on 128px thumbnails vs top-10 SSCD candidates; `work/pix_all.npz`):
  - 796 test rows (26%) have an exact train copy (NCC ≥ 0.99).
  - Exact train-train copies agree on labels 96.8% (n=3124), so random label noise is small (~1.6%/image).
- **416 source:** SSCD twins at ≥0.95 are different renders. A train Aman row's twin is Aman only 8% of the time, so twin votes there are majority-class artefacts.
  - Full-res NCC ≥ 0.99 separates true copies (8/8 agree) from different renders.
  - Model group-OOF on 416 Rusak is only 0.35.
- **Test-mix check** (advisor step 1, `work/testmix.py`): group-OOF reweighted to the test mix.
  - By size bucket: 0.923 at 27.4% Rusak.
  - By test-nearest-neighbour weights: 0.90–0.92, with the optimum at ≥30% Rusak.
  - LB instead gave 0.830 at 27.4%, rising as the share falls.
  - **So the gap is NOT the source mix. Test labels follow a different, stricter Rusak convention** (or a per-source prior shift).
- **Oct 5 slots redesigned:**
  - The old pair `d05_dial_s30` / `d05_clean_s30` was confounded: 36 and 61 of their changed rows were in the 416 source, so clean − dial is not a cleaning effect. Both are parked.
  - New decoupled pair, each paired with da_bt2 (0.845):
    1. `d05_p416_allA`: only 416 changes, all 136 R → A.
    2. `d05_share_non416`: only non-416 changes, the 122 rows that dial s26 moves R → A.

## 10-04 06:40 UTC — session 7 stopped early; cleanlab audit; 640×640 source probe
- **Session 7 (A100) terminated at ~06:35 UTC** under the "not significant → stop" rule. The 15 finished folds are on Kaggle `asuudane/datascape-colab-out7`, Drive `session7/out` and `work/colab_out7`. Colab shows 0 active sessions, 166.5 CU left.
  - Cleaned OOF (leaky in their favour): ftc_peL224 0.9049, ftc_eva02L224 0.8794, ftc_dinoL224 0.8872. The frozen-feature heads score 0.934.
  - Hill climbing gives PE-L and DINOv2 weight 0. SigLIP2@384 fold 0 reached val 0.912 at epoch 6, so it was stopped mid-fold.
  - Fine-tuning stays below frozen towers (discoveries #13).
- **cleanlab 2.9** (`work/cleanlab_audit.py`; honest h2 group-OOF, original labels):
  - `find_label_issues` flags 437–447 rows (272 R→A, 144 A→R). 333 of them overlap `labels_clean_v2`.
  - Global latent Rusak prior 0.235 vs given 0.2455.
  - Datalab: 15 outliers, 2656 near-duplicate rows (known).
  - **Conclusion: global cleanlab ≈ our hand-rolled CL, so it is not a new lever.** On macOS it needs `n_jobs=1` (spawn re-runs the script).
- **Per-source latent Rusak (CL):** 640×640 web 0.469 → **0.305**; every other source changes by less than 0.08.
  - Per-source `find_label_issues` on 640 flags 449 rows: 355 labelled R / predicted A, 93 A→R.
  - Montage of flagged R rows: new open boxes, product adverts, mock-ups, infographics. This is keyword-scrape convention noise.
- **Test 640:**
  - 347 rows; 246 have a pixel-exact train copy (da_bt2 calls 72% of those Rusak).
  - **95 of the 101 non-exact rows are predicted Rusak (94%).**
  - `d05_share_non416` changes only 29 of its 122 rows in 640; the rest are spread over "other".
- Literature:
  - chaii 1st (`chaii-hindi-and-tamil-question-answering/writeups/r001`): OOF relabelling degraded LB. This matches our closed CL-relabel result.
  - Polymer 1st (`neurips-open-polymer-prediction-2025/writeups/r001`): found a systematic test-label shift by LB probing, and pruned high-error rows rather than relabelling them.
  - **So: prune, don't relabel, and read per-source conventions from paired probes.**
- **640 cleanlab probe: empty, so nothing is submitted.** The DA heads retrained with the 449 flagged 640 rows pruned (`PRUNE=work/cl640_flags.csv TAG=da_cl640 python work/da_heads.py`; `work/p640.py`) still call **0 of the 95** non-exact 640 test Rusak rows Aman. Their mean P(R) actually rises, 0.845 → 0.869.
  - So the test 640 non-exact rows look like real damage to every model. The keyword-scrape noise sits in train, not in these test rows.
  - Exact-copy check: only 2 of the 242 exact-copy test rows predicted Rusak have a cleanlab-flagged train copy (`work/exact_flagR_rows.csv`). Group-consistent copies are invisible to CL.
  - **Oct 5 slots stay as registered:** `d05_p416_allA` and `d05_share_non416`.
- **Session 8 (A100, started 06:40 UTC):** frozen DINOv3-H+ and DINOv3-7B (bf16, `colab/emb_giant.py`), then `colab/probe8.py`.
  - **Pre-registered bar:** adding the tower to cat6+PE-G must give ≥ +0.005 group-OOF vs original labels (MLP512+CE, 3 seeds, same folds) **and** no rise in natural test Rusak share. Otherwise close DINOv3.
  - Code went in through the notebook cell (base64), and the cell ends with `runtime.unassign()`. Outputs go to Drive `session8` and `asuudane/datascape-colab-out8`.
- **Per-source check of da_bt2 (test predicted Rusak vs train given Rusak):**

  | source | test pred | train given |
  |---|---|---|
  | 224 | 0 | 0 |
  | 416 | 0.705 | 0.708 |
  | 512 | 0.13 | 0.08 |
  | 640 | 0.78 | 0.47 |
  | 675×900 | **0.068** | **0.784** |
  | other | 0.24 | 0.15 |

  - **675×900 is explained by exact copies:**
    - Every test 675 row with a train copy at NCC > 0.8 (103 of 280; 51 at ≥ 0.99) has an Aman copy.
    - Train 675 Aman rows are noisy, augmented variants of photos whose clean originals sit in test; train 675 Rusak rows are clean photos.
    - The DA lock learns "clean 675 photo with an Aman copy = Aman". This is the −107 Rusak in 675 behind da_bt2's +0.015 over h2_nolock, and it is consistent with exact-copy evidence.
  - **416:** all 21 exact (≥ 0.99) copies are Aman, yet da_bt2 calls 9 of them Rusak. These are included in `d05_p416_allA`.
- **Exact-copy vs da_bt2** (`work/exact_vs_dabt2.csv`): 796 rows have NCC ≥ 0.99 copies (776 unanimous); 24 disagree with da_bt2.
  - Copy Aman / da Rusak: 13 (416: 9, 640: 4).
  - Copy Rusak / da Aman: 10 (640: 2, other: 8).
  - Copy Lainnya / da Aman: 1.
  - A cheap component for the final combination: about +7 public rows if copies are ~90% right when they disagree with the model (discoveries #21).
- **640 non-exact test rows predicted Rusak (95):** the montage shows clearly damaged stock photos (crushed, torn, wet) and their grey or rotated variants. They are correct, so 640 is not a lever. The suspects left are the "other" source (1672 test rows, 24% predicted Rusak vs 15% in train) and 416. These are exactly what the Oct 5 pair tests.
- **Session 8 result: DINOv3 fails the pre-registered bar, so it is closed.** MLP512+CE, 3 seeds, group folds; scores are vs original / vs cleaned labels / test Rusak share / F1 on twin-locked test rows.

  | config | orig | cleaned | test Rusak | twin-locked |
  |---|---|---|---|---|
  | base cat6+PE-G | 0.8922 | 0.9271 | 0.395 | 0.9299 |
  | + DINOv3-H+ | 0.8925 (+0.0003) | 0.9271 | 0.391 | 0.931 |
  | + DINOv3-7B | 0.8940 (+0.0018) | 0.928 | 0.391 | 0.9314 |
  | H+ alone | 0.8337 | | | |
  | 7B alone | 0.8586 | | | |

  - Both gains are under +0.005, so the result is noise level.
  - The A100 ran about 17 min (≈1.5 CU).
  - Saved: Drive `datascape_2026/session8` and the new private dataset `asuudane/datascape-colab-out8` (the first `version` call returned 403 because the dataset did not exist yet; `create` then succeeded). The cell ran 16.5 min. The runtime was released: Colab shows 0 active sessions, 165.32 CU left.

## 10-04 07:30 UTC — Oct 5 slots changed: `d05_p416_allA` + `d05_p675_R` (share probe parked)
- **`d05_share_non416` is parked.** A montage of its 122 flipped rows (81 other, 29 640, 12 675) shows mostly visibly damaged boxes: torn, crushed, wet, one stamped "Damaged". It would flip probably-correct rows.
  - The share trend (42% → 0.768 … 27.4% → 0.830) mixes different models, so "fewer Rusak is better" is not a paired finding.
- **Copy structure per source** (pixel NCC ≥ 0.99, test→train):

  | source | test rows with copy (copy label) | train R with test copy | train A with test copy |
  |---|---|---|---|
  | 224 | 53 (all A) | 0/0 | 28/253 |
  | 416 | 21 (all A) | **0/211** | 11/87 |
  | 512 | 5 (all A) | **0/205** | 11/2259 |
  | 675 | 51 (all A) | **0/391** | 32/108 |
  | 640 | 246 (71 A / 174 R) | 238/1305 | 122/1478 |
  | other | 420 (361 A / 59 R) | 68/889 | 502/4615 |

  - In the synthetic and curated sources (416, 512, 675), only Aman crosses the split as copies.
  - Train 675 Aman rows are noisy copies (median 600 KB vs 75 KB clean) of clean test photos.
  - The 177 no-copy test 675 rows (da_bt2 calls 89% of them Aman) are indistinguishable by eye and by file size from both train Rusak and the copied Aman. q is between ~0.55 (copy-rate symmetry) and ~0.95 (only-Aman-copied structure).
- **`d05_p675_R`:** da_bt2 with the 158 no-copy 675 Aman predictions set to Rusak. `lbdelta` (A→R support added): breakeven q ≈ 0.45; q 0.6 → 0.856, q 0.9 → 0.875. Expected 0.855 [0.81, 0.88].
- **Pre-registered paired read for 00:00:**

  | outcome | p675_R | p416_allA | then on Oct 6 |
  |---|---|---|---|
  | copy-structure holds | ≥ +0.01 | ≤ −0.01 | set every no-copy row in synthetic and curated sources to Rusak and keep 416 as is |
  | stricter convention | ≤ −0.01 | ≥ +0.01 | keep the DA lock and push 416 to Aman |
  | per-source priors differ | up | up | apply both changes |
  | bases already right | down | down | keep da_bt2 and add the exact-copy fix only (pseudo-labelling closed 10-04) |

  Only the 24-row exact-copy fix is held back (it would contaminate both reads).

## 10-04 07:45 UTC — session 9: VLM towers re-probed (bf16, mean token pooling); closed
MLP512+CE, 3 seeds, group folds; each row is vs original labels.

| tower | alone | added to cat6+PE-G (base 0.8922) |
|---|---|---|
| InternViT-300M (InternVL2.5) | 0.7321 (was 0.28 in fp16) | 0.8920 (−0.0002) |
| AIMv2-H@448 | 0.8271 (AIMv2-L was 0.67) | 0.8903 (−0.0019) |
| Qwen3-VL ViT-306M with its merger | 0.8391 | 0.8892 (−0.0030) |

- The extraction bug is fixed: the old collapse was fp16 plus the pooling.
- None of the three adds to the frozen ensemble, so all fail the +0.005 bar and are closed.
- Session 9 ran 12.4 min. Saved to Drive `session9` and `asuudane/datascape-colab-out9`; the runtime was released (0 active sessions, 164.4 CU).
- **Notebook** `notebook/Datascape_Pipeline.ipynb` (generator `work/make_notebook.py`, executed with outputs) rebuilds `d04_da_bt2`, `d05_p416_allA` and `d05_p675_R` row for row from cached artifacts.
  - New caveat it surfaced: 6.7% of pixel-exact train pairs straddle group90 folds, so the group-OOF is slightly optimistic.
- **MLOps board:** https://claude.ai/artifact/NMNSBS6y6WJdfcj7smQohW (`report/`, built by `work/build_report.py` from `work/experiments.csv`, `submissions/ledger.csv` and the score CSVs).
- **Iterative pseudo-labelling** (`work/pl_iter.py`; DA heads on cat6+PE-G, 2 seeds; τ = 0.9; 5 folds over 876 SSCD-twin groups of the 1534 locked test rows): see `work/pl_iter_scores.csv`.
  - Held-out F1 round 0 → 3: 0.9604 → best +0.0024, and +0.001 at bias −0.8.
  - The unlocked Rusak share stays at 0.262–0.263, with 749–806 pseudo rows per round.
  - **Below the +0.005 bar, so it is closed.**

## 10-04 15:00 UTC — session 10 PRE-REGISTERED: DINOv3 ViT-H+ full fine-tune (pig-posture 1st-place recipe) + exhaustive copy matching
Ref: Kaggle multi-view-pig-posture-recognition public 1st (sugupoko, 2026): `vit_huge_plus_patch16_dinov3.lvd1689m` @384, AdamW lr 3e-5 / head 3e-4, LLRD 1.0, cosine 5% warmup, 12 ep, EMA 0.9985, LS 0.1, drop-path 0.1, grad ckpt, dihedral TTA, equal-weight LLRD-ladder ensemble.
Adaptation: no ConditionalDesaturate (test chroma ≥ train for every source, `meta_noise.csv`); dihedral 8-view TTA without remap (no chiral classes); `ft.py` noise aug kept.
Reopens a closed idea (L fine-tunes, frozen H+ +0.0003) only because the recipe differs (H+ size, uniform LR, EMA, cosine); the gate decides.

Gate (fold 0 only, two runs: **A** train-only, **B** train + twin-locked test rows of the other 4 `pl_iter` folds = pseudo-label arm). Baselines from `work/gate10.py` → `work/gate10_baselines.csv`:
- (a) group90 fold-0 val, original labels: h2 0.9035 all (n 2445), **0.9017 clean** (n 2399, no pixel-exact copy in train folds ≠ 0).
- (b) fold-0 held-out locked test rows (n 278, `pl_iter` seed-0 split): h2 train-only 0.889 / 0.9174 @ bias −0.8; **DA head r0 0.9466 / 0.9853 @ −0.8**.
- PASS = any of: A or B ≥ +0.005 over h2 on (a)-clean; B ≥ +0.005 over DA r0 on (b) at the same bias; equal-weight blend with h2 (a) or DA (b) ≥ +0.005 — AND unlocked-test Rusak share not above the baseline's. n 278 → SE ≈ 0.015, so (b) alone is weak evidence.
- FAIL on all → close the DINOv3-H+ fine-tune; no further folds. PASS → next session: remaining folds of the passing arm + LLRD 0.95 member.
- 0.99 is not a usable stop rule: exact train-train copies agree 96–97% (label noise ceiling).

## 10-04 17:30 UTC — "Pillar 4" (Lainnya prior matching to 3.5%) checked; candidate built, PARKED (user kept the planned probes)
- Claim (pasted 4-pillar write-up): Lainnya is under-predicted (1.8% vs train 3.46%), so F1_Lainnya ≈ 0.65 caps LB at 0.84; prior matching to 3.5% would lift macro-F1 to 0.97+. The arithmetic does not hold: 0.65 → 0.97 adds +0.107, i.e. 0.845 → 0.95, and only if every new Lainnya call is right.
- Evidence:
  - Reaching 3.5% flips 52 non-copy test rows whose DA-head P(L) is only 0.056–0.080. Test has just 6 non-Lainnya rows with P(L) ≥ 0.1.
  - Train group-OOF (h2) rows at P(L) 0.05–0.1 are truly Lainnya 0.3% of the time (n = 1456); at 0.1–0.2, 2.8% (n = 108).
  - Lainnya rarely has pixel copies (5.7% of Lainnya train rows vs 28.4% overall), so copy structure says nothing about the test Lainnya share.
- `d05_lainnya_pm35` is registered: expected LB 0.765 [0.74, 0.957]. Nearly all 52 flips should be false positives, F1_Lainnya ~0.8 → ~0.55, so about −0.08.
- The user chose to keep the 10-05 slots for `d05_p416_allA` + `d05_p675_R`. This strengthens the 10-03 closed idea "Lainnya share correction".
- Separately, `datascape/ground_truth_test_set.csv` (created 17:09 UTC by another process) is not ground truth. Its tiers are SHA-256/dihedral/SSCD copy labels plus "DA-Ensemble Model Prediction", the last identical to `d04_da_bt2`. It scores `d04_da_bt2` at 0.9906 while the real LB is 0.845. `submission_from_ground_truth.csv` = `d04_da_bt2` + 24 copy-label flips, 20 of them shared with the 10-06 exact-copy fix.

## 10-04 17:50 UTC — "test = augmented train?" precise-mapping audit + LB estimate (`d06_map_max` registered)
- Coverage (SSCD top-10 + dihedral pixel NCC + SIFT/RANSAC):

  | tier | test rows | share | train-side agreement with own label |
  |---|---|---|---|
  | exact pixel copy | 796 | 26.0% | 0.968 |
  | SSCD twin, pure vote | 1345 | 44.0% | 0.977 |
  | SSCD twin, mixed vote | 216 | 7.1% | 0.767 |
  | SIFT ≥ 30 inliers | 60 | 2.0% | 0.981 |
  | no train source | 640 | 20.9% | — |

  So the user's suspicion holds for ~79% of test.
- If test labels equalled the copied train labels, mapping + OOF-level model would give ≈ 0.93 accuracy.
- The LB says otherwise:
  - `d04_da_bt2` vs `d05_h2_nolock_s337` differ by a balanced swap of 253 rows (126 A→R, 127 R→A) and +0.0152 LB. At ≈ 0.00055 macro-F1 per row, the copy/DA label is right in only ≈ 55% of contested rows (LOO promised 88–98%).
  - Test labels of copies therefore often differ from their train labels: group-consistent label noise, or a different labelling convention (discoveries #24, #27).
- `d06_map_max` = `d04_da_bt2` + every remaining mapping disagreement: 29 rows (15 exact, 11 SIFT, 3 twin).
  - Expected LB 0.85 [0.845, 0.861]: the upper end is q = 1, the centre uses the train calibration discounted toward the LB-anchored 55%.
  - It replaces the 24-row exact-copy fix as the 10-06 mapping candidate. Mapping alone cannot reach 0.99: the remaining gap is label convention/noise on rows that already carry mapped labels.

## 10-04 18:10 UTC — session 10 results (A100, 194 min, 17.2 CU; runtime released, 0 active sessions, 147.17 CU left)
Executed notebook: Drive `datascape_2026/session10/Datascape_S10_DINOv3_A100.ipynb` (also `asuudane/datascape-colab-out10`).

| model | (a) fold-0 all | (a) clean | (b) locked held-out | (b) @ −0.8 | unlocked Rusak | @ −0.8 |
|---|---|---|---|---|---|---|
| h2 (baseline) | 0.9035 | 0.9017 | 0.889 | 0.9174 | 0.363 | 0.303 |
| h3A DINOv3-H+ FT, train only | 0.8874 | 0.8855 | 0.9387 | 0.9398 | 0.274 | 0.236 |
| h3B = A + 1256 locked test rows | 0.9034 | 0.9016 | **0.965** | 0.9763 | 0.284 | 0.260 |
| h3A + h2 (equal) | 0.9037 | 0.9020 | 0.9413 | 0.9466 | 0.302 | 0.248 |
| h3B + h2 (equal) | 0.9077 | 0.9061 | 0.9492 | 0.9735 | 0.309 | 0.255 |

DA head constants on (b): 0.9466, and 0.9853 @ −0.8.
- **Gate:**
  - A FAILS: (a) −0.016, blend +0.0003.
  - B PASSES the pre-registered rule on (b) at argmax (+0.018 vs DA head), but it is −0.009 at the DA bias. n = 278 (SE ≈ 0.015), so B ≈ DA head within noise.
- **Pseudo-labels vs none (user question):**
  - +0.016 on (a) and +0.026 on (b), with the Rusak share still low.
  - Adding the locked test rows helps the fine-tune, as it did the heads.
- **Caveat:** (b) labels are copied train labels, and a fully fine-tuned H+ memorises near-duplicates better than frozen heads (A gets 0.939 on (b) without seeing a single locked row). On the LB, copied labels are right only ~55% of the time where they disagree with the model (10-04 17:50). So (b) overstates B's LB value. Only an LB probe can decide.
- **Copy matching** (exhaustive all-pairs dihedral NCC, 64 px over all pairs → 128 px verify):
  - 800 test rows have an exact copy (was 796). SSCD top-10 already held 99.5% of them, so mapping is saturated.
  - Train-train exact agreement is 0.963; 6.9% of those pairs straddle group90 folds.
  - The near tier (0.95–0.99) is visibly augmented copies (Gaussian noise, colour shift, vertical flip + recolour), and some carry another label.
  - The 20 conflicting copy groups are colour-vs-grayscale versions of one image with different train labels.
  - 15 unanimous copy labels differ from `d04_da_bt2` (11 R→A, 3 A→R, 1 A→L).
- **Frozen DINOv3-H+ for conflicts:** LR OOF 0.8207 vs h2 0.8926. On 156 train rows whose copies disagree, DINO gets 0.532 and h2 0.564, so h2 is used. Labels inside conflict groups are coin flips.
- **Next (pre-registered PASS → B):**
  - First, a free LB probe from the existing fold-0 B test predictions, blended with the DA heads on unlocked rows. This needs `pred_h3B_f0.npz` downloaded from Drive.
  - Only if that probe beats `d04_da_bt2`: full-fit B (~100 min) and an LLRD 0.95 member.

## 10-05 00:00 UTC — LB: per-source probes read "stricter convention"
| sub | LB | Δ vs 0.84521 | read |
|---|---|---|---|
| `d05_p416_allA` | **0.87739** | +0.0322 | above the q=0 centre (0.8747): all 136 flipped 416 rows are Aman. **New best** |
| `d05_p675_R` | 0.80201 | −0.0432 | below the q=0 floor (0.8045): the no-copy 675 rows are Aman, so the DA lock was right |

- Copy-structure hypothesis refuted. In the curated/synthetic test sources (416, 675) Aman makes up ~100%, while train 416 is 71% Rusak and train 675 is 78% Rusak. The model's 416 Rusak calls were domain shift.
- lbdelta's public-LB model is slightly off: the 675 read landed under its band. Trust direction, not q to two decimals.
- **10-06 pair**, both built on `d05_p416_allA` and each changing one hypothesis:
  - `d06_cur_A`: the remaining 25 Rusak in 512 and 675 → Aman. Expected 0.881 [0.868, 0.889].
  - `d06_p640_A`: 121 no-copy (NCC ≤ 0.8) 640 Rusak → Aman. Test 640 is predicted 78% Rusak against 47% in train. Breakeven q ≈ 0.4; expected 0.869 [0.832, 0.903].
- `d06_map_max` is retired: it was built on `d04_da_bt2`. Its 5 A→R flips in 675 contradict this read.

## 10-05 08:30 UTC — error-pattern study (local, light)
- **Augmentation shortcut (montage of 416).**
  - Train 416 Rusak are raw renders of dented boxes.
  - Train 416 Aman are intact boxes that are mostly augmented: Gaussian or impulse noise, rotation with black corners, brightness.
  - Test 416 rows are raw intact boxes; the LB shows they are all Aman.
  - The model learned "augmented ⇒ Aman, raw ⇒ Rusak", and every test image is raw.
- **Noise confound in numbers.**
  - Train noisy share for Aman vs Rusak: 416 0.46 vs 0.04; 675 0.57 vs 0.03; 640 0.13 vs 0.04; other 0.34 vs 0.17.
  - Test noisy share is 0–3% everywhere except 224 (0.73) and 512 (0.38).
- **The noise-only fix fails.** A quick test (frozen PE-G + EVA02-E + DINOv2-L, LR, group folds):

  | variant | OOF clean | R rate 416 (truly A) | R rate 675 (truly A) |
  |---|---|---|---|
  | all rows | 0.8851 | 0.974 | 0.868 |
  | clean-only | 0.8773 | 0.974 | 0.886 |
  | noise-reweighted | 0.8858 | 0.927 | 0.800 |

  The noise flag misses the rotation and brightness augmentations.
- **640 and "other" montage.**
  - Test no-copy Rusak predictions are mostly visibly crushed or torn, so they are probably right.
  - Train Rusak includes many intact boxes and stock illustrations, which is label noise.
  - `d06_p640_A` is parked.
- **Joint LB inversion** of base + 416 + 675 (both q = 0, binomial public counts).
  - Implied public counts: k ≈ 36 of 41 expected and k2 ≈ 54 of 47 expected, which explains the 675 read landing below its floor.
  - F1 A 0.89 [0.77, 0.97], R 0.81 [0.70, 0.90], L 0.85 [0.71, 0.97]; true public Lainnya ≈ 17, close to the predicted count.
  - Underdetermined, so no class can be singled out as the weak one.
- **Next gate (session 11, Colab).** Score h3A (fine-tune, train-only) on the 473 LB-confirmed Aman rows (all 193 test 416 + all 280 test 675).
  - Two-sided: Rusak on the 473 rows must fall sharply, while the Rusak share on copy-labelled 640 Rusak rows and on "other" stays near the base.
  - h3A's augmentation was RRC, flips, rot90, ColorJitter and grayscale, with no noise and no fill-rotation.

## 10-05 09:30 UTC — session 11 PRE-REGISTERED: augmentation-equalised fine-tune (Colab A100)
- **Notebook.** `colab/ds11.ipynb` (generator `colab/make_nb11.py`, runner `colab/run11.py`); the local dry run passed.
- **Arms.** DINOv3 ViT-L/16 @384, train-only, fold 0, pig recipe, 10 epochs:
  - **h4 `augeq`:** every train image and every TTA view (8 stochastic) gets Gaussian noise σ 8–45, plus 50% impulse noise and a ±45° rotation with black or white fill.
  - **h4c:** the same with `d8` aug.
  - h3A/h3B (session 10, H+, `d8`) are read from Drive.
- **Gate (train-only models).** PASS needs all three:
  1. Rusak rate on G1n ≤ 0.50. G1n = 233 LB-confirmed Aman rows (test 416/675 with no train copy at NCC ≥ 0.95). h2 = 0.923, da_h2 = 0.498.
  2. Rusak rate on CRx ≥ 0.917. CRx = 244 rows whose exact copies unanimously say Rusak, outside 416/675. h2 = 0.967.
  3. 640 Rusak share ≥ 0.65 (h2 = 0.81).
- **On PASS.**
  - Candidate = `d06_cur_A` plus the model's ≥ 0.70-confidence disagreements on 640/other rows with no copy. It goes in the 10-06 slot 2 and is read paired against `d06_cur_A`.
  - Then: full-fit, and augeq for the frozen towers.
- **On FAIL.** Fine-tuning does not break the shortcut; 10-06 slot 2 stays empty.
- **10-05 10:05 amendment (before any new training).**
  - Run 1 (8 min, ~1 CU) produced no fine-tune: `ft.py` could not find `meta_sorted.csv` under `/content/in`, because the aux download was dropped. Fixed by copying it into `extra/`.
  - Run 1 did read the session-10 fine-tunes against the gate:

    | model | G1n_R | CRx_R | R_640 | gate |
    |---|---|---|---|---|
    | h3A (H+, d8, train-only) | 0.476 | 0.922 | 0.781 | **PASS** (fold-0 val F1 0.8874) |
    | h3B (+ locked test) | 0.399 | 0.943 | 0.767 | not eligible |

    Fine-tuning roughly halves the shortcut, against frozen h2 at 0.923.
  - The h4/h4c ViT-L pair is replaced by **h4H**: h3A's exact H+ recipe with only the augmentation changed to `augeq`. h3A is the control.
  - Same gate. Candidates come in two variants: R→A only, and both directions.

## 10-05 03:00 UTC — session 11b result (gate pre-registered 09:30 UTC; A100 93.4 min; Drive `datascape_2026/session11`)
| model (fold 0, train-only) | G1n_R (≤ 0.50) | CRx_R (≥ 0.917) | R_640 (≥ 0.65) | R_other | fold-0 val F1 | gate |
|---|---|---|---|---|---|---|
| h2 frozen (reference) | 0.923 | 0.967 | 0.810 | 0.285 | — | — |
| h3A H+ `d8` | **0.476** | 0.922 | 0.781 | 0.265 | 0.8874 | **PASS** |
| h4H H+ `augeq` | 0.515 | 0.963 | 0.795 | 0.273 | 0.8871 | FAIL (G1n) |

- Always-on corruption did **not** reduce the shortcut further: 0.515 vs 0.476. It kept more Rusak overall (CRx 0.963), so it is a more Rusak-leaning model, not a less biased one.
- The gain comes from full fine-tuning itself (h2 0.923 → h3A 0.476), not from equalising augmentation. `augeq` is closed.
- **Session 12 rule** (h4H must beat h3A by ≥ 0.05 on G1n): not met, so the final model F1 uses `d8`, h3A's recipe.

## 10-05 05:00 UTC — CORRECTION: row-order bug in today's copy-based subsets
- `work/pix_all.npz` is in **meta_sorted order** (`work/pixall.py`): 796 exact test rows, train-train agreement 0.962. Today's s11 rows file, the gate sets and the `d06_p640_A` selection remapped it through `meta.csv` order. That scrambles it: 1,309 exact rows, agreement 0.823.
- **Affected:**
  - G1n (the no-copy subset of the LB-confirmed Aman rows) and CRx/CAx (copy-labelled sets) in sessions 11/12;
  - the candidates' "no copy" eligibility in sessions 11/12 (`cand_*` in Drive are superseded);
  - the parked `d06_p640_A` set, and the "no-copy" label of the 640 montage. That montage was effectively a random sample of 640 Rusak predictions, which still look damaged.
- **Not affected:**
  - G1-all (source only): h2 0.925, DA 0.490, h3A **0.471**, h3B 0.396, h4H 0.512. The h3A conclusion stands on this set.
  - F1's pseudo-labels (twin vote plus source).
  - `d06_cur_A` (source only), `lbdelta` and the LB inversion.
- **Fix:** `work/make_final_nb.py` §9/§10 recompute the gate and the candidates in meta_sorted order from the Drive predictions.

## 10-05 07:00 UTC — autoresearch loop formalised + finalist notebook v2 (Drive `datascape_2026/final`)
- **Research** (kaggle-competition-research RAG):
  - UBC-OCEAN r001: frozen foundation features plus large seed ensembles; entropy outliers; domain-consistent inputs; CV→LB gap from domain shift.
  - BirdCLEF-2025 r005: self-distillation for noisy labels.
  - BirdCLEF-2024 r004: build a test-like validation set.
  - None replaces the paired LB probe here.
- **Evaluator** (advisor-agreed): the paired LB probe is the fixed autoresearch evaluator, logged in `work/results.tsv` (KEEP/DISCARD; the base moves only on KEEP). Offline gate numbers are guardrails only.
- **F1 predictions downloaded** with the user's OK: `work/drive_in/session12/pred_F1_f0.npz`, 233 KB.
  - F1 vs `d06_cur_A` on all rows: 96 disagreements (82 A→R, 13 R→A, 1 L→A).
  - Eligible (640/other, no copy at NCC < 0.95, conf ≥ 0.70): 25 A→R (all "other") and 6 R→A.
- **10-06 slot 2 = `d06_F1_AR25`** (registered): `d06_cur_A` + those 25 A→R.
  - Montage, group-level: ~13–15 of 25 look damaged, so q ≈ 0.55–0.6; breakeven q ≈ 0.42.
  - Expected vs p416: q 0.5 → 0.8849, q 0.6 → 0.8864, q 0.7 → 0.8881; `d06_cur_A` alone is 0.8837 at q = 0.
  - Rule: AR25 − cur_A ≥ +0.003 KEEP (widen F1 A→R); ≤ −0.001 DISCARD.
  - program.md: `SUBMIT: d06_cur_A, d06_F1_AR25`.
- **Finalist notebook v2.**
  - Added an executive summary, a pipeline diagram, 11 matplotlib charts and an autoresearch-ratchet section.
  - Executed on Colab A100: no cell errors, runtime released.
  - The runner cell is now fail-safe (try/finally `runtime.unassign()`, mount retry, Kaggle fallback). An earlier `mount failed` left an A100 idle ~3 min before a manual release.

## 10-05 07:45 UTC — session 13 PRE-REGISTERED: DINOv3-7B (largest variant, user request) vs DINOv3-H+
- **h5:** `vit_7b_patch16_dinov3.lvd1689m` @384, group-fold 0, train-only, d8 aug, cosine 6 epochs, lr 5e-5, head ×10, LS 0.1, flip TTA.
  - Bottom 34/40 blocks frozen in bf16; top 6 blocks + norm/head trained in fp32 (≈1.0B params); no EMA. A full fine-tune of 6.7B does not fit 40 GB.
- **Gate** (correct meta_sorted rows). 7B is "better than H+" only if all hold:
  - G1n Rusak ≤ 0.437 (h3A 0.487 − 0.05);
  - CRx Rusak ≥ 0.97;
  - 640 Rusak ≥ 0.65.
- **On PASS:** the next session is a 7B full-fit + pseudo-labels (F2). **On FAIL:** H+ (F1) stays the final backbone.
- **Code:** `ft.py` options `unfreeze_last`, `frozen_bf16`, `ema=0`; notebook `colab/make_nb13.py`; runner `colab/run13.py`. The local dry run passed (tiny model).

## 10-05 09:10 UTC — session 13 retry running; session 14 re-specified by the user
- **Session 13, attempt 1:** CUDA OOM at the epoch-1 validation (38.5/39.5 GiB used; predict batch 128).
  - Fix: `pred_bs=24`, plus gradients freed (`zero_grad(set_to_none)`) and `empty_cache()` before validation; per-epoch GPU-peak print.
  - The retry passed the epoch-1 validation. Raw fold-0 val F1: 0.868 after epoch 1, 0.903 after epoch 4 (h3A final 0.887).
- **User 10-05:** "naikkan epochnya sampe konvergen patiencenya 1, gunakan varian high ram untuk dinov3 7b"; "tunggu selesai dulu runtime sekarang baru dilanjut".
- **Session 14 re-specified** (runs after session 13 finishes, on an A100 **High-RAM** runtime):
  - Model: 7B F2.
  - Training data: train folds 1–4 + pseudo-labels (twin lock + LB-confirmed 416/675 Aman).
  - Stopping: fold-0 validation every epoch; plateau schedule (1-epoch warmup, then constant LR); **early stop at patience 1**; best epoch restored; max 15 epochs.
  - Trained regardless of the session-13 verdict (user request). Both F1 and F2 become candidates for paired LB probes.
- **`ft.py`:**
  - For partial fine-tunes, only the trainable tensors are checkpointed.
  - The restore uses `strict=False`.
  - Local tests: early stop at patience 1 → restored best epoch → saved.

## 10-05 09:45 UTC — session 13 result (pre-registered gate 07:45 UTC; retry run, 97.4 min A100, Drive `datascape_2026/session13`)
| model (fold 0, train-only, correct rows) | G1n Rusak (≤ 0.437) | CRx Rusak (≥ 0.97) | CAx | 640 (≥ 0.65) | other | fold-0 val F1 |
|---|---|---|---|---|---|---|
| h2 frozen | 0.938 | 0.991 | 0.034 | 0.810 | 0.285 | — |
| h3A H+ full FT | 0.487 | 0.978 | 0.021 | 0.781 | 0.265 | 0.8874 |
| h4H H+ augeq | 0.522 | 0.987 | 0.023 | 0.795 | 0.273 | 0.8871 |
| **h5 DINOv3-7B top-6 FT (6 ep cosine)** | **0.484** | **0.991** | 0.015 | 0.772 | 0.277 | **0.8977** |

- **Gate: FAIL.** On the shortcut the 7B equals H+: 0.484 vs 0.487, short of the −0.05 needed.
- It is slightly better on copy-confirmed Rusak (0.991 vs 0.978), CAx (0.015 vs 0.021) and train-label val F1 (0.898 vs 0.887).
- Per the user, session 14 (7B to convergence, patience 1, High-RAM, + pseudo-labels) runs anyway. The paired LB probe will decide between F1 and F2.

## 10-05 11:15 UTC — session 14, attempt 1 terminated by Colab; relaunched with checkpoint-to-Drive
- **Attempt 1** (A100-SXM4-80GB, High-RAM): Colab ended the runtime at ~45 min, during epoch 3 ("Runtime terputus"; 0 active sessions; 110.9 CU still available). Not user- or code-initiated; no traceback.
  - Raw fold-0 val F1: epoch 1 0.8865, epoch 2 0.8883 (still improving; no early stop yet).
  - Nothing was saved, because outputs were only copied at the end.
- **Fix:**
  - `ft.py pred_on_best`: each improving epoch writes `pred_F2_f0_best.npz` (raw-view test + val predictions).
  - `run14.py` syncs `/content/out14` to Drive `datascape_2026/session14` every 5 min (logs, history, best-so-far predictions).
  - Tested locally: best saved at epochs 0 and 1 → early stop at epoch 2 → restored epoch 1.
- **Attempt 2** launched 11:12 UTC on A100 High-RAM; Drive mounted on the first attempt.
