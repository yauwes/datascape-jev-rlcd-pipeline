## >>> NEXT ACTION 10-10 (state at 10-09 01:40 UTC) <<<
- **Team best:** `d11_lbopt`, public 0.88891 (`work/make_d11.py`; d08 + 17 LB-regression rows). The regression with d11 added proposes 0 further edits.
- **Finals: SELECTED on Kaggle 10-09 ~05:05 UTC (2/2): `d11_lbopt` + `d08_grand_consensus_champion`, verified on the Submissions page.** Any team member can change this until 10-10 16:59 UTC, so tell Farhan.
- **Deliverables ready:** `deliverables/last_game_Notebook_Datascape.ipynb` (executed, 0 errors) and `deliverables/last_game_Writeup_Datascape.pdf` in the repo. Fill in the private score after the final leaderboard, then re-render with headless Chrome from `deliverables/writeup.html`.
- **10-09 slot 2 (user OK):** long-shot `d14_640A` (87 no-copy 640 rows R→A) scored 0.85698 (−0.032), so DISCARD: 640 no-copy rows are Rusak. 10-09 used 2/3 slots.
- **Slots:** hold them unless a candidate is clearly better than d11 (the user's rule). 7B variants move at most 29 free rows (README §0), so they can't qualify.
- **Session 15** (cleaned 7B folds) failed its gate and was stopped. **Session 16** (F2 recipe, cosine 4 epochs, fold 0) answers the user's "train to convergence" question only. Judge it on the LAST epoch's raw val vs F2 0.3266 / 0.8867. Session 17 is cancelled.
- d09_* are retracted (the teammate's d23 lost 0.0048 on a subset).

## >>> NEXT ACTION 10-06 00:00 UTC (survives context reset) <<<
SUBMIT: d06_cur_A, d06_F1_AR25
(The cron submits exactly the names on the SUBMIT line, at most 2 and only if each is registered in submissions/ledger.csv. Slot 2 is added here ONLY after `subreg add`.)
- **Base:** `d05_p416_allA`, public LB 0.87739. Read each submission with `/usr/local/bin/python3.11 work/lbdelta.py submissions/d05_p416_allA.csv submissions/<name>.csv 0.87739`. lbdelta integrates Binomial public counts and handles flips in both directions.
- **Slot-2 candidate (needs `pred_F1_f0.npz` from Drive `datascape_2026/session12`):** `d06_F1_AR25` = `d06_cur_A` + the 25 "other" no-copy rows that F1 calls Rusak at ≥ 0.70 where the base says Aman.
  - It tests whether the base's −0.8 no-twin Aman bias over-corrects in "other".
  - It runs AGAINST the confirmed direction (every LB win so far was R→A); this is stated when it is pre-registered.
- **Autoresearch ratchet:** `work/results.tsv` holds one paired LB probe per row (hypothesis, base, rows changed by group and direction, lbdelta expectation, actual, q, KEEP/DISCARD). The base moves only on KEEP.
- **Never submit:** `d06_p640_A` (visually damaged), `d06_map_max` (old base), `d05_share_non416`.
- **Colab artifacts (Drive `datascape_2026`):**
  - session10: h3A/h3B fold-0 predictions;
  - session11: h4H and the gate;
  - session12: F1 predictions + `weights/`;
  - final: the executed finalist notebook (QUICK, no cell errors) + submissions.
- **Team repo** (private, pushed a147f27): `yauwes/datascape-2026-pipeline`, with `reproduce_best.py` → 0.87739, the finalist notebook and FUTURE_WORK.md.

# Datascape 2026 autoresearch contract

Pattern from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and `../s6e9/autoresearch/program.md`: fixed evaluator, one hypothesis per run, every result logged in `RESULTS.md`, keep only measured gains.

## Task
3-class package condition (Aman / Rusak / Lainnya), metric Macro-F1, final = 30% public + 70% private. Max 3 submissions/day, only from team account `ajikajadeh`. Only organiser data. No VLM/LLM inference (image towers of CLIP/EVA fine-tuned or probed as classifiers are allowed by the organiser; no text prompts, no zero-shot).

## Evaluator (frozen)
- `work/folds.csv`: StratifiedGroupKFold(5, seed 42) over near-duplicate clusters (DINOv2-B, max-over-flip/rot similarity + exact MD5). Every model uses these folds.
- Score = OOF macro-F1 on all train rows. Bias/threshold tuning is fitted on 4 folds and applied to the 5th (no in-sample tuning).
- Promotion gate: OOF macro-F1 above champion, and paired bootstrap lower bound > 0.

## Compute
- GPU: `asuudane` (T4 x2, 30 h/week) via private dataset `asuudane/datascape-small-512`; `yoursbae` only as extra GPU. Neither account submits.
- **Colab A100** (user's Google account panjiseno22, 10-03). The code lives in the private dataset `asuudane/datascape-colab-code`, published by `work/push_colab_code.sh` from `kernels/ft/ft.py` + `colab/{setup.sh,emb_giant.py,driver.py}`.
  - The user stores the asuudane token as Colab Secret `KAGGLE_API_TOKEN`. Claude never types tokens.
  - Then one typed cell:
    `import os; from google.colab import userdata; os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')`
    → `!pip -q install -U kaggle && kaggle datasets download asuudane/datascape-colab-code -p /content/code --unzip -q --force`
    → `%cd /content/code`
    → `!bash setup.sh && python -u driver.py 2>&1 | tee -a /content/out/log.txt`
    → `from google.colab import runtime; runtime.unassign()` (releases the GPU the moment work ends).
  - `driver.py` is resumable and checkpoints after every fold or tower to the private dataset `asuudane/datascape-colab-out` (preds, hist, emb, `log.txt`), with weights in `asuudane/datascape-w-<tag>`. Monitor it from the Mac via the Kaggle API, no browser needed.
- Kernels are resumable: each fold writes `pred_<tag>_f<k>.npz` + fp16 weights; a rerun with the previous output attached skips finished folds.

## Relay notes (Deotte, *The AI Relay*, PS S6E9 1st)
`program.md` plan · `RESULTS.md` progress · `discoveries.md` learnings · `closed_ideas.md` do-not-retry. One shared fold file (`work/folds_group90.csv`); judge a model by its ensemble contribution (keep if <0.99 corr with current blend and it raises honest OOF); trust group-CV, not public LB.

## Expectation table (team LB history = priors for new submissions)
| recipe | LB |
|---|---|
| Hierarchical SAM-SLCA v2, ConvNeXt-S 384, 4-way TTA, Nelder-Mead | **0.82086** (team best) |
| Ensemble consensus 3-model (SAM-SLCA v2 + 14ep + multi-encoder) | 0.81858 |
| SAM-SLCA 14ep cluster-aware fold 0 | 0.81531 |
| Super ensemble v2 + DINOv2 twin lock | 0.78957 |
| SAM3-CQA 20ep | 0.78753 |
| ours: 5NN+LR probe blend | 0.78627 |
| ours: same + CL relabel | 0.76761 |
| first baseline | 0.75914 |

## Ablation matrix (run top-down; each row = one hypothesis vs current champion on frozen folds)
Cheap rows run on cached embeddings locally (seconds); GPU rows only after a cheap proxy says the axis matters.

| # | Axis | Variants | Cost | Basis | Keep if |
|---|---|---|---|---|---|
| A1 | Duplicate label transfer | kNN / cluster-majority from train for test images with sim > t, model elsewhere | CPU | DINOv2 copy detection; happy-whale r019 kNN; Barz & Denzler 2020 (near-dups CIFAR) | OOF on dup-covered rows > model |
| A2 | Backbone (frozen-probe proxy, orig view, group folds + honest bias) | done: DINOv2-B/L, CLIP-L. queued: VLM/SSL vision towers — Meta PE-Core-L (Bolya 2025), DINOv3-L (Simeoni 2025); Google SigLIP2-so400m (Tschannen 2025); Qwen3-VL ViT; InternVL2.5 InternViT-300M; EVA02-CLIP-L (Sun 2023); DFN5B CLIP-H (Fang 2023); AIMv2-L (Fini 2024). Image towers only, no text/zero-shot (organiser-approved) | ~6-20 GPU-min each | Kornblith 2019 (probe ranks transfer); Fang/Kornblith/Schmidt 2023 (pretraining data > ImageNet acc) | fine-tune only the top-1/2 towers |
| A3 | Label cleaning | none / drop CL issues / relabel to cluster majority | CPU then GPU | Confident Learning (Northcutt 2021); Cassava (≈20% noise) top teams | OOF on *original* labels of clean folds rises |
| A4 | Loss | CE+LS 0.1 / bi-tempered / Taylor-CE / GCE / SCE | GPU | Lukasik 2020 (LS vs noise); Amid 2019; Feng 2020; Zhang & Sabuncu 2018; Wang 2019 | +OOF over CE+LS |
| A5 | Robustness augs / shortcut removal | median denoise on all inputs + class-independent noise aug, low-res 224 round-trip, rot/flip | GPU | Measured: S&P noise on 19.7% Aman vs 6.4% Rusak vs 3.8% test (label-correlated shortcut; Geirhos 2020); Rusak 2020; Hendrycks & Dietterich 2019 | +OOF, esp. on clean-image rows |
| A6 | Resolution / head | 224 vs 336/448; linear vs GeM/MLP head | GPU | FT-CLIP (Dong 2022); GUIE r018 (224→240 TTA +0.003) | +OOF |
| A7 | One-stage vs two-stage | 3-way softmax vs (Lainnya-vs-rest) → (Aman-vs-Rusak) | GPU/CPU | Hierarchical OOD-first classification; Lainnya = OOD class | +macro-F1 on Lainnya |
| A8 | TTA | none / hflip / +rot90s / +scales | inference | GUIE r018; standard Kaggle practice | +OOF |
| A9 | Pseudo labels (iterative) | soft test PL round 1/2, no hard threshold | GPU | birdclef-2025 r002 (+0.02-0.03 over 2 rounds); Noisy Student (Xie 2020) | +OOF (PL never on its own fold) |
| A10 | Ensemble | mean / rank / hill-climb (Caruana 2004) / LR-or-LGBM stack (Cassava 14th, kozodoi) | CPU | fold-wise refit, scored on held-out fold | beats best single with bootstrap LB>0 |
| A12 | Class imbalance | none / class-weighted CE / sqrt-balanced sampler / logit-adjusted loss τ=1 / focal γ=2 / decoupled cRT head re-train / per-class bias at inference (A11) | GPU+CPU | Menon 2021 (logit adjustment); Cui 2019 (class-balanced loss); Lin 2017 (focal); Kang 2020 (decoupling, cRT, sqrt sampling); birdclef-2025 r002 (balanced sampling + upsampling); Lipton 2014 | +OOF macro-F1, esp. F1_Lainnya |
| A13 | Post-processing (each on group-OOF, fold-honest) | (a) test-time cluster smoothing: average *model predictions* within test↔test near-dup clusters (no train labels); (b) prior-shift EM re-weighting (Saerens 2002) / BBSE (Lipton 2018); (c) logit- vs prob-averaging + temperature; (d) TTA set: hflip / rot90s / multi-scale; (e) per-class bias (A11) | CPU | Zhou 2004 (local-global consistency); Saerens 2002; Lipton 2018; Lipton 2014 | +OOF; (a)/(b) need LB confirmation since OOF can't see test shift |
| A14 | **Prior matching on test** | bias logits so test predicted shares = train prior (Rusak 24.5%; Lainnya 3.5% only after visual check of the cutoff) | CPU | Saerens 2002; Forman 2008; discoveries #11 (5 LB points + OOF simulation) | next LB sub; expected +0.03–0.05 |
| A15 | Pre-processing / augmentation / resolution | proxy PE-Core-B@224, folds 0–1, ES 50/5, one factor each: base, strong (TrivialAugment + RandomErasing), mixup, cutmix, no-median, letterbox pad, 384px; resolution also on frozen PE-L (224/336/448) and SigLIP2 (224/384/512) | ~1.5 GPU-h each (384: ~4 h) | Mueller 2021; Zhong 2020; Zhang 2018; Yun 2019; teammate EDA (conveyor-tray shortcut → CutMix/Erasing); Touvron 2019 (train/test resolution) | 2-fold delta > ~0.01 (2-fold OOF SE ≈ 0.01); winners go to the big model |
| A11 | Decision rule | argmax vs per-class logit bias fitted on 4 folds | CPU | Lipton 2014 (F1-optimal thresholds) | +OOF out-of-fold |

## References used for hypotheses
- Near-duplicate retrieval with self-supervised embeddings: DINOv2 (Oquab et al., 2023) copy-detection results; happy-whale r019 (kNN over embeddings).
- Noisy labels: Confident Learning / cleanlab (Northcutt et al., 2021): drop or relabel train rows whose OOF prob for the given label is low.
- CLIP image-tower fine-tuning with layer-wise LR decay: FT-CLIP (Dong et al., 2022); google-universal-image-embedding r018.
- Soft pseudo labels on test, no hard thresholding: birdclef-2025 r002.
- Linear-probe ranking as a cheap proxy for which backbone to fine-tune: Kornblith et al., 2019.

## Artifact registry (everything private, checkpointed)
- `asuudane/datascape-artifacts` (Kaggle dataset, versioned by `work/sync_artifacts.sh "msg"`): code, relay notes, all embeddings (`emb/emb_<tower>.npz`, rows = `meta/meta_sorted.csv`), folds, OOF/test predictions (`preds/`, `preds/ft/<tag>/pred_<tag>_f<k>.npz`), submissions.
- Fine-tune weights (fp16, per fold): private kernel outputs `asuudane/datascape-ft-clipl`, `asuudane/datascape-ft-eva02b`, `yoursbae/datascape-ft-cnxl`, `yoursbae/datascape-ft-cnxb` → attach as `kernel_sources` for pseudo-labelling or inference.
- Images: `{asuudane,yoursbae}/datascape-small-512` (512px cache + meta.csv).

## VLM family → vision tower map (image towers only; rules allow fine-tuning or probing them, not VLM inference)
| family | tower in timm | queue |
|---|---|---|
| Qwen3-VL / Qwen3.5 | qwen3_vit_306m (VL-4B), qwen3_vit_88m (0.8B) | emb2 |
| InternVL 2.5/3.5 | vit_intern300m_patch14_448 | emb3 |
| Gemma 4 / Gemma 3n | gemma4_vit_167m, mobilenetv5_300m | emb2 / emb3 |
| PaliGemma 2, Gemma 3, MiniCPM-V, DeepSeek-VL2, LLaVA-OV, IDEFICS | SigLIP-so400m (pali2_3b_pt), SigLIP2-so400m | emb3 / emb2 |
| SmolVLM, IDEFICS3 | SigLIP2-B/16 | emb2 |
| Florence-2 | davit_base_fl | emb2 |
| Llama 4 / Meta | PE-Core B/L, PE-Spatial B, DINOv3-L, MetaCLIP | emb2 / emb3 |
| GLM-4.xV | AIMv2 | emb3 |
| OpenAI (open vision = CLIP) | CLIP-L/14 224 + 336 | done / emb3 |
| Apple | DFN5B CLIP-H, MobileCLIP2 | emb3 / emb2 |
| Molmo2, Pixtral/Ministral 3, CogVLM2, Llama 4 native | not in timm; their encoders ≈ SigLIP/CLIP/EVA families above | — |
Selection rule: rank by group-fold probe + honest bias at matched prior, per GPU-minute; fine-tune only the top 1–2 (light ones preferred at equal score).

## Training protocol (all new runs)
- `work/mk_ft.py NAME ACCOUNT MODEL IMG LR LD BS TAG` → kernel from `kernels/ft/ft.py`, **protocol v2** (discoveries #15): max **50** epochs (user, 10-03).
  - 1-epoch linear warmup → constant LR + ReduceLROnPlateau (×0.3 after 2 flat epochs, min 1% LR).
  - fp32 EMA weights (0.999) for evaluation, selection and saving.
  - **Early stopping patience 5 on EMA val NLL** (min_delta 1e-3); best EMA weights restored. Per-epoch LR logged.
  - bf16 + TF32 on A100, fp16 on T4.
  - Options: `lpft` (Kumar 2022 LP-FT), `save_w`.
  - v1 (OneCycle over 50 ep) is invalid: tags `*_es`, `pebB_*`. Per fold it saves `pred_<tag>_f<k>.npz` (OOF + test, hflip TTA), `w_<tag>_f<k>.pt` (fp16 weights, ready for re-inference or pseudo-labels) and `hist_<tag>_f<k>.json` (convergence evidence).
- The first 4 runs (`*_g90` tags) use a fixed 6 epochs. Re-run one with early stopping only if its history shows val still improving at the last epoch.

## Submission ledger (expectation vs reality)
- `python work/subreg.py add NAME CSV "strategy" OOF EXP LO HI` registers a ready CSV + strategy + expected LB (format-checked). `submit NAME` (refuses unless the CLI user is ajikajadeh) puts strategy and expectation in the Kaggle description and polls the score. `sync` / `show` print the expected vs actual table (`submissions/ledger.csv`).
- Expected LB = OOF macro-F1 at the submitted class mix − 0.015 (fitted on 2 retro points: −0.013, −0.018). Range ±0.02 (≈1.2 SE of public LB).
