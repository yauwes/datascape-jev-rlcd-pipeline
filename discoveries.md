# Discoveries (what we learned and why) — read before planning

Relay notes per Deotte, *1st Place – The AI Relay* (PS S6E9 write-up): `program.md` = plan, `RESULTS.md` = progress, this file = discoveries, `closed_ideas.md` = do-not-retry.

1. **Test is made of augmented near-copies of train.** DINOv2 test→train sim median 0.918; 57% > 0.9; 253 byte-identical files. Random-fold val→train is *less* similar (45% > 0.9).
2. **Copying train labels to test copies hurts LB.** Our kNN blend scored 0.786, and the team's "DINOv2 twin lock" scored 0.790, against the team's 0.821 without it. Test labels of near-copies are not reliably the train label.
3. **Group CV (clusters at DINOv2 sim 0.90 + MD5) tracks LB better than random CV.** For the same recipe: random 0.897, group 0.845, LB 0.786. The remaining ≈ −0.06 offset is explained by Rusak over-prediction on test (#11).
4. **Train labels contain visible errors** (crushed boxes labelled Aman, intact labelled Rusak). But relabelling Aman→Rusak by confident learning *lowered* LB by about 2σ of the paired delta, so the test labelling seems to share the train labelling style.
5. **Noise is label-correlated in train.** S&P noise: Aman 19.7%, Rusak 6.4%, test 3.8%. In clean rows Rusak is 31%; in noisy rows 9%. Test is mostly clean, and models predict ~40% Rusak there; LB evidence (#11) says that is over-prediction, not a Rusak-heavy test.
6. **Fold-honest per-class bias (Nelder-Mead) is a large lever:** +0.042 group-OOF on the LR probe.
7. **Hierarchical (Lainnya-vs-rest → Aman-vs-Rusak) loses** to 3-way: −0.037.
8. **Lainnya is a third of the metric and may be under-predicted on test.** Paper bag, kraft pouch and air-column bag got P(Lainnya) < 0.1; ranks 56–65 (egg carton, takeaway box) are borderline.
9. **Public LB SE ≈ 0.0165** for a single submission (bootstrap, n=917). Only 1 submission/day is ours, so submit only when honest group-OOF ≥ 0.885 (≈ LB 0.82+ by the offset in item 3).
10. **Kaggle allows max 2 concurrent batch GPU sessions per account.**
11. **THE CV–LB GAP IS MOSTLY OVER-PREDICTED RUSAK ON TEST (biggest lever).** Five LB points are monotone in predicted test Rusak share: 30.2% → 0.8209 (team best), 34.6% → 0.7869, 35.1% → 0.7896, 39.3% → 0.7863 (ours), 42.1% → 0.7676 (ours + relabel). Forcing our group-OOF to the same shares reproduces LB within about 0.015: 39% → 0.799, 42% → 0.786. At the train prior (24.5%) the same model scores 0.846. Fix: **prior matching**. Shift class logit biases on test so the predicted shares ≈ train prior. This assumes the test prior ≈ train prior and the model's test outputs are biased by covariate shift (cleaner, less noisy test images). Refs: Saerens 2002; classify-and-count adjustment (Forman 2008); team post-mortem in yauwes/datascape-2026 `docs/EXPERIMENT_RESULTS_CONVNEXTV2_CQA.md` (more Rusak → "false alarm trap").
12. Team repo (yauwes/datascape-2026) facts: their best 0.82086 was a **single 1,834-row holdout** model (ConvNeXt-S 384 + frozen SAM ViT-B injected at stages 2/3, weighted CE w=(1/n)^0.5, LS 0.05, 7 ep). Its own clean CV of 0.9156 overstates LB by about 0.095. The 5-fold version scored lower on LB. Lainnya test share in all their subs ≈ 1.7–1.9% (train 3.5%), so Lainnya is probably under-predicted too. Check the ranking near the cutoff before matching it.
13. **Diversity comes from different pretraining, not different fine-tuned architectures.** Fine-tuned CLIP-L, ConvNeXt-L-CLIP and ConvNeXt-B-CLIP correlate at 0.92–0.93 and add nothing to each other. The DINOv2+CLIP LR probe (corr ~0.7) adds +0.015 at the matched share. So the next lever is frozen probes on new towers (VLM/SSL); fine-tune a second network only if its pretraining differs (DINOv3/SigLIP2/PE).
14. **Byte-identical test copies are the cleanest window on the test labels.** In clean_other, 252 copies are 38.1% Rusak by train label and twins ≥0.97 are 44%, against 31.3% in train. So the test *is* somewhat Rusak-heavier, but models over-shoot it (48.5% natural). Match to the copy rate, not the train rate.
15. **"Train to convergence" with OneCycle stretched to 50 epochs + early stopping = the LR never anneals.** The 1-epoch warmup reaches peak LR, and cosine over 50 epochs keeps the LR at ≈ 97% of peak at epoch 5. Val NLL/F1 swing at peak LR: fold 3 went 0.77 → 0.62 → 0.67 → 0.63 F1. So ES (patience 3) fired at best epochs 3/12/1/0/1, and OOF fell to 0.801 from 0.849 for the fixed 6-epoch cosine run.
    - The fix (protocol v2): linear warmup 1 epoch, then constant LR + ReduceLROnPlateau.
      - Plateau patience 1 is strictly less than ES patience 5, so the LR decays at least twice before stopping.
      - `min_lr` floor.
      - Model selection, ES and plateau all run on fp32 EMA weights (Polyak & Juditsky 1992; Morales-Brotons et al. 2024, TMLR: EMA improves generalisation and label-noise robustness).
      - `min_delta` on val NLL.
      - LR logged per epoch.
    - The flawed-protocol runs (`clipL224_es`, `peL224_es`, `pebB_base/mixup/strong`) must not enter the A15 table.
16. **Native input resolution is the consistent sweet spot for frozen towers.** PE-L: 224 ≈ 336 > 448. SigLIP2: 384 ≈ 512 ≫ 224. Upscaling past native adds nothing, and below native hurts patch16 SigLIP more than patch14 PE.
17. **Preprocessing was the bottleneck, not the GPU.** On the A100, CLIP-L took ~120 s/epoch with the GPU at ~40% busy, while 10 loader workers sat at 100% CPU.
    - PIL `MedianFilter(3)` costs 44 ms per image, about 75% of the pipeline. `cv2.medianBlur(3)` is **bit-identical** (52/52 images, 512 px included) at 0.5 ms.
    - `noise()` also did a float round-trip for the 70% of images it leaves untouched. It now returns early.
    - Data cost falls from ~57 to ~12 ms/img, so data throughput should rise about 5×. Same outputs, so earlier runs stay comparable.
    - Applied to `kernels/ft/ft.py`, `colab/emb_giant.py` and the live Colab run (every job after the running canary fold). Kaggle T4 runs also benefit: they were CPU-bound with 4 vCPUs.
18. **TWIN LABEL TRANSFER IS THE BIG LEVER. The earlier "twin lock hurts" verdict was confounded.**
    - The LB is now #1 0.976 and #2 0.924; our team best is 0.821.
    - Neither earlier LB test was a paired comparison:
      - ours (0.786) also had a 39.3% Rusak share (#11);
      - the team's DINOv2 twin lock (0.790) was compared against a *different* model (0.821).
    - Train leave-one-out, nearest *other* train image by DINOv2-B max-over-5-variants:

      | sim | twin label = own label | group-OOF MLP acc | test n |
      |---|---|---|---|
      | ≥0.99 | 0.961 | 0.836 | 514 |
      | 0.97–0.99 | 0.923 | 0.887 | 424 |
      | 0.95–0.97 | 0.928 | 0.923 | 285 |
      | 0.90–0.95 | ~0.95 | ~0.93 | 533 |
      | <0.85 | 0.861 | 0.908 | 838 |

    - Where the model and twin disagree: at sim ≥0.99 the twin is right 88% vs the model 11.5%; at ≥0.95, 78% vs 21%.
    - Hybrid estimate: accuracy 0.890 → 0.921, about #2's level.
    - 63% of test (1941/3057) sits in a train cluster.
    - Visual check: many 0.79–0.82 pairs are the *same photo* after heavy augmentation (noise, zoom-out padding, blur, crop) that DINOv2's global similarity misses. So a real copy detector (SSCD, Pizzi et al. CVPR 2022) should find more twins. That is probably the gap from 0.92 to 0.976.
    - Image size is a source artifact (Aman 512², Lainnya 200², Rusak 640²), but size→majority is only 0.82 accurate and the test size mix differs, so it is not a usable key. Test file order carries no signal.
19. **Canary of protocol v2** (CLIP-L fold 3, A100): best epoch 3, LR decayed 3e-5 → 2.7e-6, ES at ep 8.
    - F1 0.8074 (NLL 0.381), against 0.7638 under v1 and 0.8207 for the fixed 6-ep cosine run.
    - The bug is fixed, but the pre-registered bar (0.8207) is missed, so the A100 driver skipped the A15 fine-tune ablations.
    - Fine-tuning stays a minor lever; twins and frozen towers come first.
20. **SSCD (Pizzi 2022) is the right twin detector.**
    - At similarity 0.55 it covers 60% of the test set (clean_other) at 95.5% LOO label accuracy.
    - It beats the twin-seen model in 74% of disagreements.
    - Its twin accuracy is flat from 0.95 down to 0.55 and only drops below 0.45; DINOv2 degrades below 0.9.
    - The hybrid (twin vote, else model) is worth about +0.03 macro-F1 on train.
    - Coverage, not precision, is now the lever: about 40% of test still relies on the model.
21. **Gate twin locks on vote purity.** Disagreements between twin and twin-seen model (`r_cat6`), SSCD 0.55:
    - purity ≥0.9: twin right **0.921** (n=302);
    - purity 0.7–0.9: twin right 0.267 (n=45);
    - purity <0.7: twin right 0.375 (n=88).

    Mixed twin votes are near-duplicate label noise, and there the model is the better judge. With the gate, locked-row accuracy goes 0.955 → 0.981.
22. **Calibrate on test-domain twin labels, not on the old 38.1% target.**
    - The 1534 locked test rows (twin labels ~98% right) form a test-domain validation set.
    - On them the h2 model predicts 42.6% Rusak against a true 31.0%; the errors are one-directional (180 A→R vs 2 R→A). The best logit bias is −1.5 (F1 0.9148 → 0.9395).
    - The MD5-identical subset is 38.8% Rusak, but all locked rows are 31.0%. So the 38.1% clean_other target came from an unrepresentative subset; holding it after locking pushed 49.7% Rusak onto the unlocked rows.
    - Unlocked rows now get their own targets:
      - no-twin rows: 0.349 via the bias transfer (agrees with the train ratio, ≈0.36);
      - mixed-vote rows: 0.555 from their twin vote, calibrated on train (0.17/0.30/0.50/0.78 per bin).
23. **Domain adaptation with twin-labelled test rows is the second big lever.** The head is trained on train plus the locked test rows.
    - Measured on held-out locked rows with strict SSCD union-find groups (850 groups, 5 folds): F1 **0.9783** vs 0.9403 for the train-only head (both with CV-calibrated bias).
    - The DA head's natural Rusak share on no-twin test rows drops from 0.483 to 0.358, so it learns the test-domain shift instead of needing a large scalar bias.
    - CV bias for DA heads: −0.70/−0.95/−1.15/−0.75/−0.80, median −0.8.
24. **Twin LOO agreement only measures agreement between train labels.** It cannot detect label noise shared across a duplicate group (flipped before the copies were made). Every expectation built on it (0.905–0.93) had no LB anchor, and `d04_da_bt2` came in at 0.845.
    - Lesson: before trusting a lever whose validation uses the same (possibly corrupted) labels as its training, anchor it with one paired LB test.
25. **LB anchors on 10-04:** twins + DA give +0.015 over the plain model at equal shares (0.845 vs 0.830). Twins help, but the LOO estimate (+0.05) was 3× too optimistic, which is consistent with partly group-consistent label noise. The base CV→LB gap of ≈ −0.065 is the dominant problem; the twin lever is secondary.
27. **The CV→LB gap is a test-label convention (or per-source prior) shift, not the source mix (10-04).**
    - Reweighting group-OOF to the test mix (size buckets, or test-nearest-neighbour importance) predicts LB 0.90–0.92 at 27.4% Rusak, with an optimum at ≥30%.
    - LB gives 0.830 and prefers fewer Rusak.
    - Image size identifies the data source, and sources differ wildly in train Rusak rate: 416 synthetic 71%, 675×900 parcels 78%, 640×640 web 47%, 512×512 9%.
    - Pixel-exact copies (dihedral NCC ≥ 0.99, `work/pixall.py`) cover 26% of test and agree 96.8% train-to-train.
    - SSCD "twins" in synthetic sources (416) are different renders and carry the majority label, not the row's label.
    - Next lever: per-source Rusak rates on test, read from source-decoupled paired LB probes.
28. **The "stricter test Rusak" story is weakened; copy structure per source leads (10-04).**
    - The montage of rows the share dial would flip shows clearly damaged boxes, and the share trend mixes different models.
    - Pixel-exact copies show that in the synthetic and curated sources (416, 512, 675) only Aman rows cross the train/test split: 0 of 807 train Rusak rows in those sources have a test copy.
    - Leading hypothesis: no-copy test rows in those sources carry a much higher Rusak rate than da_bt2 predicts (675: 11%).
    - The Oct 5 paired probes `p675_R` and `p416_allA` decide it (RESULTS 10-04 07:30).
29. **`ft.py` never applied layer decay before 10-04 (found while preparing session 10).** timm's `create_optimizer_v2(layer_decay=…)` stores each group's decay as `lr_scale` and leaves `lr` at the base value; only timm's own schedulers multiply them. `ft.py` drives LR itself from `g['lr']`, so every earlier fine-tune (ConvNeXt, EVA02, CLIP-L, DINOv3-L, SigLIP2; `layer_decay` 0.8) trained the whole backbone at one uniform LR.
    - Fixed: `g['lr'] = g['lr0'] = g['lr'] * g.get('lr_scale', 1.0)`. It is checked on timm 1.0.30: groups report `lr` 1e-3 with `lr_scale` 1e-4…1.0.
    - Consequence: the closed "L fine-tunes trail frozen heads" results tested uniform-LR fine-tuning, not LLRD. The pig-posture winner found uniform LR best anyway, so this does not by itself reopen them.
30. **Test is largely augmented train, but augmentation does not preserve train labels (session 10).**
    - Exhaustive matching confirms the mapping is saturated: 800 test rows have an exact copy, and SSCD top-10 already held 99.5% of them. 79% of test has some train source.
    - The near-copy tier shows the generator: Gaussian noise, colour shift, flips, grayscale.
    - Train itself holds the same base image as a colour copy labelled Aman and a grayscale copy labelled Rusak. Copy groups with conflicting labels are coin flips for every model (h2 0.564, frozen DINOv3 0.532).
    - A fully fine-tuned DINOv3-H+ reproduces copied train labels on near-duplicates far better than frozen heads (0.939 vs 0.889 on locked rows, without seeing them). Metrics built on copied labels therefore reward memorisation. The LB (copy label right ~55% where it disagrees with the model) is the only arbiter.
