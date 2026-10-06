# Closed ideas — do NOT retry without new evidence

| idea | evidence | date |
|---|---|---|
| ~~kNN / "twin lock"~~ **REOPENED 10-03 (discoveries #18: both LB tests were confounded; train LOO says twins are 88% right when they disagree with the model)** — copy train label to near-duplicate test images | ours LB 0.786; team super-ensemble with DINOv2 twin lock 0.790 vs 0.821 without | 10-03 |
| Confident-learning relabel Aman↔Rusak (700 rows) | LB 0.7676 vs 0.7863 same recipe (≈2σ paired) | 10-03 |
| Random StratifiedKFold as evaluator | 0.897 OOF vs 0.786 LB; rewards memorising duplicates | 10-03 |
| Hierarchical 2-stage head on embeddings | group-OOF 0.769 vs 0.805 3-way | 10-03 |
| SAM3-CQA 20 ep (team) | val 0.903 → LB 0.7875 | 10-03 |
| SAM-SLCA dual branch v1 (team) | LB 0.7869 (v2 with ConvNeXt-S 384 got 0.8209) | 10-01 |
| Group split by union-find at sim ≤ 0.86 | chaining → 2,313-row cluster | 10-03 |
| ConvNeXt-B CLIP 384 fine-tune (6 ep) | group-OOF 0.816, corr 0.918 with CLIP-L, hill-climb weight 0 | 10-03 |
| EVA02-B 448 fine-tune (6 ep) | group-OOF 0.821, corr 0.925 with CLIP-L, lowers ensemble 0.8784→0.8718 | 10-03 |
| Qwen3-ViT-306M/88M, InternViT-300M, Gemma4-ViT, Florence-2 DaViT pooled features | all collapse (test Rusak 0.000–0.14, OOF 0.28–0.40) — **extraction/pooling bug (timm pooled output without the VLM's own merger/pooling), NOT evidence the tower is weak**; reopen only with the model's native pooling | 10-03 |
| Frozen towers above native resolution (PE-L@448, SigLIP2@512) | no gain vs native (−0.008 / −0.002) | 10-03 |
| OneCycle-over-max-epochs + early stopping (patience 3, val NLL) | stops at peak LR; CLIP-L 0.801 vs 0.849 fixed 6-ep — replaced by protocol v2 (discoveries #15) | 10-03 |
| Rotation/heavy-denoise variant search for twins (SSCD, 40 db variants) | hybrid F1 0.9313 vs 0.9318, +7 test twins — global descriptors still miss rotated+noised copies | 10-03 |
| SSCD-large (ResNeXt101) instead of disc_mixup | +0.0004 hybrid F1 (< +0.003 pre-registered bar) | 10-03 |
| Holding clean_other at the 38.1% (MD5-copy) Rusak target after twin locking | MD5 subset 38.8% vs all 1534 locked rows 31.0% -> unrepresentative; forced 49.7% Rusak on unlocked rows (discoveries #22) | 10-03 |
| Lainnya share correction (under-prediction) | test has only 7 rows with P(L) 0.1–0.35; OOF precision there implies <1 hidden Lainnya; test share 1.8% looks genuine | 10-03 |
| SIFT/RANSAC verification of no-twin rows | +60 test rows at >=30 inliers where the model is already 0.958 -> ~+1 row | 10-03 |
| Global cleanlab `find_label_issues` / Datalab as a new cleaning lever | 2.9: 447 flags, 333 overlap labels_clean_v2; latent prior 0.235 vs 0.2455 given; A->R suggestions raise Rusak share (wrong LB direction). Only per-source (640) signal kept | 10-04 |
| More fine-tunes of L-size towers (SigLIP2@384, DINOv3-L ft) | session 7: ft cleaned-OOF 0.88-0.90 vs frozen heads 0.934 even with leaky OOF; hill weight 0 for 2/3; frozen DINOv3-L +0.001. Reopen only with a frozen-probe gain >= +0.005 and the ft.py per-fold pseudo filter | 10-04 |
| DINOv3 frozen towers (L, H+, 7B) | L +0.001 (10-03), H+ +0.0003, 7B +0.0018 group-OOF vs cat6+PE-G (bar +0.005); alone 0.80/0.83/0.86 < CLIP-family towers | 10-04 |
| VLM vision towers (InternViT-300M, AIMv2-H, Qwen3-VL ViT merger) re-probed bf16 + avg pool | extraction fixed but +cat6+PE-G gives -0.0002/-0.0019/-0.0030 (bar +0.005) | 10-04 |
| Iterative pseudo-labelling on DA heads (tau 0.9, 3 rounds) | held-out locked-row F1 +0.0024 max (bar +0.005); Rusak share unchanged | 10-04 |
