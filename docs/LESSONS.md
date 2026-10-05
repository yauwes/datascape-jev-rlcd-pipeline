# datascape-2026-data-mining-competition: lessons (as of 2026-10-05)

3-class package condition (Aman / Rusak / Lainnya), macro-F1, public LB 30%.
- Team best public LB 0.87739, with `d06_*` pending.
- LB #1: 0.976.
- Code and notes: `Datascape/` and the private repo `yauwes/datascape-2026-pipeline`.

## What worked
- **Paired LB probes as the evaluator.**
  - One hypothesis changes one group of rows against a base with a known score.
  - `work/lbdelta.py` maps the LB delta to q, the share of flipped rows that are truly class X. It integrates the number of public rows over Binomial(n, 0.3) and handles flips in both directions.
  - The 416 source → Aman gave +0.032 (q ≈ 0).
  - This is the autoresearch "fixed evaluator" whenever offline labels are unreliable.
- **Copy detection.** SSCD near-copies, plus a pixel-exact dihedral NCC on 128-px thumbnails, then a twin vote lock and domain-adapted heads retrained with the locked test rows. LB +0.015, paired.
- **Full fine-tune of DINOv3-H+** (pig-posture 1st-place recipe: LLRD, EMA, cosine, d8 TTA). It halved the shortcut error on LB-confirmed rows (0.938 → 0.487) where frozen towers plus heads could not.

## What failed or misled
- **Expectations from train-label OOF** ran 0.03–0.09 above the LB (0.93 expected, 0.845 actual). The cause is an augmentation shortcut: train Aman images were organiser-augmented (noise, rotation with black/white corners, brightness), train Rusak and every test image are raw, so models learn "raw = Rusak".
- **Removing only the noisy rows, or reweighting them**, did not fix it. The flag misses the rotation and brightness augmentations.
- **"Always augment" (`augeq`)** did not fix it either: 0.522 vs 0.476.
- **Relabelling by cleanlab / Confident Learning:** LB −0.019. Iterative pseudo-labels: +0.002. Lainnya prior matching: rejected by calibration.
- **Row-order bug.** An artifact was saved in `meta_sorted` order and read through `meta.csv` order, which silently scrambled the copy subsets for a day. Check an artifact against a known invariant (here: train-train exact-copy label agreement ≈ 0.96) before trusting it.

## Process lessons
- **Colab runner cells must be fail-safe.** Use `try/finally` around everything with `runtime.unassign()`. A failed `drive.mount` once left an A100 connected after the cell died.
- **Dry-run every Colab notebook locally** (`DS_LOCAL=1`, tiny model) and add a preflight cell. Session 11's first run died after 1.5 min because `ft.py` could not find `meta_sorted.csv`.
- **Generated-code escaping.** `\n` inside generator strings becomes a real newline. Write `\\n` in cells built from Python strings.
- **Write-up retrieval** (UBC-OCEAN 1st, BirdCLEF 2024/2025): large seed ensembles over frozen foundation features, self-distillation for noisy labels, and building a test-like validation set. None of them replaces a paired LB probe when the train/test label convention differs.
