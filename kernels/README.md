# Kaggle kernels (flattened)

Each file is `<kernel-name>__<file>`. To push one as a Kaggle kernel, copy it into its own folder under the original names (`ft.py` / `emb.py` / `copydet.py` + `kernel-metadata.json`), then run `kaggle kernels push -p <folder>`.

- `ft__ft.py`: fine-tune template (protocol v2: 1-epoch warmup, ReduceLROnPlateau, fp32 EMA, early stop on EMA val NLL, cv2 median filter, TTA, group folds).
  - The `abl-*`, `ft-*` and `canary-*` variants are generated from it by `work/mk_ft.py`; only the `CFG` line at the top differs.
- `emb*__emb.py`: frozen-tower embedding extraction (5 geometric views, median 3×3, native resolution). Each variant covers a different tower list.
- `copydet__copydet.py`: SSCD copy-detector embeddings.
