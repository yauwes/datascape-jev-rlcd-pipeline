# Menjalankan notebook submission terbaik

`Datascape_Pipeline.ipynb` membangun ulang **submission terbaik `d04_da_bt2` (public LB 0.84521)** beserta dua probe 10-05 (`d05_p416_allA`, `d05_p675_R`). Notebook lalu memastikan hasilnya **sama baris per baris** dengan file yang sudah disubmit, di `../submissions/`.

Notebook sudah tersimpan **beserta output-nya**, jadi bisa langsung dibaca di GitHub tanpa dijalankan.

## A. Cara cepat (default, < 1 menit, tanpa GPU)

```bash
git clone https://github.com/yauwes/datascape-2026-pipeline.git
cd datascape-2026-pipeline
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd notebook
jupyter nbconvert --to notebook --execute --inplace Datascape_Pipeline.ipynb
```

Bisa juga membuka notebook di Jupyter Lab atau VS Code lalu **Run All**. Kernel yang dipakai Python 3.11 atau lebih baru.

**Yang terjadi saat dijalankan:**

| Bagian | Isi |
|---|---|
| §1 | audit data: sumber (dari ukuran gambar), komposisi kelas, strata noise |
| §2 | duplikat dan leakage: skor pixel-exact copy, kesepakatan label copy train-train, struktur copy per sumber |
| §3 | cek kebocoran fold (group folds) |
| §4 | group-OOF model `h2` (macro-F1 0.8926 vs label asli) dan confusion matrix |
| §5 | audit noise label dengan cleanlab |
| §6 | twin lock + domain-adapted heads → `d04_da_bt2`, lalu **assert** sama dengan `submissions/d04_da_bt2.csv` |
| §7 | probe `d05_p416_allA` dan `d05_p675_R` (assert sama), estimasi LB per q, 20 baris kandidat koreksi exact-copy |
| §8 | hasil pseudo-labeling iteratif |
| §9 | ranking 48 skenario dan ledger expected vs actual LB |

**Output ada di `notebook/output/`:**

| File | Isi |
|---|---|
| `submission_d04_da_bt2.csv` | **submission terbaik**; file ini yang dipakai jika ingin submit ulang model terbaik |
| `submission_d05_p416_allA.csv` | probe 10-05 untuk sumber 416 |
| `submission_d05_p675_R.csv` | probe 10-05 untuk sumber 675 |

Submit dari akun tim (`ajikajadeh`) dengan:

```bash
kaggle competitions submit datascape-2026-data-mining-competition -f notebook/output/submission_d04_da_bt2.csv -m "d04_da_bt2 rebuild"
```

Lebih baik lewat `python work/subreg.py add ...` lalu `python work/subreg.py submit <nama>`, supaya tercatat di `submissions/ledger.csv` lengkap dengan expected LB.

## B. Mode retrain (melatih ulang domain-adapted heads)

1. Siapkan embedding tower di `work/emb/emb_<tower>.npz` (±1,2 GB, tidak ada di git). Pilih salah satu:
   - unduh dari dataset Kaggle privat `asuudane/datascape-artifacts` (folder `emb/`) dan `asuudane/datascape-colab-out` (giant towers); minta akses ke pemilik;
   - atau hitung ulang. Embedding tower biasa dihitung dengan `kernels/emb*__emb.py` di Kaggle T4, sedangkan giant tower dengan `colab/emb_giant.py <timm_name> <size>` di GPU A100 (bf16).
2. Jalankan:

   ```bash
   cd notebook
   DS_RETRAIN=1 jupyter nbconvert --to notebook --execute --inplace Datascape_Pipeline.ipynb
   ```

   Ini butuh ±10 menit di Apple MPS atau ±2 menit di GPU CUDA. Notebook melaporkan agreement dengan prediksi tersimpan dan dengan file submit; agreement-nya tidak 100% karena seed berbeda per device. `DS_EP=1` bisa dipakai untuk smoke test cepat.

## C. Menghitung ulang skor pixel-copy dari gambar

Set `RECOMPUTE_PIX = True` di sel konfigurasi. Langkah ini butuh gambar kompetisi yang sudah di-resize di `work/small/{train,test}/`, yang dibuat dengan `python work/resize_cache.py` dari data kompetisi. Path-nya mengikuti kolom `path` di `work/emb/meta_sorted.csv`.

## D. Menjalankan di Kaggle atau Colab

- **Colab:** buka notebook dari GitHub. Di sel pertama jalankan `!git clone https://github.com/yauwes/datascape-2026-pipeline.git && cd datascape-2026-pipeline && pip install -r requirements.txt`. Lalu set `os.environ['DS_ROOT'] = '/content/datascape-2026-pipeline'` sebelum sel konfigurasi.
- **Kaggle:** unggah repo sebagai dataset, tambahkan sebagai input, lalu set `DS_ROOT` ke path dataset tersebut (mis. `/kaggle/input/datascape-2026-pipeline`). Folder `notebook/output` harus bisa ditulis; salin repo ke `/kaggle/working` jika perlu.

## Troubleshooting

| Gejala | Solusi |
|---|---|
| cleanlab hang di macOS | sudah memakai `n_jobs=1` di notebook; pastikan ini juga dipakai di script lain |
| `AssertionError: d04_da_bt2 not reproduced` | pastikan `work/preds/da_h2/*.npz`, `work/nntr_sscd.npz`, `work/meta_noise.csv` tidak diubah; jangan jalankan dengan `DS_RETRAIN=1` saat ingin assert |
| `FileNotFoundError: work/emb/emb_*.npz` | hanya muncul di mode retrain; lihat langkah B.1 |
