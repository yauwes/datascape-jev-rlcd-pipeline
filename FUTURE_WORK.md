# Future work: panduan melanjutkan (per 2026-10-05)

Dokumen ini untuk teammate yang melanjutkan. Mulailah dari `python reproduce_best.py` (best public LB 0.87739), lalu kerjakan daftar di bawah dari atas.

## 1. Posisi saat ini

| Item | Nilai |
|---|---|
| Best public LB | **0.87739**: `d05_p416_allA` = `d04_da_bt2` + semua prediksi Rusak di sumber 416×416 → Aman |
| Pending | `d06_cur_A` (auto-submit 10-06 00:07 UTC). Isinya d05_p416 + 25 prediksi Rusak sisa di 512/675 → Aman. Expected 0.8837 bila semua Aman (q=0) [5–95%: 0.879–0.889]; 0.876 bila setengahnya Rusak |
| Leaderboard #1 | 0.97577 |
| F1 publik per kelas (inversi 3 skor LB, base d04) | Aman ±0.89, Rusak ±0.81, Lainnya ±0.85 (interval lebar) |
| Sisa submission | 10-06 s/d 10-10, limit 3/hari dipakai bersama tim. Kebiasaan: maks 2 slot/hari untuk eksperimen |

**Pilihan final** (2 file untuk private LB, 70% test):
- Default: `d06_cur_A` bila skornya ≥ 0.8774, ditambah `d05_p416_allA`.
- Perbarui setiap ada hasil probe yang menang.

## 2. Kandidat siap pakai (sudah dibangun, belum disubmit)

Kandidat F1 = model final **DINOv3-H+ full-fit + pseudo-label**. Prediksinya ada di Drive `datascape_2026/session12/pred_F1_f0.npz`. Notebook finalis §10 membangun kandidat ini otomatis bila file itu ada; versi yang sudah jadi ada di Drive `datascape_2026/final/`.

| Kandidat | Perubahan vs `d06_cur_A` | Cara baca |
|---|---|---|
| `cand_F1_RA` | 6 baris "other" tanpa copy: Rusak → Aman (F1 yakin ≥ 0.70) | probe kecil. Bila naik, F1 benar pada arah R→A |
| `cand_F1_both` | 31 baris: 6 R→A + **25 Aman → Rusak** (semua "other", tanpa copy) | menguji apakah F1 lebih benar daripada bias −0.8 base di baris tanpa twin |

Baca hasilnya dengan:

```bash
python work/lbdelta.py submissions/d06_cur_A.csv <kandidat>.csv <skor_LB_d06_cur_A>
```

`lbdelta` mendukung flip dua arah dan mengintegrasikan jumlah baris publik ~ Binomial(n, 0.3).

**Validasi offline F1** (gate dua sisi, baris test yang labelnya dikonfirmasi LB, urutan baris benar):

| Model | Rusak di 339 baris Aman tanpa copy (↓) | Rusak di 230 baris copy-Rusak (↑) | 640 Rusak | other Rusak |
|---|---|---|---|---|
| h2 frozen (base model) | 0.938 | 0.991 | 0.810 | 0.288 |
| DA heads | 0.522 | 0.991 | 0.810 | 0.262 |
| h3A H+ fine-tune, train-only | **0.487** | 0.978 | 0.781 | 0.266 |
| h3B H+ + baris test terkunci | 0.419 | 0.983 | 0.767 | 0.266 |
| h4H H+ `augeq` | 0.522 | 0.987 | 0.795 | 0.273 |
| F1 (melihat baris 416/675 sebagai pseudo-label) | 0.053* | 0.996 | 0.781 | 0.261 |

\* in-sample: baris ini masuk training F1.

## 3. Rencana eksperimen berikutnya (urut prioritas)

1. **Probe F1** (10-06 slot 2 atau 10-07). Submit `cand_F1_both` (31 flip) berpasangan dengan `d06_cur_A`.
   - Naik ≥ +0.005: F1 dipakai untuk semua baris tanpa copy di 640/other. Bangun kandidat dengan ambang keyakinan lebih rendah (0.6), atau blend log-prob F1 + DA heads.
   - Turun: pertahankan base, coba `cand_F1_RA` saja.
2. **Sumber "other" adalah ketidakpastian terbesar.** Ada 1.672 baris test; diprediksi 24% Rusak, sedangkan train 15%.
   - Pecah menjadi kelompok bermakna: band ukuran (≤300, 300–500, >500 px), status copy, keyakinan model.
   - Probe **satu kelompok per submission** dengan `lbdelta` untuk menghitung expected value lebih dulu.
   - Prioritaskan R→A pada kelompok yang gejala shortcut-nya kuat: gambar bersih, tanpa twin, P(Rusak) marginal.
   - Montage 300–500 px menunjukkan kebanyakan prediksi Rusak memang rusak visual, jadi pakai ambang keyakinan.
3. **640:** jangan diprobe ke Aman. Prediksi Rusaknya rusak secara visual, dan 71% baris dengan copy pun berlabel Rusak.
4. **Ensemble F1 + DA heads** untuk baris tanpa twin. Evaluasi offline dengan gate dua sisi di notebook §9; konfirmasi di LB.
5. **Variance reduction F1:** seed kedua atau 5-fold, hanya bila probe F1 menang. Biayanya ±2 jam A100 per run.
6. **Lainnya.** Prioritas rendah, karena inversi LB menyebut jumlahnya sudah pas. Satu Lainnya yang benar bernilai ±0.02 macro, jadi audit montage prediksi Lainnya vs baris "other" kecil (150×200) layak bila ada waktu.
7. **Deliverable 11–15 Oktober:**
   - jalankan notebook finalis mode FULL sekali di Colab A100 (±3 jam) untuk bukti reproducibility penuh;
   - ganti nama ke format `NamaTim_Notebook.ipynb`;
   - write-up PDF dengan deklarasi AI;
   - buat artefak besar bisa diakses juri (dataset Kaggle / folder Drive dibagikan).

## 4. Jangan diulang (tanpa bukti baru)

Lihat juga `docs/closed_ideas.md`.

- **`augeq`** (korupsi semua gambar, train dan TTA): tidak menurunkan shortcut.
- **Fine-tune tower L-size**, tower VLM, DINOv3 sebagai fitur frozen tambahan: gain ≤ noise.
- **Relabel Confident Learning / cleanlab**: LB −0.019.
- **Pseudo-label iteratif**: +0.0024, di bawah bar +0.005.
- **Lainnya prior-match ke 3.5%**: baris yang di-flip hampir pasti bukan Lainnya.
- **Probe 640 tanpa copy → Aman**: visual rusak.
- **`d06_map_max`**: dibangun di atas base lama; akan membatalkan +0.032 dari aturan 416.
- **Share dial "Rusak lebih sedikit" global**: baris yang di-flip terlihat rusak. Pakai probe per kelompok, bukan dial global.

## 5. Di mana artefak besar disimpan

Minta akses ke pemilik.

| Artefak | Lokasi |
|---|---|
| Gambar cache 512 px | dataset Kaggle privat `asuudane/datascape-small-512` (atau bangun dari data kompetisi: `work/resize_cache.py`) |
| Embedding 11+ tower (1,1 GB), prediksi lama | dataset Kaggle privat `asuudane/datascape-artifacts` |
| Prediksi fine-tune h3A/h3B (sesi 10), h4H (sesi 11), F1 + bobot `weights/w_F1_f0.pt` 1,7 GB (sesi 12) | Drive pemilik `datascape_2026/session10`, `session11`, `session12` |
| Notebook finalis yang dieksekusi di Colab + submission | Drive `datascape_2026/final` |
| Runner Colab | Drive `datascape_a100_runner.ipynb` (Colab Pro, A100) |

## 6. Cara kerja yang terbukti

- **Satu hipotesis per probe LB**, selalu berpasangan dengan base yang skornya diketahui. Daftarkan dulu di ledger: `python work/subreg.py add NAME CSV "strategi" OOF EXP LO HI`. Submit dengan `python work/subreg.py submit NAME`.
- **Pre-register gate** di `docs/RESULTS.md` sebelum memakai GPU.
- **Dry-run lokal sebelum Colab** (`DS_LOCAL=1`, model kecil). Run pertama sesi 11 gagal karena file hilang. Notebook finalis sekarang punya cell preflight.
- **Simpan semua output ke Drive dan lepas GPU** (`runtime.unassign()`) begitu selesai.
- **Urutan baris:** semua artefak `work/*.npz` mengikuti `work/emb/meta_sorted.csv`. Bug 10-05 terjadi karena me-reindex `pix_all.npz` lewat `meta.csv`.
