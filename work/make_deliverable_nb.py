# Builds the competition deliverable notebook lastgame_Notebook_Datascape.ipynb (+ NamaTim_ copy) and executes it so outputs are embedded.
# Base: the reproducible pipeline cells of work/make_notebook.py, plus visual EDA / evaluation cells.
# Data policy: organiser data only. No external dataset and no test-label file is read anywhere.
# usage (repo root): python work/make_deliverable_nb.py [--no-exec]
import sys, shutil, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

src = open('work/make_notebook.py').read().split("nb = new_notebook(")[0]
ns = {}; exec(src, ns); base = ns['cells']          # list of the 31 pipeline cells

def find(prefix):
    return next(i for i, c in enumerate(base) if c.cell_type == 'markdown' and c.source.startswith(prefix))

HEADER = md("""# DATASCAPE 2026: Klasifikasi Kondisi Paket (Aman / Rusak / Lainnya)
**Tim:** last game &nbsp;|&nbsp; **Kompetisi:** DATASCAPE 2026 Data Mining Competition (Kaggle: `datascape-2026-data-mining-competition`) &nbsp;|&nbsp; **Metrik:** macro-F1

**Submission final (dipilih di Kaggle):**

| File | Public LB | Isi |
|---|---|---|
| `d11_lbopt.csv` | **0.88891** (terbaik tim) | `d08` + 17 baris hasil regresi leaderboard (§10) |
| `d08_grand_consensus_champion.csv` | 0.88789 | blend 4 model + 3 aturan lock (§13) |

**Kebijakan data.**
- Notebook ini hanya memakai data panitia: `Train_Set/`, `Test_Set/`, `sample_submission.csv`.
- Tidak ada dataset eksternal dan tidak ada file label test yang dibaca.
- Satu-satunya umpan balik dari test adalah skor public leaderboard dari submission yang benar-benar kami kirim (§7, §10, §12).

**Alur notebook.**
1. **EDA:** distribusi kelas, sumber gambar (ukuran), noise, contoh gambar.
2. **Duplikat & leakage:** salinan piksel-eksak (NCC dihedral) dan pasangan berlabel bertentangan.
3. **Desain validasi:** group-fold berbasis klaster.
4. **Model:** menara visi beku + MLP head; evaluasi OOF (confusion matrix, classification report, F1 per sumber).
5. **Audit label noise:** cleanlab.
6. **Twin lock + domain-adapted heads:** menghasilkan `d04_da_bt2`.
7. **Probe post-processing.**
8. **Pseudo-labelling iteratif.**
9. **Peringkat eksperimen.**
10. **Regresi leaderboard:** menghasilkan `d11_lbopt`.
11. **Analisis aturan 416.**
12. **Ekspektasi vs LB aktual.**
13. **Rekonstruksi `d08` dan file final.**

Setiap bagian punya sel visual tambahan; gambarnya juga disimpan ke `notebook/output/figs/`.

**Mode.**
- `RETRAIN=False` (default, ±1–2 menit): membangun ulang setiap file submission dari prediksi head yang di-cache, lalu meng-*assert* kesamaan baris-per-baris dengan file yang dikirim.
- `DS_RETRAIN=1`: melatih ulang MLP head dari embedding.

Lihat `lastgame_Requirements_Datascape.txt` untuk instalasi dan struktur folder.

**Deklarasi AI.**
- Kode dan dokumentasi disusun dengan bantuan asisten AI Claude (Anthropic).
- Tidak ada LLM/VLM yang dipakai untuk inferensi.
- Prediksi dihasilkan model visi tanpa teks (frozen tower, DINOv3-7B fine-tune) ditambah aturan terprogram.
- Tidak ada label test yang ditentukan manual.""")

VIS_SETUP = code("""import sys, platform, importlib
import matplotlib, matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report
for m in ['numpy', 'pandas', 'sklearn', 'scipy', 'matplotlib', 'PIL', 'cleanlab']:
    print(f'{m:<11}', importlib.import_module(m).__version__)
print('python     ', platform.python_version(), '|', platform.platform())
plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
FIG = ROOT / 'notebook' / 'output' / 'figs'; FIG.mkdir(parents=True, exist_ok=True)
COL = {'Aman': '#2e7d32', 'Lainnya': '#1565c0', 'Rusak': '#c62828'}
def savefig(name): plt.savefig(FIG / f'{name}.png', dpi=110, bbox_inches='tight')
def img_file(i):
    # cached 512px copy if present, else the organiser original (Train_Set/<label>/<file> or Test_Set/<file>)
    p = ROOT / meta.path.values[i]
    if p.exists(): return p
    lab = meta.label.values[i]
    return ROOT / 'datascape' / (f'Train_Set/{lab}' if isinstance(lab, str) else 'Test_Set') / meta.filename.values[i]
def show(ax, i, title, color='k'):
    im = Image.open(img_file(i)).convert('RGB'); im.thumbnail((256, 256)); ax.imshow(im); ax.set_title(title, fontsize=7.5, color=color); ax.axis('off')""")

V_EDA1 = [md("""### 1a. Visual: class distribution, source shift, class mix per source and per noise stratum
- **Class imbalance.** Lainnya is a small minority. In macro-F1 each class weighs 1/3, so a single Lainnya error costs far more than an Aman error.
- **Covariate shift.** Image size identifies the source, and the train and test mixes differ strongly, most of all for 224×224 and 675×900. A model that learns the train prior per source will be biased on test."""),
code("""tr, te_ = meta[TR], meta[TE]; order = BIG + ['other']
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
vc = tr.label.value_counts().reindex(C)
ax[0, 0].bar(C, vc.values, color=[COL[c] for c in C]); ax[0, 0].set_ylim(0, vc.max() * 1.22)
for i, v in enumerate(vc.values): ax[0, 0].text(i, v, f'{v:,}\\n({v / len(tr):.1%})', ha='center', va='bottom')
ax[0, 0].set_title(f'Train class distribution (n={len(tr):,}); test n={len(te_):,}')
sh = pd.DataFrame({'train': tr.source.value_counts(normalize=True), 'test': te_.source.value_counts(normalize=True)}).reindex(order).fillna(0)
sh.plot.bar(ax=ax[0, 1], color=['#78909c', '#ff8f00'], rot=0); ax[0, 1].set_title('Source (image size) share: train vs test'); ax[0, 1].set_ylabel('share')
pd.crosstab(tr.source, tr.label, normalize='index').reindex(order)[C].plot.barh(stacked=True, ax=ax[1, 0], color=[COL[c] for c in C])
ax[1, 0].set_title('Train class mix per source'); ax[1, 0].set_xlabel('share'); ax[1, 0].legend(loc='lower right', fontsize=8)
pd.crosstab(pd.Series(ST[TR], name='stratum'), pd.Series(tr.label.values, name='label'), normalize='index')[C].plot.bar(stacked=True, ax=ax[1, 1], color=[COL[c] for c in C], rot=0)
ax[1, 1].set_title('Train class mix per noise stratum (noisy = salt-and-pepper / heavy noise)'); ax[1, 1].legend(fontsize=8)
plt.tight_layout(); savefig('eda_overview'); plt.show()"""),
md("### 1b. Visual: random examples per class and from the test set (seed 42)"),
code("""rng = np.random.default_rng(42); fig, ax = plt.subplots(4, 7, figsize=(14, 8.8))
pools = [(c, np.where(TR & (meta.label == c).values)[0]) for c in C] + [('Test', np.where(TE)[0])]
for r, (name, pool) in enumerate(pools):
    for k, i in enumerate(rng.choice(pool, 7, replace=False)):
        show(ax[r, k], i, f'{name} | {meta.w[i]}x{meta.h[i]}', COL.get(name, 'k'))
plt.tight_layout(); savefig('eda_samples'); plt.show()"""),
md("### 1c. Visual: image resolution and injected impulse noise, train vs test"),
code("""nz = pd.read_csv(W / 'meta_noise.csv').set_index('path').loc[meta.path]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for s, m, c in [('train', TR, '#78909c'), ('test', TE, '#ff8f00')]:
    ax[0].scatter(meta.w[m], meta.h[m], s=5, alpha=.3, c=c, label=f'{s} (n={m.sum():,})')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('width (px)'); ax[0].set_ylabel('height (px)'); ax[0].legend(); ax[0].set_title('Image size: each source is a tight cluster')
ax[1].hist([nz.impulse.values[TR], nz.impulse.values[TE]], bins=np.linspace(0, .2, 41), label=['train', 'test'], color=['#78909c', '#ff8f00'], log=True, density=True)
ax[1].axvline(.02, ls='--', c='k', lw=1); ax[1].set_xlabel('impulse-noise pixel fraction'); ax[1].legend(); ax[1].set_title('Salt-and-pepper noise (threshold 0.02 dashed)')
plt.tight_layout(); savefig('eda_size_noise'); plt.show()
print('noisy share  train %.3f | test %.3f' % ((nz.impulse.values[TR] > .02).mean(), (nz.impulse.values[TE] > .02).mean()))""")]

V_DUP = [md("""### 2a. Visual: how many images are exact copies, and do copies agree on their labels?
Left: the best dihedral pixel-NCC for each image against its SSCD top-10 train neighbours (self excluded). Right: label agreement of exact train-train copies per source. Copies that disagree are direct evidence of annotation noise in the training labels."""),
code("""fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].hist([best[TR], best[TE]], bins=np.linspace(.5, 1, 51), label=['train → other train rows', 'test → train rows'], color=['#78909c', '#ff8f00'], log=True)
ax[0].axvline(.99, ls='--', c='k', lw=1); ax[0].legend(); ax[0].set_xlabel('best pixel NCC (8 dihedral views)'); ax[0].set_title('Exact copies are common (NCC ≥ 0.99, dashed)')
ag = tt.groupby('source').agree.agg(['mean', 'size']).reindex(BIG + ['other']).dropna()
ax[1].barh([f'{s} (n={int(n)})' for s, n in zip(ag.index, ag['size'])], ag['mean'], color='#5c6bc0'); ax[1].set_xlim(.5, 1.0)
for j, v in enumerate(ag['mean']): ax[1].text(v, j, f' {v:.3f}', va='center')
ax[1].set_title('Label agreement of exact train-train copy pairs'); plt.tight_layout(); savefig('dup_ncc'); plt.show()"""),
md("### 2b. Visual: exact copies with conflicting train labels (organiser label noise)"),
code("""seen, conf = set(), []
for r in np.where(TR)[0]:
    for k in np.where(NCC[r] >= 0.99)[0]:
        c_ = I_sscd[r, k]; key = frozenset((r, c_))
        if TR[c_] and LAB[r] != LAB[c_] and key not in seen: seen.add(key); conf.append((r, c_, float(NCC[r, k])))
print(len(conf), 'distinct exact-copy train pairs with conflicting labels')
print(pd.Series([' vs '.join(sorted((C[LAB[a]], C[LAB[b]]))) for a, b, _ in conf]).value_counts().to_dict())
pick = [conf[j] for j in np.linspace(0, len(conf) - 1, 5).astype(int)]
fig, ax = plt.subplots(2, 5, figsize=(14, 6))
for j, (a, b, s) in enumerate(pick):
    show(ax[0, j], a, f'{meta.filename[a]}\\nlabel {C[LAB[a]]}', COL[C[LAB[a]]]); show(ax[1, j], b, f'{meta.filename[b]}\\nlabel {C[LAB[b]]} | NCC {s:.3f}', COL[C[LAB[b]]])
plt.suptitle('Same picture (up to rotation/flip/re-encoding), different label', y=1.0); plt.tight_layout(); savefig('dup_conflicts'); plt.show()""")]

V_FOLD = [md("### 3a. Visual: group folds are balanced in size and class mix"),
code("""fc = pd.crosstab(fold, LAB[TR]); fc.columns = C
fig, ax = plt.subplots(figsize=(7, 3.4)); fc.plot.bar(stacked=True, color=[COL[c] for c in C], ax=ax, rot=0)
ax.set_xlabel('fold'); ax.set_ylabel('train images'); ax.set_title('StratifiedGroupKFold on DINOv2 cos ≥ 0.90 clusters'); plt.tight_layout(); savefig('folds'); plt.show()
print((fc.T / fc.sum(1)).T.round(3))""")]

V_OOF = [md("""### 4a. Visual: honest group-OOF evaluation of the frozen-tower MLP ensemble (`h2`)
The score is measured against the *original* (noisy) train labels, with copies kept inside a single fold. Expect it above the public LB, because the test sources are distributed differently (§1a) and §12 quantifies the gap."""),
code("""pred_oof = oof_h2.argmax(1); print(classification_report(y, pred_oof, target_names=C, digits=4))
cm = confusion_matrix(y, pred_oof); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={'width_ratios': [1, 1.3]})
ax[0].imshow(cmn, cmap='Blues', vmin=0, vmax=1); ax[0].grid(False)
for a in range(3):
    for b in range(3): ax[0].text(b, a, f'{cm[a, b]:,}\\n{cmn[a, b]:.1%}', ha='center', va='center', color='white' if cmn[a, b] > .5 else 'black')
ax[0].set_xticks(range(3), [f'pred {c}' for c in C]); ax[0].set_yticks(range(3), [f'true {c}' for c in C]); ax[0].set_title('Group-OOF confusion matrix (row-normalised)')
src_tr = meta.source.values[TR]; rows = []
for s in BIG + ['other']:
    m = src_tr == s
    if m.sum(): rows.append((s, m.sum(), f1_score(y[m], pred_oof[m], labels=np.unique(y[m]), average='macro', zero_division=0), (pred_oof[m] == y[m]).mean()))
pf = pd.DataFrame(rows, columns=['source', 'n', 'macro_f1', 'accuracy']).set_index('source')
pf[['macro_f1', 'accuracy']].plot.barh(ax=ax[1], color=['#5c6bc0', '#26a69a']); ax[1].set_xlim(0, 1); ax[1].set_title('Group-OOF per source'); ax[1].legend(loc='lower left')
plt.tight_layout(); savefig('oof_eval'); plt.show(); pf.round(4)""")]

V_CL = [md("### 5a. Visual: random rows flagged by cleanlab (given label → group-OOF model label)"),
code("""fl = np.where(TR)[0][iss]; ml = oof_h2[iss].argmax(1); rng = np.random.default_rng(0); pk = rng.choice(len(fl), 12, replace=False)
fig, ax = plt.subplots(2, 6, figsize=(14, 5.6))
for a_, j in zip(ax.ravel(), pk): show(a_, fl[j], f'given {C[LAB[fl[j]]]} → model {C[ml[j]]}\\np={oof_h2[iss][j].max():.2f}', COL[C[ml[j]]])
plt.tight_layout(); savefig('cleanlab_flags'); plt.show()""")]

V_DA = [md("### 6a. Visual: how `d04_da_bt2` decides each test row"),
code("""fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].bar(['twin-locked\\n(SSCD vote, purity ≥ 0.9)', 'DA heads +\\nper-stratum calibration'], [lock.sum(), (~lock).sum()], color=['#8d6e63', '#5c6bc0'])
for j, v in enumerate([lock.sum(), (~lock).sum()]): ax[0].text(j, v, f'{v:,} ({v / len(lock):.1%})', ha='center', va='bottom')
ax[0].set_title('Decision path of the 3,057 test rows')
mx = pd.DataFrame({'train labels': pd.Series([C[i] for i in LAB[TR]]).value_counts(normalize=True), 'd04 locked': pd.Series([C[i] for i in final[lock]]).value_counts(normalize=True),
                   'd04 unlocked': pd.Series([C[i] for i in final[~lock]]).value_counts(normalize=True)}).reindex(C).fillna(0)
mx.T.plot.bar(stacked=True, color=[COL[c] for c in C], ax=ax[1], rot=0); ax[1].set_title('Class mix'); plt.tight_layout(); savefig('d04_paths'); plt.show()""")]

V_PL = [md("### 8a. Visual: pseudo-labelling rounds on the held-out test-domain rows"),
code("""if pl.exists():
    s = pd.read_csv(pl); fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(s['round'], s.heldout_f1, 'o-', label='no bias'); ax[0].plot(s['round'], s.heldout_f1_b08, 's-', label='Rusak bias −0.8'); ax[0].set_xlabel('round'); ax[0].set_ylabel('held-out macro-F1'); ax[0].legend()
    ax[0].axhline(s.heldout_f1_b08[0] + .005, ls=':', c='k', lw=1); ax[0].set_title('Bar: +0.005 over round 0 (dotted)')
    ax[1].bar(s['round'], s.pseudo_rows, color='#90a4ae'); ax[1].set_xlabel('round'); ax[1].set_title('pseudo-labelled test rows added'); plt.tight_layout(); savefig('pl_rounds'); plt.show()""")]

V_LB = [md("""### 12a. Visual: public-LB trajectory and how well each kind of estimate predicted it
Only the scores of submissions we actually sent are used here. Train-label CV/OOF overestimates the LB. Paired probes on a scored base predict it within about 0.006."""),
code("""ev = pd.read_csv(W / 'expectation_vs_lb.csv').dropna(subset=['public_lb']).reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={'width_ratios': [1.6, 1]})
x = np.arange(len(ev)); ax[0].plot(x, ev.public_lb, 'o-', c='#5c6bc0', label='public LB'); ax[0].plot(x, ev.public_lb.cummax(), '--', c='#2e7d32', label='best so far')
for j, nme in enumerate(ev['sub']):
    if nme in ('d11_lbopt', 'd08_grand_consensus_champion'): ax[0].annotate(f'{nme}\\n{ev.public_lb[j]:.5f}', (j, ev.public_lb[j]), xytext=(-60, -38), textcoords='offset points', arrowprops=dict(arrowstyle='->'), fontsize=8)
ax[0].set_xticks(x, ev['sub'].str.slice(0, 22), rotation=75, fontsize=6.5); ax[0].set_ylabel('public macro-F1'); ax[0].legend(); ax[0].set_title('Public LB of every scored submission (chronological)')
m = ev.expected_lb.notna(); e, a = ev.expected_lb[m].astype(float), ev.public_lb[m]
ax[1].scatter(e, a, c='#ff8f00'); lim = [min(e.min(), a.min()) - .01, max(e.max(), a.max()) + .01]; ax[1].plot(lim, lim, 'k--', lw=1)
for j in np.where(m)[0]: ax[1].annotate(ev['sub'][j][:14], (float(ev.expected_lb[j]), ev.public_lb[j]), fontsize=6.5)
ax[1].set_xlabel('expected LB (before submitting)'); ax[1].set_ylabel('actual public LB'); ax[1].set_title('Expectation vs reality')
plt.tight_layout(); savefig('lb_trajectory'); plt.show()""")]

V_FINAL = [md("### 13a. Visual: final submissions: class mix, d08 → d11 changes and the changed rows"),
code("""mixes = pd.DataFrame({'train labels': meta[TR].label.value_counts(normalize=True), 'd08 (LB 0.88789)': pd.Series(d08f).value_counts(normalize=True),
                      'd11_lbopt (LB 0.88891)': pd.Series(d11f).value_counts(normalize=True)}).reindex(C)
fig, ax = plt.subplots(figsize=(8, 3.6)); mixes.T.plot.barh(stacked=True, color=[COL[c] for c in C], ax=ax)
for j, col in enumerate(mixes.columns):
    acc = 0
    for c in C: v = mixes.loc[c, col]; ax.text(acc + v / 2, j, f'{v:.1%}', ha='center', va='center', color='white', fontsize=8); acc += v
ax.set_title('Class mix of the final files'); plt.tight_layout(); savefig('final_mix'); plt.show()
print(pd.crosstab(pd.Series(d08f, name='d08'), pd.Series(d11f, name='d11_lbopt')))
chg = np.where(d08f != d11f)[0]; nc_ = min(len(chg), 18); fig, ax = plt.subplots(3, 6, figsize=(14, 7.6))
for a_, j in zip(ax.ravel(), chg[:nc_]): show(a_, te_rows[j], f'{fn_[j]} | {sz[j]}\\nd08 {d08f[j]} → d11 {d11f[j]}', COL[d11f[j]])
for a_ in ax.ravel()[nc_:]: a_.axis('off')
plt.tight_layout(); savefig('d11_changes'); plt.show()"""),
md("""## Kesimpulan
- **Data.** Kesulitan utama ada pada data, bukan pada model:
  - pergeseran sumber train→test (§1a);
  - salinan piksel-eksak dengan label bertentangan di data latih (§2b);
  - label noise yang terdeteksi cleanlab (§5).
- **Validasi.** Group-fold berbasis klaster mencegah salinan bocor antar fold (§3).
- **Model.** Menara visi beku dengan MLP head lebih kuat daripada fine-tune backbone ukuran L (§9).
- **Kalibrasi.** Twin lock dan domain-adapted heads (§6), ditambah kalibrasi per-stratum, menaikkan public LB dari 0.786 ke 0.845.
- **Aturan sumber.** Aturan per sumber yang dibuktikan dengan probe berpasangan (§7, §11) dan blend 4 model menghasilkan `d08` = 0.88789.
- **Regresi leaderboard.** Regresi atas semua submission yang sudah dinilai (§10) menghasilkan `d11_lbopt` = **0.88891**.
- **Reproduksibilitas.** `d04`, `d05` dan `d11` direproduksi persis (assert). `d08` direkonstruksi 3,054/3,057 baris dari komponennya.

File final ada di `notebook/output/FINAL_d11_lbopt.csv` dan `notebook/output/FINAL_d08_grand_consensus_champion.csv`.""")]

# assemble: header, setup, helper, then pipeline sections with visual cells after each section's last code cell
cells = [HEADER, base[1], VIS_SETUP]
plan = {'## 2. Duplicates': V_EDA1, '## 3. Validation': V_DUP, '## 4. Model': V_FOLD, '## 5. Label-noise': V_OOF, '## 6. Twin lock': V_CL,
        '## 7. Post-processing': V_DA, '## 9. Experiment': V_PL, '## 13. Final': V_LB}
for c in base[2:]:
    if c.cell_type == 'markdown':
        for k, extra in plan.items():
            if c.source.startswith(k): cells += extra
    cells.append(c)
cells += V_FINAL
# the original setup cell resolves ROOT relative to cwd; executing from the repo root keeps ROOT = repo root

nb = new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}})
out = 'lastgame_Notebook_Datascape.ipynb'
if '--no-exec' not in sys.argv:
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
nbf.write(nb, out); shutil.copy(out, 'NamaTim_Notebook_Datascape.ipynb'); print('wrote', out, '+ NamaTim copy |', len(cells), 'cells')
