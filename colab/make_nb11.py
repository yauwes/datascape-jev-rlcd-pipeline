# Builds colab/ds11.ipynb (session 11: augmentation-shortcut study -> gate on the session-10 fine-tunes -> augmentation-equalised DINOv3-L fine-tune -> two-sided gate -> candidates).
# The A100 runner (colab/run11.py) executes it with nbconvert and saves the executed copy to Drive + asuudane/datascape-colab-out11.
# Local dry run (tiny model, 64 train rows): DS_LOCAL=1 DS_WORK=$PWD DS_CODE=work/s11/pack DS_OUT11=work/out11_local python -m nbconvert --execute --to notebook colab/ds11.ipynb
import nbformat as nbf

CELLS = [
('md', """# Datascape 2026 · Session 11 · The augmentation shortcut and an augmentation-equalised fine-tune

**Finding that motivates this session (RESULTS.md 10-05 08:30).**
- The LB shows that every test 416 row and every test 675 row is **Aman**:
  - `d05_p416_allA` scored 0.87739 (+0.032), with q ≈ 0 for its 136 flips.
  - `d05_p675_R` scored 0.80201 (−0.043).
- Our frozen-tower heads call 92% of those rows Rusak.
- Train Aman images in those sources are organiser-augmented (Gaussian or impulse noise, rotation with black or white corners, brightness). Train Rusak and every test image are raw.
- So the model learned "raw ⇒ Rusak". A noise-only fix (clean-only or reweighted training) did not remove it, because the flag misses the rotation and brightness augmentations.

**Hypothesis.** Corrupt every image the same way, every train image and every TTA view, with noise plus ±45° rotation with fill. Raw vs augmented then carries no label signal, and the model has to judge the box itself.

| § | Stage | Output |
|---|---|---|
| 1 | EDA: noise share by source × label × split, 416 montage, LB history | tables, montage |
| 2 | Gate on the session-10 fine-tunes (DINOv3-H+, `d8` aug: 30% noise, 20% rot90-fill) read from Drive | rows of the gate table |
| 3 | Fine-tune `h4H`: h3A's exact recipe (DINOv3 ViT-H+ @384, 12 epochs, train-only, fold 0) but with `augeq` (always corrupt) + 8 stochastic corrupted TTA views. h3A is the control. | `pred_h4H_f0.npz` |
| 5 | Two-sided gate (pre-registered RESULTS.md 10-05 09:30) | `gate11_result.json` |
| 6 | Candidates: `d06_cur_A` + confident model disagreements on 640/other rows without a train copy | `cand_*.csv` |

**Two-sided gate, train-only models only** (no test row may be seen in training):
1. **Specificity.** The Rusak rate on `G1n` must be ≤ 0.50. `G1n` = LB-confirmed Aman rows (test 416/675) with no train copy at NCC ≥ 0.95, so memorising copies cannot help. Frozen h2 scores 0.923, and the DA heads that saw locked test rows score 0.498.
2. **Sensitivity.** The Rusak rate on `CRx` must be ≥ 0.917, which is h2 − 0.05. `CRx` = test rows whose pixel-exact train copies unanimously say Rusak, outside 416/675.
3. **640 Rusak share** must be ≥ 0.65. In the montage the 640 Rusak calls are visibly damaged boxes.

Fold-0 validation F1 against the train labels is reported but does not gate, because train Rusak labels include intact boxes.

**AI declaration.** Code written with Claude (Anthropic) as a coding assistant. No LLM/VLM inference: DINOv3 is a pure vision backbone, fine-tuned without text."""),
('code', """import os, sys, json, time, subprocess, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix
from IPython.display import display
LOCAL = os.environ.get('DS_LOCAL') == '1'                       # tiny model + 64 train rows, CPU dry run
WORK = os.environ.get('DS_WORK', '/content/ds'); os.chdir(WORK)
CODE = os.environ.get('DS_CODE', '/content/code')               # ft.py, rows.csv, lb_history.csv, gate11_baselines.csv
ROOT = 'work/small' if LOCAL else '/content/in/small512/small'
OUT = os.environ.get('DS_OUT11', '/content/out11'); EXTRA = f'{OUT}/extra' if LOCAL else '/content/in/extra'
for d in (OUT, EXTRA): os.makedirs(d, exist_ok=True)
S10 = 'work/out10_local' if LOCAL else '/content/drive/MyDrive/datascape_2026/session10'   # session-10 fine-tune predictions
C = np.array(['Aman', 'Lainnya', 'Rusak']); T0 = time.time()
PLAN = dict(eda=2, gate10=1, ft_h4H=95, eval=3)          # planned minutes per stage -> ETA in the runner
def stage(name):  # progress file read by run11.py (live status + ETA every minute)
    ks = list(PLAN); i = ks.index(name); after = sum(PLAN[k] for k in ks[i + 1:])
    json.dump(dict(stage=name, ts=time.time(), cur_plan=PLAN[name], plan_after=after), open(f'{OUT}/progress.json', 'w'))
    print(f'== {name} | elapsed {(time.time() - T0) / 60:.1f} min | planned remaining {PLAN[name] + after} min', flush=True)
GATE = dict(G1n_max=0.50, CRx_min=0.917, R640_min=0.65)
MF = pd.read_csv('work/emb/meta_sorted.csv'); LABF = MF.label.map({c: i for i, c in enumerate(C)}).fillna(-1).astype(int).values
R = pd.read_csv(f'{CODE}/rows.csv'); R['path'] = 'work/small/test/' + R.stem + '.jpg'
assert set(R.path) == set(MF.path[MF.split == 'test']), 'rows.csv does not match meta_sorted test rows'
R['G1'] = R.src.astype(str).isin(['416', '675']); R['G1n'] = R.G1 & (R.best_ncc < 0.95)
R['CRx'] = (R.copy_label == 2) & ~R.G1; R['CAx'] = (R.copy_label == 0) & ~R.G1; POS = dict(zip(R.path, range(len(R))))
SETS = dict(G1_R=R.G1, G1n_R=R.G1n, CRx_R=R.CRx, CAx_R=R.CAx, R_640=R.src.astype(str) == '640', R_other=R.src.astype(str) == 'other', R_all=pd.Series(True, index=R.index))
def align(paths, probs):  # -> (n_test, 3) array in R order, NaN where the model gave no prediction (LOCAL smoke predicts 32 rows)
    P = np.full((len(R), 3), np.nan); P[[POS[p] for p in paths]] = probs; return P
def stats(P, name):
    ok = ~np.isnan(P[:, 0]); lab = np.where(ok, np.nan_to_num(P).argmax(1), -1)
    return dict(model=name, **{k: round(float((lab[m.values & ok] == 2).mean()), 3) if (m.values & ok).any() else np.nan for k, m in SETS.items()})
print('LOCAL dry run' if LOCAL else 'Colab', '| test rows', len(R), '| G1', R.G1.sum(), 'G1n', R.G1n.sum(), 'CRx', R.CRx.sum(), 'CAx', R.CAx.sum())"""),
('md', """## 1. EDA: the augmentation shortcut
Image size identifies the source:
- 224 = all-Aman;
- 416 = synthetic conveyor boxes;
- 512;
- 675×900 = parcel photos;
- 640 = web damaged boxes;
- other = mixed.

The tables show the share of noisy images (impulse > 0.02 or median-residual ≥ 6) by source × label in train, and by source in test."""),
('code', """stage('eda')
nz = pd.read_csv('work/meta_noise.csv').set_index('path').loc[MF.path]
MF['noisy'] = (nz.impulse.values > 0.02) | (nz.medabs.values >= 6)
sz = MF.w.astype(str) + 'x' + MF.h.astype(str)
MF['src'] = sz.map({'224x224': '224', '416x416': '416', '512x512': '512', '640x640': '640', '675x900': '675', '900x675': '675'}).fillna('other')
trm = MF.split == 'train'
tab = MF[trm].pivot_table(index='src', columns='label', values='noisy', aggfunc='mean').round(3).add_prefix('train noisy | ')
tab['test noisy'] = MF[~trm].groupby('src').noisy.mean().round(3); tab['train n'] = MF[trm].src.value_counts(); tab['test n'] = MF[~trm].src.value_counts()
print(tab.to_string())
print('LB history (team account, this pipeline):'); print(pd.read_csv(f'{CODE}/lb_history.csv').to_string())
cur = C[R.cur.values]; rng = np.random.default_rng(0)
groups = [('test 416: LB says all Aman', MF.path[(~trm) & (MF.src == '416')].values),
          ('train 416 Rusak', MF.path[trm & (MF.src == '416') & (MF.label == 'Rusak')].values),
          ('train 416 Aman', MF.path[trm & (MF.src == '416') & (MF.label == 'Aman')].values)]
fig, ax = plt.subplots(3, 8, figsize=(16, 6.6))
for r, (t, ps) in enumerate(groups):
    for j, p in enumerate(rng.choice(ps, min(8, len(ps)), replace=False)):
        ax[r, j].imshow(Image.open(f"{ROOT}/{p.split('work/small/')[1]}").convert('RGB').resize((160, 160))); ax[r, j].axis('off')
    ax[r, 0].set_title(t, fontsize=9, loc='left')
plt.suptitle('416 source: raw intact test boxes vs raw dented train Rusak vs augmented train Aman'); plt.tight_layout(); plt.show()"""),
('md', "## 2. Gate on the session-10 fine-tunes\nThese read `pred_h3A_f0.npz` and `pred_h3B_f0.npz` from Drive `session10`. h3A is train-only, so it is eligible for the gate. h3B trained on twin-locked test rows, so its G1 numbers may be contaminated and are shown for reference only."),
('code', """stage('gate10')
BASE_TAB = pd.read_csv(f'{CODE}/gate11_baselines.csv')
MODELS, VAL = {}, {}
def add(name, f):
    if not os.path.exists(f): print('missing', f); return
    z = np.load(f, allow_pickle=True); MODELS[name] = align(z['test_path'], z['test']); VAL[name] = (z['val_path'], z['val'])
add('h3A (H+, d8, train-only)', f'{S10}/pred_h3A_f0.npz'); add('h3B (H+, d8, + locked test)', f'{S10}/pred_h3B_f0.npz')
print(pd.concat([BASE_TAB, pd.DataFrame([stats(P, n) for n, P in MODELS.items()])], ignore_index=True).set_index('model').to_string())"""),
('md', """## 3. Augmentation-equalised fine-tune (h4H) against its control h3A
Recipe identical to h3A (session 10), from the pig-posture 1st-place write-up:
- `vit_huge_plus_patch16_dinov3.lvd1689m` @384, batch 24, gradient checkpointing;
- AdamW lr 3e-5 with the head at ×10, uniform layer decay;
- cosine schedule with 5% warmup, 12 epochs;
- EMA 0.9985, label smoothing 0.1, drop-path 0.1;
- median-3 denoise; folds `group90`, fold 0; train-only.

Only the augmentation differs:
- **h4H `augeq`:** every train image gets Gaussian noise (σ 8–45), and half also get impulse noise. Every image is rotated by an angle drawn from ±45° with black or white fill, then gets d8, ColorJitter and grayscale. TTA uses 8 stochastic views from the same corruption family.
- **h3A `d8` (control, already trained):** 30% noise, rot90, dihedral TTA."""),
('code', """import shutil; pd.read_csv('work/folds_group90.csv').to_csv(f'{EXTRA}/folds_group90.csv', index=False); shutil.copy('work/emb/meta_sorted.csv', f'{EXTRA}/meta_sorted.csv')  # ft.py looks for both under DS_IN
BASE = dict(model='vit_huge_plus_patch16_dinov3.lvd1689m', img=384, epochs=12, sched='cosine', warm=0.05, lr=3e-5, head_mult=10, layer_decay=1.0, ema=0.9985, ls=0.1,
            bs=24, dpr=0.1, gc=True, folds=[0], fold_file='folds_group90.csv', save_w=False, median=True)
if LOCAL: BASE.update(model='vit_tiny_patch16_224', pretrained=False, img=64, epochs=1, bs=8, tta_k=2)
def ft(tag, **kw):
    stage(f'ft_{tag}'); cfg = dict(BASE, tag=tag, **kw); env = dict(os.environ, CFG=json.dumps(cfg), DS_OUT=OUT, NW='10')
    if LOCAL: env.pop('DS_IN', None)
    else: env['DS_IN'] = '/content/in'
    p = subprocess.Popen([sys.executable, '-u', f'{CODE}/ft.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); t = time.time()
    with open(f'{OUT}/log_{tag}.txt', 'w') as f:
        for line in p.stdout:
            f.write(line); f.flush()
            if ' ep ' in line or 'saved' in line or 'rror' in line or 'Traceback' in line or 'exit' in line: print(line, end='', flush=True)
    print(f'{tag}: exit {p.wait()} in {(time.time() - t) / 60:.1f} min', flush=True)
ft('h4H', aug='augeq', augeq=True, tta='augeq')"""),
('md', "## 5. Two-sided gate"),
('code', """stage('eval')
add('h4H (H+, augeq, train-only)', f'{OUT}/pred_h4H_f0.npz')
rows = []
for n, P in MODELS.items():
    r = stats(P, n); vp, vv = VAL[n]
    if len(vv): y = LABF[MF.reset_index().set_index('path').loc[vp, 'index'].values]; r['fold0_val_F1'] = round(f1_score(y, vv.argmax(1), average='macro'), 4)
    eligible = 'locked test' not in n
    r['PASS'] = bool(eligible and r['G1n_R'] <= GATE['G1n_max'] and r['CRx_R'] >= GATE['CRx_min'] and r['R_640'] >= GATE['R640_min']) if not LOCAL else None
    rows.append(r)
TAB = pd.concat([BASE_TAB, pd.DataFrame(rows)], ignore_index=True).set_index('model'); print(TAB.to_string())   # plain text: HTML tables do not render in the Drive viewer
print('gate:', GATE)
json.dump(dict(gate=GATE, table=TAB.reset_index().astype(object).where(TAB.reset_index().notna(), None).to_dict('records')), open(f'{OUT}/gate11_result.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
TAB[['G1n_R']].plot.barh(ax=ax[0], legend=False); ax[0].axvline(GATE['G1n_max'], c='r', ls='--'); ax[0].set_title('Rusak rate on LB-confirmed Aman rows without a train copy (lower = better)')
TAB[['CRx_R', 'R_640']].plot.barh(ax=ax[1]); ax[1].axvline(GATE['CRx_min'], c='r', ls='--'); ax[1].set_title('Rusak kept on copy-confirmed Rusak and 640 (higher = better)')
plt.tight_layout(); plt.show()
for tag in ['h4H']:
    hf = f'{OUT}/hist_{tag}_f0.json'
    if os.path.exists(hf):
        h = pd.DataFrame(json.load(open(hf))['hist'])
        if len(h): plt.plot(h.ep, h.val_f1, marker='o', label=f'{tag} fold-0 val F1 (raw val, no TTA)')
plt.xlabel('epoch'); plt.legend(); plt.title('learning curves'); plt.show()"""),
('md', """## 6. Candidates
A candidate is built only from a model that passes the gate. It starts from `d06_cur_A` (= `d05_p416_allA` + the curated-source fix) and changes only rows that meet all of these:
- in 640 or other;
- no pixel copy in train (NCC < 0.95);
- the model's argmax differs from the base and its top probability is ≥ 0.70.

Rows with copies and the four curated sources keep their LB-anchored labels."""),
('code', """ss = pd.read_csv('datascape/sample_submission.csv'); stem2fn = {f.rsplit('.', 1)[0]: f for f in ss.filename}
base_lab = C[R.cur.values]; elig = R.src.astype(str).isin(['640', 'other']).values & (R.best_ncc.values < 0.95)
os.makedirs(f'{OUT}/cands', exist_ok=True); summary = []
for n, P in MODELS.items():
    passed = bool(TAB.loc[n, 'PASS']) if not LOCAL else True
    if not passed or 'locked test' in n: continue
    ok = ~np.isnan(P[:, 0]); lab = C[np.nan_to_num(P).argmax(1)]; conf = np.nan_to_num(P).max(1)
    tag = n.split()[0]
    for var, extra in [('RA', (base_lab == 'Rusak') & (lab == 'Aman')), ('both', np.ones(len(R), bool))]:   # RA: only Rusak->Aman (the shortcut biases models towards Rusak)
        flip = ok & elig & (lab != base_lab) & (conf >= 0.70) & extra; new = np.where(flip, lab, base_lab)
        sub = pd.DataFrame({'filename': [stem2fn[s] for s in R.stem], 'label': new}); sub = ss[['filename']].merge(sub, on='filename', how='left')
        assert sub.label.notna().all(); sub.to_csv(f'{OUT}/cands/cand_{tag}_{var}.csv', index=False)
        summary.append(dict(cand=f'cand_{tag}_{var}', flips=int(flip.sum()), by_src_dir=pd.Series([f'{a}:{b}>{c}' for a, b, c in zip(R.src[flip].astype(str), base_lab[flip], lab[flip])]).value_counts().to_dict(),
                            rusak_share=round(float((new == 'Rusak').mean()), 3)))
    np.save(f'{OUT}/cands/probs_{tag}.npy', np.nan_to_num(P).astype(np.float32))
print(pd.DataFrame(summary).to_string()) if summary else print('no model passed the gate: no candidate written')"""),
('md', "## 7. Outputs\nThe runner copies `/content/out11` to Drive `MyDrive/datascape_2026/session11` and to the private Kaggle dataset `asuudane/datascape-colab-out11`, then releases the A100."),
('code', """print(sorted(os.listdir(OUT))); print(f'notebook done in {(time.time() - T0) / 60:.1f} min')"""),
]

nb = nbf.v4.new_notebook(); nb.metadata.update(kernelspec=dict(name='python3', display_name='Python 3', language='python'), accelerator='GPU')
nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s) for t, s in CELLS]
nbf.write(nb, 'colab/ds11.ipynb'); print('wrote colab/ds11.ipynb', len(CELLS), 'cells')
