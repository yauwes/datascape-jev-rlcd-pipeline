# Builds colab/ds14.ipynb (session 14: DINOv3-7B full-fit + pseudo-labels = F2; runs ONLY if the session-13 gate passed).
# The A100 runner (colab/run14.py) executes it with nbconvert and saves the executed copy to Drive + asuudane/datascape-colab-out14.
# Local dry run: DS_LOCAL=1 DS_WORK=$PWD DS_CODE=work/s14/pack DS_OUT14=work/out14_local python -m nbconvert --execute --to notebook colab/ds14.ipynb
import nbformat as nbf

CELLS = [
('md', """# Datascape 2026 · Session 14 · DINOv3-7B final model F2: full-fit + pseudo-labels

**Context.** Session 13 tested whether DINOv3-7B beats DINOv3-H+ on the LB-confirmed rows: Rusak on G1n ≤ 0.437, CRx ≥ 0.97, 640 ≥ 0.65. This notebook prints that verdict from `gate13_result.json` (Kaggle `asuudane/datascape-colab-out13`) and trains F2 regardless, at the user's request. Both F1 and F2 end up as candidates, and the paired LB probe decides.

**This model F2** (user request 10-05: "train until convergence, patience 1, High-RAM runtime").
- **Training rows:** train folds 1–4 plus pseudo-labelled test rows:
  - every twin-locked test row;
  - every test 416/675 row as **Aman** (LB-confirmed).
- **Validation:** group-fold 0, every epoch.
- **Schedule:** one warm-up epoch, then a constant LR. Training stops at the **first epoch whose validation loss does not improve** (patience 1), max 15 epochs, and the best epoch's weights are restored.
- **Model:** `vit_7b_patch16_dinov3.lvd1689m` @384; top 6 of 40 blocks + norm/head trained in fp32; bottom 34 frozen in bf16; no EMA; d8 augmentation; flip TTA. Validation batches are 24, and only the ~1B trainable tensors are checkpointed in CPU RAM.

**Candidates** start from `d06_cur_A`. They only touch 640/other rows with no train copy (NCC < 0.95), where the model disagrees with confidence ≥ 0.70:
- `RA`: Rusak → Aman flips only;
- `both`: both directions.

**AI declaration.** Code written with Claude (Anthropic) as a coding assistant. No LLM/VLM inference: DINOv3 is a pure vision backbone."""),
('code', """import os, sys, json, time, shutil, subprocess, numpy as np, pandas as pd
LOCAL = os.environ.get('DS_LOCAL') == '1'
WORK = os.environ.get('DS_WORK', '/content/ds'); os.chdir(WORK); sys.path.insert(0, 'work')
CODE = os.environ.get('DS_CODE', '/content/code')
OUT = os.environ.get('DS_OUT14', '/content/out14'); EXTRA = f'{OUT}/extra' if LOCAL else '/content/in/extra'
for d in (OUT, EXTRA): os.makedirs(d, exist_ok=True)
S13 = 'work/out13_local' if LOCAL else '/content/in/out13'
C = np.array(['Aman', 'Lainnya', 'Rusak']); T0 = time.time()
PLAN = dict(prep=3, ft_F2=150, eval=3)
def stage(name):
    ks = list(PLAN); i = ks.index(name); after = sum(PLAN[k] for k in ks[i + 1:])
    json.dump(dict(stage=name, ts=time.time(), cur_plan=PLAN[name], plan_after=after), open(f'{OUT}/progress.json', 'w'))
    print(f'== {name} | elapsed {(time.time() - T0) / 60:.1f} min | planned remaining {PLAN[name] + after} min', flush=True)
MF = pd.read_csv('work/emb/meta_sorted.csv')
R = pd.read_csv(f'{CODE}/rows13.csv'); R['path'] = 'work/small/test/' + R.stem + '.jpg'; R['src'] = R.src.astype(str)
R['G1'] = R.src.isin(['416', '675']); R['CRx'] = (R.copy_label == 2) & ~R.G1; R['CAx'] = (R.copy_label == 0) & ~R.G1; POS = dict(zip(R.path, range(len(R))))
SETS = dict(CRx_R=R.CRx, CAx_R=R.CAx, R_640=R.src == '640', R_other=R.src == 'other', R_all=pd.Series(True, index=R.index))
def stats(P, name):
    ok = ~np.isnan(P[:, 0]); lab = np.where(ok, np.nan_to_num(P).argmax(1), -1)
    return dict(model=name, **{k: round(float((lab[m.values & ok] == 2).mean()), 3) if (m.values & ok).any() else np.nan for k, m in SETS.items()})
print('LOCAL dry run' if LOCAL else 'Colab')"""),
('md', "## 1. Augmentation choice (pre-registered rule) and the pseudo-label file"),
('code', """stage('prep')
g13 = json.load(open(f'{S13}/gate13_result.json')) if os.path.exists(f'{S13}/gate13_result.json') else {}; GO = True
print('session-13 gate:', g13.get('gate'), '| 7B better than H+:', g13.get('verdict'), '| training F2:', GO)
AUG = dict(aug='d8', tta='h')
sys.argv = ['x']; import twin                      # the twin lock of d04_da_bt2 (meta_sorted order)
te = np.where(~twin.TR)[0]; Dn = np.load('work/nntr_sscd.npz'); lab, pur = twin.vote(Dn['i'][te], Dn['s'][te], 0.55)
lk = (lab >= 0) & (pur >= 0.9) & (twin.ST[te] == 'clean_other')
ps = pd.DataFrame({'path': MF.path.values[te[lk]], 'y': lab[lk]})
g1 = R.path[R.G1].values; ps = pd.concat([ps[~ps.path.isin(g1)], pd.DataFrame({'path': g1, 'y': 0})], ignore_index=True)   # LB-confirmed Aman overrides twin labels
P1 = np.eye(3)[ps.y.values]; pf = pd.DataFrame(P1, columns=list(C)); pf.insert(0, 'path', ps.path.values); pf.to_csv(f'{EXTRA}/pseudo_final.csv', index=False)
print(f'pseudo rows {len(pf)}: twin-locked {lk.sum()} (of which in 416/675: {np.isin(MF.path.values[te[lk]], g1).sum()}) + LB-confirmed Aman {len(g1)} | label mix', pf[list(C)].sum().astype(int).to_dict())
pd.read_csv('work/folds_group90.csv').to_csv(f'{EXTRA}/folds_group90.csv', index=False); shutil.copy('work/emb/meta_sorted.csv', f'{EXTRA}/meta_sorted.csv')"""),
('md', "## 2. DINOv3-7B (F2) trained to convergence: early stopping with patience 1 on fold-0 validation; top 6 blocks + head in fp32, bottom 34 frozen in bf16"),
('code', """BASE = dict(model='vit_7b_patch16_dinov3.lvd1689m', img=384, epochs=15, sched='plateau', warm=0.0, patience=1, plateau=1, lr=5e-5, head_mult=10, layer_decay=1.0, ema=0, ls=0.1,
            bs=16, dpr=0.0, gc=True, folds=[0], fold_file='folds_group90.csv', save_w=False, median=True, full_fit=False, pseudo_file='pseudo_final.csv', unfreeze_last=6, frozen_bf16=True, pred_bs=24, pred_on_best=True)
# user 10-05: train until convergence -> validate on fold 0 every epoch, early-stop with patience 1, restore the best epoch (max 15 epochs)
if LOCAL: BASE.update(model='vit_tiny_patch16_224', pretrained=False, img=64, epochs=1, bs=8, unfreeze_last=2, frozen_bf16=False)
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
ft('F2', **AUG)   # trained regardless of the session-13 verdict (user request); the verdict is printed above"""),
('md', "## 3. Sanity table and candidates"),
('code', """stage('eval')
assert os.path.exists(f'{OUT}/pred_F2_f0.npz') or not GO, 'F2 predictions missing'
if not GO: raise SystemExit('nothing to evaluate')
z = np.load(f'{OUT}/pred_F2_f0.npz', allow_pickle=True); P = np.full((len(R), 3), np.nan); P[[POS[p] for p in z['test_path']]] = z['test']
Pcur = np.eye(3)[R.cur.values]   # d06_cur_A as one-hot, same rows (correct meta_sorted order)
print(pd.DataFrame([stats(Pcur, 'd06_cur_A (pending LB)'), stats(P, 'F2 (7B, full-fit + pseudo)')]).set_index('model').to_string())
ss = pd.read_csv('datascape/sample_submission.csv'); stem2fn = {f.rsplit('.', 1)[0]: f for f in ss.filename}
base_lab = C[R.cur.values]; elig = R.src.isin(['640', 'other']).values & (R.best_ncc.values < 0.95)
ok = ~np.isnan(P[:, 0]); lab = C[np.nan_to_num(P).argmax(1)]; conf = np.nan_to_num(P).max(1); os.makedirs(f'{OUT}/cands', exist_ok=True); summary = []
for var, extra in [('RA', (base_lab == 'Rusak') & (lab == 'Aman')), ('both', np.ones(len(R), bool))]:
    flip = ok & elig & (lab != base_lab) & (conf >= 0.70) & extra; new = np.where(flip, lab, base_lab)
    sub = ss[['filename']].merge(pd.DataFrame({'filename': [stem2fn[s] for s in R.stem], 'label': new}), on='filename', how='left'); assert sub.label.notna().all()
    sub.to_csv(f'{OUT}/cands/cand_F2_{var}.csv', index=False)
    summary.append(dict(cand=f'cand_F2_{var}', flips=int(flip.sum()), by_src_dir=pd.Series([f'{a}:{b}>{c}' for a, b, c in zip(R.src[flip], base_lab[flip], lab[flip])]).value_counts().to_dict(),
                        rusak_share=round(float((new == 'Rusak').mean()), 3)))
np.save(f'{OUT}/cands/probs_F2.npy', np.nan_to_num(P).astype(np.float32)); print(pd.DataFrame(summary).to_string())
print(sorted(os.listdir(OUT))); print(f'notebook done in {(time.time() - T0) / 60:.1f} min')"""),
]

nb = nbf.v4.new_notebook(); nb.metadata.update(kernelspec=dict(name='python3', display_name='Python 3', language='python'), accelerator='GPU')
nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s) for t, s in CELLS]
nbf.write(nb, 'colab/ds14.ipynb'); print('wrote colab/ds14.ipynb', len(CELLS), 'cells')
