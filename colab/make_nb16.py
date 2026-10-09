# Builds colab/ds16.ipynb (session 16, 10-09: does DINOv3-7B improve when trained to convergence? fold 0, LR decay on plateau, patience 2).
# The A100 runner (colab/run16.py) executes it with nbconvert and saves the executed copy to Drive + asuudane/datascape-colab-out16.
# Local dry run: DS_LOCAL=1 DS_WORK=$PWD DS_CODE=work/s16/pack DS_OUT15=work/out16_local python -m nbconvert --execute --to notebook colab/ds15.ipynb
import nbformat as nbf

CELLS = [
('md', """# Datascape 2026 · Session 16 · Does DINOv3-7B improve when trained to convergence?

**Question (user, 10-09).** "Many epochs until the model converges."

**What is already known.** F2 (fold 0) peaked after its single warm-up epoch:
- valLoss 0.3266 → 0.3284 at epoch 2;
- valF1 0.8867 → 0.8805;
- train acc 0.84 → 0.91.

Its LR was never decayed, because patience = plateau = 1 stopped the run before the plateau cut.

**This run** gives convergence a fair test on fold 0:
- **exactly F2's recipe and data** (no session-15 cleaning), so the only change is the schedule;
- up to 6 epochs;
- LR × 0.3 after every non-improving epoch;
- stop after **2** non-improving epochs (patience 2);
- restore the best epoch.

**Pre-registered gate** (fold 0, the same val rows as F2). Use the converged recipe for folds 1–4 only if:
- its best valLoss ≤ F2's − 0.005, **or**
- its valF1 ≥ F2's + 0.005.

Otherwise the one-epoch recipe stands and the extra epochs only memorise label noise.

**AI declaration.** Code written with Claude (Anthropic) as a coding assistant. No LLM/VLM inference: DINOv3 is a pure vision backbone."""),
('code', """import os, sys, json, time, shutil, subprocess, numpy as np, pandas as pd
LOCAL = os.environ.get('DS_LOCAL') == '1'
WORK = os.environ.get('DS_WORK', '/content/ds'); os.chdir(WORK); sys.path.insert(0, 'work')
CODE = os.environ.get('DS_CODE', '/content/code')
OUT = os.environ.get('DS_OUT15', '/content/out15'); EXTRA = f'{OUT}/extra' if LOCAL else '/content/in/extra'
for d in (OUT, EXTRA): os.makedirs(d, exist_ok=True)
C = np.array(['Aman', 'Lainnya', 'Rusak']); T0 = time.time()
PLAN = dict(prep=3, ft_F2v=100, eval=3)
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
GO = True
AUG = dict(aug='d8', tta='h')
sys.argv = ['x']; import twin                      # the twin lock of d04_da_bt2 (meta_sorted order)
te = np.where(~twin.TR)[0]; Dn = np.load('work/nntr_sscd.npz'); lab, pur = twin.vote(Dn['i'][te], Dn['s'][te], 0.55)
lk = (lab >= 0) & (pur >= 0.9) & (twin.ST[te] == 'clean_other')
ps = pd.DataFrame({'path': MF.path.values[te[lk]], 'y': lab[lk]})
g1 = R.path[R.G1].values; ps = pd.concat([ps[~ps.path.isin(g1)], pd.DataFrame({'path': g1, 'y': 0})], ignore_index=True)   # LB-confirmed Aman overrides twin labels
P1 = np.eye(3)[ps.y.values]; pf = pd.DataFrame(P1, columns=list(C)); pf.insert(0, 'path', ps.path.values); pf.to_csv(f'{EXTRA}/pseudo_final.csv', index=False)
print(f'pseudo rows {len(pf)}: twin-locked {lk.sum()} (of which in 416/675: {np.isin(MF.path.values[te[lk]], g1).sum()}) + LB-confirmed Aman {len(g1)} | label mix', pf[list(C)].sum().astype(int).to_dict())
pd.read_csv('work/folds_group90.csv').to_csv(f'{EXTRA}/folds_group90.csv', index=False); shutil.copy('work/emb/meta_sorted.csv', f'{EXTRA}/meta_sorted.csv'); shutil.copy(f'{CODE}/drop15.csv', f'{EXTRA}/drop15.csv')
print('session 16: no train rows dropped (F2 data), only the schedule changes')"""),
('md', "## 2. DINOv3-7B fold 0 trained to convergence: plateau LR decay ×0.3, patience 2, max 6 epochs, best epoch restored"),
('code', """BASE = dict(model='vit_7b_patch16_dinov3.lvd1689m', img=384, epochs=15, sched='plateau', warm=0.0, patience=1, plateau=1, lr=5e-5, head_mult=10, layer_decay=1.0, ema=0, ls=0.1,
            bs=16, dpr=0.0, gc=True, folds=[0], fold_file='folds_group90.csv', save_w=False, median=True, full_fit=False, pseudo_file='pseudo_final.csv', unfreeze_last=6, frozen_bf16=True, pred_bs=24, pred_on_best=True, drop_file='')
BASE.update(epochs=6, patience=2, plateau=1, lr_factor=0.3)   # converge: decay the LR after each non-improving epoch, stop after 2
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
ft('F2v', **AUG)"""),
('md', "## 3. Gate: converged fold 0 vs F2 (one warm-up epoch) on the same validation rows"),
('code', """stage('eval')
from sklearn.metrics import f1_score
MFi = MF.set_index('path'); f = f'{OUT}/pred_F2v_f0.npz'; ref = f'{CODE}/pred_F2_f0_best.npz'
if os.path.exists(f) and os.path.exists(ref):
    z, r = np.load(f, allow_pickle=True), np.load(ref, allow_pickle=True)
    rv = pd.DataFrame(r['val'], index=r['val_path']).loc[z['val_path']].values; y = MFi.loc[z['val_path'], 'label'].map({c: i for i, c in enumerate(C)}).values
    nll = lambda P: float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean())
    res = dict(rows=len(y), F2_valLoss=round(nll(rv), 4), F2_valF1=round(f1_score(y, rv.argmax(1), average='macro'), 4), F2v_valLoss=round(nll(z['val']), 4), F2v_valF1=round(f1_score(y, z['val'].argmax(1), average='macro'), 4))
    res['converged_better'] = bool(res['F2v_valLoss'] <= res['F2_valLoss'] - 0.005 or res['F2v_valF1'] >= res['F2_valF1'] + 0.005)
    h = json.load(open(f'{OUT}/hist_F2v_f0.json')); print(pd.DataFrame(h['hist']).round(4).to_string(index=False)); print('best epoch', h['best_epoch'])
    print(res); json.dump(res, open(f'{OUT}/gate16.json', 'w'))
else: print('missing', f, ref)
print(sorted(os.listdir(OUT))); print(f'notebook done in {(time.time() - T0) / 60:.1f} min')"""),
]

nb = nbf.v4.new_notebook(); nb.metadata.update(kernelspec=dict(name='python3', display_name='Python 3', language='python'), accelerator='GPU')
nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s) for t, s in CELLS]
nbf.write(nb, 'colab/ds16.ipynb'); print('wrote colab/ds16.ipynb', len(CELLS), 'cells')
