# Builds colab/ds17.ipynb (session 17, 10-09: DINOv3-7B folds 1-4 on F2's data with the recipe that won session 16; RECIPE env = 'onecycle' | 'cosine4').
# The A100 runner (colab/run17.py) executes it with nbconvert and saves the executed copy to Drive + asuudane/datascape-colab-out17.
# Local dry run: DS_LOCAL=1 DS_WORK=$PWD DS_CODE=work/s17/pack DS_OUT15=work/out17_local python -m nbconvert --execute --to notebook colab/ds15.ipynb
import nbformat as nbf

CELLS = [
('md', """# Datascape 2026 · Session 17 · DINOv3-7B folds 1–4, F2 data, recipe chosen by session 16

**Why.** The best LB file blends F2, a single 7B fold-0 model. Training folds 1–4 on **F2's own data** turns it into a 5-fold 7B. Session 15 showed that dropping the 416×416 Rusak block hurts: the model then under-calls Rusak.

**Recipe** (set in the runner from the session-16 gate): `onecycle` = F2's single warm-up epoch, or `cosine4` = 4 epochs with a cosine-annealed LR.

**Gate before any submission:**
- every fold has valF1 ≥ 0.86;
- the changed test rows are free rows only (no copy, not 416/675/VN).

**AI declaration.** Code written with Claude (Anthropic) as a coding assistant. No LLM/VLM inference: DINOv3 is a pure vision backbone."""),
('code', """import os, sys, json, time, shutil, subprocess, numpy as np, pandas as pd
LOCAL = os.environ.get('DS_LOCAL') == '1'
WORK = os.environ.get('DS_WORK', '/content/ds'); os.chdir(WORK); sys.path.insert(0, 'work')
CODE = os.environ.get('DS_CODE', '/content/code')
OUT = os.environ.get('DS_OUT15', '/content/out15'); EXTRA = f'{OUT}/extra' if LOCAL else '/content/in/extra'
for d in (OUT, EXTRA): os.makedirs(d, exist_ok=True)
C = np.array(['Aman', 'Lainnya', 'Rusak']); T0 = time.time()
PLAN = dict(prep=3, ft_F2d=240, eval=3)
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
print('session 17: F2 data (no drop); recipe', os.environ.get('RECIPE', 'onecycle'))"""),
('md', "## 2. DINOv3-7B folds 1–4 on F2's data, recipe from session 16"),
('code', """BASE = dict(model='vit_7b_patch16_dinov3.lvd1689m', img=384, epochs=15, sched='plateau', warm=0.0, patience=1, plateau=1, lr=5e-5, head_mult=10, layer_decay=1.0, ema=0, ls=0.1,
            bs=16, dpr=0.0, gc=True, folds=[1, 2, 3, 4], fold_file='folds_group90.csv', save_w=False, median=True, full_fit=False, pseudo_file='pseudo_final.csv', unfreeze_last=6, frozen_bf16=True, pred_bs=24, pred_on_best=False, drop_file='')
BASE.update(dict(onecycle=dict(epochs=1), cosine4=dict(epochs=4, sched='cosine', warm=0.1))[os.environ.get('RECIPE', 'onecycle')])
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
ft('F2d', **AUG)"""),
('md', "## 3. Per-fold validation and the averaged test prediction (fold 0 = F2 is added locally)"),
('code', """stage('eval')
from sklearn.metrics import f1_score
MFi = MF.set_index('path'); rows = []; Pt = []
for k in [1, 2, 3, 4]:
    f = f'{OUT}/pred_F2d_f{k}.npz'
    if not os.path.exists(f): print('missing', f); continue
    z = np.load(f, allow_pickle=True); y = MFi.loc[z['val_path'], 'label'].map({c: i for i, c in enumerate(C)}).values
    rows.append(dict(fold=k, n_val=len(y), val_macro_f1=round(f1_score(y, z['val'].argmax(1), average='macro'), 4), val_rusak_pred=round(float((z['val'].argmax(1) == 2).mean()), 3), val_rusak_true=round(float((y == 2).mean()), 3)))
    P = np.full((len(R), 3), np.nan); P[[POS[p] for p in z['test_path']]] = z['test']; Pt.append(P)
print(pd.DataFrame(rows).to_string(index=False)); print('reference F2 fold 0: valF1 0.8867 (uncleaned val)')
if Pt:
    Pm = np.nanmean(Pt, 0); np.save(f'{OUT}/probs_F2d_f1to4.npy', Pm.astype(np.float32))
    print(pd.DataFrame([stats(Pm, f'F2d mean of {len(Pt)} folds')]).set_index('model').to_string())
print(sorted(os.listdir(OUT))); print(f'notebook done in {(time.time() - T0) / 60:.1f} min')"""),
]

nb = nbf.v4.new_notebook(); nb.metadata.update(kernelspec=dict(name='python3', display_name='Python 3', language='python'), accelerator='GPU')
nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s) for t, s in CELLS]
nbf.write(nb, 'colab/ds17.ipynb'); print('wrote colab/ds17.ipynb', len(CELLS), 'cells')
