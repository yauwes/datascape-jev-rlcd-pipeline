# Builds colab/ds12.ipynb (session 12: final DINOv3-H+ model, full-fit on all train rows + pseudo-labelled test rows, then candidates).
# The A100 runner (colab/run12.py) executes it with nbconvert and saves the executed copy to Drive + asuudane/datascape-colab-out12.
# Local dry run: DS_LOCAL=1 DS_WORK=$PWD DS_CODE=work/s12/pack DS_OUT12=work/out12_local python -m nbconvert --execute --to notebook colab/ds12.ipynb
import nbformat as nbf

CELLS = [
('md', """# Datascape 2026 · Session 12 · Final DINOv3-H+ model: full-fit + pseudo-labels

**Why.** Session 11 showed that a full fine-tune of DINOv3-H+ (h3A, train-only) roughly halves the augmentation shortcut. On LB-confirmed Aman test rows without a train copy, the Rusak rate falls from 0.923 (frozen h2) to 0.476. The calls on copy-confirmed Rusak rows (0.922) and the 640 Rusak share (0.781) are kept. Adding twin-locked test rows as pseudo-labels (h3B) improves this further, to 0.399 / 0.943.

**This model (pre-registered RESULTS.md 10-05).**
- **Training rows:** all 12,224 train rows plus pseudo-labelled test rows:
  - every twin-locked test row (SSCD ≥ 0.55, purity ≥ 0.9, clean stratum; the h3B and `d04_da_bt2` lock);
  - every test 416/675 row as **Aman**, because the LB confirmed those sources (q ≈ 0 in both probes). This overrides any twin label.
- **Recipe:** h3A's, with `full_fit` (cosine, 12 epochs, final EMA weights, no validation).
- **Augmentation:** `augeq` if session 11's h4H beat h3A by ≥ 0.05 on G1n with CRx ≥ 0.917; otherwise `d8`. It is read from `gate11_result.json` on Drive.
- **Gate rows:** the G1 rows are now in training, so only CRx, CAx and the per-source Rusak shares are reported. They are sanity checks, not a gate.

**Candidates** start from `d06_cur_A`. They only touch 640/other rows with no train copy (NCC < 0.95), where the model disagrees with confidence ≥ 0.70:
- `RA`: Rusak → Aman flips only;
- `both`: both directions.

**AI declaration.** Code written with Claude (Anthropic) as a coding assistant. No LLM/VLM inference: DINOv3 is a pure vision backbone."""),
('code', """import os, sys, json, time, shutil, subprocess, numpy as np, pandas as pd
LOCAL = os.environ.get('DS_LOCAL') == '1'
WORK = os.environ.get('DS_WORK', '/content/ds'); os.chdir(WORK); sys.path.insert(0, 'work')
CODE = os.environ.get('DS_CODE', '/content/code')
OUT = os.environ.get('DS_OUT12', '/content/out12'); EXTRA = f'{OUT}/extra' if LOCAL else '/content/in/extra'
for d in (OUT, EXTRA): os.makedirs(d, exist_ok=True)
S11 = 'work/out11_local' if LOCAL else '/content/drive/MyDrive/datascape_2026/session11'
C = np.array(['Aman', 'Lainnya', 'Rusak']); T0 = time.time()
PLAN = dict(prep=2, ft_F1=125, eval=3)
def stage(name):
    ks = list(PLAN); i = ks.index(name); after = sum(PLAN[k] for k in ks[i + 1:])
    json.dump(dict(stage=name, ts=time.time(), cur_plan=PLAN[name], plan_after=after), open(f'{OUT}/progress.json', 'w'))
    print(f'== {name} | elapsed {(time.time() - T0) / 60:.1f} min | planned remaining {PLAN[name] + after} min', flush=True)
MF = pd.read_csv('work/emb/meta_sorted.csv')
R = pd.read_csv(f'{CODE}/rows.csv'); R['path'] = 'work/small/test/' + R.stem + '.jpg'; R['src'] = R.src.astype(str)
R['G1'] = R.src.isin(['416', '675']); R['CRx'] = (R.copy_label == 2) & ~R.G1; R['CAx'] = (R.copy_label == 0) & ~R.G1; POS = dict(zip(R.path, range(len(R))))
SETS = dict(CRx_R=R.CRx, CAx_R=R.CAx, R_640=R.src == '640', R_other=R.src == 'other', R_all=pd.Series(True, index=R.index))
def stats(P, name):
    ok = ~np.isnan(P[:, 0]); lab = np.where(ok, np.nan_to_num(P).argmax(1), -1)
    return dict(model=name, **{k: round(float((lab[m.values & ok] == 2).mean()), 3) if (m.values & ok).any() else np.nan for k, m in SETS.items()})
print('LOCAL dry run' if LOCAL else 'Colab')"""),
('md', "## 1. Augmentation choice (pre-registered rule) and the pseudo-label file"),
('code', """stage('prep')
g = json.load(open(f'{S11}/gate11_result.json')); T = {r['model']: r for r in g['table']}
a = next((r for k, r in T.items() if k.startswith('h3A')), None); b = next((r for k, r in T.items() if k.startswith('h4H')), None)
v = lambda r, k: r.get(k) if r and isinstance(r.get(k), (int, float)) else None
use_augeq = None not in (v(a, 'G1n_R'), v(b, 'G1n_R'), v(b, 'CRx_R')) and v(b, 'G1n_R') <= v(a, 'G1n_R') - 0.05 and v(b, 'CRx_R') >= 0.917
AUG = dict(aug='augeq', augeq=True, tta='augeq') if use_augeq else dict(aug='d8', tta='d8')
print('h3A', {k: a.get(k) for k in ('G1n_R', 'CRx_R', 'R_640')} if a else None); print('h4H', {k: b.get(k) for k in ('G1n_R', 'CRx_R', 'R_640')} if b else None)
print('augmentation for the final model:', AUG)
sys.argv = ['x']; import twin                      # the twin lock of d04_da_bt2 (meta_sorted order)
te = np.where(~twin.TR)[0]; Dn = np.load('work/nntr_sscd.npz'); lab, pur = twin.vote(Dn['i'][te], Dn['s'][te], 0.55)
lk = (lab >= 0) & (pur >= 0.9) & (twin.ST[te] == 'clean_other')
ps = pd.DataFrame({'path': MF.path.values[te[lk]], 'y': lab[lk]})
g1 = R.path[R.G1].values; ps = pd.concat([ps[~ps.path.isin(g1)], pd.DataFrame({'path': g1, 'y': 0})], ignore_index=True)   # LB-confirmed Aman overrides twin labels
P1 = np.eye(3)[ps.y.values]; pf = pd.DataFrame(P1, columns=list(C)); pf.insert(0, 'path', ps.path.values); pf.to_csv(f'{EXTRA}/pseudo_final.csv', index=False)
print(f'pseudo rows {len(pf)}: twin-locked {lk.sum()} (of which in 416/675: {np.isin(MF.path.values[te[lk]], g1).sum()}) + LB-confirmed Aman {len(g1)} | label mix', pf[list(C)].sum().astype(int).to_dict())
pd.read_csv('work/folds_group90.csv').to_csv(f'{EXTRA}/folds_group90.csv', index=False); shutil.copy('work/emb/meta_sorted.csv', f'{EXTRA}/meta_sorted.csv')"""),
('md', "## 2. Full-fit fine-tune (final model F1)"),
('code', """BASE = dict(model='vit_huge_plus_patch16_dinov3.lvd1689m', img=384, epochs=12, sched='cosine', warm=0.05, lr=3e-5, head_mult=10, layer_decay=1.0, ema=0.9985, ls=0.1,
            bs=24, dpr=0.1, gc=True, folds=[0], fold_file='folds_group90.csv', save_w=True, median=True, full_fit=True, pseudo_file='pseudo_final.csv')
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
ft('F1', **AUG)"""),
('md', "## 3. Sanity table and candidates"),
('code', """stage('eval')
z = np.load(f'{OUT}/pred_F1_f0.npz', allow_pickle=True); P = np.full((len(R), 3), np.nan); P[[POS[p] for p in z['test_path']]] = z['test']
base = pd.read_csv(f'{CODE}/gate11_baselines.csv')
print(pd.concat([base[['model'] + list(SETS)], pd.DataFrame([stats(P, 'F1 (H+, full-fit + pseudo)')])], ignore_index=True).set_index('model').to_string())
ss = pd.read_csv('datascape/sample_submission.csv'); stem2fn = {f.rsplit('.', 1)[0]: f for f in ss.filename}
base_lab = C[R.cur.values]; elig = R.src.isin(['640', 'other']).values & (R.best_ncc.values < 0.95)
ok = ~np.isnan(P[:, 0]); lab = C[np.nan_to_num(P).argmax(1)]; conf = np.nan_to_num(P).max(1); os.makedirs(f'{OUT}/cands', exist_ok=True); summary = []
for var, extra in [('RA', (base_lab == 'Rusak') & (lab == 'Aman')), ('both', np.ones(len(R), bool))]:
    flip = ok & elig & (lab != base_lab) & (conf >= 0.70) & extra; new = np.where(flip, lab, base_lab)
    sub = ss[['filename']].merge(pd.DataFrame({'filename': [stem2fn[s] for s in R.stem], 'label': new}), on='filename', how='left'); assert sub.label.notna().all()
    sub.to_csv(f'{OUT}/cands/cand_F1_{var}.csv', index=False)
    summary.append(dict(cand=f'cand_F1_{var}', flips=int(flip.sum()), by_src_dir=pd.Series([f'{a}:{b}>{c}' for a, b, c in zip(R.src[flip], base_lab[flip], lab[flip])]).value_counts().to_dict(),
                        rusak_share=round(float((new == 'Rusak').mean()), 3)))
np.save(f'{OUT}/cands/probs_F1.npy', np.nan_to_num(P).astype(np.float32)); print(pd.DataFrame(summary).to_string())
print(sorted(os.listdir(OUT))); print(f'notebook done in {(time.time() - T0) / 60:.1f} min')"""),
]

nb = nbf.v4.new_notebook(); nb.metadata.update(kernelspec=dict(name='python3', display_name='Python 3', language='python'), accelerator='GPU')
nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s) for t, s in CELLS]
nbf.write(nb, 'colab/ds12.ipynb'); print('wrote colab/ds12.ipynb', len(CELLS), 'cells')
