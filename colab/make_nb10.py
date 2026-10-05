# Builds colab/ds10.ipynb (session 10 notebook: EDA -> exhaustive copy matching -> DINOv3 frozen for conflicts -> DINOv3 ViT-H+ fine-tune gate -> evaluation).
# The A100 runner (colab/run10.py) executes it with nbconvert and saves the executed copy to Drive + asuudane/datascape-colab-out10.
# Local dry run on a 5% subset with a tiny model: DS_LOCAL=1 DS_WORK=. DS_OUT10=work/out10_local jupyter nbconvert --execute --to notebook colab/ds10.ipynb
import nbformat as nbf

CELLS = [
('md', """# Datascape 2026 · Session 10 · DINOv3 ViT-H+ fine-tune, exact-copy matching, pseudo-label vs none

**Goal.** Test whether the Multi-view Pig Posture public-1st-place recipe beats our frozen-tower heads (h2, team-best LB 0.84521 via `d04_da_bt2`), after a stricter train↔test duplicate check. The recipe is DINOv3 ViT-H+ full fine-tune, uniform LLRD, EMA, cosine schedule and dihedral TTA (sugupoko 2026, `writeups/public-1st-place-solution-dinov3-vit-huge-llrd`).

| § | Stage | Output |
|---|---|---|
| 1 | EDA: sources (image size), class mix, noise strata | tables, montage |
| 2 | Exhaustive dihedral pixel matching, all test × all train pairs and all train × train pairs | copy tiers, label agreement, fold leakage, `copy_resolution.csv` |
| 3 | Conflicting copies → frozen DINOv3-H+ features + logistic regression | conflict predictions vs h2 |
| 4 | Fine-tune gate on fold 0: arm **A** train-only, arm **B** train + twin-locked test rows (pseudo-labels) | `pred_h3A_f0.npz`, `pred_h3B_f0.npz` |
| 5 | Evaluation against the pre-registered gate (RESULTS.md 10-04 15:00) | `gate10_result.json` |

**Pre-registered gate.** PASS if any one of these holds and the Rusak share on unlocked test rows does not rise:
- A or B is ≥ +0.005 over h2 on fold-0 clean val (h2 = 0.9017).
- B is ≥ +0.005 over the DA head on fold-0 held-out locked test rows (DA = 0.9466, or 0.9853 at bias −0.8).
- An equal-weight blend with h2 (on the val rows) or with DA (on the locked rows) gains ≥ +0.005.

On FAIL the idea is closed and no further folds run. 0.99 is not a usable stop rule: exact train↔train copies agree on their label only 96–97% of the time.

**AI declaration.** Code written with Claude (Anthropic) as a coding assistant. No LLM/VLM inference is used for predictions: DINOv3 is a pure vision backbone, fine-tuned or used frozen, without text."""),
('code', """import os, sys, json, time, glob, subprocess, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix
from IPython.display import display
LOCAL = os.environ.get('DS_LOCAL') == '1'                       # 5% subset + tiny model, for a CPU dry run
WORK = os.environ.get('DS_WORK', '/content/ds'); os.chdir(WORK); sys.path.insert(0, 'work')
ROOT = 'work/small' if LOCAL else '/content/in/small512/small'   # images (long side <= 512)
OUT = os.environ.get('DS_OUT10', '/content/out10'); EXTRA = f'{OUT}/extra' if LOCAL else '/content/in/extra'
for d in (OUT, EXTRA): os.makedirs(d, exist_ok=True)
CODE = 'kernels/ft' if LOCAL else '/content/code'                # ft.py (shared fine-tune script)
EMB_H = 'work/emb/emb_vit_large_patch16_dinov3.npz' if LOCAL else '/content/in/out8/emb/emb_vit_huge_plus_patch16_dinov3.npz'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; C = ['Aman', 'Lainnya', 'Rusak']; T0 = time.time()
PLAN = dict(eda=1, dedup=4, dino=3, ft_A=70, ft_B=75, eval=2)   # planned minutes per stage -> ETA in the runner cell
def stage(name):  # progress file read by run10.py, which prints a live status + ETA line every minute
    ks = list(PLAN); i = ks.index(name); after = sum(PLAN[k] for k in ks[i + 1:])
    json.dump(dict(stage=name, ts=time.time(), cur_plan=PLAN[name], plan_after=after), open(f'{OUT}/progress.json', 'w'))
    print(f'== {name} | elapsed {(time.time() - T0) / 60:.1f} min | planned remaining {PLAN[name] + after} min', flush=True)
GATE = dict(a_h2_clean=0.9017, a_h2_all=0.9035, b_h2=0.889, b_h2_b08=0.9174, b_da=0.9466, b_da_b08=0.9853, bar=0.005)  # work/gate10.py
print(DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', '| LOCAL dry run' if LOCAL else '| Colab')"""),
('md', "## 1. EDA\nImage size identifies the data source (discoveries #27): 224 = all-Aman, 416 = synthetic conveyor boxes, 512, 675×900 = parcel photos, 640 = web damaged boxes, other = mixed phone photos. Sources differ wildly in their Rusak rate, which is why we track them separately."),
('code', """stage('eda')
from postproc import strata
MF = pd.read_csv('work/emb/meta_sorted.csv')        # full table: twin/SSCD indices refer to it
def source(m):
    w, h = m.w.values, m.h.values
    return np.select([(w == 224) & (h == 224), (w == 416) & (h == 416), (w == 512) & (h == 512), ((w == 675) & (h == 900)) | ((w == 900) & (h == 675)), (w == 640) & (h == 640)],
                     ['224', '416', '512', '675', '640'], 'other')
MF['src'] = source(MF); MF['stratum'] = strata(MF)
KEEP = np.r_[np.where(MF.split == 'train')[0][::20], np.where(MF.split == 'test')[0][::20]] if LOCAL else np.arange(len(MF))
M = MF.iloc[KEEP].reset_index(drop=True); N = len(M); TR = (M.split == 'train').values
LAB = M.label.map({c: i for i, c in enumerate(C)}).fillna(-1).astype(int).values
FG = np.full(N, -1); FG[TR] = pd.read_csv('work/folds_group90.csv').set_index('path').fold.loc[M.path[TR]].values
img = lambda i, n: Image.open(f'{ROOT}/{M.rel[i]}').convert('RGB').resize((n, n))
print(M.split.value_counts().to_dict())
display(pd.crosstab(M.src, [M.split, M.label.fillna('?')], margins=True))
display(M[TR].groupby('src').label.apply(lambda s: (s == 'Rusak').mean()).round(3).rename('train Rusak rate').to_frame().join(M[~TR].src.value_counts().rename('test rows')))
display(pd.crosstab(M.stratum, M.split))
rng = np.random.default_rng(0); fig, ax = plt.subplots(4, 8, figsize=(16, 8.6))
for r, (name, sel) in enumerate([(c, TR & (M.label == c).values) for c in C] + [('test', ~TR)]):
    for j, i in enumerate(rng.choice(np.where(sel)[0], 8, replace=False)):
        ax[r, j].imshow(img(i, 160)); ax[r, j].set_title(f'{name} | {M.src[i]}', fontsize=8); ax[r, j].axis('off')
plt.suptitle('Random train images per class (rows 1-3) and test images (row 4)'); plt.tight_layout(); plt.show()"""),
('md', """## 2. Duplicates and leakage: exhaustive dihedral pixel matching
Each image → grayscale thumbnail, zero mean, unit norm. A dot product is then the normalised cross-correlation (NCC), which is invariant to brightness, contrast, resizing and JPEG recompression. Every query is compared in its 8 dihedral orientations (4 rotations × mirror).
- **Stage 1** (all pairs on the GPU, 64 px): keep the top 20 train candidates per query with NCC ≥ 0.95. The old `pixall.py` only checked SSCD's top-10 neighbours, so its recall depended on SSCD; this pass does not.
- **Stage 2** (verification at 128 px, using stage 1's orientation): tiers are *identical* ≥ 0.999, *exact* ≥ 0.99 (same definition as `pixall.py`), *near* 0.95–0.99.

Rule (user request): a test row whose exact copies all share one label takes that label. A row whose copies disagree goes to §3."""),
('code', """stage('dedup')
from concurrent.futures import ThreadPoolExecutor
def gray(i, n=128):
    a = np.asarray(Image.open(f'{ROOT}/{M.rel[i]}').convert('L').resize((n, n), Image.BILINEAR), np.float32); a = a - a.mean(); return a / (np.linalg.norm(a) + 1e-6)
t = time.time()
with ThreadPoolExecutor(16) as ex: G128 = np.stack(list(ex.map(gray, range(N))))
G64 = G128.reshape(N, 64, 2, 64, 2).mean((2, 4)); G64 -= G64.mean((1, 2), keepdims=True); G64 /= np.linalg.norm(G64.reshape(N, -1), axis=1)[:, None, None] + 1e-6
dt = torch.float16 if DEV == 'cuda' else torch.float32; g64 = torch.tensor(G64, device=DEV, dtype=dt); g128 = torch.tensor(G128, device=DEV, dtype=dt)
tri = np.where(TR)[0]; Dtr = g64[tri].reshape(len(tri), -1)
def orient(x, v): x = x.flip(2) if v >= 4 else x; return torch.rot90(x, v % 4, (1, 2))   # v = mirror*4 + quarter turns
def search(qi, self_rows=False, K=20, thr=0.95):
    out = []
    for a in range(0, len(qi), 256):
        q = qi[a:a + 256]; V = torch.stack([orient(g64[q], v) for v in range(8)], 1).reshape(len(q) * 8, -1)
        s, view = (V @ Dtr.T).reshape(len(q), 8, -1).max(1)
        if self_rows: s[torch.as_tensor(q[:, None] == tri[None, :], device=DEV)] = -1
        top, ti = s.topk(min(K, len(tri)), 1); keep = top >= thr; r, c = torch.where(keep)
        out.append(pd.DataFrame(dict(q=q[r.cpu().numpy()], t=tri[ti[r, c].cpu().numpy()], s64=top[r, c].float().cpu().numpy(), view=view[r, ti[r, c]].cpu().numpy())))
    P = pd.concat(out, ignore_index=True); P['ncc'] = 0.0
    for v in range(8):                                   # stage 2: verify at 128 px in the stage-1 orientation
        for b in np.array_split(np.where(P.view.values == v)[0], max(1, (P.view.values == v).sum() // 4096)):
            if len(b): P.loc[b, 'ncc'] = (orient(g128[P.q.values[b]], v) * g128[P.t.values[b]]).float().sum((1, 2)).cpu().numpy()
    P['tier'] = np.select([P.ncc >= 0.999, P.ncc >= 0.99, P.ncc >= 0.95], ['identical', 'exact', 'near'], 'similar'); return P
PT = search(np.where(~TR)[0]); PP = search(tri, self_rows=True)
print(f'stage 1+2 in {time.time() - t:.0f}s | test-train candidate pairs {len(PT)}, train-train {len(PP)}')
display(pd.DataFrame({'test->train pairs': PT.tier.value_counts(), 'train->train pairs': PP.tier.value_counts()}))
X = PT[PT.ncc >= 0.99]; XX = PP[PP.ncc >= 0.99]
hit = M[~TR].assign(copy=lambda d: d.index.isin(X.q)).groupby('src').copy.agg(['sum', 'mean']).round(3); hit.columns = ['test rows with exact copy', 'share']; display(hit)
print(f'test rows with an exact train copy: {X.q.nunique()} / {(~TR).sum()} ({X.q.nunique() / (~TR).sum():.1%})')
agree = (LAB[XX.q] == LAB[XX.t]); cross = FG[XX.q] != FG[XX.t]
print(f'train-train exact pairs {len(XX)} | label agreement {agree.mean():.3f} (label-noise ceiling) | pairs straddling group90 folds {cross.mean():.3f} (fold leakage)')
display(XX.assign(src=M.src.values[XX.q], agree=agree).groupby('src').agree.agg(['size', 'mean']).round(3))
if not LOCAL:   # recall of the old candidate generator: is the exhaustive exact match inside SSCD's top-10?
    I10 = np.load('work/nntr_sscd.npz')['i']; best = X.sort_values('ncc').groupby('q').t.last()
    print(f'exhaustive exact matches also in SSCD top-10: {np.mean([t in I10[q] for q, t in best.items()]):.3f}')"""),
('code', """g = X.assign(lab=LAB[X.t]).groupby('q').lab.agg(lambda s: tuple(sorted(set(s))))
RES = pd.DataFrame(dict(q=g.index, path=M.path[g.index].values, src=M.src[g.index].values, n_copy=X.groupby('q').size().loc[g.index].values,
                        tier=X.groupby('q').ncc.max().loc[g.index].map(lambda v: 'identical' if v >= 0.999 else 'exact').values,
                        copy_label=[C[l[0]] if len(l) == 1 else 'CONFLICT' for l in g.values]))
display(pd.crosstab(RES.src, RES.copy_label, margins=True))
sub = pd.read_csv('submissions/d04_da_bt2.csv').set_index('filename').label; RES['da_bt2'] = sub.loc[M.filename[RES.q]].values
d = RES[(RES.copy_label != 'CONFLICT') & (RES.copy_label != RES.da_bt2)]
print(f'unanimous copy label differs from the team-best submission on {len(d)} rows (held back from the 10-05 paired probes)'); display(pd.crosstab(d.da_bt2, d.copy_label))
def pairs(P, title, n=6):
    P = P.sample(min(n, len(P)), random_state=0) if len(P) else P
    if not len(P): print('none:', title); return
    fig, ax = plt.subplots(2, len(P), figsize=(2.6 * len(P), 5.4), squeeze=False)
    for j, r in enumerate(P.itertuples()):
        ax[0, j].imshow(img(r.q, 160)); ax[0, j].set_title(f'test {M.src[r.q]} ncc {r.ncc:.3f}', fontsize=8)
        ax[1, j].imshow(img(r.t, 160)); ax[1, j].set_title(f'train {M.label[r.t]}', fontsize=8)
        for a in ax[:, j]: a.axis('off')
    plt.suptitle(title); plt.tight_layout(); plt.show()
pairs(X, 'exact copies (NCC >= 0.99): test (top) vs train (bottom)'); pairs(PT[PT.tier == 'near'], 'near (0.95-0.99): copies or look-alikes?')
pairs(X[X.q.isin(RES.q[RES.copy_label == 'CONFLICT'])], 'conflicting copies')"""),
('md', "## 3. Conflicting copies → frozen DINOv3-H+ (pretrained) features\nUser rule: when an image's exact copies carry different labels, predict it with a pretrained DINO feature extractor. We use DINOv3-H+ embeddings (session 8, 256 px), a logistic regression on group folds, and compare it with h2 on **train** rows whose own exact copies disagree. That comparison is the analogous situation with known labels. Pre-registered: use the method with higher accuracy there."),
('code', """stage('dino')
from sklearn.linear_model import LogisticRegression
E = np.load(EMB_H)['orig'].astype(np.float32)[KEEP]; E /= np.linalg.norm(E, axis=1, keepdims=True)
OD = np.zeros((N, 3))
for k in range(5):
    a, b = TR & (FG != k), TR & (FG == k); OD[b] = LogisticRegression(C=0.5, max_iter=3000).fit(E[a], LAB[a]).predict_proba(E[b])
PDt = LogisticRegression(C=0.5, max_iter=3000).fit(E[TR], LAB[TR]).predict_proba(E[~TR]); PD = np.zeros((N, 3)); PD[~TR] = PDt
pos = {p: i for i, p in enumerate(M.path)}; OH = np.zeros((N, 3)); TH = np.zeros((N, 3))
for k in range(5):
    z = np.load(f'work/preds/h2_peg_c11/pred_h2_peg_c11_f{k}.npz', allow_pickle=True)
    for p, v in zip(z['val_path'], z['val']):
        if p in pos: OH[pos[p]] = v
    for p, v in zip(z['test_path'], z['test']):
        if p in pos: TH[pos[p]] += v / 5
f1 = lambda y, p, b=0.0: f1_score(y, (np.log(p + 1e-9) + np.r_[0, 0, b]).argmax(1), average='macro')
print(f'group-OOF macro-F1 on train: DINOv3-H+ frozen LR {f1(LAB[TR], OD[TR]):.4f} | h2 {f1(LAB[TR], OH[TR]):.4f}')
cg = XX.assign(other=LAB[XX.t]).groupby('q').other.agg(set); conf_tr = np.array([q for q, s in cg.items() if s != {LAB[q]}], int)
if len(conf_tr):
    print(f'train rows whose exact copies carry another label: {len(conf_tr)} | accuracy vs own label: DINO {np.mean(OD[conf_tr].argmax(1) == LAB[conf_tr]):.3f}, h2 {np.mean(OH[conf_tr].argmax(1) == LAB[conf_tr]):.3f}')
use_dino = len(conf_tr) > 0 and np.mean(OD[conf_tr].argmax(1) == LAB[conf_tr]) >= np.mean(OH[conf_tr].argmax(1) == LAB[conf_tr])
RES['dino'] = [C[i] for i in PD[RES.q].argmax(1)]; RES['h2'] = [C[i] for i in TH[RES.q].argmax(1)]
RES['final'] = np.where(RES.copy_label != 'CONFLICT', RES.copy_label, RES.dino if use_dino else RES.h2)
print('conflict rows resolved with', 'DINOv3 frozen LR' if use_dino else 'h2 (DINO lost on train conflicts)'); display(RES[RES.copy_label == 'CONFLICT'].head(20))
RES.drop(columns='q').to_csv(f'{OUT}/copy_resolution.csv', index=False); del g64, g128, Dtr; torch.cuda.empty_cache() if DEV == 'cuda' else None"""),
('md', """## 4. DINOv3 ViT-H+ fine-tune: gate on fold 0
Recipe (pig-posture 1st): `vit_huge_plus_patch16_dinov3.lvd1689m` @384, AdamW lr 3e-5 with head ×10, uniform LLRD (1.0), cosine with 5% warmup, 12 epochs, EMA 0.9985, label smoothing 0.1, drop-path 0.1, gradient checkpointing, final EMA weights (no best-epoch picking), 8-view dihedral TTA. Kept from our `ft.py`: median-3 denoise and test-corruption noise augmentation. Dropped: ConditionalDesaturate, because test chroma ≥ train chroma in every source.
- **Arm A**: train folds 1–4 only.
- **Arm B**: + twin-locked test rows (SSCD ≥ 0.55, purity ≥ 0.9, clean stratum) from the other four `pl_iter` folds. This is the pseudo-labelling arm and mirrors the DA heads that gave the team best.

`ft.py` fix found while preparing this session: timm stores layer decay as `lr_scale`, which only timm's own schedulers apply. Earlier `ft.py` runs therefore trained at a uniform LR whatever `layer_decay` said."""),
('code', """from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.argv = ['x']; import twin            # twin-lock rule of d04_da_bt2 (full table)
te = np.where(~twin.TR)[0]; Dn = np.load('work/nntr_sscd.npz'); I, S = Dn['i'][te], Dn['s'][te]; lab, pur = twin.vote(I, S, 0.55)
lk = (lab >= 0) & (pur >= 0.9) & (twin.ST[te] == 'clean_other'); L, yL = te[lk], lab[lk]
ii, jj = np.where(S[lk] >= 0.55); Aj = coo_matrix((np.ones(len(ii)), (ii, len(L) + I[lk][ii, jj])), shape=(len(L) + len(MF),) * 2)
gg = connected_components(Aj, directed=False)[1][:len(L)]; ug = np.unique(gg); gf = dict(zip(ug, np.random.default_rng(0).permutation(len(ug)) % 5)); LF = np.array([gf[x] for x in gg])
print(f'locked test rows {len(L)} in {len(ug)} groups | held-out fold 0: {(LF == 0).sum()} rows (gate set b) | pseudo rows for arm B: {(LF != 0).sum()}')
ps = pd.DataFrame(np.eye(3)[yL[LF != 0]], columns=C); ps.insert(0, 'path', MF.path.values[L[LF != 0]]); ps.to_csv(f'{EXTRA}/pseudo_locked_f0.csv', index=False)
pd.read_csv('work/folds_group90.csv').to_csv(f'{EXTRA}/folds_group90.csv', index=False)
BASE = dict(model='vit_huge_plus_patch16_dinov3.lvd1689m', img=384, epochs=12, sched='cosine', warm=0.05, lr=3e-5, head_mult=10, layer_decay=1.0, ema=0.9985, ls=0.1,
            bs=24, dpr=0.1, gc=True, aug='d8', tta='d8', folds=[0], fold_file='folds_group90.csv', save_w=False, median=True)
if LOCAL: BASE.update(model='vit_tiny_patch16_224', pretrained=False, img=64, epochs=1, bs=8)
def ft(arm, **kw):
    stage(f'ft_{arm}'); cfg = dict(BASE, tag=f'h3{arm}', **kw); env = dict(os.environ, CFG=json.dumps(cfg), DS_OUT=OUT, NW='10')
    if LOCAL: env.pop('DS_IN', None)
    else: env['DS_IN'] = '/content/in'
    p = subprocess.Popen([sys.executable, '-u', f'{CODE}/ft.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); t = time.time()
    with open(f'{OUT}/log_h3{arm}.txt', 'w') as f:
        for line in p.stdout:
            f.write(line); f.flush()
            if ' ep ' in line or 'saved' in line or 'rror' in line or 'Traceback' in line or 'exit' in line: print(line, end='', flush=True)
    print(f'arm {arm}: exit {p.wait()} in {(time.time() - t) / 60:.1f} min', flush=True)
ft('A')"""),
('code', """CAP_B = 150   # minutes since notebook start; arm B is skipped past it so the session ends inside the wall-time cap
if (time.time() - T0) / 60 < CAP_B: ft('B', pseudo_file='pseudo_locked_f0.csv')
else: print(f'arm B skipped: {(time.time() - T0) / 60:.0f} min elapsed > {CAP_B}')"""),
('md', "## 5. Evaluation against the pre-registered gate\n(a) group90 fold-0 val rows, original labels, all rows and the *clean* subset (no exact copy in train folds ≠ 0). (b) fold-0 held-out locked test rows, at argmax and at the DA bias −0.8. Blends are equal-weight probability averages with h2. The DA-head blend for (b) is computed locally from `asuudane/datascape-colab-out10`."),
('code', """stage('eval')
def load(tag):
    f = f'{OUT}/pred_{tag}_f0.npz'
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); V, T = dict(zip(z['val_path'], z['val'])), dict(zip(z['test_path'], z['test'])); return V, T
v0 = np.where(TR & (FG == 0))[0]; leak = np.isin(v0, XX.q[(FG[XX.t] != 0)].values); clean = v0[~leak]
h0 = L[LF == 0]; y0 = yL[LF == 0]; U = np.setdiff1d(te, L); TH_full = {p: TH[pos[p]] for p in MF.path.values[te] if p in pos}
rows, R = [], {}
for name in ['h2', 'h3A', 'h3B', 'h3A+h2', 'h3B+h2']:
    if name == 'h2': V = {M.path[i]: OH[i] for i in v0}; T = TH_full
    else:
        parts = [load(n) if n != 'h2' else ({M.path[i]: OH[i] for i in v0}, TH_full) for n in name.split('+')]
        if any(p is None for p in parts): continue
        V = {k: np.mean([p[0][k] for p in parts], 0) for k in parts[0][0] if all(k in p[0] for p in parts)}; T = {k: np.mean([p[1][k] for p in parts], 0) for k in parts[0][1] if all(k in p[1] for p in parts)}
    pv = lambda idx: np.stack([V[M.path[i]] for i in idx if M.path[i] in V]); yv = lambda idx: LAB[[i for i in idx if M.path[i] in V]]; pt = lambda idx: np.stack([T[MF.path[i]] for i in idx if MF.path[i] in T])
    r = dict(model=name, a_all=f1(yv(v0), pv(v0)), a_clean=f1(yv(clean), pv(clean)))  # LOCAL smoke scores only the rows ft.py predicted
    if not LOCAL: r.update(b=f1(y0, pt(h0)), b_b08=f1(y0, pt(h0), -0.8), unlocked_rusak=np.mean(pt(U).argmax(1) == 2), unlocked_rusak_b08=np.mean((np.log(pt(U) + 1e-9) + np.r_[0, 0, -0.8]).argmax(1) == 2))
    rows.append(r); R[name] = (V, T)
TAB = pd.DataFrame(rows).set_index('model').round(4); display(TAB)
print('constants: h2 a_clean', GATE['a_h2_clean'], '| DA head b', GATE['b_da'], 'b@-0.8', GATE['b_da_b08'])
gate = {}
for arm in ['h3A', 'h3B']:
    if arm not in TAB.index: continue
    t, h = TAB.loc[arm], TAB.loc['h2']; bl = TAB.loc[f'{arm}+h2']
    c = dict(a_clean=t.a_clean - GATE['a_h2_clean'] >= GATE['bar'], blend_a=bl.a_clean - GATE['a_h2_clean'] >= GATE['bar'])
    if not LOCAL and arm == 'h3B': c.update(b=t.b - GATE['b_da'] >= GATE['bar'] or t.b_b08 - GATE['b_da_b08'] >= GATE['bar'])
    share_ok = LOCAL or t.unlocked_rusak_b08 <= h.unlocked_rusak_b08 + 0.005
    gate[arm] = dict(**{k: bool(v) for k, v in c.items()}, share_ok=bool(share_ok), PASS=bool(any(c.values()) and share_ok))
print(json.dumps(gate, indent=1)); json.dump(dict(gate=gate, table=TAB.reset_index().to_dict('records'), gate_constants=GATE), open(f'{OUT}/gate10_result.json', 'w'), indent=1)
for arm in ['h3A', 'h3B']:
    hf = f'{OUT}/hist_{arm}_f0.json'
    if os.path.exists(hf):
        h = pd.DataFrame(json.load(open(hf))['hist']); plt.plot(h.ep, h.val_f1, marker='o', label=f'{arm} val F1 (EMA, no TTA)')
plt.axhline(GATE['a_h2_all'], ls='--', c='k', label='h2 fold-0 (all val)'); plt.xlabel('epoch'); plt.legend(); plt.title('fold-0 learning curve'); plt.show()
for arm in [a for a in ['h3A', 'h3B'] if a in R]:
    V = R[arm][0]; vv = np.array([i for i in v0 if M.path[i] in V]); p = np.stack([V[M.path[i]] for i in vv]).argmax(1)
    print(arm, 'confusion (rows = true Aman/Lainnya/Rusak)'); print(confusion_matrix(LAB[vv], p, labels=[0, 1, 2]))
    display(pd.DataFrame({'src': M.src.values[vv], 'y': LAB[vv], 'p': p}).groupby('src')[['y', 'p']].apply(lambda d: round(f1_score(d.y, d.p, average='macro'), 4)).rename(f'{arm} fold-0 F1 by source').to_frame())"""),
('md', "## 6. Outputs\nThe runner copies `/content/out10` to Drive `MyDrive/datascape_2026/session10` and to the private Kaggle dataset `asuudane/datascape-colab-out10`. It contains this executed notebook, `copy_resolution.csv`, `pred_h3{A,B}_f0.npz`, `hist_*.json`, `log_*.txt` and `gate10_result.json`. The runner then releases the A100."),
('code', """print(sorted(os.listdir(OUT))); print(f'notebook done in {(time.time() - T0) / 60:.1f} min')"""),
]

nb = nbf.v4.new_notebook(); nb.metadata.update(kernelspec=dict(name='python3', display_name='Python 3', language='python'), accelerator='GPU')
nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s) for t, s in CELLS]
nbf.write(nb, 'colab/ds10.ipynb'); print('wrote colab/ds10.ipynb', len(CELLS), 'cells')
