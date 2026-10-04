# Iterative label cleaning (discoveries #26). Confident Learning (Northcutt 2021): rows whose group-OOF prob of the given label is
# below that class's mean self-confidence while another class clears its own threshold get relabelled to that class; then duplicate
# clusters (DINOv2 >=0.90, size 2-50) take their majority cleaned label. Two rounds with GPU MLP heads (recipe of work/mlp_head.py).
# Then: twin votes with cleaned labels (SSCD >=0.55, purity >=0.9, clean_other), domain-adapted heads on cleaned train + locked test rows.
# Run from the mini-repo root (/content/ds). Writes work/labels_clean_v2.csv, work/preds/c2_h2 (group-OOF + test), work/preds/c2_da (test),
# work/models_c2/*.pt.
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; C = ['Aman', 'Lainnya', 'Rusak']
M = pd.read_csv('work/emb/meta_sorted.csv'); TR = (M.split == 'train').values; tr, te = np.where(TR)[0], np.where(~TR)[0]
y0 = M.label.map({c: i for i, c in enumerate(C)}).values[tr].astype(int)
fold = pd.read_csv('work/folds_group90.csv').set_index('path').loc[M.path.values[tr], 'fold'].values
clus = pd.read_csv('work/clusters.csv').set_index('path').cluster.loc[M.path.values[tr]].values
CAT6 = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224']
GIANT = ['vit_pe_core_gigantic_patch14_448', 'eva02_enormous_patch14_clip_224', 'vit_huge_patch14_clip_378', 'vit_giantopt_patch16_siglip_384', 'vit_gigantic_patch14_clip_378']
def feat(t):
    E = np.load(f'work/emb/emb_{t}.npz'); X = sum(E[k].astype(np.float32) for k in E.files); return X / np.linalg.norm(X, axis=1, keepdims=True)
XS = [torch.tensor(np.concatenate([feat(t) for t in s], 1), device=dev) for s in (CAT6 + GIANT[:1], CAT6 + GIANT)]

EP = int(os.environ.get('EP', 40)); NS = os.environ.get('SMOKE') == '1'  # smoke: 1 epoch, 1 seed
def fit(X, rows, yy, seed, ep=EP, bs=256):
    torch.manual_seed(seed); Xa = X[rows]; ya = torch.tensor(yy, device=dev)
    m = nn.Sequential(nn.Dropout(0.5), nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ep * ((len(ya) + bs - 1) // bs))
    for _ in range(ep):
        for b in torch.randperm(len(ya), device=dev).split(bs):
            loss = F.cross_entropy(m(Xa[b]), ya[b], label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()

def oof_and_test(yy, seeds):  # geometric mean over the two feature sets; group folds
    Lo, Lt = [], []
    for X in XS:
        o = np.zeros((len(tr), 3)); t = np.zeros((len(te), 3))
        for k in range(5):
            a, b = fold != k, fold == k
            for s in range(seeds):
                m = fit(X, tr[a], yy[a], s)
                with torch.no_grad(): o[b] += m(X[tr[b]]).softmax(1).cpu().numpy() / seeds; t += m(X[te]).softmax(1).cpu().numpy() / (5 * seeds)
        Lo.append(np.log(o + 1e-9)); Lt.append(np.log(t + 1e-9))
    o, t = np.exp(np.mean(Lo, 0)), np.exp(np.mean(Lt, 0)); return o / o.sum(1, keepdims=True), t / t.sum(1, keepdims=True)

def cl_clean(P, yy):
    th = np.array([P[yy == j, j].mean() for j in range(3)]); cand = np.where(P >= th, P, -1.0); cand[np.arange(len(yy)), yy] = -1
    flag = (cand.max(1) > 0) & (P[np.arange(len(yy)), yy] < th[yy]); new = np.where(flag, cand.argmax(1), yy)
    df = pd.DataFrame({'c': clus, 'y': new}); sz = df.groupby('c').y.transform('size'); maj = df.groupby('c').y.agg(lambda s: s.value_counts().index[0])
    new = np.where((sz >= 2) & (sz <= 50), maj.loc[clus].values, new)  # duplicate groups share one label
    return new, th, flag.sum()

yy = y0.copy()
for r in range(2):
    P, _ = oof_and_test(yy, 1 if NS else 3); new, th, nf = cl_clean(P, yy)
    print(f'round {r}: thresholds {th.round(3)} flagged {nf} | changed vs prev {(new != yy).sum()} | vs original {(new != y0).sum()} | shares', np.bincount(new, minlength=3) / len(new), flush=True)
    print(pd.crosstab(pd.Series([C[i] for i in y0], name='orig'), pd.Series([C[i] for i in new], name='clean')), flush=True)
    yy = new
pd.DataFrame({'path': M.path.values[tr], 'label_orig': [C[i] for i in y0], 'label_clean': [C[i] for i in yy]}).to_csv('work/labels_clean_v2.csv', index=False)
o, t = oof_and_test(yy, 1 if NS else 5)  # final group-OOF on cleaned labels + test preds
os.makedirs('work/preds/c2_h2', exist_ok=True)
for k in range(5):
    mm = fold == k; np.savez(f'work/preds/c2_h2/pred_c2_h2_f{k}.npz', val_path=M.path.values[tr][mm], val=o[mm], test_path=M.path.values[te], test=t)
from sklearn.metrics import f1_score
print('c2_h2 group-OOF macroF1 vs cleaned labels %.4f | vs original %.4f | test Rusak share %.3f' % (f1_score(yy, o.argmax(1), average='macro'), f1_score(y0, o.argmax(1), average='macro'), (t.argmax(1) == 2).mean()), flush=True)

# twin votes with cleaned labels -> domain-adapted heads (cleaned train + locked test rows)
LAB = np.full(len(M), -1); LAB[tr] = yy; ST = np.load('work/strata.npy', allow_pickle=True)
D = np.load('work/nntr_sscd.npz'); I, S = D['i'][te], D['s'][te]; w = np.where(S >= 0.55, S, 0)
Pv = np.stack([(w * (LAB[I] == c)).sum(1) for c in range(3)], 1); tot = Pv.sum(1); lab = np.where(tot > 0, Pv.argmax(1), -1); pur = Pv.max(1) / np.maximum(tot, 1e-9)
lk = (lab >= 0) & (pur >= 0.9) & (ST[te] == 'clean_other'); print('locked test rows with cleaned labels', lk.sum(), 'Rusak share', round(float((lab[lk] == 2).mean()), 3), flush=True)
rows = np.r_[tr, te[lk]]; yda = np.r_[yy, lab[lk]]; os.makedirs('work/models_c2', exist_ok=True); Lt = []
for j, X in enumerate(XS):
    Pt = 0
    for s in range(1 if NS else 5):
        m = fit(X, rows, yda, s); torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f'work/models_c2/da_set{j}_s{s}.pt')
        with torch.no_grad(): Pt = Pt + m(X[te]).softmax(1).cpu().numpy() / 5
    Lt.append(np.log(Pt + 1e-9))
t2 = np.exp(np.mean(Lt, 0)); t2 /= t2.sum(1, keepdims=True); os.makedirs('work/preds/c2_da', exist_ok=True)
for k in range(5):
    mm = fold == k; np.savez(f'work/preds/c2_da/pred_c2_da_f{k}.npz', val_path=M.path.values[tr][mm], val=o[mm], test_path=M.path.values[te], test=t2)
np.save('work/twin_clean_lock.npy', np.c_[te[lk], lab[lk]]); print('c2_da saved | test Rusak share raw', round(float((t2.argmax(1) == 2).mean()), 3), flush=True)
