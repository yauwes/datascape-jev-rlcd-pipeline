# Session 6: head-architecture x noise-robust-loss x label-version sweep on frozen features (cat6 + PE-G), group 5-fold, 3 seeds.
# Losses: ce (+LS 0.1, the current recipe), gce q=0.7 (Zhang & Sabuncu 2018), sce a=0.1 b=1 (Wang 2019), boot soft beta=0.8 (Reed 2015).
# Heads: mlp512 (current), mlp1024x2 (wider/deeper), lr (linear softmax), knn (cosine k=10, no training).
# Scores per config: OOF F1 vs original and cleaned labels, test natural Rusak share, F1 vs cleaned twin labels on locked test rows.
# Run from the mini-repo root. Writes work/preds/sw_<cfg>/ (ens.py format) and work/sweep_scores.csv.
import os, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; C = ['Aman', 'Lainnya', 'Rusak']; T0 = time.time()
M = pd.read_csv('work/emb/meta_sorted.csv'); TR = (M.split == 'train').values; tr, te = np.where(TR)[0], np.where(~TR)[0]
y0 = M.label.map({c: i for i, c in enumerate(C)}).values[tr].astype(int)
yc = pd.read_csv('work/labels_clean_v2.csv').set_index('path').label_clean.loc[M.path.values[tr]].map({c: i for i, c in enumerate(C)}).values
fold = pd.read_csv('work/folds_group90.csv').set_index('path').loc[M.path.values[tr], 'fold'].values
lock = np.load('work/twin_clean_lock.npy'); pos_te = {r: i for i, r in enumerate(te)}; li = np.array([pos_te[r] for r in lock[:, 0]]); ly = lock[:, 1]
TW = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224', 'vit_pe_core_gigantic_patch14_448']
def feat(t):
    E = np.load(f'work/emb/emb_{t}.npz'); X = sum(E[k].astype(np.float32) for k in E.files); return X / np.linalg.norm(X, axis=1, keepdims=True)
X = torch.tensor(np.concatenate([feat(t) for t in TW], 1), device=dev)

def loss_fn(kind, z, yb, ep_frac):
    if kind == 'ce': return F.cross_entropy(z, yb, label_smoothing=0.1)
    p = z.softmax(1); py = p.gather(1, yb[:, None]).squeeze(1).clamp_min(1e-7)
    if kind == 'gce': return ((1 - py ** 0.7) / 0.7).mean()
    if kind == 'sce': return 0.1 * F.cross_entropy(z, yb) + 1.0 * (4.0 * (1 - py)).mean()  # RCE with log(0) clipped to -4
    if kind == 'boot':  # soft bootstrapping, switched on after a 30% warm-up of plain CE
        if ep_frac < 0.3: return F.cross_entropy(z, yb, label_smoothing=0.1)
        tgt = 0.8 * F.one_hot(yb, 3).float() + 0.2 * p.detach(); return -(tgt * F.log_softmax(z, 1)).sum(1).mean()

def head(kind):
    d = X.shape[1]
    if kind == 'mlp512': return nn.Sequential(nn.Dropout(0.5), nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3))
    if kind == 'mlp1024x2': return nn.Sequential(nn.Dropout(0.5), nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.3), nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(0.3), nn.Linear(1024, 3))
    if kind == 'lr': return nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 3))

def fit(hk, lk, rows, yy, seed, ep=40, bs=256):
    torch.manual_seed(seed); Xa = X[rows]; ya = torch.tensor(yy, device=dev); m = head(hk).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3 if hk != 'lr' else 3e-3, weight_decay=1e-2 if hk != 'lr' else 1e-4)
    n = (len(ya) + bs - 1) // bs; sch = torch.optim.lr_scheduler.OneCycleLR(opt, opt.param_groups[0]['lr'], total_steps=ep * n)
    for e in range(ep):
        for b in torch.randperm(len(ya), device=dev).split(bs):
            loss = loss_fn(lk, m(Xa[b]), ya[b], e / ep); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()

def knn_prob(qi, ri, yy, k=10):  # cosine kNN vote with softmax(sim/0.05) weights
    out = []
    for a in range(0, len(qi), 2048):
        s = X[qi[a:a + 2048]] @ X[ri].T / len(TW); v, ix = s.topk(k, 1); w = (v / 0.05).softmax(1)
        P = torch.zeros(len(v), 3, device=dev); P.scatter_add_(1, torch.tensor(yy, device=dev)[ix], w); out.append(P.cpu().numpy())
    return np.concatenate(out)

rows_out = []
CFGS = [(h, l, lab) for lab in ('orig', 'clean') for h, l in [('mlp512', 'ce'), ('mlp512', 'gce'), ('mlp512', 'sce'), ('mlp512', 'boot'), ('mlp1024x2', 'ce'), ('lr', 'ce'), ('knn', '-')]]
for hk, lk, lab in CFGS:
    yy = y0 if lab == 'orig' else yc; tag = f'sw_{hk}_{lk}_{lab}'.replace('-', 'na'); o = np.zeros((len(tr), 3)); t = np.zeros((len(te), 3)); t1 = time.time()
    for k in range(5):
        a, b = fold != k, fold == k
        if hk == 'knn': o[b] = knn_prob(tr[b], tr[a], yy[a]); t += knn_prob(te, tr[a], yy[a]) / 5; continue
        for s in range(3):
            m = fit(hk, lk, tr[a], yy[a], s)
            with torch.no_grad(): o[b] += m(X[tr[b]]).softmax(1).cpu().numpy() / 3; t += m(X[te]).softmax(1).cpu().numpy() / 15
    d = f'work/preds/{tag}'; os.makedirs(d, exist_ok=True)
    for k in range(5):
        mm = fold == k; np.savez(f'{d}/pred_{tag}_f{k}.npz', val_path=M.path.values[tr][mm], val=o[mm], test_path=M.path.values[te], test=t)
    r = dict(tag=tag, f1_orig=round(f1_score(y0, o.argmax(1), average='macro'), 4), f1_clean=round(f1_score(yc, o.argmax(1), average='macro'), 4),
             test_rusak=round(float((t.argmax(1) == 2).mean()), 3), f1_twin_locked=round(f1_score(ly, t[li].argmax(1), average='macro'), 4), sec=round(time.time() - t1))
    rows_out.append(r); print(r, flush=True); pd.DataFrame(rows_out).to_csv('work/sweep_scores.csv', index=False)
print(f'SWEEP DONE {(time.time() - T0) / 60:.1f} min', flush=True)
