# Domain adaptation check (discoveries #22 follow-up): does adding twin-labelled test rows to the head's training data fix the test-domain
# shift better than a scalar Rusak bias? 5-fold CV over the 1534 locked test rows (twin labels ~98% right):
#   A = head on train only + bias calibrated on the other locked folds;  B = head on train + other locked folds (+ same bias calibration).
# Same MLP recipe as work/mlp_head.py (2 seeds, CPU). Features cat6 + PE-G.
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
sys.path.insert(0, 'work'); _argv = list(sys.argv); sys.argv = ['x']
from twin import M, TR, ST, LAB, vote
torch.set_num_threads(8)
T6 = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224', 'vit_pe_core_gigantic_patch14_448']
def feat(t):
    E = np.load(f'work/emb/emb_{t}.npz'); X = sum(E[k].astype(np.float32) for k in E.files); return X / np.linalg.norm(X, axis=1, keepdims=True)
X = torch.tensor(np.concatenate([feat(t) for t in T6], 1))
te = np.where(~TR)[0]; D = np.load('work/nntr_sscd.npz'); lab, pur = vote(D['i'][te], D['s'][te], 0.55)
lk = te[(lab >= 0) & (pur >= 0.9) & (ST[te] == 'clean_other')]; ylk = lab[(lab >= 0) & (pur >= 0.9) & (ST[te] == 'clean_other')]
tr = np.where(TR)[0]; ytr = LAB[tr]

def fit(idx, y, seed, ep=40):
    torch.manual_seed(seed); Xa, ya = X[idx], torch.tensor(y)
    m = nn.Sequential(nn.Dropout(0.5), nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3))
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ep * ((len(ya) + 255) // 256))
    for _ in range(ep):
        for b in torch.randperm(len(ya)).split(256):
            loss = F.cross_entropy(m(Xa[b]), ya[b], label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()

def logp(models, idx):
    with torch.no_grad(): return np.log(np.mean([m(X[idx]).softmax(1).numpy() for m in models], 0) + 1e-9)

def best_bias(L, y):
    bs = np.linspace(-3, 1, 81); f = [f1_score(y, (L + np.r_[0, 0, b]).argmax(1), average='macro') for b in bs]; return bs[int(np.argmax(f))]

# strict groups: union-find over SSCD links >= 0.55 (test<->train twins and test<->test), so no source image spans two folds
par = list(range(len(M)))
def f_(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for q in te:
    for c, s in zip(D['i'][q], D['s'][q]):
        if s >= 0.55: par[f_(q)] = f_(int(c))
E = np.load('work/emb_sscd.npz'); V = {k: E[k].astype(np.float32) for k in E.files}; St = np.max(np.stack([V['orig'][te] @ V[k][te].T for k in V]), 0); np.fill_diagonal(St, -1)
for a_, b_ in zip(*np.where(St >= 0.55)): par[f_(int(te[a_]))] = f_(int(te[b_]))
g = np.array([f_(int(x)) for x in lk]); print('locked rows', len(lk), 'groups', len(set(g)), 'largest', np.bincount(np.unique(g, return_inverse=True)[1]).max(), flush=True)
fo = (g * 2654435761 % 2**32) % 5; ONLY_B = os.environ.get('DA') == 'onlyB'; A = [] if ONLY_B else [fit(tr, ytr, s) for s in range(2)]; BB = []
pa, pb, pa0, pb0 = np.zeros(len(lk), int), np.zeros(len(lk), int), np.zeros(len(lk), int), np.zeros(len(lk), int)
for k in range(5):
    o = fo != k; h = fo == k
    if A: La_o, La_h = logp(A, lk[o]), logp(A, lk[h]); pa0[h] = La_h.argmax(1); pa[h] = (La_h + np.r_[0, 0, best_bias(La_o, ylk[o])]).argmax(1)
    B = [fit(np.r_[tr, lk[o]], np.r_[ytr, ylk[o]], s) for s in range(2)]
    Lb_o, Lb_h = logp(B, lk[o]), logp(B, lk[h]); pb0[h] = Lb_h.argmax(1); bb = best_bias(Lb_o, ylk[o]); BB.append(bb); pb[h] = (Lb_h + np.r_[0, 0, bb]).argmax(1)
    print(k, 'B bias', BB[-1], flush=True)
f = lambda p: f1_score(ylk, p, average='macro')
print(f'locked test rows n={len(lk)} | A train-only: raw {f(pa0):.4f} +bias {f(pa):.4f} (Rusak {np.mean(pa==2):.3f}) | '
      f'B +locked rows: raw {f(pb0):.4f} +bias {f(pb):.4f} (Rusak {np.mean(pb==2):.3f}) | true Rusak {np.mean(ylk==2):.3f}')
