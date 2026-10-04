# Iterative pseudo-labelling on the DA heads (Lee 2013 pseudo-label; Xie 2020 noisy-student style rounds), scored on the test domain.
# Eval set = the 1534 twin-locked test rows (labels ~98% right), split into 5 folds by SSCD-twin connected components so a held-out
# group never has a near-copy in training. Round 0 = DA head (train + other locked folds). Round r>0 adds unlocked test rows whose
# previous-round prob >= TAU as pseudo labels. Bar (pre-registered, RESULTS 10-04): held-out F1 >= r0 + 0.005 and no rise in the
# natural Rusak share of unlocked rows. Writes work/pl_iter_scores.csv.
import sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import f1_score
sys.path.insert(0, 'work'); sys.argv = ['x']
from twin import M, TR, ST, LAB, vote
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'; TAU, ROUNDS, SEEDS, B = 0.9, 3, 2, np.r_[0, 0, -0.8]
TOW = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224', 'vit_pe_core_gigantic_patch14_448']
def feat(t):
    E = np.load(f'work/emb/emb_{t}.npz'); X = sum(E[k].astype(np.float32) for k in E.files); return X / np.linalg.norm(X, axis=1, keepdims=True)
X = torch.tensor(np.concatenate([feat(t) for t in TOW], 1), device=dev)
te = np.where(~TR)[0]; D = np.load('work/nntr_sscd.npz'); I, S = D['i'][te], D['s'][te]; lab, pur = vote(I, S, 0.55)
lk = (lab >= 0) & (pur >= 0.9) & (ST[te] == 'clean_other'); L, yL, U = te[lk], lab[lk], te[~lk]
# groups: locked test rows sharing any train twin (SSCD >= 0.55) are one component
ii, jj = np.where(S[lk] >= 0.55); A = coo_matrix((np.ones(len(ii)), (ii, len(L) + I[lk][ii, jj])), shape=(len(L) + len(M),) * 2)
g = connected_components(A, directed=False)[1][:len(L)]; ug = np.unique(g); gf = dict(zip(ug, np.random.default_rng(0).permutation(len(ug)) % 5)); fold = np.array([gf[x] for x in g])
print(f'locked {len(L)} in {len(ug)} groups | unlocked {len(U)} | dev {dev}', flush=True)

def fit(rows, yy, seed, ep=40):
    torch.manual_seed(seed); Xa = X[rows]; ya = torch.tensor(yy, device=dev)
    m = nn.Sequential(nn.Dropout(0.5), nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ep * ((len(ya) + 255) // 256))
    for _ in range(ep):
        for b in torch.randperm(len(ya), device=dev).split(256):
            loss = F.cross_entropy(m(Xa[b]), ya[b], label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()

def predict(rows, yy, q):
    P = 0
    for s in range(SEEDS):
        m = fit(rows, yy, s)
        with torch.no_grad(): P = P + m(X[q]).softmax(1).cpu().numpy() / SEEDS
    return P

trn = np.where(TR)[0]; out = []; PU = {k: None for k in range(5)}
for r in range(ROUNDS + 1):
    pred = np.zeros(len(L), int); predb = np.zeros(len(L), int); shares, npl = [], []
    for k in range(5):
        keep = fold != k; rows, yy = np.r_[trn, L[keep]], np.r_[LAB[trn], yL[keep]]
        if r > 0:
            sel = PU[k].max(1) >= TAU; rows, yy = np.r_[rows, U[sel]], np.r_[yy, PU[k][sel].argmax(1)]; npl.append(sel.sum())
        P = predict(rows, yy, np.r_[L[~keep], U]); PL, PU[k] = P[:(~keep).sum()], P[(~keep).sum():]
        pred[~keep] = PL.argmax(1); predb[~keep] = (np.log(PL + 1e-9) + B).argmax(1); shares.append(((np.log(PU[k] + 1e-9) + B).argmax(1) == 2).mean())
    row = dict(round=r, heldout_f1=round(f1_score(yL, pred, average='macro'), 4), heldout_f1_b08=round(f1_score(yL, predb, average='macro'), 4),
               unlocked_rusak_b08=round(float(np.mean(shares)), 3), pseudo_rows=int(np.mean(npl)) if npl else 0)
    out.append(row); print(row, flush=True); pd.DataFrame(out).to_csv('work/pl_iter_scores.csv', index=False)
