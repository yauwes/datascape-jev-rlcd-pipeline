# Domain-adapted heads (discoveries #23): same MLP recipe as work/mlp_head.py, trained on ALL train rows + the twin-locked test rows
# (SSCD >= 0.55, purity >= 0.9, clean_other; twin labels ~98% right), so the head sees the test-domain shift.
# PRUNE=<csv with path column> drops those train rows (cleanlab-pruned retrain), TAG=<out tag> (default da_h2).
# Output: work/preds/da_h2/pred_da_h2_f{k}.npz in ens.py format. val = the h2 group-OOF (unchanged, not a DA estimate); test = geometric
# mean of the two DA heads (cat6+PE-G, cat11; 5 seeds each). Also prints the CV-calibrated Rusak bias to use on unlocked rows.
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, 'work'); sys.argv = ['x']
from twin import M, TR, ST, LAB, vote
torch.set_num_threads(8); dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'; os.makedirs('work/models_da', exist_ok=True)
CAT6 = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224']
GIANT = ['vit_pe_core_gigantic_patch14_448', 'eva02_enormous_patch14_clip_224', 'vit_huge_patch14_clip_378', 'vit_giantopt_patch16_siglip_384', 'vit_gigantic_patch14_clip_378']
def feat(t):
    E = np.load(f'work/emb/emb_{t}.npz'); X = sum(E[k].astype(np.float32) for k in E.files); return X / np.linalg.norm(X, axis=1, keepdims=True)
te = np.where(~TR)[0]; D = np.load('work/nntr_sscd.npz'); lab, pur = vote(D['i'][te], D['s'][te], 0.55)
m_ = (lab >= 0) & (pur >= 0.9) & (ST[te] == 'clean_other'); lk, ylk = te[m_], lab[m_]
TAG = os.environ.get('TAG', 'da_h2'); keep = TR & ~np.isin(M.path.values, pd.read_csv(os.environ['PRUNE']).path) if os.environ.get('PRUNE') else TR
idx = np.r_[np.where(keep)[0], lk]; y = torch.tensor(np.r_[LAB[keep], ylk], device=dev); print('train rows', keep.sum(), '+ locked', len(lk), flush=True)

def fit(X, seed, ep=40):
    torch.manual_seed(seed); Xa = X[idx]
    m = nn.Sequential(nn.Dropout(0.5), nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ep * ((len(y) + 255) // 256))
    for _ in range(ep):
        for b in torch.randperm(len(y), device=dev).split(256):
            loss = F.cross_entropy(m(Xa[b]), y[b], label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()

L = []
for towers in [CAT6 + GIANT[:1], CAT6 + GIANT]:
    X = torch.tensor(np.concatenate([feat(t) for t in towers], 1), device=dev); P = 0
    for s in range(5):
        m = fit(X, s); torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f'work/models_da/{"" if TAG == "da_h2" else TAG + "_"}head_{len(towers)}t_s{s}.pt')
        with torch.no_grad(): P = P + m(X[te]).softmax(1).cpu().numpy() / 5
        print(len(towers), 'towers seed', s, flush=True)
    L.append(np.log(P + 1e-9))
t = np.exp(np.mean(L, 0)); t /= t.sum(1, keepdims=True)
exec(open('work/ens.py').read().split("if __name__")[0], g := {'__name__': 'x'})
o, _ = g['load']('h2_peg_c11'); assert (g['te'].path.values == M.path.values[te]).all()
d = f'work/preds/{TAG}'; os.makedirs(d, exist_ok=True)
for k in range(5):
    mm = g['fold'] == k; np.savez(f'{d}/pred_{TAG}_f{k}.npz', val_path=g['tr'].path.values[mm], val=o[mm], test_path=g['te'].path.values, test=t)
print('saved', TAG, ' | test Rusak share raw', round(float((t.argmax(1) == 2).mean()), 3), '| on locked rows (in-sample)', round(float((t[m_].argmax(1) == 2).mean()), 3), 'true', round(float((ylk == 2).mean()), 3))
