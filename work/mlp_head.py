# Non-linear head on concatenated frozen tower features (CPU). group90 folds; saves preds for ens.py.
import numpy as np, sys, os, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, 'work'); sys.argv = ['x']
exec(open('work/ens.py').read().split("if __name__")[0])
torch.set_num_threads(6); trm = (meta.split == 'train').values
def feat(t):
    E = np.load(f'work/emb/emb_{t}.npz'); X = sum(E[k].astype(np.float32) for k in E.files); return X / np.linalg.norm(X, axis=1, keepdims=True)
towers = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224'] + [t for t in os.environ.get('EXTRA', '').split(',') if t]
X = torch.tensor(np.concatenate([feat(t) for t in towers], 1)); Xtr, Xte = X[trm], X[~trm]; Y = torch.tensor(y)
def fit(Xa, ya, seed, ep=40):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Dropout(0.5), nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3))
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ep * ((len(ya) + 255) // 256))
    for _ in range(ep):
        for idx in torch.randperm(len(ya)).split(256):
            loss = F.cross_entropy(m(Xa[idx]), ya[idx], label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()
oof = np.zeros((len(y), 3)); tst = np.zeros((len(Xte), 3))
for k in range(5):
    a, b = torch.tensor(fold != k), torch.tensor(fold == k)
    for s in range(3):
        m = fit(Xtr[a], Y[a], s)
        with torch.no_grad(): oof[b.numpy()] += m(Xtr[b]).softmax(1).numpy() / 3; tst += m(Xte).softmax(1).numpy() / 15
    print('fold', k, round(f1(oof[fold == k] if False else oof, slice(None)) if False else 0, 1), flush=True)
def at_share(L, share=0.258):
    lo, hi = -6, 6
    for _ in range(60):
        bb = (lo + hi) / 2; pr = (L + np.r_[0, 0, bb]).argmax(1); lo, hi = (bb, hi) if (pr == 2).mean() < share else (lo, bb)
    return pr
L = np.log(oof + 1e-9); clip_ft = np.log(load('clipL224_g90')[0] + 1e-9); lr6 = np.log(load('cat6_dfn_eva_C2.0')[0] + 1e-9)
print(f'MLP cat6: OOF {f1(oof):.4f} @0.258 {f1(at_share(L)):.4f} | +clipL @0.258 {f1(at_share((L + clip_ft) / 2)):.4f} | +clipL+LR6 @0.258 {f1(at_share((L + clip_ft + lr6) / 3)):.4f}')
tag = os.environ.get('TAG', 'mlp_cat6'); d = f'work/preds/{tag}'; os.makedirs(d, exist_ok=True)
for k in range(5):
    mm = fold == k; np.savez(f'{d}/pred_{tag}_f{k}.npz', val_path=tr.path.values[mm], val=oof[mm], test_path=te.path.values, test=tst)
