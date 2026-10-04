# GPU MLP head on concatenated frozen-tower features. Same recipe as work/mlp_head.py:
#   Dropout .5 -> Linear 512 -> GELU -> Dropout .3 -> Linear 3; AdamW 1e-3, wd 1e-2, OneCycle 40 ep, LS 0.1.
# Takes any fold file and N seeds. Writes $DS_OUT/preds/<tag>/pred_<tag>_f<k>.npz in ens.py format and one line to heads_scores.csv.
# usage: python heads.py TAG FOLDS_CSV SEEDS tower1,tower2,...
import glob, os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
tag, folds_csv, seeds, towers = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(',')
IN, OUT = os.environ['DS_IN'], os.environ['DS_OUT']; dev = 'cuda' if torch.cuda.is_available() else 'cpu'
def find(n): return glob.glob(f'{IN}/**/{n}', recursive=True)[0]
meta = pd.read_csv(find('meta_sorted.csv')); trm = (meta.split == 'train').values; C = ['Aman', 'Lainnya', 'Rusak']
tr, te = meta[trm], meta[~trm]; y = tr.label.map({c: i for i, c in enumerate(C)}).values
fold = pd.read_csv(find(folds_csv)).set_index('path').loc[tr.path, 'fold'].values

def feat(t):
    E = np.load(find(f'emb_{t}.npz')); X = sum(E[k].astype(np.float32) for k in E.files); return X / np.linalg.norm(X, axis=1, keepdims=True)
X = torch.tensor(np.concatenate([feat(t) for t in towers], 1), device=dev); Xtr, Xte = X[trm], X[~trm]; Y = torch.tensor(y, device=dev)

def fit(Xa, ya, seed, ep=40, bs=256):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Dropout(0.5), nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ep * ((len(ya) + bs - 1) // bs))
    for _ in range(ep):
        for idx in torch.randperm(len(ya), device=dev).split(bs):
            loss = F.cross_entropy(m(Xa[idx]), ya[idx], label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()

def at_share(P, share=0.258):  # bisect the Rusak logit bias to a fixed predicted share (as work/towers.py)
    L, lo, hi = np.log(P + 1e-9), -6, 6
    for _ in range(60):
        b = (lo + hi) / 2; pr = (L + np.r_[0, 0, b]).argmax(1); lo, hi = (b, hi) if (pr == 2).mean() < share else (lo, b)
    return pr

t0 = time.time(); oof = np.zeros((len(y), 3)); tst = np.zeros((len(Xte), 3))
for k in range(5):
    a, b = torch.tensor(fold != k, device=dev), torch.tensor(fold == k, device=dev)
    for s in range(seeds):
        m = fit(Xtr[a], Y[a], s)
        with torch.no_grad(): oof[fold == k] += m(Xtr[b]).softmax(1).cpu().numpy() / seeds; tst += m(Xte).softmax(1).cpu().numpy() / (5 * seeds)
d = f'{OUT}/preds/{tag}'; os.makedirs(d, exist_ok=True)
for k in range(5):
    mm = fold == k; np.savez(f'{d}/pred_{tag}_f{k}.npz', val_path=tr.path.values[mm], val=oof[mm], test_path=te.path.values, test=tst)
r = dict(tag=tag, folds=folds_csv, seeds=seeds, dim=X.shape[1], towers='+'.join(towers), oof=round(f1_score(y, oof.argmax(1), average='macro'), 4),
         oof_at_258=round(f1_score(y, at_share(oof), average='macro'), 4), test_rusak=round(float((tst.argmax(1) == 2).mean()), 3), sec=round(time.time() - t0))
pd.DataFrame([r]).to_csv(f'{OUT}/heads_scores.csv', mode='a', header=not os.path.exists(f'{OUT}/heads_scores.csv'), index=False); print(r, flush=True)
