# Clusters = union-find over (DINOv2 max-variant sim > THR) edges + exact MD5 matches, across train AND test.
# Writes work/clusters.csv (all rows) and work/folds.csv (train rows, StratifiedGroupKFold 5, seed 42).
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
THR = float(sys.argv[1])
meta = pd.read_csv('work/emb/meta_sorted.csv'); md5 = pd.read_csv('work/meta_md5.csv').set_index('path').md5
nn = np.load('work/nn_dinob.npz'); I, S = nn['i'], nn['s']
par = np.arange(len(meta))
def find(a):
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
for a, b in zip(*np.nonzero(S > THR)): ra, rb = find(a), find(I[a, b]); par[ra] = rb
first = {}
for i, h in enumerate(md5.loc[meta.path].values):
    if h in first: ra, rb = find(i), find(first[h]); par[ra] = rb
    else: first[h] = i
meta['cluster'] = [find(i) for i in range(len(meta))]
meta[['path', 'split', 'label', 'cluster']].to_csv('work/clusters.csv', index=False)
tr = meta[meta.split == 'train']
sz = tr.groupby('cluster').size(); conf = tr.groupby('cluster').label.nunique()
print(f'THR {THR}: train clusters {len(sz)} (multi {int((sz > 1).sum())}, rows in multi {int(sz[sz > 1].sum())}), conflicting {int((conf > 1).sum())} clusters / {int(sz[conf > 1].sum())} rows')
te = meta[meta.split == 'test']; trc = set(tr.cluster)
print(f'test rows sharing a cluster with train: {te.cluster.isin(trc).mean():.3f}; largest cluster {sz.max()}')
f = np.zeros(len(tr), int)
for k, (_, v) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, tr.label, tr.cluster)): f[v] = k
pd.DataFrame({'path': tr.path.values, 'fold': f}).to_csv('work/folds.csv', index=False)
print(pd.crosstab(f, tr.label.values))
