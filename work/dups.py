# Near-duplicate structure from DINOv2-B embeddings: top-k neighbours (max over geometric variants), clusters, label conflicts.
import numpy as np, pandas as pd, sys
E = np.load('work/emb/emb_vit_base_patch14_dinov2.npz')
meta = pd.read_csv('work/emb/meta_sorted.csv')
V = {k: E[k].astype(np.float32) for k in E.files}
for k in V: V[k] /= np.linalg.norm(V[k], axis=1, keepdims=True)
N = len(meta); K = 10
nn_i = np.zeros((N, K), np.int32); nn_s = np.zeros((N, K), np.float32)
for a in range(0, N, 1500):
    q = V['orig'][a:a + 1500]
    s = np.max(np.stack([q @ V[k].T for k in V]), 0)
    s[np.arange(len(q)), np.arange(a, a + len(q))] = -1
    idx = np.argpartition(-s, K, axis=1)[:, :K]
    ss = np.take_along_axis(s, idx, 1); o = np.argsort(-ss, 1)
    nn_i[a:a + len(q)] = np.take_along_axis(idx, o, 1); nn_s[a:a + len(q)] = np.take_along_axis(ss, o, 1)
np.savez('work/nn_dinob.npz', i=nn_i, s=nn_s)
is_tr = (meta.split == 'train').values
# nearest train neighbour for every image
tr_best = np.where(is_tr[nn_i], nn_s, -1).max(1)
te = ~is_tr
print('test -> nearest-train sim quantiles', np.quantile(tr_best[te], [.05, .1, .25, .5, .75, .9]).round(3))
for thr in (0.80, 0.85, 0.90, 0.95):
    sm = (meta.w == 224) & (meta.h == 224)
    print(f'thr {thr}: test with train-dup {np.mean(tr_best[te] > thr):.3f} | 224px {np.mean(tr_best[te & sm.values] > thr):.3f} (n={int((te & sm.values).sum())}) | other {np.mean(tr_best[te & ~sm.values] > thr):.3f}')
