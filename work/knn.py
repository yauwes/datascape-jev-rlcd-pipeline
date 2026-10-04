# A1: kNN label transfer on frozen embeddings (max over geometric variants), OOF on frozen folds; hybrid with probe.
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
C = ['Aman', 'Lainnya', 'Rusak']
meta = pd.read_csv('work/emb/meta_sorted.csv'); tr = np.nonzero((meta.split == 'train').values)[0]
y = meta.label.values[tr]; y = np.array([C.index(v) for v in y])
f = pd.read_csv('work/folds.csv').set_index('path').loc[meta.path.values[tr], 'fold'].values
def load(tag):
    E = np.load(f'work/emb/emb_{tag}.npz'); V = [E[k].astype(np.float32) for k in E.files]
    return [v / np.linalg.norm(v, axis=1, keepdims=True) for v in V]
tags = sys.argv[1:] or ['vit_base_patch14_dinov2']
Vs = [load(t) for t in tags]
def sims(rows, pool):  # average over backbones of max-over-variants cosine
    return np.mean([np.max(np.stack([V[0][rows] @ v[pool].T for v in V]), 0) for V in Vs], 0)
K = 10; oof_s = np.zeros((len(tr), K)); oof_l = np.zeros((len(tr), K), int)
for k in range(5):
    va, trn = np.nonzero(f == k)[0], np.nonzero(f != k)[0]
    for a in range(0, len(va), 1000):
        s = sims(tr[va[a:a + 1000]], tr[trn]); idx = np.argsort(-s, 1)[:, :K]
        oof_s[va[a:a + 1000]] = np.take_along_axis(s, idx, 1); oof_l[va[a:a + 1000]] = y[trn][idx]
np.savez(f'work/knn_{"+".join(tags)}.npz', s=oof_s, l=oof_l)
print('1-NN macroF1', round(f1_score(y, oof_l[:, 0], average='macro'), 4))
for kk in (3, 5):
    vote = np.zeros((len(y), 3))
    for j in range(kk): np.add.at(vote, (np.arange(len(y)), oof_l[:, j]), np.exp(20 * oof_s[:, j]))
    print(f'{kk}-NN weighted macroF1', round(f1_score(y, vote.argmax(1), average='macro'), 4))
pr = np.load(sys.argv[-1] if False else 'work/probe_vit_large_patch14_dinov2+vit_large_patch14_clip_224_C2.0.npz')['oof']
for t in (0.85, 0.9, 0.93, 0.95, 0.97):
    h = np.where(oof_s[:, 0] > t, oof_l[:, 0], pr.argmax(1))
    m = oof_s[:, 0] > t
    print(f'hybrid t={t}: covered {m.mean():.3f}, 1NN acc on covered {np.mean(oof_l[m, 0] == y[m]):.4f}, probe acc on covered {np.mean(pr[m].argmax(1) == y[m]):.4f}, macroF1 {f1_score(y, h, average="macro"):.4f}')
