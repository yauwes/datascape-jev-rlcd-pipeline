# Confident-learning style label cleaning (Northcutt 2021) from group-held-out OOF probs, iterated.
# Flag row labelled i if argmax j != i and P(j|x) >= t_j (t_j = mean self-confidence of class j). Relabel to j.
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
C = ['Aman', 'Lainnya', 'Rusak']
meta = pd.read_csv('work/emb/meta_sorted.csv'); trm = (meta.split == 'train').values
y0 = np.array([C.index(v) for v in meta.label.values[trm]])
def feats(t):
    E = np.load(f'work/emb/emb_{t}.npz'); X = E['orig'].astype(np.float32) + E['hflip'].astype(np.float32); return X / np.linalg.norm(X, axis=1, keepdims=True)
Xall = np.concatenate([feats('vit_large_patch14_dinov2'), feats('vit_large_patch14_clip_224')], 1); X, Xte = Xall[trm], Xall[~trm]
f = pd.read_csv('work/folds_group90.csv').set_index('path').loc[meta.path.values[trm], 'fold'].values
y = y0.copy()
for it in range(3):
    oof = np.zeros((len(y), 3))
    for k in range(5):
        m = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(X[f != k], y[f != k]); oof[f == k] = m.predict_proba(X[f == k])
    t = np.array([oof[y == j, j].mean() for j in range(3)])
    am = oof.argmax(1); flag = (am != y) & (oof[np.arange(len(y)), am] >= t[am])
    print(f'iter {it}: groupF1 vs current labels {f1_score(y, am, average="macro"):.4f} | vs original {f1_score(y0, am, average="macro"):.4f} | thresholds {t.round(3)} | flagged {flag.sum()}')
    y = np.where(flag, am, y)
print('changed vs original', (y != y0).sum(), pd.crosstab(np.array(C)[y0], np.array(C)[y], rownames=['orig'], colnames=['clean']))
pd.DataFrame({'path': meta.path.values[trm], 'label_orig': np.array(C)[y0], 'label_clean': np.array(C)[y]}).to_csv('work/labels_clean_v1.csv', index=False)
np.save('work/oof_clean_v1.npy', oof)
