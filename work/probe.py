# Linear probe per frozen backbone on fixed group folds (work/folds.csv). Reports OOF macro-F1; saves OOF/test probs.
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
C = ['Aman', 'Lainnya', 'Rusak']
meta = pd.read_csv('work/emb/meta_sorted.csv'); folds = pd.read_csv('work/folds.csv')
tr = (meta.split == 'train').values
y = meta.label[tr].map({c: i for i, c in enumerate(C)}).values
f = folds.set_index('path').loc[meta.path[tr], 'fold'].values
def feats(tag):
    E = np.load(f'work/emb/emb_{tag}.npz'); X = E['orig'].astype(np.float32) + E['hflip'].astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)
tags = sys.argv[1:]
X = np.concatenate([feats(t) for t in tags], 1)
Xtr, Xte = X[tr], X[~tr]
for Cr in (0.5, 2.0):
    oof = np.zeros((len(y), 3)); pte = np.zeros((len(Xte), 3))
    for k in range(5):
        m = LogisticRegression(C=Cr, max_iter=3000, class_weight='balanced').fit(Xtr[f != k], y[f != k])
        oof[f == k] = m.predict_proba(Xtr[f == k]); pte += m.predict_proba(Xte) / 5
    per = f1_score(y, oof.argmax(1), average=None)
    print('+'.join(tags), f'C={Cr}', f'macroF1={per.mean():.4f}', dict(zip(C, per.round(4))), flush=True)
    np.savez(f'work/probe_{"+".join(tags)}_C{Cr}.npz', oof=oof, test=pte)
