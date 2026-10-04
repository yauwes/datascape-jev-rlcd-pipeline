# Ensemble + decision rule on group90 OOF. Everything fitted is fold-honest: fit on 4 folds' OOF, score the 5th.
# usage: python ens.py tagA tagB ...   (expects work/preds/<tag>/pred_<tag>_f{0..4}.npz)
import os, sys, glob, numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score
C = ['Aman', 'Lainnya', 'Rusak']
meta = pd.read_csv('work/emb/meta_sorted.csv'); tr = meta[meta.split == 'train'].reset_index(drop=True); te = meta[meta.split == 'test'].reset_index(drop=True)
y = tr.label.map({c: i for i, c in enumerate(C)}).values
fold = pd.read_csv(os.environ.get('FOLDS', 'work/folds_group90.csv')).set_index('path').loc[tr.path, 'fold'].values
pos = {p: i for i, p in enumerate(tr.path)}; tpos = {p: i for i, p in enumerate(te.path)}

def load(tag):
    oof = np.full((len(tr), 3), np.nan); tst = np.zeros((len(te), 3)); n = 0
    for f in sorted(glob.glob(f'work/preds/{tag}/pred_{tag}_f*.npz')):
        d = np.load(f, allow_pickle=True)
        oof[[pos[p] for p in d['val_path']]] = d['val']; tst[[tpos[p] for p in d['test_path']]] += d['test']; n += 1
    assert n == 5 and not np.isnan(oof).any(), (tag, n)
    return oof, tst / n

def f1(p, m=slice(None)): return f1_score(y[m], p[m].argmax(1) if p.ndim == 2 else p[m], average='macro')

def fit_bias(L, m):
    obj = lambda b: -f1_score(y[m], (L[m] + np.r_[0, b]).argmax(1), average='macro')
    return min((minimize(obj, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-5)) for x0 in ([0, 0], [.5, .5], [-.5, .5], [.5, -.5])), key=lambda r: r.fun).x

def hill(P, m, iters=30):  # Caruana 2004 ensemble selection with replacement, on log-probs
    w = np.zeros(len(P)); cur = None
    for _ in range(iters):
        best = None
        for j in range(len(P)):
            w2 = w.copy(); w2[j] += 1; L = sum(wk * np.log(Pk + 1e-9) for wk, Pk in zip(w2, P)) / w2.sum()
            s = f1(L, m)
            if best is None or s > best[0]: best = (s, j)
        if cur is not None and best[0] <= cur: break
        cur = best[0]; w[best[1]] += 1
    return w / w.sum()

def honest(P):
    """fold-honest: weights + bias fitted on 4 folds, applied to held-out fold"""
    pred = np.zeros(len(y), int)
    for k in range(5):
        a, b = fold != k, fold == k
        w = hill(P, a); L = sum(wk * np.log(Pk + 1e-9) for wk, Pk in zip(w, P))
        pred[b] = (L[b] + np.r_[0, fit_bias(L, a)]).argmax(1)
    return pred

if __name__ == '__main__':
    tags = sys.argv[1:]; O, T = zip(*[load(t) for t in tags])
    for t, o in zip(tags, O):
        print(f'{t:20s} argmax {f1(o):.4f}  per-class {f1_score(y, o.argmax(1), average=None).round(3)}')
    mean = np.mean([np.log(o + 1e-9) for o in O], 0); print(f'{"logit-mean":20s} argmax {f1(mean):.4f}')
    hp = honest(list(O)); print(f'{"hill+bias (honest)":20s} {f1(hp):.4f}  per-class {f1_score(y, hp, average=None).round(3)}')
    # final fit on all folds for test
    w = hill(list(O), slice(None)); L = sum(wk * np.log(o + 1e-9) for wk, o in zip(w, O)); b = fit_bias(L, np.ones(len(y), bool))
    Lt = sum(wk * np.log(t + 1e-9) for wk, t in zip(w, T)) + np.r_[0, b]
    print('weights', dict(zip(tags, w.round(3))), 'bias', b.round(3), 'test pred mix', (np.bincount(Lt.argmax(1), minlength=3) / len(Lt)).round(3))
    np.save('work/ens_test_logits.npy', Lt)
