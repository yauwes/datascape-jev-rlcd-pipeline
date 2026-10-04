# Ensembling on CLEANED labels (discoveries #26): blend (log-mean), Caruana hill climbing, LR stacking. All are fold-honest
# (weights/stacker fit on 4 group folds, scored on the 5th). Each method also reports F1 vs the twin labels on locked test rows.
# Saves the chosen ensemble's test probabilities to work/preds/<out>/ in ens.py format (val = honest OOF).
# usage: python work/ens2.py OUT_TAG tag1 tag2 ...      (tags under work/preds/<tag>/ or work/colab_out7/pred_<tag>_f*.npz)
import glob, os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
out_tag, tags = sys.argv[1], sys.argv[2:]; sys.argv = ['x']
exec(open('work/ens.py').read().split("if __name__")[0])
yc = pd.read_csv('work/labels_clean_v2.csv').set_index('path').label_clean.loc[tr.path].map({c: i for i, c in enumerate(C)}).values
lock = np.load('work/twin_clean_lock.npy'); M_ = pd.read_csv('work/emb/meta_sorted.csv'); tpos_ = {p: i for i, p in enumerate(te.path)}
li = np.array([tpos_[M_.path.values[r]] for r in lock[:, 0]]); ly = lock[:, 1]

def load2(tag):
    if glob.glob(f'work/preds/{tag}/pred_{tag}_f*.npz'): return load(tag)
    oof = np.full((len(tr), 3), np.nan); tst = np.zeros((len(te), 3)); fs = sorted(glob.glob(f'work/colab_out7/pred_{tag}_f*.npz'))
    for f in fs:
        d = np.load(f, allow_pickle=True); oof[[pos[p] for p in d['val_path']]] = d['val']; tst[[tpos[p] for p in d['test_path']]] += d['test']
    assert len(fs) == 5 and not np.isnan(oof).any(), (tag, len(fs)); return oof, tst / 5

O, T = zip(*[load2(t) for t in tags]); LO = [np.log(o + 1e-9) for o in O]; LT = [np.log(t + 1e-9) for t in T]
F = lambda yy, p: f1_score(yy, p, average='macro')
def report(name, oof_pred, test_L):
    print(f'{name:28s} OOF clean {F(yc, oof_pred):.4f} orig {F(y, oof_pred):.4f} | twin-locked {F(ly, test_L[li].argmax(1)):.4f} | test Rusak {np.mean(test_L.argmax(1) == 2):.3f}', flush=True)
for t, lo, lt in zip(tags, LO, LT): report(t, lo.argmax(1), lt)
mean_o, mean_t = np.mean(LO, 0), np.mean(LT, 0); report('blend: log-mean', mean_o.argmax(1), mean_t)

def hill_c(Ls, m, iters=40):  # Caruana with replacement vs cleaned labels
    w = np.zeros(len(Ls)); cur = -1
    for _ in range(iters):
        sc = [F(yc[m], (sum(wk * L for wk, L in zip(w + np.eye(len(Ls))[j], Ls)))[m].argmax(1)) for j in range(len(Ls))]
        j = int(np.argmax(sc))
        if sc[j] <= cur: break
        cur = sc[j]; w[j] += 1
    return w / w.sum()
hp = np.zeros(len(y), int)
for k in range(5):
    w = hill_c(LO, fold != k); hp[fold == k] = sum(wk * L for wk, L in zip(w, LO))[fold == k].argmax(1)
wall = hill_c(LO, np.ones(len(y), bool)); hill_t = sum(wk * L for wk, L in zip(wall, LT)); report('hill (honest)', hp, hill_t)
print('  hill weights (all folds):', dict(zip(tags, wall.round(3))))

Xo, Xt = np.concatenate(LO, 1), np.concatenate(LT, 1); sp = np.zeros((len(y), 3))
for k in range(5):
    st = LogisticRegression(C=0.3, max_iter=3000).fit(Xo[fold != k], yc[fold != k]); sp[fold == k] = st.predict_proba(Xo[fold == k])
st = LogisticRegression(C=0.3, max_iter=3000).fit(Xo, yc); stack_t = np.log(st.predict_proba(Xt) + 1e-9); report('stack LR (honest)', sp.argmax(1), stack_t)

best = max([('blend', F(yc, mean_o.argmax(1)), np.exp(mean_o), np.exp(mean_t)), ('hill', F(yc, hp), None, np.exp(hill_t)), ('stack', F(yc, sp.argmax(1)), sp, np.exp(stack_t))], key=lambda r: r[1])
oof_save = best[2] if best[2] is not None else np.exp(sum(wk * L for wk, L in zip(wall, LO)))
oof_save = oof_save / oof_save.sum(1, keepdims=True); tst_save = best[3] / best[3].sum(1, keepdims=True); d = f'work/preds/{out_tag}'; os.makedirs(d, exist_ok=True)
for k in range(5):
    mm = fold == k; np.savez(f'{d}/pred_{out_tag}_f{k}.npz', val_path=tr.path.values[mm], val=oof_save[mm], test_path=te.path.values, test=tst_save)
print(f'saved {out_tag} = {best[0]} (honest OOF clean {best[1]:.4f})')
