# Per-fold validation report for session-15 7B folds vs F2 (fold 0) and the frozen h2 heads, on the same (cleaned) val rows.
# usage (Datascape/): python work/s15/fold_report.py [dir]
import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import f1_score
D = sys.argv[1] if len(sys.argv) > 1 else 'work/s15/out15'; C = ['Aman', 'Lainnya', 'Rusak']
M = pd.read_csv('work/emb/meta_sorted.csv').set_index('path'); drop = set(pd.read_csv('work/s15/pack/drop15.csv').path)
def src(paths):
    m = M.loc[paths]; w, h = m.w.values, m.h.values; sz = np.char.add(np.char.add(w.astype(str), 'x'), h.astype(str))
    return np.where(np.isin(sz, ['640x640', '512x512', '675x900', '224x224']), sz, np.where((h >= 250) & (h <= 270) & (w > h), 'fam251', 'other'))
rows = []
for f in sorted(glob.glob(f'{D}/pred_F2c_f*.npz')) + ['work/s14/out14/pred_F2_f0.npz']:
    z = np.load(f, allow_pickle=True); vp = pd.Index(z['val_path']); keep = ~vp.isin(drop); vp = vp[keep]; P = z['val'][keep]
    y = M.loc[vp, 'label'].map({c: i for i, c in enumerate(C)}).values; p = P.argmax(1); k = int(f.split('_f')[-1][0])
    h2 = np.load(f'work/preds/h2_peg_c11/pred_h2_peg_c11_f{k}.npz', allow_pickle=True); hp = pd.DataFrame(h2['val'], index=h2['val_path']).loc[vp].values.argmax(1)
    r = dict(model=f.split('/')[-1][:-4], n=len(y), macroF1=f1_score(y, p, average='macro'), h2_macroF1=f1_score(y, hp, average='macro'))
    r.update({f'F1_{c[0]}': v for c, v in zip(C, f1_score(y, p, average=None, labels=[0, 1, 2]))}); r['predR'] = (p == 2).mean(); r['trueR'] = (y == 2).mean()
    s = src(vp)
    for g in ['640x640', 'fam251', 'other']: r[f'acc_{g}'] = (p[s == g] == y[s == g]).mean(); r[f'h2acc_{g}'] = (hp[s == g] == y[s == g]).mean()
    rows.append(r)
pd.set_option('display.width', 250); print(pd.DataFrame(rows).round(4).to_string(index=False))
