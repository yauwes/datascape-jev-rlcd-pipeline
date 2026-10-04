# Rank frozen vision towers (A2): group90 LR probe per tower -> OOF, OOF@matched share, corr with current blend, ensemble gain.
# Saves each as work/preds/<tower>_lr_g90/ so ens.py can use it. usage: python work/towers.py [base_tags...]
import glob, os, sys, numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, 'work'); argv = sys.argv[1:] or ['clipL224_g90', 'lrprobe_g90']; sys.argv = ['x']
exec(open('work/ens.py').read().split("if __name__")[0])
trm = (meta.split == 'train').values
def at_share(L, share=0.258):
    lo, hi = -6, 6
    for _ in range(60):
        b = (lo + hi) / 2; pr = (L + np.r_[0, 0, b]).argmax(1); lo, hi = (b, hi) if (pr == 2).mean() < share else (lo, b)
    return pr
base = np.mean([np.log(load(t)[0] + 1e-9) for t in argv], 0); b0 = f1(at_share(base))
print(f'base {argv} OOF@0.258 {b0:.4f}')
rows = []
for f in sorted(glob.glob('work/emb/emb_*.npz')):
    tag = os.path.basename(f)[4:-4]; E = np.load(f)
    X = sum(E[k].astype(np.float32) for k in E.files); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    out = f'work/preds/{tag}_lr_g90'
    if not os.path.exists(f'{out}/pred_{tag}_lr_g90_f4.npz'):
        os.makedirs(out, exist_ok=True); tst = np.zeros(((~trm).sum(), 3)); oof = np.zeros((len(y), 3))
        for k in range(5):
            m = LogisticRegression(C=0.5, max_iter=3000).fit(X[trm][fold != k], y[fold != k]); oof[fold == k] = m.predict_proba(X[trm][fold == k]); tst += m.predict_proba(X[~trm]) / 5
        for k in range(5):
            mm = fold == k; np.savez(f'{out}/pred_{tag}_lr_g90_f{k}.npz', val_path=tr.path.values[mm], val=oof[mm], test_path=te.path.values, test=tst)
    o, t = load(f'{tag}_lr_g90'); L = np.log(o + 1e-9)
    gain = f1(at_share((base * len(argv) + L) / (len(argv) + 1))) - b0
    rows.append((tag, X.shape[1], round(f1(o), 4), round(f1(at_share(L)), 4), round(np.corrcoef(base.ravel(), L.ravel())[0, 1], 3), round(gain, 4), round((t.argmax(1) == 2).mean(), 3)))
import pandas as pd
r = pd.DataFrame(rows, columns=['tower', 'dim', 'oof_argmax', 'oof@0.258', 'corr_base', 'ens_gain@0.258', 'test_rusak']).sort_values('ens_gain@0.258', ascending=False)
print(r.to_string(index=False)); r.to_csv('work/towers_rank.csv', index=False)
