# 640x640 source probe (RESULTS 10-04 06:40): from d04_da_bt2, flip R->A only on 640 test rows WITHOUT a pixel-exact train copy,
# where the DA head retrained with the cleanlab-flagged 640 rows pruned (da_cl640) says Aman at the DA CV bias (-0.8). Nothing else changes.
# usage: python work/p640.py TAG OUT_NAME
import sys, numpy as np, pandas as pd
tag, name = sys.argv[1], sys.argv[2]; sys.argv = ['x']; exec(open('work/ens.py').read().split("if __name__")[0])
B = np.r_[0, 0, -0.8]; Lp = np.log(load(tag)[1] + 1e-9) + B; Lr = np.log(load('da_h2')[1] + 1e-9) + B
base = pd.read_csv('submissions/d04_da_bt2.csv').set_index('filename').label.loc[te.filename]
ncc = pd.Series(np.load('work/pix_all.npz')['ncc'].max(1), index=meta.filename).loc[te.filename].values
lock = np.load('work/twin_clean_lock.npy'); locked = np.isin(meta.index[meta.split == 'test'], lock[:, 0])
m = ((te.w == 640) & (te.h == 640)).values & (ncc < 0.99) & (base.values == 'Rusak')
flip = m & (Lp.argmax(1) == 0); print(f'candidates {m.sum()} | pruned head says Aman {flip.sum()} (of which twin-locked {(flip & locked).sum()}) | unpruned da_h2 says Aman {(m & (Lr.argmax(1) == 0)).sum()}')
print('P(R) pruned vs unpruned on candidates (mean):', np.exp(Lp[m] - np.logaddexp.reduce(Lp[m], 1, keepdims=True))[:, 2].mean().round(3), np.exp(Lr[m] - np.logaddexp.reduce(Lr[m], 1, keepdims=True))[:, 2].mean().round(3))
out = base.copy(); out[flip] = 'Aman'; out.reset_index().to_csv(f'submissions/{name}.csv', index=False)
print(name, 'Rusak share', round((out == 'Rusak').mean(), 3), '| changed', (out != base).sum())
