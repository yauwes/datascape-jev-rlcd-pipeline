# 10-09 slot 2 candidate: d12_7b5 = d11_lbopt (public 0.88891) with d08's blend recomputed on a 5-fold DINOv3-7B
# (F2 = fold 0 from session 14, F2c folds 1-4 from session 15: same recipe + LB-backed train cleaning, work/s15/pack/drop15.csv).
# Blend = d08 weights: .35 7B + .30 r_cat11 + .20 ens_pe_heads + .15 da_h2. Only "free" rows may change: no pixel/SSCD train copy,
# not 416x416 / 675x900 / VN-parcel sizes (LB-locked), not one of d11's 17 regression edits. Everything else keeps d11's label.
# usage (Datascape/): python work/make_d12.py [dir with pred_F2c_f1..4.npz]   -> submissions/d12_7b5.csv + a change table
import sys, glob, numpy as np, pandas as pd
exec(open('work/make_d09.py').read().split("ms = [m for m")[0])        # meta/test order, fn, size, vn, copy, ncc, S, d08 ...
F2C = sys.argv[1] if len(sys.argv) > 1 else 'work/s15/out15'
def avg(files):
    P = np.zeros((len(te), 3))
    for f in files:
        z = np.load(f, allow_pickle=True); Q = np.zeros((len(te), 3)); Q[[tpos[p] for p in z['test_path']]] = z['test']; P += Q
    return P / len(files)
f2c = sorted(glob.glob(f'{F2C}/pred_F2c_f[1-4].npz')); assert len(f2c) == 4, f'need 4 fold files in {F2C}, found {len(f2c)}'
P7_old = avg(['work/s14/out14/pred_F2_f0_best.npz']); P7_new = (P7_old + 4 * avg(f2c)) / 5
rest = .30 * avg(sorted(glob.glob('work/artifacts/preds/ft/r_cat11/pred_r_cat11_f*.npz'))) + .20 * avg(sorted(glob.glob('work/preds/ens_pe_heads/pred_ens_pe_heads_f*.npz'))) \
     + .15 * avg(sorted(glob.glob('work/preds/da_h2/pred_da_h2_f*.npz')))
C = np.array(['Aman', 'Lainnya', 'Rusak']); old, new = C[(.35 * P7_old + rest).argmax(1)], C[(.35 * P7_new + rest).argmax(1)]
d11 = get('d11_lbopt'); ncc_all = np.load('work/pix_all.npz')['ncc'][te]
free = (copy == -1) & (S[:, 0] < .9) & (ncc_all.max(1) < .95) & ~np.isin(size, ['416x416', '675x900']) & ~vn & (d11 == d08) & (old == d08)
lab = np.where(free, new, d11); ch = lab != d11
assert len(lab) == 3057 and set(lab) <= set(C)
print(f'free rows {free.sum()} | d12_7b5 changes vs d11: {ch.sum()}', pd.Series([f'{a[0]}->{b[0]}' for a, b in zip(d11[ch], lab[ch])]).value_counts().to_dict())
fam = (h >= 250) & (h <= 270) & (w > h); src = np.where(size == '640x640', '640', np.where(size == '512x512', '512', np.where(fam, 'fam251', np.where(size == '224x224', '224', 'other'))))
print(pd.crosstab(src[ch], [d11[ch], lab[ch]]).to_string())
pd.DataFrame({'filename': fn, 'label': lab}).set_index('filename').loc[ss.filename].reset_index().to_csv('submissions/d12_7b5.csv', index=False)
pd.DataFrame(dict(filename=fn[ch], src=src[ch], d11=d11[ch], d12=lab[ch], p7_old=(P7_old[ch, 2]).round(3), p7_new=(P7_new[ch, 2]).round(3))).to_csv('work/s15/d12_changes.csv', index=False)
