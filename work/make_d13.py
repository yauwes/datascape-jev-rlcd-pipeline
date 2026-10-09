# Candidate builder after sessions 16/17: d11_lbopt (public 0.88891) with d08's blend recomputed on a NEW 7B component =
# mean of the given 7B prediction files (e.g. F2 fold 0 + session-17 folds 1-4). Blend weights = d08's (.35 7B + .30 r_cat11
# + .20 ens_pe_heads + .15 da_h2). Only free rows may change: no pixel/SSCD train copy, not 416/675/VN sizes, not a d11 edit.
# usage (Datascape/): python work/make_d13.py NAME file1.npz [file2.npz ...]   -> submissions/NAME.csv + work/NAME_changes.csv
import sys, glob, numpy as np, pandas as pd
exec(open('work/make_d09.py').read().split("ms = [m for m")[0])        # meta/test order, fn, size, vn, copy, ncc, S, d08 ...
NAME, FILES = sys.argv[1], sys.argv[2:]; assert FILES, 'give the 7B prediction files'
def avg(files):
    P = np.zeros((len(te), 3))
    for f in files:
        z = np.load(f, allow_pickle=True); Q = np.zeros((len(te), 3)); Q[[tpos[p] for p in z['test_path']]] = z['test']; P += Q
    return P / len(files)
P7_old, P7_new = avg(['work/s14/out14/pred_F2_f0_best.npz']), avg(FILES)
rest = .30 * avg(sorted(glob.glob('work/artifacts/preds/ft/r_cat11/pred_r_cat11_f*.npz'))) + .20 * avg(sorted(glob.glob('work/preds/ens_pe_heads/pred_ens_pe_heads_f*.npz'))) \
     + .15 * avg(sorted(glob.glob('work/preds/da_h2/pred_da_h2_f*.npz')))
C = np.array(['Aman', 'Lainnya', 'Rusak']); old, new = C[(.35 * P7_old + rest).argmax(1)], C[(.35 * P7_new + rest).argmax(1)]
d11 = get('d11_lbopt'); ncc_all = np.load('work/pix_all.npz')['ncc'][te]
free = (copy == -1) & (S[:, 0] < .9) & (ncc_all.max(1) < .95) & ~np.isin(size, ['416x416', '675x900']) & ~vn & (d11 == d08) & (old == d08)
lab = np.where(free, new, d11); ch = lab != d11
assert len(lab) == 3057 and set(lab) <= set(C)
fam = (h >= 250) & (h <= 270) & (w > h); src = np.where(size == '640x640', '640', np.where(size == '512x512', '512', np.where(fam, 'fam251', np.where(size == '224x224', '224', 'other'))))
print(f'{NAME}: 7B files {len(FILES)} | free rows {free.sum()} | changes vs d11 {ch.sum()}', pd.Series([f'{a[0]}->{b[0]}' for a, b in zip(d11[ch], lab[ch])]).value_counts().to_dict())
print(pd.crosstab(src[ch], [d11[ch], lab[ch]]).to_string() if ch.any() else 'no changes')
pd.DataFrame({'filename': fn, 'label': lab}).set_index('filename').loc[ss.filename].reset_index().to_csv(f'submissions/{NAME}.csv', index=False)
pd.DataFrame(dict(filename=fn[ch], src=src[ch], d11=d11[ch], new=lab[ch], p7_old_R=P7_old[ch, 2].round(3), p7_new_R=P7_new[ch, 2].round(3))).to_csv(f'work/{NAME}_changes.csv', index=False)
