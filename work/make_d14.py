# 10-09 long-shot probe (user OK): d14_640A = d11_lbopt (0.88891) with every 640x640 test row that has NO train copy
# (no SSCD twin >= .9, no dihedral pixel NCC >= .95, no copy lock) and d11 = Rusak flipped to Aman. 87 rows.
# Hypothesis: like 675x900 (LB-proven), these are "new" images of a family that train labels Rusak; train 640 singletons
# are 80% Aman, while 95.7% of train 640 Rusak have augmented siblings. Against it: models AND kNN vote say Rusak (kNN 0.77).
# Prior ~5%. Usage (Datascape/ or repo root): python work/make_d14.py
import numpy as np, pandas as pd
exec(open('work/make_d09.py').read().split("ms = [m for m")[0])        # fn, size, copy, S, te, get(), ss ...
d11 = get('d11_lbopt'); ncc_max = np.load('work/pix_all.npz')['ncc'][te].max(1)
flip = (size == '640x640') & (copy == -1) & (S[:, 0] < .9) & (ncc_max < .95) & (d11 == 'Rusak')
assert flip.sum() == 87, flip.sum()
lab = np.where(flip, 'Aman', d11)
out = pd.DataFrame({'filename': fn, 'label': lab}).set_index('filename').loc[ss.filename].reset_index()
assert len(out) == 3057 and set(out.label) <= {'Aman', 'Rusak', 'Lainnya'}
out.to_csv('submissions/d14_640A.csv', index=False); print('d14_640A:', int(flip.sum()), 'rows Rusak -> Aman vs d11')
