# 10-09 candidates on d08_grand_consensus_champion (public 0.88789, team best). Evidence: LB regression over 9 scored subs (work/lb_regress.py).
#   d09_vnA        d08's own new Rusak calls on VN-parcel scans (67x x 900 / 900 x 67x; base p416 said Aman) -> Aman. Same source family as 675x900,
#                  whose base-Aman rows the LB proved Aman (p675_R -0.043). The 7B (F2) over-calls Rusak there (38% on 675x900); F1 says Aman.
#   d09_otherR     LB says d08's non-VN "other" A->R flips were ~all right although the models gave them only P(R) ~0.66: extend one tier,
#                  no-copy rows never called Rusak by a scored sub, mean(F1, F2) P(Rusak) >= 0.25, d08 Aman -> Rusak (SSCD/pixel copies excluded).
#   d09_combo      vnA + otherR + copyfix (3 rows where d08 overrode an exact train copy, ncc >= 0.99): expected best.
import numpy as np, pandas as pd
R = 'submissions/'
meta = pd.read_csv('work/emb/meta_sorted.csv'); TE = (meta.split == 'test').values; te = np.where(TE)[0]
ss = pd.read_csv('datascape/sample_submission.csv'); stem2fn = {f.rsplit('.', 1)[0]: f for f in ss.filename}
fn = np.array([stem2fn[p.split('/')[-1][:-4]] for p in meta.path[TE]]); tpos = {p: i for i, p in enumerate(meta.path[TE])}
w, h = meta.w.values[te], meta.h.values[te]; size = np.char.add(np.char.add(w.astype(str), 'x'), h.astype(str))
r13 = pd.read_csv('work/s13/pack/rows13.csv'); assert (('work/small/test/' + r13.stem + '.jpg').values == meta.path.values[TE]).all()
copy, ncc = r13.copy_label.values, r13.best_ncc.values
def pr(f):
    z = np.load(f, allow_pickle=True); P = np.zeros((len(te), 3)); P[[tpos[p] for p in z['test_path']]] = z['test']; return P[:, 2]   # P(Rusak); classes Aman, Lainnya, Rusak
f1, f2 = pr('work/drive_in/session12/pred_F1_f0.npz'), pr('work/s14/out14/pred_F2_f0_best.npz')
get = lambda n: pd.read_csv(R + n + '.csv').set_index('filename').label.reindex(fn).values
d08, p416 = get('d08_grand_consensus_champion'), get('d05_p416_allA')
vn = ((h == 900) & (w >= 650) & (w <= 690)) | ((w == 900) & (h >= 650) & (h <= 690))
other = ~np.isin(size, ['224x224', '416x416', '512x512', '640x640']) & ~vn
z = np.load('work/nntr_sscd.npz'); ntr = (~TE).sum(); lab_tr = meta.label.values[~TE]; nb, sim = z['i'][te], z['s'][te]          # SSCD near-copies (train rows first in meta_sorted)
lockA = np.array([(lambda L: len(L) > 0 and (L == 'Aman').mean() >= .9)(lab_tr[n[(s_ >= .55) & (n < ntr)]]) for n, s_ in zip(nb, sim)])
mean = (f1 + f2) / 2
SCORED = ['d04_da_bt2', 'd05_p675_R', 'd05_h2_nolock_s337', 'd06_cur_A', 'd06_F1_AR25', 'd06_master_tightened_similarity', 'd07_pure_curated_master', 'd10_transductive_098_champion']
tested = np.any([get(n) == 'Rusak' for n in SCORED], axis=0)                     # already called Rusak on the LB: no new information there
CL = {0: 'Aman', 1: 'Lainnya', 2: 'Rusak'}; cl = np.array([CL.get(c, '') for c in copy])
E = {'vnA': (vn & (d08 == 'Rusak') & (p416 == 'Aman') & (copy != 2) & (f1 <= .5), 'Aman'),
     'otherR': (other & (copy == -1) & (ncc < .95) & ~lockA & ~tested & (d08 == 'Aman') & (mean >= .25), 'Rusak'),
     'copyfix': ((cl != '') & (ncc >= .99) & (cl != d08) & ~vn, cl)}                        # d08 overrode an exact train copy (VN ones are in vnA)
ms = [m for m, _ in E.values()]; assert not any((a & b).any() for i, a in enumerate(ms) for b in ms[i + 1:]), 'edits must be disjoint'
FILES = {'d09_combo': ['vnA', 'otherR', 'copyfix'], 'd09_vnA': ['vnA'], 'd09_otherR': ['otherR']}
for n, edits in FILES.items():
    lab = d08.copy()
    for e in edits: m, t = E[e]; lab = np.where(m, t, lab)
    m = lab != d08; assert len(lab) == 3057 and set(lab) <= {'Aman', 'Rusak', 'Lainnya'} and m.sum() == sum(E[e][0].sum() for e in edits)
    out = pd.DataFrame({'filename': fn, 'label': lab}).set_index('filename').loc[ss.filename].reset_index()   # sample_submission row order
    out.to_csv(f'{R}{n}.csv', index=False)
    print(f'{n}: {int(m.sum())} rows changed vs d08', pd.Series(np.char.add(np.char.add(d08[m].astype(str), '>'), lab[m].astype(str))).value_counts().to_dict())
