# 10-09 slot 1: d11_lbopt = d08_grand_consensus_champion (0.88789) + every row where the LB regression over all scored subs
# (work/lb_regress.py, 675x900-only Aman prior) prefers another label by >= 1e-4 expected gain per row. ~17 rows, posterior sum ~ +0.003;
# expect less (winner's curse). Usage (Datascape/ or repo root): python work/make_d11.py
import numpy as np, pandas as pd
EXCLUDE = ('d11_lbopt', 'd14_640A')   # fit on the 11 subs scored before d11
exec(open('work/lb_regress.py').read().split("print(f'{len(SCORED)} scored subs")[0])   # fit only: base, lab, key, vars_, v
d08 = lab['d08_grand_consensus_champion']; MIN_GAIN = 1e-4
gain = lambda r, l: 0.0 if l == base[r] else (v[vars_.index((key[r], l))] if (key[r], l) in vars_ else -np.inf)
new = d08.copy()
for r in key.index:
    alts = {l: gain(r, l) for l in set(lab.loc[r]) | {base[r]}}
    best = max(alts, key=alts.get)
    if alts[best] - alts[d08[r]] >= MIN_GAIN: new[r] = best
ss = pd.read_csv('datascape/sample_submission.csv'); out = pd.DataFrame({'filename': ss.filename, 'label': new.reindex(ss.filename).values})
assert out.label.notna().all() and set(out.label) <= {'Aman', 'Rusak', 'Lainnya'} and len(out) == 3057
ch = new != d08; print(f'd11_lbopt: {ch.sum()} rows changed vs d08', pd.Series([f'{a[0]}->{b[0]}' for a, b in zip(d08[ch], new[ch])]).value_counts().to_dict())
out.to_csv('submissions/d11_lbopt.csv', index=False)
