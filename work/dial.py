# Share dial on the best LB recipe (d04_da_bt2 = 0.845): DA heads + SOFT twin lock (purity>=0.9 twin vote enters as a strong log-prior,
# lam=3) + one Rusak logit bias for clean_other bisected to a target share. 224 strata and noisy_other as before. Lower targets flip the
# least-supported Rusak calls first, whether they come from the model or from twins (discoveries #25: LB improves monotonically as the share drops).
# usage: python work/dial.py TARGET_CLEAN_OTHER NAME [LAM] [BASE_TAG] [LABELS_CSV]   (LABELS_CSV: cleaned train labels for the twin votes)
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'work'); from twin import M, TR, ST, LAB, vote, C
from postproc import match, write_sub
tgt, name = float(sys.argv[1]), sys.argv[2]; lam = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
base = sys.argv[4] if len(sys.argv) > 4 else 'da_h2'; lab_csv = sys.argv[5] if len(sys.argv) > 5 else ''
if lab_csv:  # twin votes with cleaned train labels (discoveries #26)
    cl = pd.read_csv(lab_csv).set_index('path').label_clean; trr = np.where(TR)[0]; LAB = LAB.copy(); LAB[trr] = [C.index(v) for v in cl.loc[M.path.values[trr]]]
    import twin; twin.LAB = LAB
sys.argv = ['x']; exec(open('work/ens.py').read().split("if __name__")[0], g := {'__name__': 'x'})
_, t = g['load'](base); te = np.where(~TR)[0]; L = np.log(t + 1e-9)
D = np.load('work/nntr_sscd.npz'); I, S = D['i'][te], D['s'][te]; lab, pur = vote(I, S, 0.55)
lk = (lab >= 0) & (pur >= 0.9) & (ST[te] == 'clean_other')
w = np.where(S >= 0.55, S, 0); P = np.stack([(w * (LAB[I] == c)).sum(1) for c in range(3)], 1); P /= np.maximum(P.sum(1, keepdims=True), 1e-9)
L[lk] += lam * np.log(0.9 * P[lk] + 0.1 / 3)
st = ST[te]; ytr = LAB[TR]; Lm, rep = match(L, st, ytr, ST[TR], targets={'clean_other': tgt, 'noisy_other': 0.096}, min_n=50)
pred = Lm.argmax(1); write_sub(M.iloc[te], pred, f'work/sub_{name}.csv')
ref = pd.read_csv('work/sub_d04_da_bt2.csv').set_index('filename').label; me = pd.read_csv(f'work/sub_{name}.csv').set_index('filename').label.loc[ref.index]
print(rep.round(3).to_string(index=False)); print(f'{name}: vs d04_da_bt2 differ {(me != ref).sum()} {pd.crosstab(ref[me != ref], me[me != ref]).to_dict()} | mix', me.value_counts(normalize=True).round(3).to_dict(),
      '| locked rows whose twin label was overridden:', int((pred[lk] != lab[lk]).sum()))
