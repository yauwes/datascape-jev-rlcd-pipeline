# Read a paired LB probe (discoveries #27): base sub with LB S0 vs probe that flips k rows Rusak->Aman (or Aman->Rusak). Public = ~30% random rows.
# Enumerate public confusion states consistent with S0 (Lainnya F1 fixed), then report for each hypothesised fraction q of truly-Rusak
# rows among the flipped ones the implied probe LB. Invert: observed probe LB -> range of q.
# usage: python work/lbdelta.py BASE_CSV PROBE_CSV S0 [S1]
import sys, numpy as np, pandas as pd
def f1(tp, fp, fn): return 2 * tp / max(2 * tp + fp + fn, 1e-9)
def states(pA, pR, pL, S0, fL=0.96, tol=0.004):  # public predicted counts -> (nR_true, TP_R) pairs with macro F1 ~ S0 (A/R confusions only)
    out = []
    for nR in range(int(0.4 * pR), int(1.6 * pR)):
        for tpR in range(max(0, nR - pA), min(nR, pR) + 1):
            fpR, fnR = pR - tpR, nR - tpR; tpA = pA - fnR; fpA, fnA = fnR, fpR
            if tpA < 0: continue
            s = (f1(tpA, fpA, fnA) + fL + f1(tpR, fpR, fnR)) / 3
            if abs(s - S0) < tol: out.append((nR, tpR))
    return out
if __name__ == '__main__':
    b = pd.read_csv(sys.argv[1]).set_index('filename').label; p = pd.read_csv(sys.argv[2]).set_index('filename').label.loc[b.index]; S0 = float(sys.argv[3])
    ch = (b != p); up = set(b[ch]) == {'Aman'} and set(p[ch]) == {'Rusak'}
    assert up or (set(b[ch]) == {'Rusak'} and set(p[ch]) == {'Aman'}), 'only one-direction A<->R probes supported'
    f = 0.3; pA, pR, pL = [round(f * (b == c).sum()) for c in ('Aman', 'Rusak', 'Lainnya')]; k = round(f * ch.sum())
    st = states(pA, pR, pL, S0); print(f'public approx: pred A {pA} R {pR} L {pL}; flipped k={k}; {len(st)} consistent states; nR range {min(s[0] for s in st)}-{max(s[0] for s in st)}')
    for q in np.linspace(0, 1, 11):
        r = round(q * k); vals = []
        for nR, tpR in st:
            fpR_b, fnR_b = pR - tpR, nR - tpR; tpA_b = pA - fnR_b
            if up:  # A->R flips: r_eff were FN_R (true R), a_eff were TP_A (true A)
                r_eff = min(r, fnR_b); a_eff = min(k - r_eff, tpA_b)
                tpR2, fpR2, fnR2 = tpR + r_eff, fpR_b + a_eff, fnR_b - r_eff; tpA2, fpA2, fnA2 = tpA_b - a_eff, fnR_b - r_eff, fpR2
            else:  # R->A flips: r_eff were TP_R (true R), a_eff were FP_R (true A)
                r_eff = min(r, tpR); a_eff = min(k - r_eff, fpR_b)
                tpR2, fpR2, fnR2 = tpR - r_eff, fpR_b - a_eff, fnR_b + r_eff; tpA2, fpA2, fnA2 = tpA_b + a_eff, fnR_b + r_eff, fpR2
            vals.append((f1(tpA2, fpA2, fnA2) + 0.96 + f1(tpR2, fpR2, fnR2)) / 3)
        print(f'  q(truly Rusak among flipped)={q:.1f}: probe LB {np.mean(vals):.4f} [{np.min(vals):.4f}, {np.max(vals):.4f}]')
