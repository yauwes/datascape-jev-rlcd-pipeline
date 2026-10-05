# Read a paired LB probe (discoveries #27): base sub with LB S0 vs a probe that flips rows Aman<->Rusak (either or both directions). Public = ~30% random rows.
# Enumerate public confusion states consistent with S0 (Lainnya F1 fixed), then report, for hypothesised fractions q of truly-Rusak rows among
# the flipped ones, the implied probe LB. 10-05: the number of PUBLIC rows among the flipped ones is integrated over Binomial(n, 0.3) instead of
# being fixed at 0.3*n (the 675 read fell below the fixed-k floor: the joint inversion needed ~54 public rows where 47 were expected).
# usage: python work/lbdelta.py BASE_CSV PROBE_CSV S0
import sys, numpy as np, pandas as pd
from scipy.stats import binom
F = 0.3
def f1(tp, fp, fn): return 2 * tp / max(2 * tp + fp + fn, 1e-9)
def states(pA, pR, S0, fL=0.96, tol=0.004):  # public predicted counts -> (nR_true, TP_R) pairs with macro F1 ~ S0 (A/R confusions only)
    out = []
    for nR in range(int(0.4 * pR), int(1.6 * pR)):
        for tpR in range(max(0, nR - pA), min(nR, pR) + 1):
            fpR, fnR = pR - tpR, nR - tpR; tpA = pA - fnR
            if tpA >= 0 and abs((f1(tpA, fnR, fpR) + fL + f1(tpR, fpR, fnR)) / 3 - S0) < tol: out.append((nR, tpR))
    return out
def apply(pA, pR, nR, tpR, k_up, r_up, k_dn, r_dn, fL=0.96):
    """k_up public A->R flips of which r_up truly Rusak; k_dn public R->A flips of which r_dn truly Rusak. Returns the probe macro F1 (None if infeasible)."""
    fpR, fnR = pR - tpR, nR - tpR; tpA = pA - fnR          # base: FN_R rows are predicted A, truly R; FP_R rows are predicted R, truly A
    if r_up > fnR or k_up - r_up > tpA or r_dn > tpR or k_dn - r_dn > fpR: return None
    tpR2 = tpR + r_up - r_dn; fpR2 = fpR + (k_up - r_up) - (k_dn - r_dn); fnR2 = fnR - r_up + r_dn
    tpA2 = tpA - (k_up - r_up) + (k_dn - r_dn)
    return (f1(tpA2, fnR2, fpR2) + fL + f1(tpR2, fpR2, fnR2)) / 3
def curve(pA, pR, st, n_up, n_dn, q_up, q_dn):
    """weighted mean and 5-95% of the probe LB over consistent states and Binomial public counts."""
    ks_up = range(max(0, int(F * n_up - 4 * np.sqrt(n_up * F * (1 - F)) - 1)), int(F * n_up + 4 * np.sqrt(n_up * F * (1 - F)) + 2)) if n_up else [0]
    ks_dn = range(max(0, int(F * n_dn - 4 * np.sqrt(n_dn * F * (1 - F)) - 1)), int(F * n_dn + 4 * np.sqrt(n_dn * F * (1 - F)) + 2)) if n_dn else [0]
    vals, ws = [], []
    for ku in ks_up:
        wu = binom.pmf(ku, n_up, F) if n_up else 1.0
        for kd in ks_dn:
            wd = binom.pmf(kd, n_dn, F) if n_dn else 1.0
            for nR, tpR in st:
                v = apply(pA, pR, nR, tpR, ku, round(q_up * ku), kd, round(q_dn * kd))
                if v is not None: vals.append(v); ws.append(wu * wd)
    v, w = np.array(vals), np.array(ws); w = w / w.sum(); o = np.argsort(v); c = np.cumsum(w[o])
    return float((v * w).sum()), float(v[o][np.searchsorted(c, 0.05)]), float(v[o][min(np.searchsorted(c, 0.95), len(v) - 1)])
if __name__ == '__main__':
    b = pd.read_csv(sys.argv[1]).set_index('filename').label; p = pd.read_csv(sys.argv[2]).set_index('filename').label.loc[b.index]; S0 = float(sys.argv[3])
    up = ((b == 'Aman') & (p == 'Rusak')).sum(); dn = ((b == 'Rusak') & (p == 'Aman')).sum(); other = ((b != p) & ~((b == 'Aman') & (p == 'Rusak')) & ~((b == 'Rusak') & (p == 'Aman'))).sum()
    pA, pR = [round(F * (b == c).sum()) for c in ('Aman', 'Rusak')]; st = states(pA, pR, S0)
    print(f'public approx: pred A {pA} R {pR}; flips A->R {up}, R->A {dn}, other {other} (ignored); {len(st)} consistent states; nR range {min(s[0] for s in st)}-{max(s[0] for s in st)}')
    if up and dn:
        for qu in (0.0, 0.5, 1.0):
            print('  q_up(A->R truly R)=%.1f |' % qu, ' '.join('q_dn=%.2f: %.4f' % (qd, curve(pA, pR, st, up, dn, qu, qd)[0]) for qd in (0, 0.25, 0.5, 0.75, 1)))
    else:
        for q in np.linspace(0, 1, 11):
            m, lo, hi = curve(pA, pR, st, up, dn, q, q)
            print(f'  q(truly Rusak among flipped)={q:.1f}: probe LB {m:.4f} [5% {lo:.4f}, 95% {hi:.4f}]')
