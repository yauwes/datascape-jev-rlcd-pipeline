# Twin label transfer (discoveries #18).
# 1) nn():    top-K *train* neighbours for every row, max cosine over the 5 geometric variants (as work/dups.py).
# 2) loo():   train leave-one-out table: does a twin vote at sim >= thr agree with the row's own label, and who wins when it
#             disagrees with a model that has (random folds) or has not (group folds) seen the twin.
# 3) build(): lock *clean*-stratum test rows whose twins vote clearly, then prior-match only the unlocked rows so every stratum keeps
#             its pm38 Rusak target (no share confound, discoveries #11). 224/noisy strata are never locked: their twins are
#             artifact-driven (all-Aman 224px train images).
# usage: python work/twin.py nn dinob|sscd   |   python work/twin.py loo SRC MODEL_TAG   |   python work/twin.py build SRC THR BASE_TAG NAME
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'work'); from postproc import strata, match, write_sub, C
from sklearn.metrics import f1_score
M = pd.read_csv('work/emb/meta_sorted.csv'); TR = (M.split == 'train').values
LAB = np.array([C.index(l) if isinstance(l, str) else -1 for l in M.label]); ST = strata(M)
SRC = {'dinob': 'work/emb/emb_vit_base_patch14_dinov2.npz', 'sscd': 'work/emb_sscd.npz', 'sscdL': 'work/out/copydet/emb_sscd_disc_large.npz'}

def nn(src, K=10):
    E = np.load(SRC[src]); V = {k: E[k].astype(np.float32) for k in E.files}
    for k in V: V[k] /= np.linalg.norm(V[k], axis=1, keepdims=True)
    tr = np.where(TR)[0]; I = np.zeros((len(M), K), np.int32); S = np.zeros((len(M), K), np.float32)
    for a in range(0, len(M), 1000):
        s = np.max(np.stack([V['orig'][a:a + 1000] @ V[k][tr].T for k in V]), 0)
        q = np.arange(a, min(a + 1000, len(M))); s[np.isin(q, tr)[:, None] & (tr[None, :] == q[:, None])] = -1  # drop self
        idx = np.argpartition(-s, K, 1)[:, :K]; ss = np.take_along_axis(s, idx, 1); o = np.argsort(-ss, 1)
        I[q] = tr[np.take_along_axis(idx, o, 1)]; S[q] = np.take_along_axis(ss, o, 1)
    np.savez(f'work/nntr_{src}.npz', i=I, s=S); return I, S

def vote(I, S, thr):
    """similarity-weighted label vote over all train twins >= thr. returns label (-1 = no twin), weight share of the winner."""
    w = np.where(S >= thr, S, 0.0); P = np.zeros((len(I), 3))
    for c in range(3): P[:, c] = (w * (LAB[I] == c)).sum(1)
    tot = P.sum(1); lab = np.where(tot > 0, P.argmax(1), -1); return lab, np.where(tot > 0, P.max(1) / np.maximum(tot, 1e-9), 0)

def model_pred(tag):  # train-row argmax from a saved OOF tag, aligned to M
    exec(open('work/ens.py').read().split("if __name__")[0], g := {'__name__': 'x'})
    o, _ = g['load'](tag); pos = {p: i for i, p in enumerate(g['tr'].path.values)}; out = np.full(len(M), -1)
    rows = np.where(TR)[0]; out[rows] = o[[pos[p] for p in M.path.values[rows]]].argmax(1); return out

def loo(src, tag):
    d = np.load(f'work/nntr_{src}.npz'); I, S = d['i'], d['s']; mp = model_pred(tag)
    print(f'{src} twins vs {tag} (train rows, clean stratum only)')
    for thr in (0.99, 0.97, 0.95, 0.93, 0.90, 0.85, 0.80):
        lab, pur = vote(I, S, thr); k = TR & (lab >= 0) & (ST == 'clean_other'); dis = k & (mp != lab)
        print(f'  thr {thr}: n {k.sum():5d} twin-acc {np.mean(lab[k] == LAB[k]):.3f} model-acc {np.mean(mp[k] == LAB[k]):.3f} | disagree {dis.sum():4d}: '
              f'twin right {np.mean(lab[dis] == LAB[dis]):.3f} model right {np.mean(mp[dis] == LAB[dis]):.3f} | test rows {((~TR) & (S[:, 0] >= thr) & (ST == "clean_other")).sum()}')

def build(src, thr, base, name, targets=None, purity=0.0, bias_transfer=False, bstar=None):
    exec(open('work/ens.py').read().split("if __name__")[0], g := {'__name__': 'x'})
    _, t = g['load'](base); te = np.where(~TR)[0]; assert (g['te'].path.values == M.path.values[te]).all()
    d = np.load(f'work/nntr_{src}.npz'); lab, pur = vote(d['i'][te], d['s'][te], thr)
    lock = (lab >= 0) & (pur >= purity) & (ST[te] == 'clean_other')  # mixed twin votes: the model wins 2/3 of disagreements (discoveries #21)
    targets = targets or {'clean_other': 0.381}  # pm38 rule (discoveries #14); other strata: train P(Rusak|stratum)
    L = np.log(t + 1e-9); ytr = LAB[TR]; sttr = ST[TR]; final = L.argmax(1).copy(); tg = {}
    if bias_transfer:  # discoveries #22: calibrate the Rusak logit bias on twin-locked test rows (twin labels ~98% right), reuse it for unlocked rows
        m = lock & (ST[te] == 'clean_other'); bs = np.linspace(-3, 1, 81)
        f = [f1_score(lab[m], (L[m] + np.r_[0, 0, b]).argmax(1), average='macro') for b in bs]
        if bstar is None: bstar = bs[int(np.argmax(f))]  # a base trained on the locked rows (DA) must pass a CV-calibrated bstar instead
        u = ~lock & (ST[te] == 'clean_other')
        bt_share = float(((L[u] + np.r_[0, 0, bstar]).argmax(1) == 2).mean()); print(f'bias transfer: b*={bstar:.2f} (locked F1 {max(f):.4f}) -> unlocked clean_other Rusak {bt_share:.3f}')
    for s in np.unique(ST[te]):  # stratum target -> share the unlocked rows must reach so the stratum total keeps the pm38 target
        m = ST[te] == s; T = targets.get(s, (ytr[sttr == s] == 2).mean()); n_l = (m & lock).sum()
        tg[s] = float(np.clip((T * m.sum() - (lab[m & lock] == 2).sum()) / max((m & ~lock).sum(), 1), 0, 1)) if n_l else T
        if bias_transfer and s == 'clean_other': tg[s] = bt_share
    stu = ST[te].astype(object).copy()
    if bias_transfer:  # split unlocked clean_other: mixed-vote rows get a target from their own twin vote (train-calibrated bins), no-twin rows the bias-transfer share
        d0 = np.load(f'work/nntr_{src}.npz'); Itr, Str = d0['i'][TR], d0['s'][TR]; lt, pt = vote(Itr, Str, thr)
        def wr(I_, S_): w = np.where(S_ >= thr, S_, 0); return (w * (LAB[I_] == 2)).sum(1) / np.maximum(w.sum(1), 1e-9)
        mtr = (lt >= 0) & (pt < 0.9) & (ST[TR] == 'clean_other'); rtr = wr(Itr, Str); edges = [0, 0.3, 0.5, 0.7, 1.01]
        cal = [float((LAB[TR][mtr & (rtr >= a) & (rtr < b)] == 2).mean()) for a, b in zip(edges[:-1], edges[1:])]
        mix = (lab >= 0) & ~lock & (ST[te] == 'clean_other'); nt = (lab < 0) & (ST[te] == 'clean_other')
        rte = wr(d['i'][te], d['s'][te]); stu[mix] = 'co_mixed'; stu[nt] = 'co_notwin'
        tg['co_mixed'] = float(np.mean([cal[np.searchsorted(edges, r, 'right') - 1] for r in rte[mix]]))
        tg['co_notwin'] = float(((L[nt] + np.r_[0, 0, bstar]).argmax(1) == 2).mean()); print('mixed-vote calibration bins', np.round(cal, 3), '| targets', {k: round(tg[k], 3) for k in ('co_mixed', 'co_notwin')})
    Lm, rep = match(L[~lock], stu[~lock].astype(str), ytr, sttr, targets=tg, min_n=50)
    final[~lock] = Lm.argmax(1); final[lock] = lab[lock]
    write_sub(M.iloc[te], final, f'work/sub_{name}.csv')
    base_sub = pd.read_csv('submissions/d04_mlp6_pm38.csv').set_index('filename').label
    me = pd.read_csv(f'work/sub_{name}.csv').set_index('filename').label.loc[base_sub.index]
    print(rep.round(3).to_string(index=False)); print(f'{name}: locked {lock.sum()} | rows differing from d04_mlp6_pm38: {(me != base_sub).sum()} |',
          'mix', pd.Series([C[i] for i in final]).value_counts(normalize=True).round(3).to_dict(),
          '| per-stratum Rusak', {s: round(float((final[ST[te] == s] == 2).mean()), 3) for s in np.unique(ST[te])})

if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'nn': nn(a[1])
    elif a[0] == 'loo': loo(a[1], a[2])
    elif a[0] == 'build': build(a[1], float(a[2]), a[3], a[4], purity=float(a[5]) if len(a) > 5 else 0.0, bias_transfer=len(a) > 6 and a[6] == 'bt', bstar=float(a[7]) if len(a) > 7 else None)
