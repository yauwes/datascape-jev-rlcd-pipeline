# A14 per-stratum prior matching (Saerens 2002 / Forman 2008 quantification): in each test stratum (noise x 224px),
# shift the Rusak logit so the predicted Rusak share equals P_train(Rusak | stratum). Lainnya stays at its natural share.
import numpy as np, pandas as pd
C = ['Aman', 'Lainnya', 'Rusak']

def strata(meta):
    nz = pd.read_csv('work/meta_noise.csv').set_index('path').loc[meta.path]
    noisy = (nz.impulse.values > 0.02) | (nz.medabs.values >= 6)
    s224 = ((meta.w == 224) & (meta.h == 224)).values
    return np.where(noisy, 'noisy', 'clean') + '_' + np.where(s224, '224', 'other')

def match(Lte, st_te, ytr, st_tr, min_n=200, targets=None):
    """Lte: test log-probs (n,3). Returns adjusted log-probs and a report. Strata with < min_n test rows stay natural."""
    L = Lte.copy(); rep = []
    for s in np.unique(st_te):
        m = st_te == s; tgt = (targets or {}).get(s, (ytr[st_tr == s] == 2).mean() if (st_tr == s).any() else None)
        nat = (L[m].argmax(1) == 2).mean()
        if tgt is None or m.sum() < min_n or tgt == 0: rep.append((s, m.sum(), nat, nat, tgt)); continue
        lo, hi = -8.0, 8.0
        for _ in range(60):
            b = (lo + hi) / 2
            if ((L[m] + np.r_[0, 0, b]).argmax(1) == 2).mean() < tgt: lo = b
            else: hi = b
        L[m, 2] += (lo + hi) / 2; rep.append((s, m.sum(), nat, (L[m].argmax(1) == 2).mean(), tgt))
    return L, pd.DataFrame(rep, columns=['stratum', 'n_test', 'rusak_natural', 'rusak_matched', 'target'])

def write_sub(meta_te, pred, path):
    ss = pd.read_csv('datascape/sample_submission.csv'); stem = {f.rsplit('.', 1)[0]: f for f in ss.filename}
    sub = pd.DataFrame({'filename': [stem[p.split('/')[-1][:-4]] for p in meta_te.path.values], 'label': [C[i] for i in pred]})
    sub = ss[['filename']].merge(sub, on='filename', how='left')
    assert sub.label.notna().all() and len(sub) == len(ss) and set(sub.label) <= set(C)
    sub.to_csv(path, index=False); return sub

if __name__ == '__main__':  # self-check on the calibration recipe
    meta = pd.read_csv('work/emb/meta_sorted.csv'); trm = (meta.split == 'train').values; st = strata(meta)
    ytr = np.array([C.index(v) for v in meta.label.values[trm]])
    kt = np.load('work/knn_test_dinob.npz'); pr = np.load('work/probe_vit_large_patch14_dinov2+vit_large_patch14_clip_224_C2.0.npz')
    n = (~trm).sum(); v = np.zeros((n, 3))
    for j in range(5): np.add.at(v, (np.arange(n), kt['l'][:, j]), np.exp(20 * kt['s'][:, j]))
    L, rep = match(np.log(0.5 * v / v.sum(1, keepdims=True) + 0.5 * pr['test'] + 1e-9), st[~trm], ytr, st[trm])
    print(rep.round(3).to_string(index=False))
    sub = write_sub(meta[~trm], L.argmax(1), 'work/sub_calib_priormatch.csv')
    print(sub.label.value_counts(normalize=True).round(3).to_dict())
    assert abs((L.argmax(1) == 2).mean() - 0.27) < 0.06
