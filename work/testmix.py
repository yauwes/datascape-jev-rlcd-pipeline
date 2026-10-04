# Test-mix-weighted group-OOF (discoveries #27): weight each train row by p_test(bucket)/p_train(bucket), bucket = image size (sources)
# x noise stratum; sizes with < 50 train rows merge into 'other'. Answers whether the source mix explains the CV->LB gap.
import sys, numpy as np, pandas as pd
sys.argv = ['x']; exec(open('work/ens.py').read().split("if __name__")[0])
sys.path.insert(0, 'work'); from twin import M as MM, ST
from sklearn.metrics import f1_score
MM['sz'] = MM.w.astype(str) + 'x' + MM.h.astype(str); MM['st'] = ST; MM['nz'] = np.where(pd.Series(ST).str.startswith('noisy'), 'N', 'C')
cnt = MM[MM.split == 'train'].sz.value_counts(); big = set(cnt[cnt >= 50].index)
MM['bk'] = np.where(MM.sz.isin(big), MM.sz, 'other') + '_' + MM.nz
ptr = MM[MM.split == 'train'].bk.value_counts(normalize=True); pte = MM[MM.split == 'test'].bk.value_counts(normalize=True)
bk_tr = MM.set_index('path').bk.loc[tr.path].values; W = (pte.reindex(bk_tr).fillna(0) / ptr.reindex(bk_tr)).values
print('test mass with no train bucket', round(1 - pte[pte.index.isin(ptr.index)].sum(), 4))
def at_share(o, share, w):  # Rusak logit bias so that the weighted predicted Rusak share = share
    L = np.log(o + 1e-9); bs = np.linspace(-4, 3, 141); sh = [np.average((L + np.r_[0, 0, b]).argmax(1) == 2, weights=w) for b in bs]
    b = bs[int(np.argmin(np.abs(np.array(sh) - share)))]; return (L + np.r_[0, 0, b]).argmax(1)
if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'h2_peg_c11'
    for tag in ['h2_peg_c11', 'c2_h2']:
        o, _ = load(tag)
        for share in (0.274, 0.244, 0.212):
            p = at_share(o, share, W); p0 = at_share(o, share, np.ones(len(y)))
            print(f'{tag} share {share}: unweighted F1 {f1_score(y, p0, average="macro"):.4f} | test-mix weighted F1 {f1_score(y, p, average="macro", sample_weight=W):.4f} per-class', f1_score(y, p, average=None, sample_weight=W).round(3))
    o, _ = load('h2_peg_c11'); p = at_share(o, 0.274, W)
    df = pd.DataFrame({'bk': bk_tr, 'y': y, 'p': p, 'w': W}); df['err_A2R'] = (df.y == 0) & (df.p == 2); df['err_R2A'] = (df.y == 2) & (df.p == 0); df['err_L'] = ((df.y == 1) | (df.p == 1)) & (df.y != df.p)
    g = df.groupby('bk').apply(lambda d: pd.Series({'n_tr': len(d), 'test_rows': round(d.w.sum() * len(MM[MM.split == 'test']) / len(y)), 'R%tr': round((d.y == 2).mean(), 2), 'acc': round((d.y == d.p).mean(), 3),
                                                   'A->R@test': round((d.err_A2R * d.w).sum() * 3057 / len(y)), 'R->A@test': round((d.err_R2A * d.w).sum() * 3057 / len(y)), 'L err@test': round((d.err_L * d.w).sum() * 3057 / len(y))}))
    print(g.sort_values('test_rows', ascending=False).to_string())
