# Per-source cleanlab on the 640x640 web source (given Rusak 0.469 vs CL latent 0.305, work/cleanlab_audit.py) + overlap checks.
import sys, numpy as np, pandas as pd; OUTD = sys.argv[1]
from PIL import Image
exec(open('work/testmix.py').read().split("if __name__ ==")[0])
from cleanlab.filter import find_label_issues
o, _ = load('h2_peg_c11'); sz_tr = MM.set_index('path').sz.loc[tr.path].values; m = sz_tr == '640x640'
fl = np.zeros(len(y), bool); fl[m] = find_label_issues(y[m], o[m], filter_by='both', n_jobs=1)
print('640 train rows', m.sum(), '| per-source flagged', fl.sum(), pd.crosstab(y[fl], o[fl].argmax(1)).to_dict())
rng = np.random.default_rng(0); R = np.where(m & (y == 2))[0]
def mont(ix, f, n=24, s=160):
    ix = rng.choice(ix, min(n, len(ix)), replace=False); g = Image.new('RGB', (6 * s, ((len(ix) + 5) // 6) * s))
    for j, i in enumerate(ix): g.paste(Image.open(tr.path[i]).convert('RGB').resize((s, s)), ((j % 6) * s, (j // 6) * s))
    g.save(f)
mont(R[fl[R]], OUTD + '/m640_flagR.jpg'); mont(R[~fl[R]], OUTD + '/m640_keptR.jpg')
pd.DataFrame({'path': tr.path[fl], 'label': [C[i] for i in y[fl]], 'pred': [C[i] for i in o[fl].argmax(1)]}).to_csv('work/cl640_flags.csv', index=False)
# overlap: rows changed by d05_share_non416 vs da_bt2, by test source
a, b = pd.read_csv('submissions/d04_da_bt2.csv'), pd.read_csv('submissions/d05_share_non416.csv'); TE = MM[MM.split == 'test'].set_index('filename')
d = a.label != b.label; print('share_non416 changes by source:', TE.sz.loc[a.filename[d]].value_counts().to_dict())
ncc = np.load('work/pix_all.npz')['ncc'].max(1); exact = pd.Series(ncc, index=MM.filename)
t640 = TE.sz == '640x640'; f640 = TE.index[t640]; ex = exact.loc[f640].values >= 0.99; lab = a.set_index('filename').label.loc[f640].values
print('test 640 rows', t640.sum(), '| exact copies', ex.sum(), '| da_bt2 Rusak among non-exact', (lab[~ex] == 'Rusak').sum(), 'of', (~ex).sum(), '| among exact', (lab[ex] == 'Rusak').sum())
