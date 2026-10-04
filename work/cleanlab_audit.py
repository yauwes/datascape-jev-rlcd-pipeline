# cleanlab (Northcutt 2021, cleanlab 2.9) audit of train labels on honest group-OOF probs (original labels, h2_peg_c11):
# find_label_issues under 3 filters, latent true-class prior per source (estimate_latent), Datalab label/outlier/near-duplicate
# issues on DINOv2-L features. Compares with our hand-rolled CL (labels_clean_v2). Writes work/cleanlab_issues.csv.
import sys, numpy as np, pandas as pd
exec(open('work/testmix.py').read().split("if __name__ ==")[0])
from cleanlab.filter import find_label_issues
from cleanlab.count import compute_confident_joint, estimate_latent
from cleanlab.rank import get_label_quality_scores
from cleanlab import Datalab
o, _ = load('h2_peg_c11'); src = MM.set_index('path').sz.loc[tr.path].values; src = np.where(pd.Series(src).isin(big), src, 'other')
q = get_label_quality_scores(y, o)
out = pd.DataFrame({'path': tr.path, 'src': src, 'label': [C[i] for i in y], 'pred': [C[i] for i in o.argmax(1)], 'quality': q.round(4)})
for f in ['prune_by_noise_rate', 'confident_learning', 'both']:
    out[f] = find_label_issues(y, o, filter_by=f, n_jobs=1); print(f, out[f].sum(), out[out[f]].groupby(['label', 'pred']).size().to_dict())
cj = compute_confident_joint(y, o); py, nm, inm = estimate_latent(cj, y)
print('all: given prior', np.bincount(y, minlength=3) / len(y), '| latent true prior', py.round(3), '\nP(given=row | true=col)\n', nm.round(3))
for s in sorted(set(src)):  # per-source latent Rusak prior (needs both classes present)
    m = src == s
    if (np.bincount(y[m], minlength=3)[[0, 2]] > 5).all():
        try: pys = estimate_latent(compute_confident_joint(y[m], o[m]), y[m])[0]; print(f'{s:9s} n={m.sum():5d} given R {np.mean(y[m] == 2):.3f} latent R {pys[2]:.3f}')
        except Exception as e: print(s, 'skip', e)
E = np.load('work/emb/emb_vit_large_patch14_dinov2.npz'); X = sum(E[k].astype(np.float32) for k in E.files); X = X / np.linalg.norm(X, axis=1, keepdims=True)
lab = Datalab(data={'label': out.label.values}, label_name='label', verbosity=0)
lab.find_issues(pred_probs=o, features=X[(MM.split == 'train').values], issue_types={'label': {}, 'outlier': {}, 'near_duplicate': {}})
iss = lab.get_issues(); print(lab.get_issue_summary().to_string())
for k in ['label', 'outlier', 'near_duplicate']: out[f'dl_{k}'] = iss[f'is_{k}_issue'].values
cv2 = pd.read_csv('work/labels_clean_v2.csv').set_index('path').loc[tr.path]; out['v2_changed'] = (cv2.label_clean != cv2.label_orig).values
print('overlap with labels_clean_v2 changes:', {f: int((out[f] & out.v2_changed).sum()) for f in ['prune_by_noise_rate', 'both', 'dl_label']}, 'v2 total', out.v2_changed.sum())
print(out[out.both].groupby(['src', 'label']).size().unstack(fill_value=0).to_string())
out.to_csv('work/cleanlab_issues.csv', index=False)
