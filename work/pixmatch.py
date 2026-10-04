# Pixel-level twin verification (beyond SSCD): for a query row and its top-K SSCD train candidates, max normalised cross-correlation
# over the 8 dihedral transforms of 64x64 grayscale thumbnails. NCC ~1 = same picture up to flip/90-rotation/resize/brightness.
import sys, numpy as np, pandas as pd
from PIL import Image
M = pd.read_csv('work/emb/meta_sorted.csv'); D = np.load('work/nntr_sscd.npz'); I, S = D['i'], D['s']
_c = {}
def thumb(r, n=64):
    if r not in _c:
        a = np.asarray(Image.open(M.path.values[r]).convert('L').resize((n, n), Image.BILINEAR), np.float32); a = a - a.mean(); _c[r] = a / (np.linalg.norm(a) + 1e-6)
    return _c[r]
def dihedral(a): return [np.rot90(b, k) for b in (a, a[:, ::-1]) for k in range(4)]
def best(q, K=10):  # -> (cand rows, max NCC per cand)
    Q = dihedral(thumb(q)); c = I[q, :K]; return c, np.array([max(float((x * thumb(r)).sum()) for x in Q) for r in c])
if __name__ == '__main__':
    sz = sys.argv[1]; M['sz'] = M.w.astype(str) + 'x' + M.h.astype(str); rows = np.where((M.sz == sz).values)[0]
    out = []
    for q in rows:
        c, n = best(q); j = int(np.argmax(n)); out.append((q, M.split.values[q], M.label.values[q], c[j], M.label.values[c[j]], n[j], S[q, 0]))
    df = pd.DataFrame(out, columns=['row', 'split', 'label', 'nn', 'nnlab', 'ncc', 'sscd']); df.to_csv(f'work/pix_{sz}.csv', index=False)
    for sp in ('train', 'test'):
        d = df[df.split == sp]; print(sp, len(d), 'NCC quantiles', d.ncc.quantile([.1, .25, .5, .75, .9]).round(3).tolist())
        if sp == 'train':
            for t in (0.98, 0.95, 0.9, 0.8): k = d.ncc >= t; print(f'  ncc>={t}: n {k.sum()} label agree {np.mean(d.label[k] == d.nnlab[k]):.3f}')
