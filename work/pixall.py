# Pixel-exact twin verification for ALL rows (discoveries #27 draft): each row vs its top-10 SSCD train candidates (self excluded),
# max NCC over 8 dihedral transforms on 128x128 grayscale thumbnails (mean/std normalised -> robust to brightness/contrast/resize/JPEG).
# Output work/pix_all.npz: ncc[N,10] aligned with nntr_sscd 'i'.
import numpy as np, pandas as pd, time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
M = pd.read_csv('work/emb/meta_sorted.csv'); D = np.load('work/nntr_sscd.npz'); I = D['i']; N = len(M); n = 128; T0 = time.time()
def load(r):
    a = np.asarray(Image.open(M.path.values[r]).convert('L').resize((n, n), Image.BILINEAR), np.float32); a = a - a.mean(); return a / (np.linalg.norm(a) + 1e-6)
with ThreadPoolExecutor(8) as ex: TH = np.stack(list(ex.map(load, range(N)))).astype(np.float32)
print('thumbs', TH.shape, round(time.time() - T0), 's', flush=True)
DH = np.stack([np.rot90(b, k, axes=(1, 2)) for b in (TH, TH[:, :, ::-1]) for k in range(4)], 1)  # [N,8,n,n] (views)
ncc = np.zeros((N, 10), np.float32)
for a in range(0, N, 500):
    q = np.arange(a, min(a + 500, N)); Q = DH[q].reshape(len(q), 8, -1); Cn = TH[I[q]].reshape(len(q), 10, -1)
    ncc[q] = np.einsum('qvd,qkd->qkv', Q, Cn).max(2)
np.savez('work/pix_all.npz', ncc=ncc); print('done', round(time.time() - T0), 's')
