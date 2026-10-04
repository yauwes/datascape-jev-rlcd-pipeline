# Rotation/noise-robust twin search (discoveries #22). Unlocked test rows include copies rotated by arbitrary angles (black corners),
# heavily noised or washed out, which SSCD on the 5 right-angle variants misses.
# Database = train rows x {median3, median5+blur} x {0,±15,±30,±45,90,180,270 deg} x {orig, hflip} (SSCD disc_mixup, 288px, GPU rotation).
# Queries  = all rows x {median3, median5+blur}.
# Similarity = max over query x database variants. Output: top-10 train neighbours per row (self excluded) -> $DS_OUT/nntr_sscd2.npz
# (same format as work/twin.py nn()).
import glob, os, time, subprocess, numpy as np, pandas as pd, torch, cv2, torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
IN, OUT = os.environ['DS_IN'], os.environ['DS_OUT']; os.makedirs(OUT, exist_ok=True); dev = 'cuda'; S = 288; t0 = time.time()
meta = pd.read_csv(glob.glob(f'{IN}/**/meta_sorted.csv', recursive=True)[0])
ROOT = os.path.dirname(os.path.dirname(glob.glob(f'{IN}/**/train/Aman__Aman_00001.jpg', recursive=True)[0]))
TR = (meta.split == 'train').values; tr = np.where(TR)[0]

class DS(Dataset):
    def __len__(s): return len(meta)
    def __getitem__(s, i):
        a = np.asarray(Image.open(f'{ROOT}/{meta.rel[i]}').convert('RGB'))
        m3 = cv2.resize(cv2.medianBlur(a, 3), (S, S), interpolation=cv2.INTER_CUBIC)
        m5 = cv2.resize(cv2.GaussianBlur(cv2.medianBlur(a, 5), (0, 0), 1.5), (S, S), interpolation=cv2.INTER_AREA)  # strong denoise for noised copies
        return torch.from_numpy(np.stack([m3, m5])).permute(0, 3, 1, 2)  # 2 x 3 x S x S uint8

subprocess.run(['wget', '-q', '-O', '/content/sscd.pt', 'https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt'], check=True)
m = torch.jit.load('/content/sscd.pt').eval().to(dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
ANG = [0, 15, -15, 30, -30, 45, -45, 90, 180, 270]
def enc(x): return torch.nn.functional.normalize(m(x).float(), dim=1).half()
Q = torch.zeros(len(meta), 2, 512, dtype=torch.half, device=dev); DB = torch.zeros(len(tr), 2 * len(ANG) * 2, 512, dtype=torch.half, device=dev)
pos = np.full(len(meta), -1); pos[tr] = np.arange(len(tr)); i0 = 0
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for bi, x in enumerate(DataLoader(DS(), batch_size=128, num_workers=10, pin_memory=True)):
        n = len(x); x = (x.to(dev).float() / 255).flatten(0, 1); x = ((x - mean) / std)  # (2n) x 3 x S x S, order: row-major (row, denoise)
        Q[i0:i0 + n] = enc(x).view(n, 2, 512)
        rows = np.arange(i0, i0 + n); k = TR[rows]
        if k.any():
            xs = x.view(n, 2, 3, S, S)[torch.from_numpy(k).to(dev)].flatten(0, 1); outs = []
            for a in ANG:
                xr = TF.rotate(xs, a, fill=[float(-mu / sd) for mu, sd in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # black fill, like the copies
                outs += [enc(xr), enc(xr.flip(3))]
            DB[pos[rows[k]]] = torch.stack(outs, 1).view(int(k.sum()), 2, len(ANG) * 2, 512).flatten(1, 2)
        i0 += n
        if bi % 20 == 0: print('enc', i0, len(meta), f'{time.time() - t0:.0f}s', flush=True)
K = 10; I = np.zeros((len(meta), K), np.int32); Sm = np.zeros((len(meta), K), np.float32); dbf = DB.flatten(0, 1)  # (ntr*V) x 512
V = DB.shape[1]
with torch.no_grad():
    for a in range(0, len(meta), 256):
        q = Q[a:a + 256].flatten(0, 1)  # (b*2) x 512
        s = (q @ dbf.T).view(-1, 2, len(tr), V).amax(dim=(1, 3)).float()  # b x ntr
        b = s.shape[0]; qi = np.arange(a, a + b); self_ = pos[qi] >= 0
        if self_.any(): s[torch.from_numpy(np.where(self_)[0]).to(dev), torch.from_numpy(pos[qi[self_]]).to(dev)] = -1
        v, ix = s.topk(K, dim=1); I[a:a + b] = tr[ix.cpu().numpy()]; Sm[a:a + b] = v.cpu().numpy()
        if a % 2560 == 0: print('knn', a, len(meta), f'{time.time() - t0:.0f}s', flush=True)
np.savez(f'{OUT}/nntr_sscd2.npz', i=I, s=Sm); print('saved nntr_sscd2', f'{time.time() - t0:.0f}s', flush=True)
