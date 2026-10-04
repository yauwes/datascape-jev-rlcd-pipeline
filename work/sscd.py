# Copy-detection descriptors (SSCD, Pizzi et al. CVPR 2022: robust to crop/noise/blur/pad/colour edits) for twin search.
# Same row order as work/emb/meta_sorted.csv; 5 geometric variants like work/dups.py -> work/emb_sscd.npz (kept out of work/emb so
# towers.py does not treat it as a classification tower). Local MPS, ~10 min.
import numpy as np, pandas as pd, torch, cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
meta = pd.read_csv('work/emb/meta_sorted.csv'); S = 288
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class DS(Dataset):
    def __len__(s): return len(meta)
    def __getitem__(s, i):  # median 3x3 (removes the S&P corruption, as for every other embedding) + square resize
        a = cv2.medianBlur(np.asarray(Image.open(meta.path[i]).convert('RGB')), 3)
        return torch.from_numpy(np.asarray(Image.fromarray(a).resize((S, S), Image.BICUBIC)).copy()).permute(2, 0, 1)

if __name__ == '__main__':
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = torch.jit.load('work/models/sscd_disc_mixup.torchscript.pt').eval().to(dev)
    V = {'orig': lambda x: x, 'hflip': lambda x: x.flip(3), 'rot90': lambda x: x.rot90(1, (2, 3)), 'rot180': lambda x: x.rot90(2, (2, 3)), 'rot270': lambda x: x.rot90(3, (2, 3))}
    out = {k: [] for k in V}
    with torch.no_grad():
        for bi, x in enumerate(DataLoader(DS(), batch_size=64, num_workers=4)):
            x = ((x.float() / 255 - mean) / std).to(dev)
            for k, f in V.items(): out[k].append(torch.nn.functional.normalize(m(f(x)), dim=1).cpu().numpy().astype(np.float16))
            if bi % 40 == 0: print(bi * 64, len(meta), flush=True)
    E = {k: np.concatenate(v) for k, v in out.items()}
    assert all(np.isfinite(v.astype(np.float32)).all() for v in E.values())
    np.savez('work/emb_sscd.npz', **E); print('saved', {k: v.shape for k, v in E.items()})
