# Frozen giant-tower embeddings (A100, bf16) in the shared row order (meta_sorted.csv) -> $DS_OUT/emb/emb_<tag>.npz.
# Same preprocessing as kernels/emb*: median 3x3 + square bicubic resize at the tower's native size, 'orig' view. Resumable per tower.
# usage: DS_IN=... DS_OUT=... python emb_giant.py TIMM_NAME SIZE
import glob, os, sys, time, numpy as np, pandas as pd, torch, timm
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
IN, OUT = os.environ['DS_IN'], os.environ['DS_OUT']; os.makedirs(f'{OUT}/emb', exist_ok=True)
meta = pd.read_csv(glob.glob(f'{IN}/**/meta_sorted.csv', recursive=True)[0])
ROOT = os.path.dirname(os.path.dirname(glob.glob(f'{IN}/**/train/Aman__Aman_00001.jpg', recursive=True)[0]))

class DS(Dataset):
    def __init__(s, size): s.size = size
    def __len__(s): return len(meta)
    def __getitem__(s, i):
        im = Image.fromarray(cv2.medianBlur(np.asarray(Image.open(f'{ROOT}/{meta.rel[i]}').convert('RGB')), 3)).resize((s.size, s.size), Image.BICUBIC)  # == PIL MedianFilter(3)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)

def run(name, size):
    tag = name.split('.')[0]; f = f'{OUT}/emb/emb_{tag}.npz'
    if os.path.exists(f): print('skip', tag, flush=True); return
    t0 = time.time(); kw = dict(global_pool=os.environ['GP']) if os.environ.get('GP') else {}  # GP=avg: mean over patch tokens (VLM towers have no CLS/pool head)
    try: m = timm.create_model(name, pretrained=True, num_classes=0, **kw)
    except TypeError: m = timm.create_model(name, pretrained=True, num_classes=0)
    m = m.eval().cuda().to(torch.bfloat16)  # bf16: fp32 range, no fp16 NaN (DINOv3/MobileNetV5 lesson)
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg['mean'], device='cuda').view(1, 3, 1, 1); std = torch.tensor(cfg['std'], device='cuda').view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for bi, x in enumerate(DataLoader(DS(size), batch_size=64, num_workers=int(os.environ.get('NW', 8)), pin_memory=True)):
            out.append(m(((x.cuda().float() / 255 - mean) / std).to(torch.bfloat16)).float().cpu().numpy())
            if bi % 50 == 0: print(tag, bi * 64, len(meta), f'{time.time() - t0:.0f}s', flush=True)
    E = np.concatenate(out).astype(np.float16)
    assert np.isfinite(E).all() and E.astype(np.float32).std(0).mean() > 1e-4, f'{tag}: degenerate features'
    np.savez(f, orig=E); print('saved', tag, E.shape, f'{time.time() - t0:.0f}s', flush=True)

if __name__ == '__main__': run(sys.argv[1], int(sys.argv[2]))
