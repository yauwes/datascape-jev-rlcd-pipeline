# Datascape: frozen-backbone embeddings for duplicate analysis + linear probes. Resumable: each model saved on completion.
import glob, os, time, subprocess, sys
import numpy as np, pandas as pd, torch, timm
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader

def find_meta():
    hits = glob.glob('/kaggle/input/**/meta.csv', recursive=True)
    assert hits, 'meta.csv not found'
    return hits[0]
META = find_meta()
hits = glob.glob('/kaggle/input/**/train/Aman__Aman_00001.jpg', recursive=True)
if not hits:  # dataset zip not auto-extracted
    z = glob.glob('/kaggle/input/**/small.zip', recursive=True)[0]
    subprocess.run(['unzip', '-q', '-o', z, '-d', '/kaggle/tmp/small'], check=True)
    hits = ['/kaggle/tmp/small/train/Aman__Aman_00001.jpg']
ROOT = os.path.dirname(os.path.dirname(hits[0]))
print('meta', META, 'root', ROOT, flush=True)
meta = pd.read_csv(META)
meta['rel'] = meta.path.str.split('work/small/').str[1]
meta = meta.sort_values(['split', 'rel']).reset_index(drop=True)
meta.to_csv('/kaggle/working/meta_sorted.csv', index=False)

class DS(Dataset):
    def __init__(s, size): s.size = size
    def __len__(s): return len(meta)
    def __getitem__(s, i):
        im = Image.open(f'{ROOT}/{meta.rel[i]}').convert('RGB').filter(ImageFilter.MedianFilter(3)).resize((s.size, s.size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)

def done(tag):
    return any(glob.glob(f'{d}/**/emb_{tag}.npz', recursive=True) for d in ('/kaggle/working', '/kaggle/input'))

MODELS = [  # (timm name, img size, variants)
    ('vit_base_patch14_dinov2.lvd142m', 224, ('orig', 'hflip', 'rot90', 'rot180', 'rot270')),
    ('vit_large_patch14_dinov2.lvd142m', 224, ('orig', 'hflip')),
    ('vit_large_patch14_clip_224.openai', 224, ('orig', 'hflip')),
    ('eva02_large_patch14_clip_224.merged2b_s4b_b131k', 224, ('orig', 'hflip')),
]
dev = 'cuda'
for name, size, variants in MODELS:
    tag = name.split('.')[0]
    if done(tag): print('skip', tag); continue
    t0 = time.time()
    try:
        m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size).eval().to(dev).half()
    except TypeError:
        m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(dev).half()
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1).half(); std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1).half()
    out = {v: [] for v in variants}
    dl = DataLoader(DS(size), batch_size=128, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for bi, x in enumerate(dl):
            x = (x.to(dev).half() / 255 - mean) / std
            for v in variants:
                xv = {'orig': x, 'hflip': x.flip(3), 'rot90': x.rot90(1, (2, 3)), 'rot180': x.rot90(2, (2, 3)), 'rot270': x.rot90(3, (2, 3))}[v]
                out[v].append(m(xv).float().cpu().numpy().astype(np.float16))
            if bi % 20 == 0: print(tag, bi * 128, len(meta), f'{time.time() - t0:.0f}s', flush=True)
    np.savez(f'/kaggle/working/emb_{tag}.npz', **{v: np.concatenate(a) for v, a in out.items()})
    print('saved', tag, f'{time.time() - t0:.0f}s', flush=True)
    del m; torch.cuda.empty_cache()
print('ALL DONE')
