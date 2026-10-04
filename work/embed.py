# DINOv2 embeddings for all cached images under 5 geometric variants (orig, hflip, rot90, rot180, rot270), median-filtered.
import sys, time, numpy as np, pandas as pd, torch, timm
from PIL import Image, ImageFilter
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'vit_base_patch14_dinov2.lvd142m'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval().to(dev)
cfg = timm.data.resolve_data_config({}, model=m); mean = torch.tensor(cfg['mean']).view(1, 3, 1, 1).to(dev); std = torch.tensor(cfg['std']).view(1, 3, 1, 1).to(dev)
meta = pd.read_csv('work/meta.csv').sort_values(['split', 'path']).reset_index(drop=True)
if N: meta = meta.iloc[:N]
meta.to_csv('work/meta_sorted.csv', index=False)
def load(p):
    im = Image.open(p).convert('RGB').filter(ImageFilter.MedianFilter(3)).resize((224, 224), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1)
out = {k: [] for k in ('orig', 'hflip', 'rot90', 'rot180', 'rot270')}
t0 = time.time(); B = 64
with torch.no_grad():
    for i in range(0, len(meta), B):
        x = torch.stack([load(p) for p in meta.path[i:i + B]]).to(dev).float().div(255)
        x = (x - mean) / std
        for k, v in (('orig', x), ('hflip', x.flip(3)), ('rot90', x.rot90(1, (2, 3))), ('rot180', x.rot90(2, (2, 3))), ('rot270', x.rot90(3, (2, 3)))):
            f = m(v); out[k].append(torch.nn.functional.normalize(f, dim=1).cpu().half().numpy())
        if i % (B * 20) == 0: print(i, len(meta), f'{time.time() - t0:.0f}s', flush=True)
tag = MODEL.split('.')[0]
np.savez(f'work/emb_{tag}.npz', **{k: np.concatenate(v) for k, v in out.items()})
print('done', time.time() - t0)
