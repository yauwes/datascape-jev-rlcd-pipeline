# Datascape: frozen-backbone embeddings for duplicate analysis + linear probes. Resumable: each model saved on completion.
import glob, os, time, subprocess, sys
import numpy as np, pandas as pd, torch, timm
try:  # optional: HF_TOKEN from Kaggle Secrets (higher HF rate limit); silently skipped if absent
    from kaggle_secrets import UserSecretsClient; os.environ.setdefault('HF_TOKEN', UserSecretsClient().get_secret('HF_TOKEN'))
except Exception: pass
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

MODELS = [  # (timm name, img size, variants) -- VLM/CLIP vision towers, batch 3
    ('vit_pe_core_base_patch16_224.fb', 224, ('orig',)),            # Meta Perception Encoder B
    ('vit_pe_spatial_base_patch16_512.fb', 512, ('orig',)),         # Meta PE-Spatial B (dense/local features)
    ('vit_so400m_patch14_siglip_gap_224.pali2_3b_pt', 224, ('orig',)),  # PaliGemma2-3B vision tower
    ('mobilenetv5_300m.gemma3n', 512, ('orig',)),                   # Gemma 3n MobileNetV5
    ('vit_intern300m_patch14_448.ogvl_2pt5', 448, ('orig',)),       # InternVL2.5 InternViT-300M
    ('eva02_large_patch14_clip_224.merged2b', 224, ('orig',)),      # EVA02-CLIP-L (Sun 2023)
    ('vit_huge_patch14_clip_224.dfn5b', 224, ('orig',)),            # Apple DFN5B CLIP-H (Fang 2023)
    ('aimv2_large_patch14_224.apple_pt', 224, ('orig',)),           # Apple AIMv2-L (Fini 2024)
    ('vit_large_patch14_clip_336.openai', 336, ('orig',)),          # OpenAI CLIP-L@336 (Radford 2021) - OpenAI's open vision tower
]
dev = 'cuda'
def run(name, size, variants):
    tag = name.split('.')[0]
    if done(tag): print('skip', tag); return
    t0 = time.time()
    m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(dev).half()
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1).half(); std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1).half()
    out = {v: [] for v in variants}
    with torch.no_grad():
        for bi, x in enumerate(DataLoader(DS(size), batch_size=64, num_workers=4, pin_memory=True)):
            x = (x.to(dev).half() / 255 - mean) / std
            for v in variants:
                out[v].append(m(x if v == 'orig' else x.flip(3)).float().cpu().numpy().astype(np.float16))
            if bi % 40 == 0: print(tag, bi * 64, len(meta), f'{time.time() - t0:.0f}s', flush=True)
    np.savez(f'/kaggle/working/emb_{tag}.npz', **{v: np.concatenate(a) for v, a in out.items()})
    print('saved', tag, f'{time.time() - t0:.0f}s', flush=True)
    del m; torch.cuda.empty_cache()
for spec in MODELS:
    try: run(*spec)
    except Exception as e: print('FAILED', spec[0], repr(e)[:300], flush=True)
print('ALL DONE')
