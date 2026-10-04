# SSCD copy-detection descriptors (Pizzi et al. CVPR 2022) on Kaggle T4: disc_large (ResNeXt101) + disc_mixup (ResNet50), 5 geometric variants,
# median 3x3 + 288px square, rows = meta_sorted.csv -> /kaggle/working/emb_sscd_<model>.npz
import glob, os, subprocess, numpy as np, pandas as pd, torch, cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
meta = pd.read_csv(glob.glob('/kaggle/input/**/meta_sorted.csv', recursive=True)[0])
hits = glob.glob('/kaggle/input/**/train/Aman__Aman_00001.jpg', recursive=True)
if not hits:
    z = glob.glob('/kaggle/input/**/small.zip', recursive=True)[0]; subprocess.run(['unzip', '-q', '-o', z, '-d', '/kaggle/tmp/small'], check=True)
    hits = ['/kaggle/tmp/small/train/Aman__Aman_00001.jpg']
ROOT = os.path.dirname(os.path.dirname(hits[0])); S = 288
class DS(Dataset):
    def __len__(s): return len(meta)
    def __getitem__(s, i):
        a = cv2.medianBlur(np.asarray(Image.open(f'{ROOT}/{meta.rel[i]}').convert('RGB')), 3)
        return torch.from_numpy(np.asarray(Image.fromarray(a).resize((S, S), Image.BICUBIC)).copy()).permute(2, 0, 1)
mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
V = {'orig': lambda x: x, 'hflip': lambda x: x.flip(3), 'rot90': lambda x: x.rot90(1, (2, 3)), 'rot180': lambda x: x.rot90(2, (2, 3)), 'rot270': lambda x: x.rot90(3, (2, 3))}
for name in ['sscd_disc_large', 'sscd_disc_mixup']:
    subprocess.run(['wget', '-q', '-O', f'/kaggle/working/{name}.pt', f'https://dl.fbaipublicfiles.com/sscd-copy-detection/{name}.torchscript.pt'], check=True)
    m = torch.jit.load(f'/kaggle/working/{name}.pt').eval().cuda(); out = {k: [] for k in V}
    with torch.no_grad():
        for bi, x in enumerate(DataLoader(DS(), batch_size=64, num_workers=4, pin_memory=True)):
            x = (x.cuda().float() / 255 - mean) / std
            for k, f in V.items(): out[k].append(torch.nn.functional.normalize(m(f(x)).float(), dim=1).cpu().numpy().astype(np.float16))
            if bi % 40 == 0: print(name, bi * 64, len(meta), flush=True)
    E = {k: np.concatenate(v) for k, v in out.items()}; assert all(np.isfinite(v.astype(np.float32)).all() for v in E.values())
    np.savez(f'/kaggle/working/emb_{name}.npz', **E); os.remove(f'/kaggle/working/{name}.pt'); print('saved', name, flush=True)
print('ALL DONE')
