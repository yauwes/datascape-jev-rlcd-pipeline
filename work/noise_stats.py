# Per-image corruption stats: impulse fraction, gaussian-noise proxy, grayscale flag.
import numpy as np, pandas as pd
from multiprocessing import Pool
from PIL import Image, ImageFilter
def f(p):
    im = Image.open(p).convert('RGB'); a = np.asarray(im).astype(np.int16); md = np.asarray(im.filter(ImageFilter.MedianFilter(3))).astype(np.int16)
    d = np.abs(a - md).max(2)
    gray = float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())
    return p, float((d > 80).mean()), float(np.median(d)), gray
if __name__ == '__main__':
    m = pd.read_csv('work/meta.csv')
    with Pool(5) as pool: r = pool.map(f, m.path, chunksize=64)
    s = pd.DataFrame(r, columns=['path', 'impulse', 'medabs', 'chroma']); m = m.merge(s, on='path'); m.to_csv('work/meta_noise.csv', index=False)
    m['grp'] = np.where(m.split == 'test', 'test', m.label)
    m['sp'] = m.impulse > 0.02; m['gauss'] = (m.medabs >= 6) & ~m.sp; m['gray'] = m.chroma < 1
    print(m.groupby('grp')[['sp', 'gauss', 'gray']].mean().round(3))
