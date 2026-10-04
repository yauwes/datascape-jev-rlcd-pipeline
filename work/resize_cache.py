# Resize every image once to max side 512 (JPEG q93) + record metadata. Uses PIL draft() so huge JPEGs decode fast.
import os, csv, sys
from multiprocessing import Pool
from PIL import Image, ImageOps
SRC = 'datascape'; DST = 'work/small'; S = 512

def job(a):
    split, label, fn = a
    src = f'{SRC}/Train_Set/{label}/{fn}' if split == 'train' else f'{SRC}/Test_Set/{fn}'
    stem = os.path.splitext(fn)[0]
    dst = f'{DST}/{split}/{stem}.jpg' if split == 'test' else f'{DST}/{split}/{label}__{stem}.jpg'
    try:
        im = Image.open(src); w, h = im.size; mode = im.mode
        im.draft('RGB', (S, S))
        im = im.convert('RGB'); im.thumbnail((S, S), Image.BICUBIC)
        if not os.path.exists(dst): im.save(dst, quality=93)
        return (split, label, fn, dst, w, h, mode, os.path.getsize(src), '')
    except Exception as e:
        return (split, label, fn, dst, 0, 0, '', 0, repr(e))

if __name__ == '__main__':
    os.makedirs(f'{DST}/train', exist_ok=True); os.makedirs(f'{DST}/test', exist_ok=True)
    jobs = [('train', c, f) for c in ('Aman', 'Rusak', 'Lainnya') for f in sorted(os.listdir(f'{SRC}/Train_Set/{c}')) if not f.startswith('.')]
    jobs += [('test', '', f) for f in sorted(os.listdir(f'{SRC}/Test_Set')) if not f.startswith('.')]
    with Pool(5) as p, open('work/meta.csv', 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(['split', 'label', 'filename', 'path', 'w', 'h', 'mode', 'bytes', 'err'])
        for i, r in enumerate(p.imap_unordered(job, jobs, chunksize=32)):
            w.writerow(r)
            if i % 2000 == 0: print(i, len(jobs), flush=True)
    print('done')
