# File-level forensics for every raw image: format, JPEG quantization-table hash + quality estimate, chroma subsampling,
# progressive flag, JFIF/EXIF/ICC/Adobe presence. Output: work/forensics.csv aligned with work/emb/meta_sorted.csv.
import os, glob, hashlib, numpy as np, pandas as pd
from PIL import Image, JpegImagePlugin
M = pd.read_csv('work/emb/meta_sorted.csv')
raw_tr = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob('datascape/Train_Set/*/*')}
raw_te = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob('datascape/Test_Set/*')}
def raw(path, split):
    st = path.split('/')[-1].rsplit('.', 1)[0]; return raw_te[st] if split == 'test' else raw_tr[st.split('__', 1)[1]]
rows = []
for p, s in zip(M.path, M.split):
    f = raw(p, s); r = dict(path=p, raw=f, ext=f.rsplit('.', 1)[-1].lower(), kb=os.path.getsize(f) / 1024)
    try:
        im = Image.open(f); info = im.info; r.update(fmt=im.format, mode=im.mode, prog=int(bool(info.get('progressive') or info.get('progression'))),
                                                       jfif=int('jfif' in info), exif=int('exif' in info), icc=int('icc_profile' in info), adobe=int('adobe' in info),
                                                       dpi=str(info.get('dpi', '')), comment=str(info.get('comment', ''))[:40])
        if im.format == 'JPEG':
            q = im.quantization; t0 = np.array(q.get(0, []))
            r.update(qhash=hashlib.md5(str([q[k] for k in sorted(q)]).encode()).hexdigest()[:8], q0mean=round(float(t0.mean()), 2) if t0.size else np.nan,
                     ntab=len(q), samp=JpegImagePlugin.get_sampling(im))
        else: r.update(qhash=im.format, q0mean=np.nan, ntab=0, samp=-1)
    except Exception as e: r.update(fmt='ERR', qhash='ERR')
    rows.append(r)
F = pd.DataFrame(rows); F.to_csv('work/forensics.csv', index=False); print(F.shape, F.fmt.value_counts().to_dict(), '| distinct qhash', F.qhash.nunique())
