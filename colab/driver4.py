# A100 session 4 (run from /content/ds = unzipped repo.zip): domain-adapted heads (train + twin-locked test rows) -> candidate d04_da_bt2.
# Everything (heads, preds, submissions) goes to Google Drive (MyDrive/datascape_2026/session4) and the private dataset asuudane/datascape-colab-out4.
import glob, json, os, shutil, subprocess, time
T0 = time.time()
def sh(c): print(f'[{(time.time() - T0) / 60:.1f}m] $ {c}', flush=True); return subprocess.run(c, shell=True).returncode
CAT6 = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224']
GIANT = ['vit_pe_core_gigantic_patch14_448', 'eva02_enormous_patch14_clip_224', 'vit_huge_patch14_clip_378', 'vit_giantopt_patch16_siglip_384', 'vit_gigantic_patch14_clip_378']
for t in CAT6: sh(f'kaggle datasets download asuudane/datascape-artifacts -f emb/emb_{t}.npz -p work/emb -q --force')
for t in GIANT: sh(f'kaggle datasets download asuudane/datascape-colab-out -f emb/emb_{t}.npz -p work/emb -q --force')
sh('ls work/emb | wc -l; nvidia-smi --query-gpu=name --format=csv,noheader')
sh('python -u work/da_heads.py')
sh(f'python -u work/twin.py build sscd 0.55 da_h2 d04_da_bt2 0.9 bt {os.environ["BSTAR"]}')
OUT = '/content/out4'; os.makedirs(OUT, exist_ok=True)
for src in ['work/preds/da_h2', 'work/models_da', 'submissions']: shutil.copytree(src, f'{OUT}/{os.path.basename(src)}', dirs_exist_ok=True)
for f in glob.glob('work/sub_d04_*.csv'): shutil.copy(f, OUT)
json.dump(dict(title='datascape-colab-out4', id='asuudane/datascape-colab-out4', licenses=[dict(name='CC0-1.0')]), open(f'{OUT}/dataset-metadata.json', 'w'))
if sh(f'kaggle datasets version -p {OUT} -m "session4 DA" -r zip -q'): sh(f'kaggle datasets create -p {OUT} -r zip -q')
if os.path.isdir('/content/drive/MyDrive'):
    shutil.copytree(OUT, '/content/drive/MyDrive/datascape_2026/session4', dirs_exist_ok=True); sh('ls -R /content/drive/MyDrive/datascape_2026 | head -40')
print(f'DRIVER4 DONE {(time.time() - T0) / 60:.1f} min', flush=True)
