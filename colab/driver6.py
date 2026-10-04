# A100 session 6: head x noise-robust-loss x label sweep (sweep.py).
# Outputs go to Google Drive MyDrive/datascape_2026/session6 and the private dataset asuudane/datascape-colab-out6; the cell then releases the runtime.
import glob, json, os, shutil, subprocess, time
T0 = time.time()
def sh(c): print(f'[{(time.time() - T0) / 60:.1f}m] $ {c}', flush=True); return subprocess.run(c, shell=True).returncode
CAT6 = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224']
GIANT = ['vit_pe_core_gigantic_patch14_448', 'eva02_enormous_patch14_clip_224', 'vit_huge_patch14_clip_378', 'vit_giantopt_patch16_siglip_384', 'vit_gigantic_patch14_clip_378']
for t in CAT6: sh(f'kaggle datasets download asuudane/datascape-artifacts -f emb/emb_{t}.npz -p work/emb -q --force')
for t in GIANT: sh(f'kaggle datasets download asuudane/datascape-colab-out -f emb/emb_{t}.npz -p work/emb -q --force')
sh('ls work/emb | wc -l; nvidia-smi --query-gpu=name --format=csv,noheader')
sh('python -u /content/code/sweep.py 2>&1 | tee work/sweep_log.txt')
OUT = '/content/out6'; os.makedirs(OUT, exist_ok=True)
for src in glob.glob('work/preds/sw_*'):
    if os.path.isdir(src): shutil.copytree(src, f'{OUT}/{os.path.basename(src)}', dirs_exist_ok=True)
for f in ['work/sweep_scores.csv', 'work/sweep_log.txt']:
    if os.path.exists(f): shutil.copy(f, OUT)
json.dump(dict(title='datascape-colab-out6', id='asuudane/datascape-colab-out6', licenses=[dict(name='CC0-1.0')]), open(f'{OUT}/dataset-metadata.json', 'w'))
if sh(f'kaggle datasets version -p {OUT} -m "session6 clean" -r zip -q'): sh(f'kaggle datasets create -p {OUT} -r zip -q')
if os.path.isdir('/content/drive/MyDrive'): shutil.copytree(OUT, '/content/drive/MyDrive/datascape_2026/session6', dirs_exist_ok=True); print('copied to Drive', flush=True)
print(f'DRIVER6 DONE {(time.time() - T0) / 60:.1f} min', flush=True)
