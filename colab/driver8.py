# A100 frozen-tower gate (session 8: DINOv3-H+/7B; session 9: VLM towers re-probed in bf16 with mean token pooling, GP=avg).
# emb_giant.py -> probe8.py gate vs cat6+PE-G. Outputs -> Drive MyDrive/datascape_2026/session<S> and asuudane/datascape-colab-out<S>. The cell then releases the runtime.
import glob, json, os, shutil, subprocess, time
S = os.environ.get('SESSION', '8'); IN, OUT = '/content/in', f'/content/out{S}'; T0 = time.time()
for d in (IN, OUT, 'work/emb'): os.makedirs(d, exist_ok=True)
def sh(c, env=None): print(f'[{(time.time() - T0) / 60:.1f}m] $ {c}', flush=True); return subprocess.run(c, shell=True, env=env).returncode
sh('pip -q install -U timm'); sh(f'kaggle datasets download asuudane/datascape-small-512 -p {IN}/small512 --unzip -q')
sh(f'cd {IN}/small512 && for z in *.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm "$z"; done; ls {IN}/small512')
sh(f'kaggle datasets download asuudane/datascape-aux -p {IN}/aux --unzip -q')
NEW = [('vit_huge_plus_patch16_dinov3.lvd1689m', 256), ('vit_7b_patch16_dinov3.lvd1689m', 256)] if S == '8' else \
      [('vit_intern300m_patch14_448.ogvl_2pt5', 448), ('aimv2_huge_patch14_448.apple_pt', 448), ('qwen3_vit_306m_merge.qwen3_vl_4b', 448)]
ENV = dict(os.environ, DS_IN=IN, DS_OUT=OUT, NW='10', GP='' if S == '8' else 'avg')
for name, size in NEW: sh(f'python -u /content/code/emb_giant.py {name} {size}', ENV); sh(f'cp {OUT}/emb/*.npz work/emb/ 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader')
CAT6 = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224']
for t in CAT6: sh(f'kaggle datasets download asuudane/datascape-artifacts -f emb/emb_{t}.npz -p work/emb -q --force')
sh('kaggle datasets download asuudane/datascape-colab-out -f emb/emb_vit_pe_core_gigantic_patch14_448.npz -p work/emb -q --force')
sh('python -u /content/code/probe8.py ' + ' '.join(n.split('.')[0] for n, _ in NEW) + f' 2>&1 | tee {OUT}/probe8_log.txt')
for src in glob.glob('work/preds/p8_*'): shutil.copytree(src, f'{OUT}/{os.path.basename(src)}', dirs_exist_ok=True)
if os.path.exists('work/probe8_scores.csv'): shutil.copy('work/probe8_scores.csv', OUT)
json.dump(dict(title=f'datascape-colab-out{S}', id=f'asuudane/datascape-colab-out{S}', licenses=[dict(name='CC0-1.0')]), open(f'{OUT}/dataset-metadata.json', 'w'))
if sh(f'kaggle datasets version -p {OUT} -m "session{S}" -r zip -q'): sh(f'kaggle datasets create -p {OUT} -r zip -q')
if os.path.isdir('/content/drive/MyDrive'): shutil.copytree(OUT, f'/content/drive/MyDrive/datascape_2026/session{S}', dirs_exist_ok=True); print('copied to Drive', flush=True)
print(f'DRIVER{S} DONE {(time.time() - T0) / 60:.1f} min', flush=True)
