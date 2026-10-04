# A100 session 7: fine-tune diverse backbones (discoveries #13: diversity comes from pretraining) on CLEANED labels (#26) plus twin pseudo-labels
# (test rows whose SSCD train-twins vote with cleaned labels; DA, #23). Protocol v2, 4-view TTA, group 5-fold.
# One fold per ft.py call; preds + log are versioned to asuudane/datascape-colab-out7 after every fold; weights go to Drive when there is room.
# No new fold starts after MAX_H hours. The notebook cell releases the runtime afterwards.
import glob, json, os, shutil, subprocess, time
IN, OUT = '/content/in', '/content/out7'; T0 = time.time(); MAX_H = float(os.environ.get('MAX_H', 7))
for d in (IN, OUT, f'{IN}/extra'): os.makedirs(d, exist_ok=True)
def sh(c, env=None): print(f'[{(time.time() - T0) / 3600:.2f}h] $ {c}', flush=True); return subprocess.run(c, shell=True, env=env).returncode
sh(f'kaggle datasets download asuudane/datascape-small-512 -p {IN}/small512 --unzip -q')
sh(f'cd {IN}/small512 && for z in *.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm "$z"; done; ls {IN}/small512')
sh(f'kaggle datasets download asuudane/datascape-aux -p {IN}/aux --unzip -q')
for f in ['work/labels_clean_v2.csv', 'work/pseudo_twin_clean.csv']: shutil.copy(f, f'{IN}/extra/')
DRIVE = '/content/drive/MyDrive/datascape_2026/session7'; W = '/content/w'
if os.path.isdir('/content/drive/MyDrive'):
    free = shutil.disk_usage('/content/drive/MyDrive').free / 1e9; print(f'Drive free {free:.0f} GB', flush=True)
    if free > 30: W = f'{DRIVE}/weights'
os.makedirs(W, exist_ok=True); print('weights ->', W, flush=True)
ENV = dict(os.environ, DS_IN=IN, DS_OUT=OUT, NW='10')
COMMON = dict(label_file='labels_clean_v2.csv', pseudo_file='pseudo_twin_clean.csv', tta='hv4', save_w=True)
CFGS = [dict(model='vit_pe_core_large_patch14_336.fb', img=224, lr=3e-5, layer_decay=0.8, bs=32, aug='base', tag='ftc_peL224'),
        dict(model='eva02_large_patch14_clip_224.merged2b', img=224, lr=3e-5, layer_decay=0.8, bs=32, aug='strong', tag='ftc_eva02L224'),
        dict(model='vit_large_patch14_dinov2.lvd142m', img=224, lr=2e-5, layer_decay=0.8, bs=32, aug='base', mix='cutmix', tag='ftc_dinoL224'),
        dict(model='vit_so400m_patch16_siglip_384.v2_webli', img=384, lr=2e-5, layer_decay=0.8, bs=24, aug='base', tag='ftc_sig384')]
def save(msg):
    json.dump(dict(title='datascape-colab-out7', id='asuudane/datascape-colab-out7', licenses=[dict(name='CC0-1.0')]), open(f'{OUT}/dataset-metadata.json', 'w'))
    if sh(f'kaggle datasets version -p {OUT} -m "{msg}" -q'): sh(f'kaggle datasets create -p {OUT} -q')
    if os.path.isdir('/content/drive/MyDrive'): shutil.copytree(OUT, f'{DRIVE}/out', dirs_exist_ok=True)
for cfg in CFGS:
    for k in range(5):
        if glob.glob(f'{OUT}/pred_{cfg["tag"]}_f{k}.npz'): continue
        if (time.time() - T0) / 3600 > MAX_H: print('time budget reached', flush=True); break
        os.makedirs(f'{W}/{cfg["tag"]}', exist_ok=True)
        sh('python /content/code/ft.py 2>&1 | tee -a ' + f'{OUT}/log_{cfg["tag"]}.txt', dict(ENV, CFG=json.dumps(dict(COMMON, **cfg, folds=[k])), DS_WOUT=f'{W}/{cfg["tag"]}'))
        save(f'{cfg["tag"]} f{k}')
print(f'DRIVER7 DONE {(time.time() - T0) / 3600:.2f}h', flush=True); save('driver7 done')
