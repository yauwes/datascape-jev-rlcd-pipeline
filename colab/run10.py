# Session 10 A100 runner, started from datascape_a100_runner.ipynb after drive.mount + the Kaggle token (Colab Secret) + repo -> /content/ds.
# Data -> execute ds10.ipynb with nbconvert (executed copy keeps every EDA/eval output) -> live status + ETA every minute -> Drive + Kaggle.
# The runner cell's next line calls runtime.unassign(), so the A100 is released even if this script fails.
import glob, json, os, shutil, subprocess, sys, time
T0 = time.time(); IN, OUT, CAP = '/content/in', '/content/out10', 270   # CAP: hard wall-time cap in minutes
os.makedirs(OUT, exist_ok=True); os.makedirs(f'{IN}/extra', exist_ok=True)
def sh(c): print(f'[{(time.time() - T0) / 60:.1f}m] $ {c}', flush=True); return subprocess.run(c, shell=True).returncode
sh('pip -q install -U timm; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')
sh(f'kaggle datasets download asuudane/datascape-small-512 -p {IN}/small512 --unzip -q && cd {IN}/small512 && for z in *.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm "$z"; done; ls')
sh(f'kaggle datasets download asuudane/datascape-aux -p {IN}/aux --unzip -q')
sh(f'kaggle datasets download asuudane/datascape-colab-out8 -f emb/emb_vit_huge_plus_patch16_dinov3.npz -p {IN}/out8/emb -q --force; ls -la {IN}/out8/emb')
nb_out = f'{OUT}/Datascape_S10_DINOv3_A100.ipynb'
p = subprocess.Popen([sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', '--allow-errors', '--ExecutePreprocessor.timeout=-1', '--output', nb_out, '/content/code/ds10.ipynb'],
                     env=dict(os.environ, DS_WORK='/content/ds', DS_OUT10=OUT))
def status():
    try: s = json.load(open(f'{OUT}/progress.json'))
    except Exception: return 'starting notebook'
    el = (time.time() - T0) / 60; line = f"[{el:.0f}m] stage {s['stage']}"
    ft = f"{OUT}/progress_h3{s['stage'][-1]}.json" if s['stage'].startswith('ft_') else ''
    if ft and os.path.exists(ft):
        f = json.load(open(ft)); eta = f['eta'] / 60 + 3   # + final dihedral-TTA prediction
        line += f" | ep {f['ep'] + 1} it {f['it']}/{f['total']} loss {f['loss']} val F1 {f['val_f1'] if f['val_f1'] is None else round(f['val_f1'], 4)}"
    else: eta = max(s['cur_plan'] - (time.time() - s['ts']) / 60, 0.5)
    return line + f" | stage ETA {eta:.0f}m | total ETA {eta + s['plan_after']:.0f}m (finish ~{time.strftime('%H:%M', time.gmtime(time.time() + 60 * (eta + s['plan_after'])))} UTC)"
while p.poll() is None:
    time.sleep(60); print(status(), flush=True)
    if (time.time() - T0) / 60 > CAP: p.kill(); print(f'wall-time cap {CAP} min reached: notebook killed', flush=True); break
print(f'nbconvert exit {p.returncode}', flush=True); shutil.copy('/content/code/ds10.ipynb', OUT)
if os.path.exists(f'{OUT}/gate10_result.json'): print(open(f'{OUT}/gate10_result.json').read()[:3000], flush=True)
json.dump(dict(title='datascape-colab-out10', id='asuudane/datascape-colab-out10', licenses=[dict(name='CC0-1.0')]), open(f'{OUT}/dataset-metadata.json', 'w'))
if sh(f'kaggle datasets version -p {OUT} -m "session10" -r zip -q'): sh(f'kaggle datasets create -p {OUT} -r zip -q')
if os.path.isdir('/content/drive/MyDrive'): shutil.copytree(OUT, '/content/drive/MyDrive/datascape_2026/session10', dirs_exist_ok=True); print('copied to Drive datascape_2026/session10', flush=True)
print(f'RUN10 DONE {(time.time() - T0) / 60:.1f} min', flush=True)
