# Session 12 A100 runner, started from datascape_a100_runner.ipynb after drive.mount + the Kaggle token (Colab Secret) + repo -> /content/ds.
# Images -> execute ds12.ipynb with nbconvert (the executed copy keeps every output) -> live status + ETA every minute -> Drive + Kaggle.
# The runner cell's next line calls runtime.unassign(), so the A100 is released even if this script fails.
import json, os, shutil, subprocess, sys, time
T0 = time.time(); IN, OUT, CAP, WOUT = '/content/in', '/content/out12', 190, '/content/w12'   # CAP: hard wall-time cap in minutes
os.makedirs(OUT, exist_ok=True); os.makedirs(WOUT, exist_ok=True); os.makedirs(f'{IN}/extra', exist_ok=True)
def sh(c): print(f'[{(time.time() - T0) / 60:.1f}m] $ {c}', flush=True); return subprocess.run(c, shell=True).returncode
sh('pip -q install -U timm; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')
sh(f'kaggle datasets download asuudane/datascape-small-512 -p {IN}/small512 --unzip -q && cd {IN}/small512 && for z in *.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm "$z"; done; ls')
sh('kaggle datasets files asuudane/datascape-artifacts --page-size 200 > /content/out12/artifacts_inventory.txt 2>&1; kaggle datasets files asuudane/datascape-aux --page-size 200 >> /content/out12/artifacts_inventory.txt 2>&1; wc -l /content/out12/artifacts_inventory.txt')
print('session-11 gate on Drive:', os.path.exists('/content/drive/MyDrive/datascape_2026/session11/gate11_result.json'), flush=True)
nb_out = f'{OUT}/Datascape_S12_final_H+_A100.ipynb'
p = subprocess.Popen([sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', '--allow-errors', '--ExecutePreprocessor.timeout=-1', '--output', nb_out, '/content/code/ds12.ipynb'],
                     env=dict(os.environ, DS_WORK='/content/ds', DS_OUT12=OUT, DS_CODE='/content/code', DS_WOUT=WOUT))
def status():
    try: s = json.load(open(f'{OUT}/progress.json'))
    except Exception: return 'starting notebook'
    line = f"[{(time.time() - T0) / 60:.0f}m] stage {s['stage']}"
    ft = f"{OUT}/progress_{s['stage'][3:]}.json" if s['stage'].startswith('ft_') else ''
    if ft and os.path.exists(ft):
        f = json.load(open(ft)); eta = f['eta'] / 60 + 3   # + final TTA prediction
        vf = 'full-fit (no hold-out)' if f.get('mode') == 'full-fit' else ('n/a until epoch 1 ends' if f['val_f1'] is None else round(f['val_f1'], 4))
        line += f" | ep {f['ep'] + 1} it {f['it']}/{f['total']} loss {f['loss']} (floor 0.291 with LS 0.1) train acc {f.get('acc', '-')} | val F1 {vf}"
    else: eta = max(s['cur_plan'] - (time.time() - s['ts']) / 60, 0.5)
    return line + f" | stage ETA {eta:.0f}m | total ETA {eta + s['plan_after']:.0f}m (finish ~{time.strftime('%H:%M', time.gmtime(time.time() + 60 * (eta + s['plan_after'])))} UTC)"
while p.poll() is None:
    time.sleep(60); print(status(), flush=True)
    if (time.time() - T0) / 60 > CAP: p.kill(); print(f'wall-time cap {CAP} min reached: notebook killed', flush=True); break
print(f'nbconvert exit {p.returncode}', flush=True); shutil.copy('/content/code/ds12.ipynb', OUT)
json.dump(dict(title='datascape-colab-out12', id='asuudane/datascape-colab-out12', licenses=[dict(name='CC0-1.0')]), open(f'{OUT}/dataset-metadata.json', 'w'))
if sh(f'kaggle datasets version -p {OUT} -m "session12" -r zip -q'): sh(f'kaggle datasets create -p {OUT} -r zip -q')
if os.path.isdir('/content/drive/MyDrive'):
    shutil.copytree(OUT, '/content/drive/MyDrive/datascape_2026/session12', dirs_exist_ok=True); shutil.copytree(WOUT, '/content/drive/MyDrive/datascape_2026/session12/weights', dirs_exist_ok=True)
    print('copied to Drive datascape_2026/session12 (+ weights/)', flush=True)
print(f'RUN12 DONE {(time.time() - T0) / 60:.1f} min', flush=True)
