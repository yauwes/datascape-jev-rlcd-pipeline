# Final-notebook runner (Colab), started from datascape_a100_runner.ipynb after drive.mount + the Kaggle token (Colab Secret) + repo -> /content/ds.
# Overlays the pack (/content/code: latest submissions, DA-head and pixel-copy caches, ft.py, lbdelta.py, the notebook), downloads the image cache,
# executes notebook/Datascape_Final_Pipeline.ipynb with nbconvert (status + ETA every minute) and saves everything to Drive datascape_2026/final.
# Mode: QUICK by default; set FULL=1 below (A100, ~3 h) to recompute embeddings, pixel copies, heads and the H+ fine-tune from the images.
import json, os, shutil, subprocess, sys, time
FULL = os.environ.get('DS_FULL') == '1'
T0 = time.time(); IN, OUT, CAP, DRV = '/content/in', '/content/out_final', 240 if FULL else 45, '/content/drive/MyDrive/datascape_2026'
os.makedirs(OUT, exist_ok=True)
def sh(c): print(f'[{(time.time() - T0) / 60:.1f}m] $ {c}', flush=True); return subprocess.run(c, shell=True).returncode
sh('pip -q install -U timm cleanlab; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')
sh(f'kaggle datasets download asuudane/datascape-small-512 -p {IN}/small512 --unzip -q && cd {IN}/small512 && for z in *.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm "$z"; done; ls')
if FULL: sh(f'kaggle datasets download asuudane/datascape-artifacts -p {IN}/artifacts --unzip -q')     # cached embeddings, only needed for partial retrains
nb_in, nb_out = '/content/code/Datascape_Final_Pipeline.ipynb', f'{OUT}/Datascape_Final_Pipeline_executed.ipynb'
env = dict(os.environ, DS_ROOT='/content/ds', DS_IMG=f'{IN}/small512/small', DS_DRIVE=DRV, DS_OUTF=OUT, **({'DS_FULL': '1'} if FULL else {}))
p = subprocess.Popen([sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', '--allow-errors', '--ExecutePreprocessor.timeout=-1', '--output', nb_out, nb_in], env=env)
def status():
    try: s = json.load(open(f'{OUT}/progress.json'))
    except Exception: return 'starting notebook'
    eta = max(s['cur_plan'] - (time.time() - s['ts']) / 60, 0.5)
    return f"[{(time.time() - T0) / 60:.0f}m] stage {s['stage']} | total ETA {eta + s['plan_after']:.0f}m (finish ~{time.strftime('%H:%M', time.gmtime(time.time() + 60 * (eta + s['plan_after'])))} UTC)"
while p.poll() is None:
    time.sleep(30); print(status(), flush=True)
    if (time.time() - T0) / 60 > CAP: p.kill(); print(f'wall-time cap {CAP} min reached: notebook killed', flush=True); break
print(f'nbconvert exit {p.returncode}', flush=True)
errs = [(i, o['ename'], o['evalue'][:300]) for i, c in enumerate(json.load(open(nb_out))['cells']) if c['cell_type'] == 'code' for o in c.get('outputs', []) if o.get('output_type') == 'error'] if os.path.exists(nb_out) else [('-', 'no executed notebook', '')]
print('CELL ERRORS:', errs if errs else 'none', flush=True)
shutil.copy(nb_in, OUT)
if os.path.isdir('/content/drive/MyDrive'):
    shutil.copytree(OUT, f'{DRV}/final', dirs_exist_ok=True); print('copied to Drive datascape_2026/final', flush=True)
else:  # Drive not mounted: never lose the run, push it to a private Kaggle dataset instead
    json.dump(dict(title='datascape-colab-final', id='asuudane/datascape-colab-final', licenses=[dict(name='CC0-1.0')]), open(f'{OUT}/dataset-metadata.json', 'w'))
    if sh(f'kaggle datasets version -p {OUT} -m final -r zip -q'): sh(f'kaggle datasets create -p {OUT} -r zip -q')
    print('Drive not mounted: outputs saved to Kaggle dataset asuudane/datascape-colab-final', flush=True)
fin = f'{OUT}/final'
if os.path.isdir(fin): print('final submissions:', sorted(os.listdir(fin)), flush=True)
print(f'RUN_FINAL DONE {(time.time() - T0) / 60:.1f} min', flush=True)
