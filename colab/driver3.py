# A100 session 3: rotation/noise-robust twin search (twin2.py). Saves to private dataset asuudane/datascape-colab-out3; the cell then releases the runtime.
import json, os, subprocess, time
IN, OUT = '/content/in', '/content/out3'; os.makedirs(IN, exist_ok=True); os.makedirs(OUT, exist_ok=True); T0 = time.time()
ENV = dict(os.environ, DS_IN=IN, DS_OUT=OUT)
def sh(c): print(f'[{(time.time() - T0) / 60:.1f}m] $ {c}', flush=True); return subprocess.run(c, shell=True, env=ENV).returncode
sh(f'kaggle datasets download asuudane/datascape-small-512 -p {IN}/small512 --unzip -q')
sh(f'cd {IN}/small512 && for z in *.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm "$z"; done; ls {IN}/small512')
sh(f'kaggle datasets download asuudane/datascape-artifacts -f meta/meta_sorted.csv -p {IN} -q --force')
sh('python -u twin2.py')
json.dump(dict(title='datascape-colab-out3', id='asuudane/datascape-colab-out3', licenses=[dict(name='CC0-1.0')]), open(f'{OUT}/dataset-metadata.json', 'w'))
if sh(f'kaggle datasets version -p {OUT} -m "twin2" -q'): sh(f'kaggle datasets create -p {OUT} -q')
print(f'DRIVER3 DONE {(time.time() - T0) / 60:.1f} min', flush=True)
