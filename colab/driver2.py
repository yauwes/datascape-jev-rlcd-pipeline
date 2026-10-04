# A100 session 2: MLP-head search over frozen-tower feature sets (group90 = evaluator; random folds = twin-seen proxy for discoveries #18).
# Inputs are fetched per file: whole-dataset downloads 404 while a version is still processing.
# Every config is saved to the private dataset asuudane/datascape-colab-out2. The notebook cell releases the runtime afterwards.
import json, os, subprocess, time
IN, OUT = '/content/in', '/content/out2'; os.makedirs(IN, exist_ok=True); os.makedirs(OUT, exist_ok=True); T0 = time.time()
ENV = dict(os.environ, DS_IN=IN, DS_OUT=OUT)

def sh(c): print(f'[{(time.time() - T0) / 60:.1f}m] $ {c}', flush=True); return subprocess.run(c, shell=True, env=ENV).returncode

def get(ds, f): sh(f'kaggle datasets download {ds} -f {f} -p {IN} -q --force')

def save(msg):
    json.dump(dict(title='datascape-colab-out2', id='asuudane/datascape-colab-out2', licenses=[dict(name='CC0-1.0')]), open(f'{OUT}/dataset-metadata.json', 'w'))
    if sh(f'kaggle datasets version -p {OUT} -m "{msg}" -r zip -q'): sh(f'kaggle datasets create -p {OUT} -r zip -q')

CAT6 = ['vit_pe_core_large_patch14_336', 'vit_so400m_patch16_siglip_384', 'vit_large_patch14_dinov2', 'vit_large_patch14_clip_224', 'vit_huge_patch14_clip_224', 'eva02_large_patch14_clip_224']
GIANT = ['vit_pe_core_gigantic_patch14_448', 'eva02_enormous_patch14_clip_224', 'vit_huge_patch14_clip_378', 'vit_giantopt_patch16_siglip_384', 'vit_gigantic_patch14_clip_378']
for f in ['meta/meta_sorted.csv', 'meta/folds_group90.csv', 'meta/folds_random.csv'] + [f'emb/emb_{t}.npz' for t in CAT6]: get('asuudane/datascape-artifacts', f)
for t in GIANT: get('asuudane/datascape-colab-out', f'emb/emb_{t}.npz')
sh(f'ls -la {IN}')

PEG, EVAE, MC2H, SIGG, MC2G = GIANT
SETS = {'cat6': CAT6, 'cat6_peg': CAT6 + [PEG], 'cat6x': [PEG] + CAT6[1:], 'cat9g': CAT6 + [PEG, EVAE, MC2H], 'cat11': CAT6 + GIANT,
        'giants5': GIANT, 'cat4g': [PEG, 'vit_so400m_patch16_siglip_384', 'vit_huge_patch14_clip_224', EVAE]}
for name, towers in SETS.items():  # group90 evaluator, 5 seeds
    sh(f'python heads.py g_{name} folds_group90.csv 5 {",".join(towers)}'); save(f'g_{name}')
for name in ['cat6', 'cat9g', 'cat11']:  # random folds: the model has seen each row's twins (decisive twin check)
    sh(f'python heads.py r_{name} folds_random.csv 5 {",".join(SETS[name])}'); save(f'r_{name}')
sh(f'cat {OUT}/heads_scores.csv'); print(f'DRIVER2 DONE {(time.time() - T0) / 60:.1f} min', flush=True)
