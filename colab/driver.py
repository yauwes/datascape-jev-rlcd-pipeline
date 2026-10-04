# A100 plan runner (Colab, cwd /content/code after setup.sh). Every step is resumable: it skips outputs that already exist.
# Every result is versioned to the private Kaggle dataset asuudane/datascape-colab-out.
# No new work is launched after MAX_H hours. The notebook then releases the runtime.
# Order = expected value per GPU-hour:
#   canary (protocol v2) -> giant frozen towers (A2) -> A15/A12 proxy ablations -> PE-L 5-fold -> SigLIP2 5-fold.
import glob, json, os, subprocess, time
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

IN, OUT, W = '/content/in', '/content/out', '/content/w'
MAX_H = float(os.environ.get('MAX_H', 8)); T0 = time.time()
ENV = dict(os.environ, DS_IN=IN, DS_OUT=OUT, NW=str(max(4, (os.cpu_count() or 8) - 2)))
C = ['Aman', 'Lainnya', 'Rusak']
meta = pd.read_csv(glob.glob(f'{IN}/**/meta_sorted.csv', recursive=True)[0]); lab = dict(zip(meta.path, meta.label))

def sh(c, env=ENV):
    print(f'[{(time.time() - T0) / 3600:.2f}h] $ {c}', flush=True); return subprocess.run(c, shell=True, env=env).returncode

def save(msg, d=OUT, slug='datascape-colab-out'):  # private by default; first call creates the dataset, later calls add versions
    json.dump(dict(title=slug, id=f'asuudane/{slug}', licenses=[dict(name='CC0-1.0')]), open(f'{d}/dataset-metadata.json', 'w'))
    if sh(f'kaggle datasets version -p {d} -m "{msg}" -r zip -q'): sh(f'kaggle datasets create -p {d} -r zip -q')

def left(): return MAX_H - (time.time() - T0) / 3600

def done(tag, k): return bool(glob.glob(f'{OUT}/pred_{tag}_f{k}.npz'))

def f1(tag, folds):
    ps = [np.load(f'{OUT}/pred_{tag}_f{k}.npz', allow_pickle=True) for k in folds]
    yt = np.concatenate([[C.index(lab[p]) for p in d['val_path']] for d in ps])
    return f1_score(yt, np.concatenate([d['val'] for d in ps]).argmax(1), average='macro')

def ft(cfg, folds, wslug=None):  # one fold per call -> checkpoint to Kaggle after every fold
    for k in folds:
        if done(cfg['tag'], k): continue
        if left() < 0.25: print('time budget reached', flush=True); return False
        env = dict(ENV, CFG=json.dumps(dict(cfg, folds=[k])), DS_WOUT=f'{W}/{cfg["tag"]}'); os.makedirs(env['DS_WOUT'], exist_ok=True)
        sh('python ft.py', env); save(f'{cfg["tag"]} f{k}')
    ok = all(done(cfg['tag'], k) for k in folds)
    if ok and wslug: save(f'{cfg["tag"]} weights', f'{W}/{cfg["tag"]}', wslug)
    return ok

# 1) canary: CLIP-L fold 3 under protocol v2. Old runs on fold 3: 6-ep fixed 0.8207; OneCycle-50 + ES best_epoch 0 -> 0.7638.
CANARY = dict(model='vit_large_patch14_clip_224.openai', img=224, lr=3e-5, layer_decay=0.8, bs=32, tag='clipL224_v2', save_w=False)
proto_ok = os.environ.get('SKIP_CANARY') == '1'
if not proto_ok and ft(CANARY, [3]):
    h = json.load(open(f'{OUT}/hist_clipL224_v2_f3.json')); lrs = [e['lr'] for e in h['hist']]; s = f1('clipL224_v2', [3])
    proto_ok = h['best_epoch'] >= 2 and min(lrs) < max(lrs) and s >= 0.8207
    print(f'CANARY best_epoch {h["best_epoch"]} lr {max(lrs):.1e}->{min(lrs):.1e} F1 {s:.4f}', 'PASS' if proto_ok else 'FAIL', flush=True)

# 2) giant frozen towers (Kornblith 2019 / Fang 2023: better pretraining -> better transfer); T4 can't hold these at speed
GIANTS = [('vit_pe_core_gigantic_patch14_448.fb', 448),                    # Meta PE-Core-G (1.9B), best family here (PE-L 0.843)
          ('vit_giantopt_patch16_siglip_384.v2_webli', 384),               # Google SigLIP2-g (1.1B)
          ('vit_gigantic_patch14_clip_378.metaclip2_worldwide', 378),      # Meta MetaCLIP2-G (new data -> diversity)
          ('eva02_enormous_patch14_clip_224.laion2b_plus', 224),           # EVA02-CLIP-E (4.4B)
          ('vit_huge_patch14_clip_378.metaclip2_worldwide', 378)]          # Meta MetaCLIP2-H
for n, s in GIANTS:
    if left() < 0.25: break
    if not glob.glob(f'{OUT}/emb/emb_{n.split(".")[0]}.npz'): sh(f'python emb_giant.py {n} {s}'); save(f'emb {n}')

# 3) A15 preprocessing/augmentation/resolution + A12 imbalance ablations, PE-Core-B@224 proxy, folds 0-1 (keep rule: 2-fold delta > +0.01)
P = dict(model='vit_pe_core_base_patch16_224.fb', img=224, lr=5e-5, layer_decay=0.75, bs=48, save_w=False)
ABL = {'base': {}, 'strong': dict(aug='strong'), 'mixup': dict(mix='mixup'), 'cutmix': dict(mix='cutmix'), 'nomed': dict(median=False),
       'pad': dict(resize='pad'), 'la': dict(balance='la'), 'sqrt': dict(balance='sqrt'), 'lpft': dict(lpft=True), 'light': dict(aug='light'),
       'r384': dict(img=384), 'r512': dict(img=512)}
SLOTS = [['strong', 'light'], ['mixup', 'cutmix'], ['nomed'], ['pad'], ['la', 'sqrt'], ['lpft']]
if proto_ok:
    for a, c in ABL.items(): ft(dict(P, tag=f'pebB2_{a}', **c), [0, 1])

# 4) final fine-tunes 5-fold with the pre-registered combination rule: per slot, take the best option only if it beats base by > 0.01
if proto_ok and done('pebB2_base', 0) and done('pebB2_base', 1):
    base = f1('pebB2_base', [0, 1]); sc = {a: f1(f'pebB2_{a}', [0, 1]) for a in ABL if done(f'pebB2_{a}', 0) and done(f'pebB2_{a}', 1)}
    print('A15/A12 2-fold OOF F1 (delta vs base):', {a: round(v - base, 4) for a, v in sc.items()}, flush=True)
    best = {}
    for slot in SLOTS:
        b = max(slot, key=lambda a: sc.get(a, -1))
        if sc.get(b, -1) > base + 0.01: best.update(ABL[b])
    hi = max(sc.get('r384', 0), sc.get('r512', 0)) > base + 0.01
    json.dump(dict(scores=sc, base=base, chosen=best, hires=hi), open(f'{OUT}/a15_choice.json', 'w')); save('a15 choice')
    FINALS = [dict(model='vit_pe_core_large_patch14_336.fb', img=448 if hi else 336, lr=3e-5, layer_decay=0.75, bs=24 if hi else 32, tag='peL_v2', **best),
              dict(model='vit_so400m_patch16_siglip_384.v2_webli', img=384, lr=3e-5, layer_decay=0.8, bs=32, tag='sigL384_v2', **best)]
    for cfg in FINALS:
        if not ft(cfg, [0, 1, 2, 3, 4], wslug='datascape-w-' + cfg['tag'].lower().replace('_', '-')): break
        print(cfg['tag'], '5-fold OOF F1', round(f1(cfg['tag'], range(5)), 4), flush=True)
print(f'DRIVER DONE {(time.time() - T0) / 3600:.2f}h', flush=True); save('driver done')
