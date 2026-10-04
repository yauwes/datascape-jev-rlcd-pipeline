# Datascape fine-tune: one fold per child process (2 GPUs in parallel), resumable, group folds, noise-aware augs.
import glob, json, os, sys, time, subprocess
import numpy as np, pandas as pd
try:  # optional: HF_TOKEN from Kaggle Secrets (higher HF rate limit); silently skipped if absent
    from kaggle_secrets import UserSecretsClient; os.environ.setdefault('HF_TOKEN', UserSecretsClient().get_secret('HF_TOKEN'))
except Exception: pass

CFG = dict(model='vit_large_patch14_clip_224.openai', img=224, epochs=50, patience=5, aug='base', mix='', median=True, resize='squash', lr=3e-5, layer_decay=0.8, bs=32,
           ls=0.1, wd=0.05, folds=[0, 1, 2, 3, 4], seed=42, tag='clipL_224', balance='none', drop_file='', pseudo_file='', full_fit=False)
CFG.update({'model': 'vit_pe_core_base_patch16_224.fb', 'img': 224, 'lr': 5e-05, 'layer_decay': 0.75, 'bs': 48, 'tag': 'pebB_strong', 'aug': 'strong', 'folds': [0, 1]})
CFG.update(json.loads(os.environ.get('CFG', '{}')))
for p in glob.glob('/kaggle/input/**/cfg_override.json', recursive=True): CFG.update(json.load(open(p)))
OUT = '/kaggle/working' if os.path.exists('/kaggle/working') else 'work/ft_smoke'
SMOKE = not os.path.exists('/kaggle/working')
C = ['Aman', 'Lainnya', 'Rusak']

def find(name):
    roots = ['/kaggle/input'] if not SMOKE else ['work']
    for r in roots:
        h = glob.glob(f'{r}/**/{name}', recursive=True)
        if h: return h[0]

def done_files(fold):
    pat = f'pred_{CFG["tag"]}_f{fold}.npz'
    return glob.glob(f'{OUT}/{pat}') + (glob.glob(f'/kaggle/input/**/{pat}', recursive=True) if not SMOKE else [])

def run_fold(fold, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch, timm, timm.optim, torch.nn as nn, torch.nn.functional as F
    from PIL import Image, ImageFilter, ImageOps
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.v2 as T
    torch.manual_seed(CFG['seed'] + fold); np.random.seed(CFG['seed'] + fold)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = pd.read_csv(find('meta_sorted.csv')); folds = pd.read_csv(find('folds.csv')).set_index('path')
    root = os.path.dirname(os.path.dirname(find('Aman__Aman_00001.jpg')))
    tr = meta[meta.split == 'train'].copy(); te = meta[meta.split == 'test'].copy()
    tr['fold'] = folds.loc[tr.path, 'fold'].values; tr['y'] = tr.label.map({c: i for i, c in enumerate(C)})
    if CFG['drop_file']: tr = tr[~tr.path.isin(set(pd.read_csv(find(CFG['drop_file'])).path))]
    trn = tr[tr.fold != fold] if not CFG['full_fit'] else tr
    val = tr[tr.fold == fold]
    Ytrn = np.eye(3, dtype=np.float32)[trn.y.values]
    if CFG['pseudo_file']:  # soft pseudo labels on test (columns path, Aman, Lainnya, Rusak)
        ps = pd.read_csv(find(CFG['pseudo_file']))
        trn = pd.concat([trn, te.set_index('path').loc[ps.path].reset_index()]); Ytrn = np.concatenate([Ytrn, ps[C].values.astype(np.float32)])
    if SMOKE: trn, val, te, Ytrn = trn.iloc[:64], val.iloc[:32], te.iloc[:32], Ytrn[:64]
    S = CFG['img']
    m = timm.create_model(CFG['model'], pretrained=True, num_classes=3, img_size=S) if 'eva' in CFG['model'] or 'vit' in CFG['model'] else timm.create_model(CFG['model'], pretrained=True, num_classes=3)
    m = m.to(dev)
    cfg = timm.data.resolve_data_config({}, model=m); norm = T.Normalize(cfg['mean'], cfg['std'])

    def noise(im):  # mimic test-set corruptions
        a = np.asarray(im).astype(np.float32); r = np.random.rand()
        if r < 0.15:
            msk = np.random.rand(*a.shape[:2]); p = np.random.uniform(0.01, 0.1)
            a[msk < p / 2] = 0; a[msk > 1 - p / 2] = 255
        elif r < 0.3: a = a + np.random.randn(*a.shape) * np.random.uniform(5, 40)
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))
    base = [T.RandomResizedCrop(S, scale=(0.4, 1), ratio=(0.6, 1.66)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.2),
            T.RandomApply([T.RandomRotation(90)], p=0.2), T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.RandomGrayscale(0.1)]
    aug = T.Compose({'light': [T.RandomResizedCrop(S, scale=(0.7, 1)), T.RandomHorizontalFlip()],
                     'base': base,
                     'strong': base + [T.TrivialAugmentWide()]}[CFG['aug']])  # Mueller 2021 TrivialAugment
    erase = T.RandomErasing(p=0.25) if CFG['aug'] == 'strong' else None   # Zhong 2020 random erasing
    class DS(Dataset):
        def __init__(s, df, Y=None, train=False): s.p, s.Y, s.train = df.path.values, Y, train
        def __len__(s): return len(s.p)
        def __getitem__(s, i):
            im = Image.open(f'{root}/{s.p[i].split("work/small/")[1]}').convert('RGB')
            if s.train:
                if np.random.rand() < 0.2: im = im.resize((224, 224)).resize(im.size)  # low-res copies seen in test
                im = noise(im)
            if CFG['median']: im = im.filter(ImageFilter.MedianFilter(3))
            if CFG['resize'] == 'pad': im = ImageOps.pad(im, (max(im.size),) * 2, color=(128, 128, 128))  # letterbox keeps aspect ratio
            x = T.functional.to_image(im)
            x = aug(x) if s.train else T.functional.resize(x, [S, S], antialias=True)
            x = norm(T.functional.to_dtype(x, torch.float32, scale=True))
            if s.train and erase is not None: x = erase(x)
            return (x, s.Y[i]) if s.Y is not None else x
    nw = 4 if not SMOKE else 0
    prior = torch.tensor(Ytrn.mean(0) + 1e-6, device=dev)
    smp = None
    if CFG['balance'] == 'sqrt':  # Kang 2020: square-root sampling
        w = (Ytrn @ (1 / np.sqrt(Ytrn.sum(0) + 1)));  smp = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)
    cw = (1 / prior) / (1 / prior).sum() * 3 if CFG['balance'] == 'cw' else torch.ones(3, device=dev)
    dl = DataLoader(DS(trn, Ytrn, True), batch_size=CFG['bs'], shuffle=smp is None, sampler=smp, num_workers=nw, drop_last=True, pin_memory=True)
    opt = timm.optim.create_optimizer_v2(m, 'adamw', lr=CFG['lr'], weight_decay=CFG['wd'], layer_decay=CFG['layer_decay'])
    steps = CFG['epochs'] * len(dl); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in opt.param_groups], total_steps=max(steps, 1), pct_start=1 / CFG['epochs'])  # 1-epoch warmup, cosine to max epochs
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda')
    def predict(df, tta=True):
        m.eval(); out = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
            for x in DataLoader(DS(df), batch_size=64, num_workers=nw):
                x = x.to(dev); p = m(x).float().softmax(1)
                if tta: p = (p + m(x.flip(3)).float().softmax(1)) / 2
                out.append(p.cpu().numpy())
        m.train(); return np.concatenate(out)
    from sklearn.metrics import f1_score
    t0 = time.time(); m.train(); hist, best, bad = [], (np.inf, -1, None), 0
    for ep in range(CFG['epochs']):
        tl = 0
        for x, yb in dl:
            x, yb = x.to(dev, non_blocking=True), yb.to(dev)
            if CFG['mix'] and np.random.rand() < 0.5:  # Zhang 2018 mixup / Yun 2019 cutmix (both noted robust to label noise)
                idx = torch.randperm(x.size(0), device=dev)
                if CFG['mix'] == 'mixup': lam = float(np.random.beta(0.4, 0.4)); x = lam * x + (1 - lam) * x[idx]
                else:
                    lam = float(np.random.beta(1.0, 1.0)); H, W = x.shape[2:]; ch, cw_ = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
                    cy, cx = np.random.randint(H), np.random.randint(W); y0, y1, x0, x1 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw_ // 2, 0), min(cx + cw_ // 2, W)
                    x = x.clone(); x[:, :, y0:y1, x0:x1] = x[idx, :, y0:y1, x0:x1]; lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
                yb = lam * yb + (1 - lam) * yb[idx]
            yb = yb * (1 - CFG['ls']) + CFG['ls'] / 3
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
                z = m(x).float()
                if CFG['balance'] == 'la': z = z + torch.log(prior)  # Menon 2021 logit-adjusted loss, tau=1
                lp = F.log_softmax(z, 1)
                if CFG['balance'] == 'focal': lp = lp * (1 - lp.exp()) ** 2  # Lin 2017, gamma=2
                loss = torch.sum(-yb * cw * lp, 1).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); tl += loss.item()
        msg = f'fold {fold} ep {ep} loss {tl / max(len(dl), 1):.4f} {time.time() - t0:.0f}s'
        if not CFG['full_fit']:  # early stopping on val NLL (smoother than F1 -> less selection bias on OOF F1)
            pv = predict(val, tta=False); vl = float(-np.log(pv[np.arange(len(val)), val.y.values] + 1e-9).mean())
            vf = f1_score(val.y, pv.argmax(1), average='macro'); hist.append(dict(ep=ep, train_loss=tl / max(len(dl), 1), val_loss=vl, val_f1=vf))
            msg += f' valLoss {vl:.4f} valF1 {vf:.4f}'
            if vl < best[0]: best = (vl, ep, {k: v.detach().to('cpu', copy=True) for k, v in m.state_dict().items()}); bad = 0
            else: bad += 1
            if bad >= CFG['patience']: print(msg + ' -> early stop', flush=True); break
        print(msg, flush=True)
    if best[2] is not None: m.load_state_dict(best[2]); print(f'fold {fold} restored best epoch {best[1]} valLoss {best[0]:.4f}', flush=True)
    json.dump(dict(hist=hist, best_epoch=best[1]), open(f'{OUT}/hist_{CFG["tag"]}_f{fold}.json', 'w'))
    pv = predict(val) if not CFG['full_fit'] else np.zeros((0, 3)); pt = predict(te)
    np.savez(f'{OUT}/pred_{CFG["tag"]}_f{fold}.npz', val_path=val.path.values, val=pv, test_path=te.path.values, test=pt)
    torch.save({k: v.half() for k, v in m.state_dict().items()}, f'{OUT}/w_{CFG["tag"]}_f{fold}.pt')
    print(f'fold {fold} saved {time.time() - t0:.0f}s', flush=True)

if __name__ == '__main__':
    if len(sys.argv) > 2: run_fold(int(sys.argv[1]), int(sys.argv[2])); sys.exit()
    os.makedirs(OUT, exist_ok=True); print(CFG, flush=True)
    todo = [f for f in CFG['folds'] if not done_files(f)]
    ngpu = 1 if SMOKE else max(1, len(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip().splitlines()))
    running = {}
    while todo or running:
        for g in range(ngpu):
            if g not in running and todo:
                f = todo.pop(0); running[g] = (f, subprocess.Popen([sys.executable, __file__, str(f), str(g)]))
        time.sleep(5)
        for g, (f, p) in list(running.items()):
            if p.poll() is not None: print(f'fold {f} exit {p.returncode}', flush=True); del running[g]
    print('ALL DONE', flush=True)
