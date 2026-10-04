# Datascape fine-tune: one fold per child process (one per GPU), resumable, group folds, noise-aware augs.
# Protocol v2 (discoveries #15): 1-ep linear warmup -> constant LR + ReduceLROnPlateau (patience < ES patience) on fp32-EMA val NLL.
# Runs on Kaggle (/kaggle/input) or anywhere with DS_IN/DS_OUT env vars (e.g. Colab A100, bf16 auto).
import glob, json, os, sys, time, subprocess
import numpy as np, pandas as pd
try:  # optional: HF_TOKEN from Kaggle Secrets (higher HF rate limit); silently skipped if absent
    from kaggle_secrets import UserSecretsClient; os.environ.setdefault('HF_TOKEN', UserSecretsClient().get_secret('HF_TOKEN'))
except Exception: pass

CFG = dict(model='vit_large_patch14_clip_224.openai', img=224, epochs=50, patience=5, aug='base', mix='', median=True, resize='squash', lr=3e-5, layer_decay=0.8, bs=32,
           ls=0.1, wd=0.05, folds=[0, 1, 2, 3, 4], seed=42, tag='clipL_224', balance='none', drop_file='', pseudo_file='', full_fit=False,
           ema=0.999, plateau=1, lr_factor=0.3, min_lr_frac=0.01, min_delta=1e-3, lpft=False, save_w=True, label_file='', tta='h')
CFG.update(json.loads(os.environ.get('CFG', '{}')))
for p in glob.glob('/kaggle/input/**/cfg_override.json', recursive=True): CFG.update(json.load(open(p)))
IN = os.environ.get('DS_IN', '/kaggle/input')
SMOKE = 'DS_IN' not in os.environ and not os.path.exists('/kaggle/working')
OUT = os.environ.get('DS_OUT') or ('work/ft_smoke' if SMOKE else '/kaggle/working')
C = ['Aman', 'Lainnya', 'Rusak']

def find(name):
    roots = [IN] if not SMOKE else ['work']
    for r in roots:
        h = glob.glob(f'{r}/**/{name}', recursive=True)
        if h: return h[0]

def done_files(fold):
    pat = f'pred_{CFG["tag"]}_f{fold}.npz'
    return glob.glob(f'{OUT}/{pat}') + (glob.glob(f'{IN}/**/{pat}', recursive=True) if not SMOKE else [])

def run_fold(fold, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch, timm, timm.optim, torch.nn as nn, torch.nn.functional as F
    from timm.utils import ModelEmaV3
    from PIL import Image, ImageOps
    import cv2
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.v2 as T
    torch.manual_seed(CFG['seed'] + fold); np.random.seed(CFG['seed'] + fold)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16 = dev == 'cuda' and torch.cuda.get_device_capability()[0] >= 8  # A100/L4/H100: bf16 + TF32, no loss scaling
    amp = torch.bfloat16 if bf16 else torch.float16; torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    meta = pd.read_csv(find('meta_sorted.csv')); folds = pd.read_csv(find('folds.csv')).set_index('path')
    root = os.path.dirname(os.path.dirname(find('Aman__Aman_00001.jpg')))
    tr = meta[meta.split == 'train'].copy(); te = meta[meta.split == 'test'].copy()
    tr['fold'] = folds.loc[tr.path, 'fold'].values; tr['y'] = tr.label.map({c: i for i, c in enumerate(C)})
    if CFG['label_file']:  # cleaned train labels (discoveries #26): csv path,label_clean
        lc = pd.read_csv(find(CFG['label_file'])).set_index('path').label_clean; tr['y'] = lc.loc[tr.path].map({c: i for i, c in enumerate(C)}).values
    if CFG['drop_file']: tr = tr[~tr.path.isin(set(pd.read_csv(find(CFG['drop_file'])).path))]
    trn = tr[tr.fold != fold] if not CFG['full_fit'] else tr
    val = tr[tr.fold == fold]
    Ytrn = np.eye(3, dtype=np.float32)[trn.y.values]
    if CFG['pseudo_file']:  # soft pseudo labels on test (columns path, Aman, Lainnya, Rusak)
        ps = pd.read_csv(find(CFG['pseudo_file']))
        trn = pd.concat([trn, te.set_index('path').loc[ps.path].reset_index()]); Ytrn = np.concatenate([Ytrn, ps[C].values.astype(np.float32)])
    if SMOKE: ix = np.random.RandomState(0).choice(len(trn), 64, replace=False); trn, val, te, Ytrn = trn.iloc[ix], val.sample(32, random_state=0), te.iloc[:32], Ytrn[ix]
    S = CFG['img']
    m = timm.create_model(CFG['model'], pretrained=True, num_classes=3, img_size=S) if 'eva' in CFG['model'] or 'vit' in CFG['model'] else timm.create_model(CFG['model'], pretrained=True, num_classes=3)
    m = m.to(dev)
    cfg = timm.data.resolve_data_config({}, model=m); norm = T.Normalize(cfg['mean'], cfg['std'])

    def noise(im):  # mimic test-set corruptions
        r = np.random.rand()
        if r >= 0.3: return im  # 70%: untouched (skips a float round-trip)
        a = np.asarray(im).astype(np.float32)
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
            if CFG['median']: im = Image.fromarray(cv2.medianBlur(np.asarray(im), 3))  # == PIL MedianFilter(3) bit-exact, ~80x faster
            if CFG['resize'] == 'pad': im = ImageOps.pad(im, (max(im.size),) * 2, color=(128, 128, 128))  # letterbox keeps aspect ratio
            x = T.functional.to_image(im)
            x = aug(x) if s.train else T.functional.resize(x, [S, S], antialias=True)
            x = norm(T.functional.to_dtype(x, torch.float32, scale=True))
            if s.train and erase is not None: x = erase(x)
            return (x, s.Y[i]) if s.Y is not None else x
    nw = int(os.environ.get('NW', 4)) if not SMOKE else 0
    prior = torch.tensor(Ytrn.mean(0) + 1e-6, device=dev)
    smp = None
    if CFG['balance'] == 'sqrt':  # Kang 2020: square-root sampling
        w = (Ytrn @ (1 / np.sqrt(Ytrn.sum(0) + 1)));  smp = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)
    cw = (1 / prior) / (1 / prior).sum() * 3 if CFG['balance'] == 'cw' else torch.ones(3, device=dev)
    dl = DataLoader(DS(trn, Ytrn, True), batch_size=CFG['bs'], shuffle=smp is None, sampler=smp, num_workers=nw, drop_last=True, pin_memory=True)
    if CFG['lpft']:  # Kumar 2022 LP-FT: fit the linear head on frozen features first -> full FT starts from a good head, less feature distortion
        from sklearn.linear_model import LogisticRegression
        m.eval(); fs = []
        with torch.no_grad(), torch.autocast(dev, dtype=amp, enabled=dev == 'cuda'):
            for x in DataLoader(DS(trn), batch_size=128, num_workers=nw): fs.append(m.forward_head(m.forward_features(x.to(dev)), pre_logits=True).float().cpu().numpy())
        lp = LogisticRegression(C=0.5, max_iter=3000).fit(np.concatenate(fs), Ytrn.argmax(1)); hd = m.get_classifier()
        with torch.no_grad(): hd.weight.copy_(torch.tensor(lp.coef_)); hd.bias.copy_(torch.tensor(lp.intercept_))
        m.train(); print(f'fold {fold} LP-FT head fitted', flush=True)
    opt = timm.optim.create_optimizer_v2(m, 'adamw', lr=CFG['lr'], weight_decay=CFG['wd'], layer_decay=CFG['layer_decay'])
    for g in opt.param_groups: g['lr0'] = g['lr']
    plat = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=CFG['lr_factor'], patience=CFG['plateau'], threshold=CFG['min_delta'], threshold_mode='abs',
                                                      min_lr=[g['lr'] * CFG['min_lr_frac'] for g in opt.param_groups])
    warm, total = len(dl), CFG['epochs'] * len(dl)
    def set_lr(it):  # per step: linear warmup over epoch 0; full_fit (no val) instead cosine-decays over the fixed epochs
        if it < warm: f = (it + 1) / warm
        elif CFG['full_fit']: f = 0.5 * (1 + np.cos(np.pi * (it - warm) / max(1, total - warm)))
        else: return
        for g in opt.param_groups: g['lr'] = g['lr0'] * f
    ema = ModelEmaV3(m, decay=CFG['ema'], use_warmup=True)  # fp32 copy; all eval/selection/saving uses it
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda' and not bf16)
    def predict(df, tta=True):
        net = ema.module.eval(); out = []
        with torch.no_grad(), torch.autocast(dev, dtype=amp, enabled=dev == 'cuda'):
            for x in DataLoader(DS(df), batch_size=128, num_workers=nw):
                x = x.to(dev); p = net(x).float().softmax(1)
                if tta and CFG['tta'] == 'hv4': p = (p + net(x.flip(3)).float().softmax(1) + net(x.flip(2)).float().softmax(1) + net(x.flip(2).flip(3)).float().softmax(1)) / 4
                elif tta: p = (p + net(x.flip(3)).float().softmax(1)) / 2
                out.append(p.cpu().numpy())
        return np.concatenate(out)
    from sklearn.metrics import f1_score
    t0 = time.time(); m.train(); hist, best, bad, it = [], (np.inf, -1, None), 0, 0
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
            set_lr(it)
            with torch.autocast(dev, dtype=amp, enabled=dev == 'cuda'):
                z = m(x).float()
                if CFG['balance'] == 'la': z = z + torch.log(prior)  # Menon 2021 logit-adjusted loss, tau=1
                lp = F.log_softmax(z, 1)
                if CFG['balance'] == 'focal': lp = lp * (1 - lp.exp()) ** 2  # Lin 2017, gamma=2
                loss = torch.sum(-yb * cw * lp, 1).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update(); ema.update(m, step=it); it += 1; tl += loss.item()
        lr = max(g['lr'] for g in opt.param_groups); msg = f'fold {fold} ep {ep} lr {lr:.2e} loss {tl / max(len(dl), 1):.4f} {time.time() - t0:.0f}s'
        if not CFG['full_fit']:  # EMA val NLL drives plateau decay, ES and checkpoint choice (smoother than F1 -> less selection bias)
            pv = predict(val, tta=False); vl = float(-np.log(pv[np.arange(len(val)), val.y.values] + 1e-9).mean())
            vf = f1_score(val.y, pv.argmax(1), average='macro'); hist.append(dict(ep=ep, lr=lr, train_loss=tl / max(len(dl), 1), val_loss=vl, val_f1=vf))
            msg += f' valLoss {vl:.4f} valF1 {vf:.4f}'
            if vl < best[0] - CFG['min_delta']: best = (vl, ep, {k: v.detach().to('cpu', copy=True) for k, v in ema.module.state_dict().items()}); bad = 0
            else: bad += 1
            if ep > 0: plat.step(vl)
            if bad >= CFG['patience']: print(msg + ' -> early stop', flush=True); break
        print(msg, flush=True)
    if best[2] is not None: ema.module.load_state_dict(best[2]); print(f'fold {fold} restored best epoch {best[1]} valLoss {best[0]:.4f}', flush=True)
    json.dump(dict(hist=hist, best_epoch=best[1]), open(f'{OUT}/hist_{CFG["tag"]}_f{fold}.json', 'w'))
    pv = predict(val) if not CFG['full_fit'] else np.zeros((0, 3)); pt = predict(te)
    np.savez(f'{OUT}/pred_{CFG["tag"]}_f{fold}.npz', val_path=val.path.values, val=pv, test_path=te.path.values, test=pt)
    if CFG['save_w']: torch.save({k: v.half() for k, v in ema.module.state_dict().items()}, f'{os.environ.get("DS_WOUT", OUT)}/w_{CFG["tag"]}_f{fold}.pt')
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
