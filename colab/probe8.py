# Session 8: frozen-probe gate for new towers (DINOv3-7B, DINOv3-H+; bf16 embeddings from emb_giant.py). Pre-registered bar (RESULTS 10-04):
# a tower counts only if adding it to cat6+PE-G raises group-OOF macro-F1 vs ORIGINAL labels by >= +0.005 (same MLP512+CE recipe, 3 seeds)
# without raising the natural test Rusak share. Writes work/preds/p8_*/ (ens.py format) and work/probe8_scores.csv.
import sys, numpy as np, pandas as pd, torch
exec(open('/content/code/sweep.py').read().split('rows_out = []')[0])
X0, rows = X, []
def run(Xs, tag):
    global X; X = Xs; o = np.zeros((len(tr), 3)); t = np.zeros((len(te), 3))
    for k in range(5):
        a, b = fold != k, fold == k
        for s in range(3):
            m = fit('mlp512', 'ce', tr[a], y0[a], s)
            with torch.no_grad(): o[b] += m(X[tr[b]]).softmax(1).cpu().numpy() / 3; t += m(X[te]).softmax(1).cpu().numpy() / 15
    d = f'work/preds/{tag}'; os.makedirs(d, exist_ok=True)
    for k in range(5):
        mm = fold == k; np.savez(f'{d}/pred_{tag}_f{k}.npz', val_path=M.path.values[tr][mm], val=o[mm], test_path=M.path.values[te], test=t)
    r = dict(tag=tag, f1_orig=round(f1_score(y0, o.argmax(1), average='macro'), 4), f1_clean=round(f1_score(yc, o.argmax(1), average='macro'), 4),
             test_rusak=round(float((t.argmax(1) == 2).mean()), 3), f1_twin_locked=round(f1_score(ly, t[li].argmax(1), average='macro'), 4))
    rows.append(r); print(r, flush=True); pd.DataFrame(rows).to_csv('work/probe8_scores.csv', index=False)
run(X0, 'p8_base')
for t in sys.argv[1:]:
    if not os.path.exists(f'work/emb/emb_{t}.npz'): print('missing', t, flush=True); continue
    F_ = torch.tensor(feat(t), device=dev); run(F_, f'p8_{t}'); run(torch.cat([X0, F_], 1), f'p8_base_{t}')
print(f'PROBE8 DONE {(time.time() - T0) / 60:.1f} min', flush=True)
