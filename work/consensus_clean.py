# Multi-backbone consensus label cleaning (user idea, 10-09): 11 independent "jurors" = group-OOF predictions of 8 frozen-tower
# LR heads + 3 fine-tuned CNN/ViT models already in work/preds. A train label is corrected only if >= MIN_VOTES jurors agree on
# another class AND their mean probability for it >= MIN_P. Then one strong head (LR on PE-G + SigLIP-giant + DINOv3-L embeddings)
# is refit with original vs cleaned labels to measure how many test predictions the cleaning moves.
# Usage (Datascape/): python work/consensus_clean.py [MIN_VOTES MIN_P]   -> work/labels_consensus.csv
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
MIN_VOTES, MIN_P = (int(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (10, 0.9)
C = np.array(['Aman', 'Lainnya', 'Rusak'])
JURORS = ['vit_pe_core_gigantic_patch14_448_lr_g90', 'vit_giantopt_patch16_siglip_384_lr_g90', 'vit_large_patch16_dinov3_lr_g90', 'vit_large_patch14_dinov2_lr_g90',
          'eva02_enormous_patch14_clip_224_lr_g90', 'vit_gigantic_patch14_clip_378_lr_g90', 'aimv2_large_patch14_224_lr_g90', 'vit_intern300m_patch14_448_lr_g90',
          'cnxB384_g90', 'cnxL320_g90', 'eva02B448_g90']
M = pd.read_csv('work/emb/meta_sorted.csv'); TR = (M.split == 'train').values; trp = M.path.values[TR]; y = M.label.values[TR]
P = []
for j in JURORS:
    V = []
    for k in range(5):
        z = np.load(f'work/preds/{j}/pred_{j}_f{k}.npz', allow_pickle=True); V.append(pd.DataFrame(z['val'], index=z['val_path'], columns=C))
    P.append(pd.concat(V).reindex(trp).values)
P = np.stack(P)                                   # [jurors, train rows, 3]
assert not np.isnan(P).any(), 'a juror misses OOF rows'
votes = np.stack([(P.argmax(2) == c).sum(0) for c in range(3)], 1); mp = P.mean(0)
cons = votes.argmax(1); flag = (C[cons] != y) & (votes.max(1) >= MIN_VOTES) & (mp[np.arange(len(y)), cons] >= MIN_P)
yc = np.where(flag, C[cons], y)
pd.DataFrame({'path': trp, 'label_orig': y, 'label_clean': yc}).to_csv('work/labels_consensus.csv', index=False)
sz = (M.w.astype(str) + 'x' + M.h.astype(str)).values[TR]
print(f'jurors {len(JURORS)} | rule >= {MIN_VOTES} votes & mean p >= {MIN_P} | corrected {flag.sum()} of {len(y)} train labels')
print(pd.crosstab(np.where(np.isin(sz, ['416x416', '512x512', '640x640', '675x900', '224x224']), sz, 'other')[flag], [y[flag], yc[flag]]).to_string())
# effect on the test set: refit one strong head with original vs cleaned labels
X = np.hstack([np.load(f'work/emb/emb_{t}.npz')['orig'].astype(np.float32) for t in ['vit_pe_core_gigantic_patch14_448', 'vit_giantopt_patch16_siglip_384', 'vit_large_patch16_dinov3']])
sc = StandardScaler().fit(X[TR]); Xtr, Xte = sc.transform(X[TR]), sc.transform(X[~TR])
pt = {}
for nm, lab in [('orig', y), ('clean', yc)]:
    pt[nm] = LogisticRegression(C=0.05, max_iter=400).fit(Xtr, lab).predict_proba(Xte); print('fitted', nm, flush=True)
np.savez('work/consensus_clean_testprobs.npz', test_path=M.path.values[~TR], orig=pt['orig'], clean=pt['clean'], classes=C)
