import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

# 1. Load metadata
meta = pd.read_csv('work/emb/meta_sorted.csv')
TR = (meta.split == 'train').values
TE = ~TR
C = ['Aman', 'Lainnya', 'Rusak']
y_tr = np.array([C.index(l) for l in meta[TR].label])

print(f"Total train: {TR.sum()}, Test: {TE.sum()}")

# 2. Load high-quality multi-tower features
print("Loading multi-tower embeddings...")
emb_files = [
    'work/emb/emb_vit_pe_core_large_patch14_336.npz',
    'work/emb/emb_vit_so400m_patch16_siglip_384.npz',
    'work/emb/emb_vit_large_patch14_dinov2.npz',
    'work/emb/emb_vit_large_patch14_clip_224.npz'
]

features = []
for ef in emb_files:
    d = np.load(ef)
    f = d['emb'] if 'emb' in d else d[list(d.keys())[0]]
    # Normalize features
    f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-7)
    features.append(f_norm)

X = np.concatenate(features, axis=1)
X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-7)

X_tr = X_norm[TR]
X_te = X_norm[TE]

print(f"Embedding shape: {X.shape}, X_tr: {X_tr.shape}")

# 3. Compute Class Centroids
centroids = np.zeros((3, X_tr.shape[1]), dtype=np.float32)
for c in range(3):
    mask_c = (y_tr == c)
    centroids[c] = X_tr[mask_c].mean(axis=0)
    centroids[c] /= (np.linalg.norm(centroids[c]) + 1e-7)

# Compute similarity of each train sample to all centroids
sims = np.dot(X_tr, centroids.T) # (N_tr, 3)
own_sim = sims[np.arange(len(y_tr)), y_tr]

# Margin to other centroids
other_sim = np.zeros(len(y_tr))
for i in range(len(y_tr)):
    other_classes = [c for c in range(3) if c != y_tr[i]]
    other_sim[i] = np.max(sims[i, other_classes])

margin = other_sim - own_sim # Positive means closer to another centroid!

print("\n--- Centroid Distance Analysis ---")
print(f"Samples closer to another class centroid (margin > 0): {(margin > 0).sum()} ({(margin > 0).mean()*100:.2f}%)")
print(f"Severe outliers (margin > 0.10): {(margin > 0.10).sum()}")
print(f"Extreme outliers (margin > 0.20): {(margin > 0.20).sum()}")

# 4. Check label breakdown of severe outliers
for c in range(3):
    m_c = (y_tr == c)
    outliers_c = (margin[m_c] > 0.10).sum()
    print(f"Class {C[c]}: {outliers_c} / {m_c.sum()} outliers ({outliers_c/m_c.sum()*100:.2f}%)")

# 5. Overlapping / Exact near-duplicate conflicts
# From dihedral NCC or SSCD
sscd = np.load('work/nntr_sscd.npz')
I_sscd, S_sscd = sscd['i'], sscd['s']

# Find train samples with top-1 neighbor having high similarity (>0.90) but different label
tr_indices = np.where(TR)[0]
conflict_twins = np.zeros(len(y_tr), dtype=bool)

for idx_in_tr, orig_idx in enumerate(tr_indices):
    top_cand = I_sscd[orig_idx, 0] # nearest other train neighbor
    top_sim = S_sscd[orig_idx, 0]
    cand_label = meta.iloc[top_cand]['label']
    cand_idx_c = C.index(cand_label)
    if top_sim >= 0.85 and cand_idx_c != y_tr[idx_in_tr]:
        conflict_twins[idx_in_tr] = True

print(f"\nTrain samples with near-duplicate conflict (SSCD >= 0.85, diff label): {conflict_twins.sum()}")

# Combine outlier filter
noisy_mask = (margin > 0.10) | conflict_twins
clean_mask = ~noisy_mask

print(f"\nTotal clean samples: {clean_mask.sum()} / {len(y_tr)} ({clean_mask.mean()*100:.2f}%)")
print(f"Total flagged noisy/outlier samples: {noisy_mask.sum()} ({noisy_mask.mean()*100:.2f}%)")

# Save cleaned indices
np.savez('work/clean_data_mask.npz', clean_mask=clean_mask, noisy_mask=noisy_mask, margin=margin)
print("Saved clean data mask to work/clean_data_mask.npz")
