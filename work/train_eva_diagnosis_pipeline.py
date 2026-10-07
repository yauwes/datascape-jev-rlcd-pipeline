import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

print("=== Datascape 2026: EVA002-Diagnosis Integrated Training Pipeline ===")
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")

# 1. Load metadata
meta = pd.read_csv('work/emb/meta_sorted.csv')
TR = (meta.split == 'train').values
TE = ~TR
tr_idx = np.where(TR)[0]
te_idx = np.where(TE)[0]
C = ['Aman', 'Lainnya', 'Rusak']
y_tr = np.array([C.index(l) for l in meta.iloc[tr_idx]['label']])

# 2. Load Multi-Tower Embeddings
print("\n[Step 1] Loading Multi-Tower Foundation Embeddings...")
emb_list = [
    'work/emb/emb_vit_pe_core_large_patch14_336.npz',
    'work/emb/emb_vit_so400m_patch16_siglip_384.npz',
    'work/emb/emb_vit_large_patch14_dinov2.npz',
    'work/emb/emb_vit_large_patch14_clip_224.npz',
    'work/emb/emb_eva02_enormous_patch14_clip_224.npz',
    'work/emb/emb_vit_pe_core_gigantic_patch14_448.npz'
]

features = []
for ef in emb_list:
    if os.path.exists(ef):
        d = np.load(ef)
        f = d['emb'] if 'emb' in d else d[list(d.keys())[0]]
        # Normalize
        f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-7)
        features.append(f_norm)
        print(f" - Loaded {os.path.basename(ef)}: shape {f.shape}")

X_all = np.concatenate(features, axis=1)
X_norm = X_all / (np.linalg.norm(X_all, axis=1, keepdims=True) + 1e-7)
X_tr = torch.tensor(X_norm[tr_idx], dtype=torch.float32)
X_te = torch.tensor(X_norm[te_idx], dtype=torch.float32)
print(f"Concatenated Embedding Shape: {X_norm.shape} (Train: {X_tr.shape}, Test: {X_te.shape})")

# 3. Diagnosis & Noise Audit (from eva002-diagnosis-integrated)
print("\n[Step 2] Applying Diagnosis & Denoising...")
clean_mask = np.load('work/clean_data_mask.npz')['clean_mask']
print(f" - Clean Mask: {clean_mask.sum()} / {len(clean_mask)} train samples retained")

# Cluster Grouping & Group Weights
folds = pd.read_csv('work/folds_group90.csv').set_index('path').loc[meta.path.values[tr_idx], 'fold'].values
if 'cluster' in meta.columns:
    clusters = meta.iloc[tr_idx]['cluster'].values
else:
    # Use folds_group90 clusters or SSCD groups
    sscd = np.load('work/nntr_sscd.npz')
    top1_tr = sscd['i'][tr_idx, 0]
    clusters = np.minimum(tr_idx, top1_tr)

group_sizes = pd.Series(clusters).map(pd.Series(clusters).value_counts()).values
gw = (group_sizes.astype(float) ** -0.5) # Penalty for giant duplicate clusters
gw = gw / gw.mean()

# 4. Domain Adversarial Shift Weighting
print("\n[Step 3] Computing Adversarial Test-Distribution Shift Weights...")
y_adv = np.r_[np.zeros(len(tr_idx)), np.ones(len(te_idx))]
X_adv_sample = X_norm[:, :512] # use first 512 PCA/features for fast adversarial density ratio
adv_clf = HistGradientBoostingClassifier(max_iter=50, random_state=42)
adv_clf.fit(X_adv_sample, y_adv)
p_adv = adv_clf.predict_proba(X_adv_sample[tr_idx])[:, 1]
p_adv = np.clip(p_adv, 0.05, 0.95)
# Density ratio: p_test(x) / p_train(x)
density_ratio = (p_adv / (1.0 - p_adv)) * (len(tr_idx) / len(te_idx))
shift_w = np.clip(density_ratio ** 0.5, 0.2, 5.0)
shift_w = shift_w / shift_w.mean()
print(f" - Shift Weights: min={shift_w.min():.2f}, median={np.median(shift_w):.2f}, max={shift_w.max():.2f}")

# 5. Training Loop with LayerNorm, Dropout & Cosine Annealing on MPS
print(f"\n[Step 4] Training Multi-Tower Model across 5 Group Folds on {device}...")

class DeepMultiTowerHead(nn.Module):
    def __init__(self, in_features, hidden_dim=768, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

oof_probs = np.zeros((len(tr_idx), 3), dtype=np.float32)
test_probs_folds = np.zeros((len(te_idx), 3), dtype=np.float32)

epochs = 25
bs = 256

for fold in range(5):
    val_idx = np.where(folds == fold)[0]
    # Train on clean subset with group weights
    train_idx = np.where((folds != fold) & clean_mask)[0]
    
    # Calculate sampler weights: class_freq^-0.5 * group_weight * shift_weight
    class_counts = pd.Series(y_tr[train_idx]).value_counts()
    cw = pd.Series(y_tr[train_idx]).map(lambda c: class_counts[c] ** -0.5).values
    sample_weights = cw * gw[train_idx] * shift_w[train_idx]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(train_idx), replacement=True)
    
    train_ds = TensorDataset(X_tr[train_idx], torch.tensor(y_tr[train_idx], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler)
    
    model = DeepMultiTowerHead(X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_loss = 1e9
    best_state = None
    
    X_val_dev = X_tr[val_idx].to(device)
    y_val_dev = torch.tensor(y_tr[val_idx], dtype=torch.long, device=device)
    
    for ep in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by, label_smoothing=0.08)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val_dev)
            v_loss = F.cross_entropy(v_logits, y_val_dev).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
    # Restore best
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        oof_probs[val_idx] = F.softmax(model(X_val_dev), dim=-1).cpu().numpy()
        test_probs_folds += F.softmax(model(X_te.to(device)), dim=-1).cpu().numpy() / 5.0
        
    f1_f = f1_score(y_tr[val_idx], np.argmax(oof_probs[val_idx], axis=1), average='macro')
    print(f"Fold {fold}: Val Loss {best_loss:.4f} | Macro F1: {f1_f:.5f}")

# 6. Evaluasi OOF & Shift-Weighted Macro F1
raw_f1 = f1_score(y_tr, np.argmax(oof_probs, axis=1), average='macro')
clean_f1 = f1_score(y_tr[clean_mask], np.argmax(oof_probs[clean_mask], axis=1), average='macro')
print(f"\n[Step 5] OOF Results:")
print(f" - Raw Overall OOF Macro F1: {raw_f1:.5f}")
print(f" - Clean Evaluated Macro F1: {clean_f1:.5f}")

# 7. Bias Optimization on Shift Weights (eva002-diagnosis §11)
print("\n[Step 6] Optimizing Class Bias Vector on Shift-Weighted Macro F1...")
def eval_bias(bias):
    adj_logits = np.log(np.clip(oof_probs, 1e-7, 1.0)) + bias
    preds = np.argmax(adj_logits, axis=1)
    return f1_score(y_tr[clean_mask], preds[clean_mask], average='macro', sample_weight=shift_w[clean_mask])

best_bias = np.zeros(3)
best_score = eval_bias(best_bias)

for b_a in np.linspace(-0.5, 0.5, 11):
    for b_l in np.linspace(-0.5, 0.5, 11):
        for b_r in np.linspace(-0.5, 0.5, 11):
            cand_b = np.array([b_a, b_l, b_r])
            score = eval_bias(cand_b)
            if score > best_score:
                best_score = score
                best_bias = cand_b.copy()

print(f" - Optimized Bias Vector: {best_bias.round(3)} -> Shift-Weighted Clean F1: {best_score:.5f}")

# 8. Apply Geometric & Domain Constraints to Test Predictions
print("\n[Step 7] Constructing Final Test Predictions with Geometric Locks...")
adj_test_logits = np.log(np.clip(test_probs_folds, 1e-7, 1.0)) + best_bias
test_pred_classes = np.argmax(adj_test_logits, axis=1)

te_meta = meta.iloc[te_idx].reset_index(drop=True)
te_meta['sz'] = te_meta['w'].astype(str) + 'x' + te_meta['h'].astype(str)

# Dihedral matches
pix = np.load('work/pix_all.npz')['ncc']
ncc_max = pix[te_idx].max(1)
ncc_best_k = pix[te_idx].argmax(1)

sscd = np.load('work/nntr_sscd.npz')
I_sscd = sscd['i']
ncc_match_tr_idx = np.array([I_sscd[orig_idx, k] for orig_idx, k in zip(te_idx, ncc_best_k)])
ncc_match_label = meta.iloc[ncc_match_tr_idx]['label'].values

# Byte dupes
byte_dupes = pd.read_csv('datascape/duplicates_test_to_train.csv').set_index('test_filename')

curated_sources = ['416x416', '675x900', '224x224', '3000x4000', '2816x1536', '4000x3000']

final_preds = []
tier_counts = {}

for i in range(len(te_meta)):
    fname = te_meta.iloc[i]['filename']
    sz = te_meta.iloc[i]['sz']
    ncc = ncc_max[i]
    m_lbl = ncc_match_label[i]
    model_lbl = C[test_pred_classes[i]]
    
    # Tier 1: 416 Raw Conveyor Render
    if sz == '416x416':
        lbl = 'Aman'
        tier = 'T1_416_Aman'
    # Tier 2: Pristine Dimensions
    elif sz in curated_sources and ncc < 0.999:
        lbl = 'Aman'
        tier = 'T2_Pristine_Aman'
    # Tier 3: Byte exact
    elif fname in byte_dupes.index:
        classes = byte_dupes.loc[fname, 'train_classes'].split(';')
        lbl = classes[0] if len(classes) == 1 else model_lbl
        tier = 'T3_Byte_Exact'
    # Tier 4: Exact Dihedral Copy (only flip Rusak -> Aman if NCC >= 0.999)
    elif ncc >= 0.999 and m_lbl == 'Aman':
        lbl = 'Aman'
        tier = 'T4_Dihedral_Aman'
    # Tier 5: Multi-Tower Model Prediction
    else:
        lbl = model_lbl
        tier = 'T5_MultiTower_Model'
        
    tier_counts[tier] = tier_counts.get(tier, 0) + 1
    final_preds.append(lbl)

sub_df = pd.DataFrame({'filename': te_meta['filename'], 'label': final_preds})
print("\n--- Final Tier Breakdown ---")
for t, cnt in sorted(tier_counts.items()):
    print(f" - {t}: {cnt} ({cnt/len(sub_df)*100:.2f}%)")

print("\n--- Final Label Distribution ---")
print(sub_df['label'].value_counts())
print(f"Lainnya count: {(sub_df['label'] == 'Lainnya').sum()} ({(sub_df['label'] == 'Lainnya').mean()*100:.2f}%)")

out_file = 'submissions/d07_eva_diagnosis_integrated.csv'
sub_df.to_csv(out_file, index=False)
print(f"\nSaved new submission to {out_file}")

# Compare vs champion d06_F1_AR25
f1_ar25 = pd.read_csv('submissions/d06_F1_AR25.csv').set_index('filename')
diff_cnt = (sub_df.set_index('filename')['label'] != f1_ar25['label']).sum()
print(f"Total differences vs d06_F1_AR25 (0.88047): {diff_cnt}")
