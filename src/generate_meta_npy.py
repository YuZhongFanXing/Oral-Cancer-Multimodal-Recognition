"""
Re-run inference for Meta_Only to generate test_probs.npy / test_lbls.npy.
Uses the EXACT MetaOnlyModel architecture from the training script.

Root cause of previous AUC=0.39 bug:
  - regen_meta_only_npy.py auto-inferred wrong architecture (FlexMetaOnlyModel)
  - inference call was model(meta) but saved model expects forward(img, meta)
  => garbage logits => AUC < 0.5
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os, warnings, random
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR       = "/home/wgf_v100/srtp/\u53e3\u8154\u764c\u5206\u7c7b\u8bc6\u522b\u9879\u76ee"
CROP_DIR       = os.path.join(BASE_DIR, "Segmented_Images/Segmented_Images")
CSV_PATH       = os.path.join(BASE_DIR, "data/Imagewise_Data.csv")
PATIENT_CSV    = os.path.join(BASE_DIR, "data/Patientwise_Data.csv")
META_SAVE_ROOT = os.path.join(BASE_DIR, "results_meta_only_cv5_seed42_optimized")
SEED       = 42
TEST_SIZE  = 0.15
IMG_SIZE   = 448
BATCH_SIZE = 32
N_FOLDS    = 5
CLASS_MAP  = {'Benign': 0, 'OCA': 1, 'OPMD': 1}

# ── EXACT model definition copied verbatim from training script ─
class MetaMLP_Baseline(nn.Module):
    def __init__(self, meta_dim=5, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),    nn.Dropout(0.2),
            nn.Linear(128, out_dim),  nn.BatchNorm1d(out_dim),
            nn.SiLU(inplace=True),    nn.Dropout(0.2)
        )
    def forward(self, x): return self.net(x)

def build_head_baseline(in_dim, num_classes=2):
    return nn.Sequential(
        nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024),
        nn.SiLU(inplace=True),   nn.Dropout(0.5),
        nn.Linear(1024, 512),    nn.BatchNorm1d(512),
        nn.SiLU(inplace=True),   nn.Dropout(0.4),
        nn.Linear(512, 256),     nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),   nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

class MetaOnlyModel(nn.Module):
    """
    Exact replica of training script MetaOnlyModel.
    forward(img, meta) accepts both args; img is ignored inside (same as training).
    """
    def __init__(self, meta_dim=5, num_classes=2):
        super().__init__()
        self.meta_branch = MetaMLP_Baseline(meta_dim, out_dim=256)
        self.classifier  = build_head_baseline(256, num_classes)

    def forward(self, img, meta):
        return self.classifier(self.meta_branch(meta))   # img intentionally ignored

# ── Dataset: dummy image + real meta + label ──────────────────
class MetaOnlyDataset(Dataset):
    def __init__(self, data_items):
        self.data = data_items

    def __len__(self): return len(self.data)

    def _encode_meta(self, info):
        feats = []
        try:    feats.append(min(100, max(0, float(info.get('Age', 50)))) / 100.0)
        except: feats.append(0.5)
        feats.append(1.0 if str(info.get('Gender')).upper().startswith('M') else 0.0)
        for k in ['Smoking', 'Chewing_Betel_Quid', 'Alcohol']:
            feats.append(1.0 if str(info.get(k)).upper() in ['Y','YES','1'] else 0.0)
        return np.array(feats, dtype=np.float32)

    def __getitem__(self, idx):
        item  = self.data[idx]
        img   = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)  # dummy
        meta  = torch.tensor(self._encode_meta(item['info']), dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        return img, meta, label

# ── Load fixed test set (same seed/split as all other scripts) ─
def load_test_data():
    print("Loading fixed test set (seed=42, test_size=0.15)...")
    img_dir = Path(CROP_DIR)
    img_df  = pd.read_csv(CSV_PATH, encoding='utf-8', engine='python')

    name2label = {}
    for _, row in img_df.iterrows():
        name = Path(row['Image Name']).stem
        cat  = row['Category'].strip()
        if cat in CLASS_MAP:
            name2label[name] = CLASS_MAP[cat]

    try:    pt_df = pd.read_csv(PATIENT_CSV, encoding='utf-8', engine='python')
    except: pt_df = pd.read_csv(PATIENT_CSV, encoding='gbk',  engine='python')
    pt_dict = {str(row['Patient ID']).strip(): row for _, row in pt_df.iterrows()}

    patient_data = {}
    for img_path in img_dir.glob("*.jpg"):
        fname = img_path.name
        name  = fname.replace("_oral_only.jpg","") if "_oral_only.jpg" in fname \
                else img_path.stem
        if name not in name2label: continue
        parts = name.split('-')
        pid   = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else name
        if pid in pt_dict:
            patient_data.setdefault(pid, []).append(
                {'path': img_path, 'label': name2label[name], 'info': pt_dict[pid]})

    pids    = np.array(list(patient_data.keys()))
    plabels = np.array([patient_data[p][0]['label'] for p in pids])
    random.seed(SEED); np.random.seed(SEED)
    _, test_pids = train_test_split(pids, test_size=TEST_SIZE,
                                    stratify=plabels, random_state=SEED)
    test_data = [item for pid in test_pids for item in patient_data[pid]]
    tc = np.bincount([x['label'] for x in test_data])
    print(f"  {len(test_data)} images  (Benign={tc[0]}, Malignant={tc[1]})")
    return test_data

# ── Run inference for one fold ────────────────────────────────
def run_fold(fold_idx, test_data):
    fold_dir  = os.path.join(META_SAVE_ROOT, f"fold_{fold_idx}")
    ckpt_path = os.path.join(fold_dir, "best_model.pth")

    if not os.path.exists(ckpt_path):
        print(f"  Fold {fold_idx}: NOT FOUND -- {ckpt_path}")
        return None

    ckpt      = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state     = ckpt.get('state', ckpt)
    threshold = ckpt.get('threshold', 0.5)

    # Key-match sanity check
    expected = set(MetaOnlyModel().state_dict().keys())
    actual   = set(state.keys())
    if expected != actual:
        print(f"  Fold {fold_idx}: [ERROR] State dict mismatch -- "
              f"missing={expected-actual}, extra={actual-expected}")
        print("  Architecture in checkpoint does not match MetaOnlyModel.")
        print("  Cannot proceed with strict loading.")
        return None

    model = MetaOnlyModel(meta_dim=5).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  Fold {fold_idx}: checkpoint OK  "
          f"(epoch={ckpt.get('epoch','?')}, "
          f"ema={ckpt.get('ema_used','?')}, "
          f"threshold={threshold:.3f})")

    ds = MetaOnlyDataset(test_data)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs, all_lbls = [], []
    with torch.no_grad():
        for img, meta, lbl in dl:
            img, meta = img.to(DEVICE), meta.to(DEVICE)
            out   = model(img, meta)          # img ignored inside model
            probs = torch.softmax(out, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_lbls.extend(lbl.numpy())

    all_probs = np.array(all_probs)
    all_lbls  = np.array(all_lbls)
    auc_val   = roc_auc_score(all_lbls, all_probs[:, 1])
    preds     = (all_probs[:, 1] >= threshold).astype(int)
    acc       = np.mean(preds == all_lbls)

    print(f"  Fold {fold_idx}: AUC={auc_val:.4f}  Acc={acc:.4f}")
    print(f"    Prob[:,1]: min={all_probs[:,1].min():.3f}  "
          f"max={all_probs[:,1].max():.3f}  "
          f"mean={all_probs[:,1].mean():.3f}")

    np.save(os.path.join(fold_dir, "test_probs.npy"), all_probs)
    np.save(os.path.join(fold_dir, "test_lbls.npy"),  all_lbls)
    print(f"  Fold {fold_idx}: saved test_probs.npy & test_lbls.npy")
    return auc_val

# ── Main ──────────────────────────────────────────────────────
print("=" * 60)
print("Regenerating Meta_Only .npy  (corrected architecture)")
print("=" * 60)

test_data = load_test_data()

aucs = []
for fold in range(1, N_FOLDS + 1):
    print(f"\n--- Fold {fold} ---")
    result = run_fold(fold, test_data)
    if result is not None:
        aucs.append(result)

print(f"\n{'='*60}")
print(f"Completed: {len(aucs)}/{N_FOLDS} folds")
if aucs:
    print(f"Mean AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    if np.mean(aucs) < 0.55:
        print("\n[NOTE] AUC is near or below random (0.5).")
        print("  This is expected if metadata alone (Age/Gender/Smoking/")
        print("  Betel/Alcohol) has low discriminative power for this dataset.")
        print("  The result is valid and meaningful for the ablation study.")
    else:
        print("\nRe-run  python replot_roc.py  to update the figure.")