"""Data loading and patient-level stratified splitting for oral cancer dataset."""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import sys


class MultiModalDataset(Dataset):
    """Dataset returning (image, metadata, label) tuples."""

    def __init__(self, data_items, transform=None):
        self.data = data_items
        self.transform = transform

    def __len__(self):
        return len(self.data)

    @staticmethod
    def encode_meta(info):
        """Encode 5-dim metadata: [Age, Gender, Smoking, Betel, Alcohol].

        Age normalized to [0, 1] (capped at 100).
        Gender: 1.0 for Male, 0.0 for Female.
        Behaviour factors: 1.0 for Yes, 0.0 for No.
        """
        feats = []
        try:
            feats.append(min(100, max(0, float(info.get('Age', 50)))) / 100.0)
        except Exception:
            feats.append(0.5)
        feats.append(1.0 if str(info.get('Gender')).upper().startswith('M') else 0.0)
        for k in ['Smoking', 'Chewing_Betel_Quid', 'Alcohol']:
            feats.append(1.0 if str(info.get(k)).upper() in ['Y', 'YES', '1'] else 0.0)
        return np.array(feats, dtype=np.float32)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            img = Image.open(item['path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (448, 448))
        if self.transform:
            img = self.transform(img)
        meta = torch.tensor(self.encode_meta(item['info']), dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        return img, meta, label


def load_all_data(img_dir, csv_path, patient_csv, class_map):
    """Load and match images with metadata and labels.

    Returns dict: {patient_id: [{'path': ..., 'label': ..., 'info': ...}, ...]}
    """
    print(">>> Loading raw data...")
    img_dir = Path(img_dir)

    try:
        img_df = pd.read_csv(csv_path, encoding='utf-8', engine='python')
    except Exception:
        sys.exit("Error: Could not read Imagewise_Data.csv")

    name2label = {}
    for _, row in img_df.iterrows():
        name = Path(row['Image Name']).stem
        cat = row['Category'].strip()
        if cat in class_map:
            name2label[name] = class_map[cat]

    try:
        pt_df = pd.read_csv(patient_csv, encoding='utf-8', engine='python')
    except Exception:
        pt_df = pd.read_csv(patient_csv, encoding='gbk', engine='python')
    pt_dict = {str(row['Patient ID']).strip(): row for _, row in pt_df.iterrows()}

    patient_data = {}
    for img_path in tqdm(list(img_dir.glob("*.jpg")), desc="Matching"):
        fname = img_path.name
        name = fname.replace("_oral_only.jpg", "") if "_oral_only.jpg" in fname else img_path.stem
        if name not in name2label:
            continue
        parts = name.split('-')
        pid = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else name
        if pid in pt_dict:
            item = {'path': img_path, 'label': name2label[name], 'info': pt_dict[pid]}
            patient_data.setdefault(pid, []).append(item)

    total = sum(len(v) for v in patient_data.values())
    print(f"Loaded {total} images from {len(patient_data)} patients.")
    return patient_data


def prepare_splits(patient_data, test_size, n_folds, seed):
    """Patient-level stratified split: fixed test set + 5-fold CV.

    Returns:
      test_data: list of all test items
      folds: list of (train_data, val_data) tuples
    """
    pids = np.array(list(patient_data.keys()))
    plabels = np.array([patient_data[p][0]['label'] for p in pids])

    tv_pids, test_pids, tv_labels, _ = train_test_split(
        pids, plabels, test_size=test_size, stratify=plabels, random_state=seed)

    test_data = [item for pid in test_pids for item in patient_data[pid]]
    tc = np.bincount([x['label'] for x in test_data])
    print(f"\n  Fixed test set: {len(test_data)} images "
          f"(Benign={tc[0]}, Malignant={tc[1]})")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    print(f"\n  {n_folds}-fold CV splits:")
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(tv_pids, tv_labels)):
        tr_data = [item for pid in tv_pids[tr_idx] for item in patient_data[pid]]
        va_data = [item for pid in tv_pids[va_idx] for item in patient_data[pid]]
        trc = np.bincount([x['label'] for x in tr_data])
        vac = np.bincount([x['label'] for x in va_data])
        print(f"    Fold {fold_idx+1}: "
              f"Train={len(tr_data)}(B={trc[0]},M={trc[1]})  "
              f"Val={len(va_data)}(B={vac[0]},M={vac[1]})")
        folds.append((tr_data, va_data))

    return test_data, folds
