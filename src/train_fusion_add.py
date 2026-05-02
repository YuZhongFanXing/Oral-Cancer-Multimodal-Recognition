"""
Img_SE + Fusion_Add 组合实验 -- 5折交叉验证
============================================
同一脚本包含两个实验，唯一差异是分类头深度:

  [1] Img_SE_Add          SEBlock + AddFusion + build_head_baseline(512)
  [2] Img_SE_Add_DeepHead SEBlock + AddFusion + build_head_deep(512)

所有组件原样复制自架构实验脚本 architecture_experiments_cv5.py。

ArchModel 对应路径:
  1. backbone  = get_efficientnet_v2s()               img_out=1280
  2. img_se    = SEBlock(1280, reduction=16)
  3. meta      = MetaMLP_Baseline(5, out_dim=256)     meta_out=256
  4. fusion    = AddFusion(1280, 256, proj_dim=512)   fusion_dim=512
  5. head      = build_head_baseline(512)  或  build_head_deep(512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os, warnings, random, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
import seaborn as sns
import shutil
from datetime import datetime
from collections import deque

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{datetime.now().strftime('%H:%M:%S')}] Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# 1. 实验注册表
# ==============================================================================
EXPERIMENTS = [
    # name                    head_type
    ('Img_SE_Add',            'baseline'),   # head: 512->1024->512->256->2
    ('Img_SE_Add_DeepHead',   'deep'),       # head: 512->2048->1024->512->256->2
]


# ==============================================================================
# 2. Config
# ==============================================================================
class Config:
    BASE_DIR    = "/home/wgf_v100/srtp/\u53e3\u8154\u764c\u5206\u7c7b\u8bc6\u522b\u9879\u76ee"
    CROP_DIR    = os.path.join(BASE_DIR, "Segmented_Images/Segmented_Images")
    CSV_PATH    = os.path.join(BASE_DIR, "data/Imagewise_Data.csv")
    PATIENT_CSV = os.path.join(BASE_DIR, "data/Patientwise_Data.csv")

    SEED      = 42
    N_FOLDS   = 5
    TEST_SIZE = 0.15
    SAVE_ROOT = os.path.join(BASE_DIR, "results_img_se_add_cv5_seed42")

    IMG_SIZE            = 448
    BATCH_SIZE          = 16
    GRAD_ACCUMULATION   = 3
    EPOCHS              = 100
    LR_MAX              = 2e-4
    WEIGHT_DECAY        = 2e-3
    EARLY_STOP_PATIENCE = 20
    NUM_WORKERS         = 4

    USE_TTA      = True
    THRESH_RANGE = np.arange(0.15, 0.95, 0.01)

    CLASS_MAP   = {'Benign': 0, 'OCA': 1, 'OPMD': 1}
    CLASS_NAMES = ['Benign', 'Malignant']

    EMA_DECAY     = 0.99
    EMA_WARMUP    = 5
    SMOOTH_WINDOW = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ==============================================================================
# 3. EMA
# ==============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.decay  = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                self.shadow[n] = (self.decay * self.shadow[n]
                                  + (1 - self.decay) * p.data)

    def apply_shadow(self, model):
        self._backup = {n: p.data.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            p.data.copy_(self._backup[n])

    class _Ctx:
        def __init__(self, ema, model): self.ema, self.model = ema, model
        def __enter__(self): self.ema.apply_shadow(self.model); return self.model
        def __exit__(self, *a): self.ema.restore(self.model)

    def get_context(self, model): return self._Ctx(self, model)


# ==============================================================================
# 4. 损失函数
# ==============================================================================
class WeightedLabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.05, dim=-1, weight=None):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing  = smoothing
        self.cls        = classes
        self.dim        = dim
        self.weight     = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.weight is not None:
            true_dist = true_dist * self.weight.to(pred.device).view(1, -1)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# ==============================================================================
# 5. Dataset
# ==============================================================================
class MultiModalDataset(Dataset):
    def __init__(self, data_items, transform=None):
        self.data      = data_items
        self.transform = transform

    def __len__(self): return len(self.data)

    def _encode_meta(self, info):
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
            img = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        meta  = torch.tensor(self._encode_meta(item['info']), dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        return img, meta, label


# ==============================================================================
# 6. 数据加载与固定划分
# ==============================================================================
def load_all_data():
    print(">>> Loading raw data...")
    img_dir = Path(Config.CROP_DIR)
    try:
        img_df = pd.read_csv(Config.CSV_PATH, encoding='utf-8', engine='python')
    except Exception:
        sys.exit("Error: Could not read Imagewise_Data.csv")

    name2label = {}
    for _, row in img_df.iterrows():
        name = Path(row['Image Name']).stem
        cat  = row['Category'].strip()
        if cat in Config.CLASS_MAP:
            name2label[name] = Config.CLASS_MAP[cat]

    try:
        pt_df = pd.read_csv(Config.PATIENT_CSV, encoding='utf-8', engine='python')
    except Exception:
        pt_df = pd.read_csv(Config.PATIENT_CSV, encoding='gbk', engine='python')
    pt_dict = {str(row['Patient ID']).strip(): row for _, row in pt_df.iterrows()}

    patient_data = {}
    for img_path in tqdm(list(img_dir.glob("*.jpg")), desc="Matching"):
        fname = img_path.name
        name  = fname.replace("_oral_only.jpg", "") if "_oral_only.jpg" in fname \
                else img_path.stem
        if name not in name2label:
            continue
        parts = name.split('-')
        pid   = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else name
        if pid in pt_dict:
            item = {'path': img_path, 'label': name2label[name], 'info': pt_dict[pid]}
            patient_data.setdefault(pid, []).append(item)

    total = sum(len(v) for v in patient_data.values())
    print(f"Loaded {total} images from {len(patient_data)} patients.")
    return patient_data


def prepare_splits(patient_data):
    """与所有前序脚本完全相同的参数"""
    pids    = np.array(list(patient_data.keys()))
    plabels = np.array([patient_data[p][0]['label'] for p in pids])

    tv_pids, test_pids, tv_labels, _ = train_test_split(
        pids, plabels,
        test_size=Config.TEST_SIZE,
        stratify=plabels,
        random_state=Config.SEED)

    test_data = [item for pid in test_pids for item in patient_data[pid]]
    tc = np.bincount([x['label'] for x in test_data])
    print(f"\n  Fixed test set: {len(test_data)} images "
          f"(Benign={tc[0]}, Malignant={tc[1]})  <-- same as all previous scripts")

    skf   = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True,
                             random_state=Config.SEED)
    folds = []
    print(f"\n  5-fold CV splits:")
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


# ==============================================================================
# 7. 模型组件 -- 原样复制自架构实验脚本
# ==============================================================================

def get_efficientnet_v2s():
    """加载EfficientNetV2-S backbone，返回模型和特征维度"""
    m   = models.efficientnet_v2_s(weights='DEFAULT')
    dim = m.classifier[1].in_features   # 1280
    m.classifier = nn.Identity()
    return m, dim


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 对1D特征做通道重标定"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // reduction, 8), dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class MetaMLP_Baseline(nn.Module):
    """基线: 5->128->256, 带BN"""
    def __init__(self, meta_dim=5, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),    nn.Dropout(0.2),
            nn.Linear(128, out_dim),  nn.BatchNorm1d(out_dim),
            nn.SiLU(inplace=True),    nn.Dropout(0.2)
        )
    def forward(self, x): return self.net(x)


class AddFusion(nn.Module):
    """
    对齐维度后直接相加.
    img和meta都投影到proj_dim, element-wise add.
    """
    def __init__(self, img_dim, meta_dim_out, proj_dim=512):
        super().__init__()
        self.img_proj  = nn.Linear(img_dim,      proj_dim)
        self.meta_proj = nn.Linear(meta_dim_out, proj_dim)
        self.norm      = nn.LayerNorm(proj_dim)
        self.out_dim   = proj_dim

    def forward(self, img_feat, meta_feat):
        return self.norm(
            F.silu(self.img_proj(img_feat)) +
            F.silu(self.meta_proj(meta_feat)))


def build_head_baseline(in_dim, num_classes=2):
    """基线头: in->1024->512->256->2"""
    return nn.Sequential(
        nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024),
        nn.SiLU(inplace=True),   nn.Dropout(0.5),
        nn.Linear(1024, 512),    nn.BatchNorm1d(512),
        nn.SiLU(inplace=True),   nn.Dropout(0.4),
        nn.Linear(512, 256),     nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),   nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )


def build_head_deep(in_dim, num_classes=2):
    """深宽头: in->2048->1024->512->256->2"""
    return nn.Sequential(
        nn.Linear(in_dim, 2048), nn.BatchNorm1d(2048),
        nn.SiLU(inplace=True),   nn.Dropout(0.5),
        nn.Linear(2048, 1024),   nn.BatchNorm1d(1024),
        nn.SiLU(inplace=True),   nn.Dropout(0.5),
        nn.Linear(1024, 512),    nn.BatchNorm1d(512),
        nn.SiLU(inplace=True),   nn.Dropout(0.4),
        nn.Linear(512, 256),     nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),   nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )


# ==============================================================================
# 8. 模型类
# ==============================================================================
class ImgSEAddModel(nn.Module):
    """
    Img_SE + Fusion_Add 组合模型。
    初始化顺序严格对应 ArchModel 的各分支:

      1. backbone  = get_efficientnet_v2s()               1280d
      2. img_se    = SEBlock(1280, reduction=16)           img_se 分支
      3. meta      = MetaMLP_Baseline(meta_dim, 256)       else 分支, 256d
      4. fusion    = AddFusion(1280, 256, proj_dim=512)    fusion_add 分支, 512d
      5. head      由 head_type 决定:
           'baseline' -> build_head_baseline(512)  else 分支
           'deep'     -> build_head_deep(512)       head_deep 分支
    """
    def __init__(self, head_type='baseline', meta_dim=5, num_classes=2):
        super().__init__()
        # step 1
        self.backbone, img_out = get_efficientnet_v2s()            # 1280d
        # step 2
        self.img_se = SEBlock(img_out, reduction=16)
        # step 3
        self.meta_branch = MetaMLP_Baseline(meta_dim, out_dim=256)
        meta_out  = 256
        # step 4
        self.fusion   = AddFusion(img_out, meta_out, proj_dim=512)
        fusion_dim    = self.fusion.out_dim                        # 512
        # step 5
        if head_type == 'deep':
            self.classifier = build_head_deep(fusion_dim, num_classes)
        else:
            self.classifier = build_head_baseline(fusion_dim, num_classes)

    def forward(self, img, meta):
        img_feat  = self.backbone(img)                  # (B, 1280)
        img_feat  = self.img_se(img_feat)               # (B, 1280)  SE重标定
        meta_feat = self.meta_branch(meta)              # (B, 256)
        fused     = self.fusion(img_feat, meta_feat)    # (B, 512)   Add融合
        return self.classifier(fused)


# ==============================================================================
# 9. 阈值搜索 & 评估
# ==============================================================================
def find_threshold(labels, probs):
    best_score, best_th = 0.0, 0.5
    labels = np.array(labels)
    for th in Config.THRESH_RANGE:
        preds = (probs[:, 1] >= th).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        bal = (tp / (tp + fn + 1e-6) + tn / (tn + fp + 1e-6)) / 2.0
        if bal > best_score:
            best_score, best_th = bal, th
    return best_th


def evaluate(model, loader, criterion, threshold=0.5, tta=False):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    with torch.no_grad():
        for img, meta, lbl in loader:
            img, meta, lbl = img.to(DEVICE), meta.to(DEVICE), lbl.to(DEVICE)
            if tta:
                out = (model(img, meta) +
                       model(torch.flip(img, [3]), meta) +
                       model(torch.flip(img, [2]), meta)) / 3.0
            else:
                out = model(img, meta)
            if criterion is not None:
                total_loss += criterion(out, lbl).item() * len(lbl)
            pb = torch.softmax(out, dim=1)
            all_probs.extend(pb.cpu().numpy())
            all_preds.extend((pb[:, 1] >= threshold).long().cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    loss = total_loss / len(loader.dataset) if criterion is not None else 0
    acc  = np.mean(np.array(all_preds) == np.array(all_labels))
    return acc, loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ==============================================================================
# 10. 单折训练
# ==============================================================================
def train_one_fold(fold_idx, exp_name, head_type,
                   train_data, val_data, test_data, save_dir):

    set_seed(Config.SEED)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n  {'─'*62}")
    print(f"  Fold {fold_idx+1}/{Config.N_FOLDS} | {exp_name} | seed={Config.SEED}")
    print(f"  Train={len(train_data)}  Val={len(val_data)}  "
          f"Test={len(test_data)}(fixed)")
    print(f"  {'─'*62}")

    train_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.15, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = MultiModalDataset(train_data, train_tf)
    val_ds   = MultiModalDataset(val_data,   eval_tf)
    test_ds  = MultiModalDataset(test_data,  eval_tf)

    targets   = [x['label'] for x in train_data]
    counts    = np.bincount(targets)
    s_weights = (1. / torch.tensor(counts, dtype=torch.float))[targets]
    sampler   = WeightedRandomSampler(s_weights, len(s_weights))

    train_dl = DataLoader(train_ds, Config.BATCH_SIZE, sampler=sampler,
                          num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   Config.BATCH_SIZE, shuffle=False,
                          num_workers=Config.NUM_WORKERS)
    test_dl  = DataLoader(test_ds,  Config.BATCH_SIZE, shuffle=False,
                          num_workers=Config.NUM_WORKERS)

    model = ImgSEAddModel(head_type=head_type, meta_dim=5).to(DEVICE)

    total   = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_str = ('512->1024->512->256->2' if head_type == 'baseline'
                else '512->2048->1024->512->256->2')
    print(f"  Params: total={total:,}  trainable={n_train:,}")
    print(f"  EfficientNetV2-S(1280) -> SE(1280) + MLP(256) "
          f"-> Add(512) -> FC({head_str})")

    ema       = ModelEMA(model, decay=Config.EMA_DECAY)
    w         = torch.tensor([2.0, 1.0]).to(DEVICE)
    criterion = WeightedLabelSmoothingLoss(weight=w, smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(),
                            lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=5e-7)

    best_val_f1   = 0.0
    best_epoch    = 0
    early_counter = 0
    f1_window     = deque(maxlen=Config.SMOOTH_WINDOW)
    history       = {'train_loss': [], 'val_loss': [], 'val_acc': [],
                     'val_f1': [], 'val_f1_smooth': [], 'val_mcc': []}
    model_path    = os.path.join(save_dir, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        model.train()
        t_loss = 0
        optimizer.zero_grad()

        for i, (img, meta, lbl) in enumerate(
                tqdm(train_dl,
                     desc=f"[{exp_name}|F{fold_idx+1}] Ep{epoch+1}",
                     leave=False)):
            img, meta, lbl = img.to(DEVICE), meta.to(DEVICE), lbl.to(DEVICE)
            loss = criterion(model(img, meta), lbl) / Config.GRAD_ACCUMULATION
            loss.backward()
            t_loss += loss.item() * Config.GRAD_ACCUMULATION
            if (i + 1) % Config.GRAD_ACCUMULATION == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step(); optimizer.zero_grad()
                ema.update(model)

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        if epoch < Config.EMA_WARMUP:
            _, val_loss, _, val_lbls, val_probs = evaluate(
                model, val_dl, criterion, threshold=0.5, tta=False)
            ema_tag = "raw"
        else:
            with ema.get_context(model):
                _, val_loss, _, val_lbls, val_probs = evaluate(
                    model, val_dl, criterion, threshold=0.5, tta=False)
            ema_tag = "EMA"

        th          = find_threshold(val_lbls, val_probs)
        final_preds = (val_probs[:, 1] >= th).astype(int)
        acc  = np.mean(final_preds == val_lbls)
        f1   = classification_report(val_lbls, final_preds,
                   output_dict=True, zero_division=0)['macro avg']['f1-score']
        mcc  = matthews_corrcoef(val_lbls, final_preds)

        tp = np.sum((final_preds == 1) & (val_lbls == 1))
        tn = np.sum((final_preds == 0) & (val_lbls == 0))
        fp = np.sum((final_preds == 1) & (val_lbls == 0))
        fn = np.sum((final_preds == 0) & (val_lbls == 1))
        sens    = tp / (tp + fn + 1e-6)
        spec    = tn / (tn + fp + 1e-6)
        bal_acc = (sens + spec) / 2.0

        f1_window.append(f1)
        f1_smooth = float(np.mean(f1_window))

        history['train_loss'].append(t_loss / max(len(train_dl), 1))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        history['val_f1_smooth'].append(f1_smooth)
        history['val_mcc'].append(mcc)

        print(f"  Ep{epoch+1:2d}[{ema_tag}] "
              f"Loss:{t_loss/max(len(train_dl),1):.4f} | "
              f"Val Acc:{acc:.4f} Bal:{bal_acc:.4f} | "
              f"F1:{f1:.4f}(sm:{f1_smooth:.4f}) MCC:{mcc:.4f} | "
              f"Sens:{sens:.3f} Spec:{spec:.3f} Thr:{th:.2f} LR:{curr_lr:.2e}")

        if f1_smooth > best_val_f1 + 0.001:
            best_val_f1 = f1_smooth; best_epoch = epoch; early_counter = 0
            if epoch >= Config.EMA_WARMUP:
                ema.apply_shadow(model)
            torch.save({'state': model.state_dict(), 'threshold': th,
                        'f1': f1, 'f1_smooth': f1_smooth, 'mcc': mcc,
                        'epoch': epoch, 'ema_used': epoch >= Config.EMA_WARMUP},
                       model_path)
            if epoch >= Config.EMA_WARMUP:
                ema.restore(model)
            print(f"  >>> Best saved [{ema_tag}] F1={f1:.4f} sm={f1_smooth:.4f} "
                  f"Thr={th:.2f} Bal={bal_acc:.4f}")
        else:
            early_counter += 1
            print(f"  >>> No improve. "
                  f"Counter:{early_counter}/{Config.EARLY_STOP_PATIENCE}")
            if early_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"  >>> Early stop @ ep{epoch+1}. Best ep{best_epoch+1}")
                break

    ckpt = torch.load(model_path, weights_only=False)
    model.load_state_dict(ckpt['state'])
    th_final = ckpt['threshold']
    print(f"\n  Loaded best ep{ckpt['epoch']+1} "
          f"(EMA={'yes' if ckpt['ema_used'] else 'no'}, thr={th_final:.2f})")
    print(f"  Evaluating on FIXED test set...")

    test_acc, _, preds, test_lbls, test_probs = evaluate(
        model, test_dl, criterion, threshold=th_final, tta=Config.USE_TTA)

    cm = confusion_matrix(test_lbls, preds)
    tn2, fp2, fn2, tp2 = cm.ravel()
    test_sens = tp2 / (tp2 + fn2 + 1e-6)
    test_spec = tn2 / (tn2 + fp2 + 1e-6)
    test_mcc  = matthews_corrcoef(test_lbls, preds)
    test_bal  = (test_sens + test_spec) / 2.0
    test_auc  = roc_auc_score(test_lbls, test_probs[:, 1])

    print(f"\n  [Fold{fold_idx+1}|{exp_name}] "
          f"Acc={test_acc:.4f} Sens={test_sens:.4f} Spec={test_spec:.4f} "
          f"Bal={test_bal:.4f} MCC={test_mcc:.4f} AUC={test_auc:.4f}")
    print(f"  CM: TN={tn2} FP={fp2} FN={fn2} TP={tp2}")

    _save_fold_plots(history, test_acc, preds, test_lbls, test_probs,
                     save_dir, exp_name, fold_idx)

    return {'fold': fold_idx + 1, 'exp': exp_name,
            'test_acc': test_acc, 'test_sens': test_sens, 'test_spec': test_spec,
            'test_bal': test_bal, 'test_mcc': test_mcc,  'test_auc': test_auc,
            'threshold': th_final, 'best_epoch': best_epoch + 1,
            'cm': cm, 'preds': preds, 'lbls': test_lbls, 'probs': test_probs}


# ==============================================================================
# 11. 图表
# ==============================================================================
def _save_fold_plots(history, test_acc, preds, test_lbls, test_probs,
                     save_dir, exp_name, fold_idx):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', color='r')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='b')
    if Config.EMA_WARMUP < len(history['val_loss']):
        axes[0].axvline(x=Config.EMA_WARMUP - 0.5, color='gray',
                        linestyle=':', alpha=0.7, label='EMA start')
    axes[0].set_title(f'Loss | {exp_name} Fold{fold_idx+1}')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_acc'], label='Val Acc',  color='g',      alpha=0.4)
    axes[1].plot(history['val_f1'],  label='Val F1',   color='orange', alpha=0.4)
    axes[1].plot(history['val_f1_smooth'],
                 label=f'F1 smooth(w={Config.SMOOTH_WINDOW})',
                 color='red', linewidth=2, linestyle='--')
    axes[1].set_title('Val Acc & F1')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['val_mcc'], label='Val MCC', color='purple')
    axes[2].set_title('Val MCC')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'{exp_name} | Fold {fold_idx+1} | '
                 f'Test Acc={test_acc:.4f} (fixed test set)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=100)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(test_lbls, preds), annot=True, fmt='d',
                cmap='Blues', xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES)
    plt.title(f'{exp_name} Fold{fold_idx+1} Acc={test_acc:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=100)
    plt.close()

    fpr, tpr, _ = roc_curve(test_lbls, test_probs[:, 1])
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC | {exp_name} Fold{fold_idx+1}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=100)
    plt.close()


def plot_summary(all_exp_results, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    metrics      = ['test_acc', 'test_sens', 'test_spec',
                    'test_bal', 'test_mcc',  'test_auc']
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity',
                    'Balanced Acc', 'MCC', 'AUC']

    exp_names = list(all_exp_results.keys())
    # 加入参照实验数据（来自多试验杂糅结果）
    reference = {
        'Img_SE':       {'test_acc': 0.8281, 'test_sens': 0.8714, 'test_spec': 0.7271,
                         'test_bal': 0.7993, 'test_mcc':  0.5944, 'test_auc': 0.8712},
        'Fusion_Add':   {'test_acc': 0.8181, 'test_sens': 0.8429, 'test_spec': 0.7604,
                         'test_bal': 0.8016, 'test_mcc':  0.5858, 'test_auc': 0.8765},
        'Fusion_Gated': {'test_acc': 0.8231, 'test_sens': 0.8589, 'test_spec': 0.7396,
                         'test_bal': 0.7993, 'test_mcc':  0.5905, 'test_auc': 0.8665},
    }

    colors_new = ['#e74c3c', '#e67e22']         # 新实验: 红/橙
    colors_ref = ['#95a5a6', '#7f8c8d', '#bdc3c7']  # 参照: 灰

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for ax, metric, mname in zip(axes.flatten(), metrics, metric_names):
        # 新实验
        new_means = [np.mean([r[metric] for r in all_exp_results[e]])
                     for e in exp_names]
        new_stds  = [np.std([r[metric]  for r in all_exp_results[e]])
                     for e in exp_names]
        # 参照
        ref_names  = list(reference.keys())
        ref_means  = [reference[r][metric] for r in ref_names]

        all_names  = exp_names + ref_names
        all_means  = new_means + ref_means
        all_stds   = new_stds  + [0.0] * len(ref_names)
        all_colors = colors_new + colors_ref

        order = np.argsort(all_means)
        ax.barh([all_names[i] for i in order],
                [all_means[i] for i in order],
                xerr=[all_stds[i] for i in order],
                color=[all_colors[i] for i in order],
                alpha=0.85, capsize=4, height=0.6)
        for i_o, i in enumerate(order):
            ax.text(all_means[i] + all_stds[i] + 0.002, i_o,
                    f'{all_means[i]:.4f}', va='center', fontsize=8.5)
        ax.set_title(mname, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.10)
        ax.grid(True, alpha=0.3, axis='x')
        # 最高值标红字
        best_i = int(np.argmax([all_means[i] for i in order]))
        ax.get_yticklabels()[best_i].set_color('red')
        ax.get_yticklabels()[best_i].set_fontweight('bold')

    plt.suptitle(
        'Img_SE + Fusion_Add 组合实验\n'
        '5-Fold CV | Fixed Test Set | seed=42\n'
        '红/橙=新实验  灰=参照(单实验均值)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ROC 叠加
    fig, ax = plt.subplots(figsize=(8, 7))
    mean_fpr = np.linspace(0, 1, 200)
    for exp, color in zip(exp_names, colors_new):
        tprs, aucs = [], []
        for r in all_exp_results[exp]:
            fpr, tpr, _ = roc_curve(r['lbls'], r['probs'][:, 1])
            aucs.append(auc(fpr, tpr))
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2.5,
                label=f'{exp}  AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Mean ROC -- Img_SE + Fusion_Add', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc.png'), dpi=150)
    plt.close()
    print(f"  Summary plots saved: {save_dir}")


# ==============================================================================
# 12. 主流程
# ==============================================================================
def main():
    set_seed(Config.SEED)

    if os.path.exists(Config.SAVE_ROOT):
        shutil.rmtree(Config.SAVE_ROOT)
    Path(Config.SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    patient_data = load_all_data()
    print(f"\n>>> Preparing fixed test set + {Config.N_FOLDS}-fold CV...")
    test_data, folds = prepare_splits(patient_data)

    metrics      = ['test_acc', 'test_sens', 'test_spec',
                    'test_bal', 'test_mcc',  'test_auc']
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity',
                    'Balanced Acc', 'MCC', 'AUC']

    all_exp_results = {}

    for exp_name, head_type in EXPERIMENTS:
        head_str = ('512->1024->512->256->2' if head_type == 'baseline'
                    else '512->2048->1024->512->256->2')
        print(f"\n{'='*65}")
        print(f"EXPERIMENT : {exp_name}")
        print(f"  Backbone : EfficientNetV2-S (1280d)")
        print(f"  Img SE   : SEBlock(1280, reduction=16)")
        print(f"  Meta     : MetaMLP_Baseline(5->128->256)")
        print(f"  Fusion   : AddFusion(img=1280, meta=256, proj=512)")
        print(f"             LayerNorm(SiLU(img_proj) + SiLU(meta_proj))")
        print(f"  Head     : FC({head_str})")
        print(f"{'='*65}")

        fold_results = []
        for fold_idx, (train_data, val_data) in enumerate(folds):
            fold_dir = os.path.join(Config.SAVE_ROOT, exp_name,
                                    f"fold_{fold_idx+1}")
            try:
                result = train_one_fold(
                    fold_idx, exp_name, head_type,
                    train_data, val_data, test_data, fold_dir)
                fold_results.append(result)
            except Exception as e:
                print(f"  [ERROR] Fold {fold_idx+1}: {e}")
                import traceback; traceback.print_exc()
                continue

        if not fold_results:
            print(f"  [SKIP] {exp_name}: no successful folds.")
            continue

        all_exp_results[exp_name] = fold_results

        print(f"\n  [{exp_name}] 5-Fold CV Results:")
        for r in fold_results:
            print(f"    Fold {r['fold']}: "
                  f"Acc={r['test_acc']:.4f} Bal={r['test_bal']:.4f} "
                  f"MCC={r['test_mcc']:.4f} AUC={r['test_auc']:.4f} "
                  f"BestEp={r['best_epoch']}")
        for m, mn in zip(metrics, metric_names):
            vals = [r[m] for r in fold_results]
            print(f"    {mn:<16}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    if all_exp_results:
        plot_summary(all_exp_results, os.path.join(Config.SAVE_ROOT, "summary"))

    sep = "=" * 80
    lines = [
        sep,
        "Img_SE + Fusion_Add 组合实验 -- 5-Fold CV (Fixed Test Set, seed=42)",
        sep,
        f"Time    : {datetime.now()}",
        f"Seed    : {Config.SEED}",
        f"Test set: fixed {Config.TEST_SIZE*100:.0f}% holdout ({len(test_data)} images)",
        f"TTA     : {Config.USE_TTA}",
        "",
        "共同组件 (两实验一致):",
        "  Backbone : EfficientNetV2-S  ->  1280d",
        "  Img SE   : SEBlock(dim=1280, reduction=16)",
        "             x_out = x * Sigmoid(FC(ReLU(FC(x))))",
        "  Meta     : MetaMLP_Baseline(5->128->256)  ->  256d",
        "  Fusion   : AddFusion(img=1280, meta=256, proj=512)",
        "             out = LayerNorm(SiLU(img_proj(x)) + SiLU(meta_proj(m)))",
        "  Loss     : WLS(w=[2,1], s=0.05)",
        "  Optimizer: AdamW(lr=2e-4, wd=2e-3) + CosineAnnealingLR",
        "  EMA      : decay=0.99, warmup=5",
        "  EarlyStop: patience=20, smooth_window=5",
        "",
        "差异:",
        "  Img_SE_Add          -> build_head_baseline(512->1024->512->256->2)",
        "  Img_SE_Add_DeepHead -> build_head_deep(512->2048->1024->512->256->2)",
        "",
        "参照 (来自多试验杂糅结果):",
        "  Img_SE      Acc=0.8281 Spec=0.7271 Bal=0.7993 MCC=0.5944 AUC=0.8712",
        "  Fusion_Add  Acc=0.8181 Spec=0.7604 Bal=0.8016 MCC=0.5858 AUC=0.8765",
        "",
        sep,
        f"{'Experiment':<22} {'Acc':>14} {'Sens':>14} {'Spec':>14} "
        f"{'BalAcc':>14} {'MCC':>14} {'AUC':>14}",
        "-" * 80,
    ]

    sorted_exps = sorted(
        all_exp_results.items(),
        key=lambda x: np.mean([r['test_acc'] for r in x[1]]),
        reverse=True)

    for exp_name, fold_results in sorted_exps:
        row = f"{exp_name:<22}"
        for m in metrics:
            vals = [r[m] for r in fold_results]
            row += f"  {np.mean(vals):.4f}+/-{np.std(vals):.4f}"
        lines.append(row)

    lines += ["", sep, "", "Per-fold detail:"]
    for exp_name, fold_results in sorted_exps:
        lines.append(f"\n  {exp_name}:")
        lines.append(f"    {'Fold':>4} {'Acc':>8} {'Sens':>8} {'Spec':>8} "
                     f"{'Bal':>8} {'MCC':>8} {'AUC':>8} {'BestEp':>7} {'Thr':>5}")
        for r in fold_results:
            lines.append(
                f"    {r['fold']:>4} {r['test_acc']:>8.4f} {r['test_sens']:>8.4f} "
                f"{r['test_spec']:>8.4f} {r['test_bal']:>8.4f} "
                f"{r['test_mcc']:>8.4f} {r['test_auc']:>8.4f} "
                f"{r['best_epoch']:>7d} {r['threshold']:>5.2f}")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(Config.SAVE_ROOT, "final_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")
    return all_exp_results


if __name__ == "__main__":
    main()