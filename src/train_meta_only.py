"""
Meta_Only 实验独立版本 -- 5折交叉验证（优化版）
=====================================
仅使用元数据（Age, Gender, Smoking, Chewing_Betel_Quid, Alcohol），
不使用图像特征。移除无效的图像加载/变换逻辑，大幅降低运行耗时。
其余训练设置与原脚本完全一致，保证消融实验的严谨性。

MetaOnlyModel 完整路径:
  1. meta_branch = MetaMLP_Baseline(5, out_dim=256)    → 256d
  2. classifier  = build_head_baseline(256)            → 256->1024->512->256->2

forward:
  meta_feat = meta_branch(meta)       # 256d
  out       = classifier(meta_feat)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import ImageFile  # 仅保留必要导入，移除Image
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
# 1. Config
# ==============================================================================
class Config:
    BASE_DIR    = "/home/wgf_v100/srtp/\u53e3\u8154\u764c\u5206\u7c7b\u8bc6\u522b\u9879\u76ee"
    CROP_DIR    = os.path.join(BASE_DIR, "Segmented_Images/Segmented_Images")
    CSV_PATH    = os.path.join(BASE_DIR, "data/Imagewise_Data.csv")
    PATIENT_CSV = os.path.join(BASE_DIR, "data/Patientwise_Data.csv")

    SEED      = 42
    N_FOLDS   = 5
    TEST_SIZE = 0.15
    SAVE_ROOT = os.path.join(BASE_DIR, "results_meta_only_cv5_seed42_optimized")

    IMG_SIZE            = 448
    BATCH_SIZE          = 16
    GRAD_ACCUMULATION   = 3
    EPOCHS              = 100
    LR_MAX              = 2e-4
    WEIGHT_DECAY        = 2e-3
    EARLY_STOP_PATIENCE = 20
    NUM_WORKERS         = 0  # 优化：无需加载图像，设为0减少进程开销

    USE_TTA      = True          # DataLoader结构不变，TTA对meta-only无额外效果
    THRESH_RANGE = np.arange(0.15, 0.95, 0.01)

    CLASS_MAP   = {'Benign': 0, 'OCA': 1, 'OPMD': 1}
    CLASS_NAMES = ['Benign', 'Malignant']

    EMA_DECAY     = 0.99
    EMA_WARMUP    = 5
    SMOOTH_WINDOW = 5

    EXP_NAME = 'Meta_Only_Optimized'


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
# 2. EMA
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
# 3. 损失函数
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
# 4. Dataset （核心优化：移除所有图像加载/变换逻辑）
# ==============================================================================
class MultiModalDataset(Dataset):
    def __init__(self, data_items, transform=None):  # transform保留但不使用
        self.data      = data_items
        self.transform = transform  # 保留参数以兼容原有接口

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
        # 核心优化1：完全移除图像加载逻辑，用空张量占位（兼容接口）
        img = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE, dtype=torch.float32)
        # 核心优化2：无需图像变换，直接生成元数据和标签
        meta  = torch.tensor(self._encode_meta(item['info']), dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        return img, meta, label


# ==============================================================================
# 5. 数据加载与固定划分 （与原脚本完全一致）
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
# 6. 模型 -- 仅使用元数据（与原脚本完全一致）
# ==============================================================================

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


class MetaOnlyModel(nn.Module):
    """
    仅使用元数据，不使用图像特征。
      1. meta_branch = MetaMLP_Baseline(5->128->256)   → 256d
      2. classifier  = build_head_baseline(256)        → 256->1024->512->256->2

    forward 接收 (img, meta) 以保持与 DataLoader/evaluate 接口一致，
    但 img 张量在此模型中被完全忽略。
    """
    def __init__(self, meta_dim=5, num_classes=2):
        super().__init__()
        # step 1: meta branch
        self.meta_branch = MetaMLP_Baseline(meta_dim, out_dim=256)
        meta_out = 256
        # step 2: classifier（仅基于meta特征）
        self.classifier = build_head_baseline(meta_out, num_classes)

    def forward(self, img, meta):
        # img 被忽略，仅使用 meta
        meta_feat = self.meta_branch(meta)   # (B, 256)
        return self.classifier(meta_feat)


# ==============================================================================
# 7. 阈值搜索 & 评估 （与原脚本完全一致）
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
            # tta 对 meta-only 无实际增益，保持接口一致即可
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
# 8. 单折训练 （核心优化：简化transform，移除图像相关计算）
# ==============================================================================
def train_one_fold(fold_idx, train_data, val_data, test_data, save_dir):

    set_seed(Config.SEED)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n  {'─'*60}")
    print(f"  Fold {fold_idx+1}/{Config.N_FOLDS} | {Config.EXP_NAME} | seed={Config.SEED}")
    print(f"  Train={len(train_data)}  Val={len(val_data)}  "
          f"Test={len(test_data)}(fixed)")
    print(f"  {'─'*60}")

    # 核心优化3：transform设为None（无需图像变换）
    train_tf = None
    eval_tf = None

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

    # 使用 MetaOnlyModel（与原脚本一致）
    model = MetaOnlyModel(meta_dim=5).to(DEVICE)

    total   = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: total={total:,}  trainable={n_train:,}")
    print(f"  MetaMLP(5->128->256) -> FC(256->1024->512->256->2)  [img IGNORED]")

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

    # 新增：记录单折开始时间，验证提速效果
    fold_start = datetime.now()

    for epoch in range(Config.EPOCHS):
        model.train()
        t_loss = 0
        optimizer.zero_grad()

        for i, (img, meta, lbl) in enumerate(
                tqdm(train_dl,
                     desc=f"[{Config.EXP_NAME}|F{fold_idx+1}] Ep{epoch+1}",
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

    # 新增：打印单折耗时
    fold_end = datetime.now()
    print(f"  Fold {fold_idx+1} 总耗时: {fold_end - fold_start}")

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

    print(f"\n  [Fold{fold_idx+1}|{Config.EXP_NAME}] "
          f"Acc={test_acc:.4f} Sens={test_sens:.4f} Spec={test_spec:.4f} "
          f"Bal={test_bal:.4f} MCC={test_mcc:.4f} AUC={test_auc:.4f}")
    print(f"  CM: TN={tn2} FP={fp2} FN={fn2} TP={tp2}")

    _save_fold_plots(history, test_acc, preds, test_lbls, test_probs,
                     save_dir, fold_idx)

    return {'fold': fold_idx + 1,
            'test_acc': test_acc, 'test_sens': test_sens, 'test_spec': test_spec,
            'test_bal': test_bal, 'test_mcc': test_mcc,  'test_auc': test_auc,
            'threshold': th_final, 'best_epoch': best_epoch + 1,
            'cm': cm, 'preds': preds, 'lbls': test_lbls, 'probs': test_probs}


# ==============================================================================
# 9. 图表 （与原脚本完全一致）
# ==============================================================================
def _save_fold_plots(history, test_acc, preds, test_lbls, test_probs,
                     save_dir, fold_idx):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', color='r')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='b')
    if Config.EMA_WARMUP < len(history['val_loss']):
        axes[0].axvline(x=Config.EMA_WARMUP - 0.5, color='gray',
                        linestyle=':', alpha=0.7, label='EMA start')
    axes[0].set_title(f'Loss | {Config.EXP_NAME} Fold{fold_idx+1}')
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

    plt.suptitle(f'{Config.EXP_NAME} | Fold {fold_idx+1} | '
                 f'Test Acc={test_acc:.4f} (fixed test set)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=100)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(test_lbls, preds), annot=True, fmt='d',
                cmap='Blues', xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES)
    plt.title(f'{Config.EXP_NAME} Fold{fold_idx+1} Acc={test_acc:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=100)
    plt.close()

    fpr, tpr, _ = roc_curve(test_lbls, test_probs[:, 1])
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC | {Config.EXP_NAME} Fold{fold_idx+1}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=100)
    plt.close()


def plot_cv_summary(all_results, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    metrics      = ['test_acc', 'test_sens', 'test_spec',
                    'test_bal', 'test_mcc',  'test_auc']
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity',
                    'Balanced Acc', 'MCC', 'AUC']
    fold_nums = [r['fold'] for r in all_results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric, mname in zip(axes.flatten(), metrics, metric_names):
        vals = [r[metric] for r in all_results]
        ax.plot(fold_nums, vals, marker='o', linewidth=2, color='#2980b9')
        ax.axhline(np.mean(vals), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean={np.mean(vals):.4f}')
        ax.fill_between(fold_nums,
                        np.mean(vals) - np.std(vals),
                        np.mean(vals) + np.std(vals),
                        alpha=0.15, color='red',
                        label=f'Std={np.std(vals):.4f}')
        ax.set_xticks(fold_nums)
        ax.set_xlabel('Fold')
        ax.set_title(mname, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'{Config.EXP_NAME} -- 5-Fold CV Summary\n'
        f'MetaMLP(5->128->256) + baseline head | seed=42',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_summary.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(all_results)))
    for r, color in zip(all_results, colors):
        fpr, tpr, _ = roc_curve(r['lbls'], r['probs'][:, 1])
        fold_auc    = auc(fpr, tpr); aucs.append(fold_auc)
        interp_tpr    = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1, alpha=0.5,
                label=f"Fold {r['fold']}  AUC={fold_auc:.4f}")

    mean_tpr     = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr      = np.std(tprs, axis=0)
    ax.plot(mean_fpr, mean_tpr, 'k-', linewidth=2.5,
            label=f'Mean AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}')
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.15, color='black', label='+/- 1 std')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC -- {Config.EXP_NAME} (All Folds)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_roc.png'), dpi=150)
    plt.close()
    print(f"  Summary plots saved: {save_dir}")


# ==============================================================================
# 10. 主流程 （与原脚本完全一致）
# ==============================================================================
def main():
    set_seed(Config.SEED)

    if os.path.exists(Config.SAVE_ROOT):
        shutil.rmtree(Config.SAVE_ROOT)
    Path(Config.SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    # 新增：记录总开始时间
    total_start = datetime.now()

    patient_data = load_all_data()
    print(f"\n>>> Preparing fixed test set + {Config.N_FOLDS}-fold CV...")
    test_data, folds = prepare_splits(patient_data)

    metrics      = ['test_acc', 'test_sens', 'test_spec',
                    'test_bal', 'test_mcc',  'test_auc']
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity',
                    'Balanced Acc', 'MCC', 'AUC']

    print(f"\n{'='*65}")
    print(f"EXPERIMENT: {Config.EXP_NAME}")
    print(f"  Meta only: MetaMLP_Baseline(5->128->256)  ← 无图像输入")
    print(f"  Head     : FC(256->1024->512->256->2) + BN+SiLU+Dropout")
    print(f"  Loss     : WLS(w=[2,1], s=0.05)")
    print(f"  Optimized: 移除无效图像加载/变换，NUM_WORKERS=0")
    print(f"{'='*65}")

    all_results = []
    for fold_idx, (train_data, val_data) in enumerate(folds):
        fold_dir = os.path.join(Config.SAVE_ROOT, f"fold_{fold_idx+1}")
        result   = train_one_fold(fold_idx, train_data, val_data,
                                  test_data, fold_dir)
        all_results.append(result)

        print(f"\n  Results so far:")
        for r in all_results:
            print(f"    Fold {r['fold']}: "
                  f"Acc={r['test_acc']:.4f} Bal={r['test_bal']:.4f} "
                  f"MCC={r['test_mcc']:.4f} AUC={r['test_auc']:.4f} "
                  f"BestEp={r['best_epoch']}")

    plot_cv_summary(all_results, os.path.join(Config.SAVE_ROOT, "summary"))

    # 新增：打印总耗时
    total_end = datetime.now()
    print(f"\n  5折总耗时: {total_end - total_start}")

    sep = "=" * 70
    lines = [
        sep,
        f"{Config.EXP_NAME} -- 5-Fold CV Report",
        sep,
        f"Time    : {datetime.now()}",
        f"Seed    : {Config.SEED}",
        f"Test set: fixed {Config.TEST_SIZE*100:.0f}% holdout "
        f"({len(test_data)} images)",
        f"TTA     : {Config.USE_TTA}",
        f"Optimized: Yes (removed image loading/transform, NUM_WORKERS=0)",
        "",
        "Architecture:",
        "  Meta branch: MetaMLP_Baseline(5->128->256)  ->  256d",
        "  Classifier : FC(256->1024->512->256->2) + BN+SiLU+Dropout",
        "  Image      : NOT USED (empty tensor placeholder)",
        "  Loss       : WeightedLabelSmoothing(w=[2,1], s=0.05)",
        "  Optimizer  : AdamW(lr=2e-4, wd=2e-3) + CosineAnnealingLR",
        "  EMA        : decay=0.99, warmup=5",
        "  Early stop : patience=20, smooth_window=5",
        "",
        sep,
        f"{'Fold':>6}{'Acc':>8}{'Sens':>8}{'Spec':>8}"
        f"{'BalAcc':>9}{'MCC':>8}{'AUC':>8}{'BestEp':>8}{'Thr':>6}",
        "-" * 70,
    ]
    for r in all_results:
        lines.append(
            f"{r['fold']:>6}{r['test_acc']:>8.4f}{r['test_sens']:>8.4f}"
            f"{r['test_spec']:>8.4f}{r['test_bal']:>9.4f}{r['test_mcc']:>8.4f}"
            f"{r['test_auc']:>8.4f}{r['best_epoch']:>8d}{r['threshold']:>6.2f}")

    lines += ["", "-" * 70, "5-Fold Summary (mean +/- std):"]
    for metric, mname in zip(metrics, metric_names):
        vals = [r[metric] for r in all_results]
        lines.append(
            f"  {mname:<16}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}"
            f"  [min={np.min(vals):.4f}  max={np.max(vals):.4f}]")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(Config.SAVE_ROOT, "cv_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")
    return all_results


if __name__ == "__main__":
    main()