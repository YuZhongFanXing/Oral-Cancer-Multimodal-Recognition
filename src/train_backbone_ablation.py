"""
对比实验 + 消融实验 -- 5折交叉验证
=====================================
固定测试集 (15%, seed=42, patient-level stratified) 与所有前序脚本一致。

【对比实验】固定 Img_SE + Concat + build_head_baseline，只换backbone:
  Backbone_ResNet50      ResNet50       (2048d)
  Backbone_ConvNeXt      ConvNeXt-Tiny  ( 768d)
  Backbone_EffB3         EfficientNet-B3(1536d)
  Backbone_MobileNetV3   MobileNet-V3-L ( 960d)
  Backbone_DenseNet121   DenseNet121    (1024d)
  参照: EfficientNetV2-S (1280d, Img_SE结果 Acc=0.8281)

【消融实验】固定 EfficientNetV2-S backbone:
  Ablation_Unimodal  仅图像 (无meta分支)，img_se=SEBlock，head(1280d)
  Ablation_NoSE      无Img_SE，有meta，Concat(1536d)，head(1536d)
  Ablation_NoEMA     完整结构(同Img_SE)，但关闭EMA
  参照: Img_SE full model (Acc=0.8281, 已有结果)

【指标】Accuracy, Sensitivity, Specificity, Balanced Acc,
        Macro F1, Macro Precision, MCC, AUC
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
# (exp_name, exp_category, exp_variant)
# exp_category: 'backbone' | 'ablation'
# exp_variant : backbone name  或  ablation type
EXPERIMENTS = [
    # ── 对比实验: backbone ──────────────────────────────────────────────────
    ('Backbone_ResNet50',    'backbone', 'resnet50'),
    ('Backbone_ConvNeXt',    'backbone', 'convnext_tiny'),
    ('Backbone_EffB3',       'backbone', 'efficientnet_b3'),
    ('Backbone_MobileNetV3', 'backbone', 'mobilenet_v3_large'),
    ('Backbone_DenseNet121', 'backbone', 'densenet121'),
    # ── 消融实验 ──────────────────────────────────────────────────────────
    ('Ref_ImgSE',            'ablation', 'ref_imgse'),  # 基线(重跑，获取真实ROC数据)
    ('Ablation_Unimodal',    'ablation', 'unimodal'),   # 无meta
    ('Ablation_NoSE',        'ablation', 'no_se'),       # 无Img_SE
    ('Ablation_NoEMA',       'ablation', 'no_ema'),      # 无EMA
    ('Ablation_WCE',         'ablation', 'wce'),         # 加权交叉熵(无label smoothing)
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
    SAVE_ROOT = os.path.join(BASE_DIR, "results_comparison_ablation_cv5_seed42")

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


class WeightedCELoss(nn.Module):
    """加权交叉熵: 无 label smoothing, weight=[2,1]"""
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        w = self.weight.to(pred.device) if self.weight is not None else None
        return F.cross_entropy(pred, target, weight=w)


# ==============================================================================
# 5. Dataset
# ==============================================================================
class MultiModalDataset(Dataset):
    def __init__(self, data_items, transform=None, return_meta=True):
        self.data        = data_items
        self.transform   = transform
        self.return_meta = return_meta   # False -> 单模态实验

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


# ── Backbone 加载函数 ─────────────────────────────────────────────────────────

def get_efficientnet_v2s():
    m   = models.efficientnet_v2_s(weights='DEFAULT')
    dim = m.classifier[1].in_features   # 1280
    m.classifier = nn.Identity()
    return m, dim

def get_resnet50():
    m   = models.resnet50(weights='DEFAULT')
    dim = m.fc.in_features               # 2048
    m.fc = nn.Identity()
    return m, dim

def get_convnext_tiny():
    m   = models.convnext_tiny(weights='DEFAULT')
    dim = m.classifier[2].in_features   # 768
    m.classifier[2] = nn.Identity()
    return m, dim

def get_efficientnet_b3():
    m   = models.efficientnet_b3(weights='DEFAULT')
    dim = m.classifier[1].in_features   # 1536
    m.classifier = nn.Identity()
    return m, dim

def get_mobilenet_v3_large():
    m   = models.mobilenet_v3_large(weights='DEFAULT')
    # classifier: [Linear(960,1280), Hardswish, Dropout, Linear(1280,1000)]
    # 取960d (avgpool后的直接输出), 去掉整个classifier
    dim = m.classifier[0].in_features   # 960
    m.classifier = nn.Identity()
    return m, dim

def get_densenet121():
    m   = models.densenet121(weights='DEFAULT')
    dim = m.classifier.in_features      # 1024
    m.classifier = nn.Identity()
    return m, dim

BACKBONE_LOADERS = {
    'efficientnet_v2_s':  get_efficientnet_v2s,
    'resnet50':           get_resnet50,
    'convnext_tiny':      get_convnext_tiny,
    'efficientnet_b3':    get_efficientnet_b3,
    'mobilenet_v3_large': get_mobilenet_v3_large,
    'densenet121':        get_densenet121,
}

BACKBONE_LABELS = {
    'resnet50':           'ResNet50',
    'convnext_tiny':      'ConvNeXt-Tiny',
    'efficientnet_b3':    'EfficientNet-B3',
    'mobilenet_v3_large': 'MobileNet-V3-L',
    'densenet121':        'DenseNet121',
    'efficientnet_v2_s':  'EfficientNetV2-S',
}


# ==============================================================================
# 8. 模型类
# ==============================================================================

class BackboneModel(nn.Module):
    """
    对比实验通用模型:
      backbone(各自out_dim) -> SEBlock(out_dim) -> Concat(out_dim+256)
      MetaMLP_Baseline(5->256)
      build_head_baseline(out_dim+256)
    """
    def __init__(self, backbone_name, meta_dim=5, num_classes=2):
        super().__init__()
        loader = BACKBONE_LOADERS[backbone_name]
        self.backbone, img_out = loader()
        self.img_se      = SEBlock(img_out, reduction=16)
        self.meta_branch = MetaMLP_Baseline(meta_dim, out_dim=256)
        fusion_dim       = img_out + 256
        self.classifier  = build_head_baseline(fusion_dim, num_classes)

    def forward(self, img, meta):
        img_feat  = self.backbone(img)
        img_feat  = self.img_se(img_feat)
        meta_feat = self.meta_branch(meta)
        fused     = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(fused)


class AblationModel(nn.Module):
    """
    消融实验模型 (backbone 固定为 EfficientNetV2-S):

      'ref_imgse': img_se=SEBlock, meta=MLP, Concat(1536d), head(1536d) + EMA  [基线复现]
      'unimodal':  img_se=SEBlock, NO meta, head(1280d)
      'no_se':     img_se=None, meta=MLP, Concat(1536d), head(1536d)
      'no_ema':    img_se=SEBlock, meta=MLP, Concat(1536d), head(1536d)  [EMA关闭]
      'wce':       img_se=SEBlock, meta=MLP, Concat(1536d), head(1536d)  [WCE损失]
    """
    def __init__(self, ablation_type, meta_dim=5, num_classes=2):
        super().__init__()
        self.ablation_type = ablation_type

        self.backbone, img_out = get_efficientnet_v2s()   # 1280d

        if ablation_type == 'no_se':
            self.img_se = None
        else:
            self.img_se = SEBlock(img_out, reduction=16)

        if ablation_type == 'unimodal':
            self.meta_branch = None
            fusion_dim = img_out                           # 1280d
        else:
            self.meta_branch = MetaMLP_Baseline(meta_dim, out_dim=256)
            fusion_dim = img_out + 256                     # 1536d

        self.classifier = build_head_baseline(fusion_dim, num_classes)

    def forward(self, img, meta):
        img_feat = self.backbone(img)
        if self.img_se is not None:
            img_feat = self.img_se(img_feat)

        if self.meta_branch is not None:
            meta_feat = self.meta_branch(meta)
            fused     = torch.cat([img_feat, meta_feat], dim=1)
        else:
            fused = img_feat   # unimodal: meta ignored

        return self.classifier(fused)


def build_model(exp_category, exp_variant):
    if exp_category == 'backbone':
        return BackboneModel(exp_variant, meta_dim=5)
    else:
        return AblationModel(exp_variant, meta_dim=5)


# ==============================================================================
# 9. 阈值搜索 & 评估 (新增 macro_f1, macro_precision)
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


def compute_metrics(labels, preds, probs):
    """返回完整指标字典"""
    labels = np.array(labels)
    preds  = np.array(preds)

    report = classification_report(
        labels, preds, output_dict=True, zero_division=0)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    sens    = tp / (tp + fn + 1e-6)
    spec    = tn / (tn + fp + 1e-6)
    bal_acc = (sens + spec) / 2.0
    mcc     = matthews_corrcoef(labels, preds)
    auc_val = roc_auc_score(labels, probs[:, 1])
    acc     = np.mean(preds == labels)

    macro_f1   = report['macro avg']['f1-score']
    macro_prec = report['macro avg']['precision']

    return {
        'acc':        acc,
        'sens':       sens,
        'spec':       spec,
        'bal_acc':    bal_acc,
        'macro_f1':   macro_f1,
        'macro_prec': macro_prec,
        'mcc':        mcc,
        'auc':        auc_val,
        'cm':         cm,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


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
    return (loss,
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


# ==============================================================================
# 10. 单折训练
# ==============================================================================
def train_one_fold(fold_idx, exp_name, exp_category, exp_variant,
                   train_data, val_data, test_data, save_dir):

    set_seed(Config.SEED)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    use_ema = (exp_variant != 'no_ema')   # 消融NoEMA时关闭

    print(f"\n  {'─'*64}")
    print(f"  Fold {fold_idx+1}/{Config.N_FOLDS} | {exp_name} | "
          f"EMA={'on' if use_ema else 'OFF'} | seed={Config.SEED}")
    print(f"  Train={len(train_data)}  Val={len(val_data)}  "
          f"Test={len(test_data)}(fixed)")
    print(f"  {'─'*64}")

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

    model = build_model(exp_category, exp_variant).to(DEVICE)

    total   = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: total={total:,}  trainable={n_train:,}")

    ema = ModelEMA(model, decay=Config.EMA_DECAY) if use_ema else None
    w   = torch.tensor([2.0, 1.0])
    if exp_variant == 'wce':
        criterion = WeightedCELoss(weight=w)
    else:
        criterion = WeightedLabelSmoothingLoss(weight=w.to(DEVICE), smoothing=0.05)
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
                if ema is not None:
                    ema.update(model)

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        # 验证
        use_ema_this_epoch = use_ema and (epoch >= Config.EMA_WARMUP)
        if use_ema_this_epoch:
            with ema.get_context(model):
                val_loss, val_preds, val_lbls, val_probs = evaluate(
                    model, val_dl, criterion, threshold=0.5, tta=False)
            ema_tag = "EMA"
        else:
            val_loss, val_preds, val_lbls, val_probs = evaluate(
                model, val_dl, criterion, threshold=0.5, tta=False)
            ema_tag = "raw"

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
            if use_ema_this_epoch:
                ema.apply_shadow(model)
            torch.save({'state': model.state_dict(), 'threshold': th,
                        'f1': f1, 'f1_smooth': f1_smooth, 'mcc': mcc,
                        'epoch': epoch,
                        'ema_used': use_ema_this_epoch},
                       model_path)
            if use_ema_this_epoch:
                ema.restore(model)
            print(f"  >>> Best [{ema_tag}] F1={f1:.4f} sm={f1_smooth:.4f} "
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
    print(f"  Evaluating on FIXED test set (TTA={Config.USE_TTA})...")

    _, preds, test_lbls, test_probs = evaluate(
        model, test_dl, criterion, threshold=th_final, tta=Config.USE_TTA)

    m = compute_metrics(test_lbls, preds, test_probs)
    print(f"\n  [Fold{fold_idx+1}|{exp_name}]")
    print(f"    Acc={m['acc']:.4f}  Sens={m['sens']:.4f}  Spec={m['spec']:.4f}  "
          f"Bal={m['bal_acc']:.4f}")
    print(f"    MacF1={m['macro_f1']:.4f}  MacPrec={m['macro_prec']:.4f}  "
          f"MCC={m['mcc']:.4f}  AUC={m['auc']:.4f}")
    print(f"    CM: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    _save_fold_plots(history, m['acc'], preds, test_lbls, test_probs,
                     save_dir, exp_name, fold_idx)

    # 保存供参照ROC加载
    np.save(os.path.join(save_dir, 'test_probs.npy'), test_probs)
    np.save(os.path.join(save_dir, 'test_lbls.npy'),  test_lbls)

    return {
        'fold': fold_idx + 1, 'exp': exp_name,
        'test_acc':   m['acc'],       'test_sens':  m['sens'],
        'test_spec':  m['spec'],      'test_bal':   m['bal_acc'],
        'macro_f1':   m['macro_f1'],  'macro_prec': m['macro_prec'],
        'test_mcc':   m['mcc'],       'test_auc':   m['auc'],
        'threshold':  th_final,       'best_epoch': best_epoch + 1,
        'cm': m['cm'], 'preds': preds, 'lbls': test_lbls, 'probs': test_probs,
    }


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
    plt.title(f'{exp_name} F{fold_idx+1} Acc={test_acc:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=100)
    plt.close()

    fpr, tpr, _ = roc_curve(test_lbls, test_probs[:, 1])
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC | {exp_name} F{fold_idx+1}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=100)
    plt.close()


def plot_summary(all_exp_results, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    metrics      = ['test_acc', 'test_sens', 'test_spec', 'test_bal',
                    'macro_f1', 'macro_prec', 'test_mcc', 'test_auc']
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Balanced Acc',
                    'Macro F1', 'Macro Precision', 'MCC', 'AUC']

    # 参照: Img_SE (EfficientNetV2-S, 架构实验最优结果, 两组实验共用同一基准)
    # macro_f1/macro_prec 原报告未记录，不含该字段，绘图时自动跳过
    REFERENCE = {
        'Img_SE(ref)': {
            'test_acc': 0.8281, 'test_sens': 0.8714, 'test_spec': 0.7271,
            'test_bal': 0.7993, 'test_mcc':  0.5944, 'test_auc':  0.8712,
        },
    }

    exp_names = list(all_exp_results.keys())
    backbone_exps  = [e for e in exp_names if e.startswith('Backbone')]
    ablation_exps  = [e for e in exp_names if e.startswith('Ablation')]

    # ── 对比实验图 ─────────────────────────────────────────────────────────
    _plot_group(backbone_exps, all_exp_results, REFERENCE, metrics, metric_names,
                title='Backbone对比实验 (固定: Img_SE + Concat + baseline_head)',
                save_path=os.path.join(save_dir, 'backbone_comparison.png'),
                new_color='#2980b9')

    # ── 消融实验图 ─────────────────────────────────────────────────────────
    _plot_group(ablation_exps, all_exp_results, REFERENCE, metrics, metric_names,
                title='消融实验 (固定backbone: EfficientNetV2-S)',
                save_path=os.path.join(save_dir, 'ablation_comparison.png'),
                new_color='#e74c3c')

    # ── Ref_ImgSE 基线ROC (从本脚本运行结果中获取) ─────────────────────
    mean_fpr_ref = np.linspace(0, 1, 200)
    ref_tprs, ref_aucs = [], []
    if 'Ref_ImgSE' in all_exp_results:
        for r in all_exp_results['Ref_ImgSE']:
            fpr_r, tpr_r, _ = roc_curve(r['lbls'], r['probs'][:, 1])
            ref_aucs.append(auc(fpr_r, tpr_r))
            ref_tprs.append(np.interp(mean_fpr_ref, fpr_r, tpr_r))
        print(f'  [ROC] Ref_ImgSE baseline: {len(ref_tprs)} folds, '
              f'mean AUC={np.mean(ref_aucs):.4f}')

    # ── ROC 叠加: backbone组 和 ablation组各一张，都叠加基线曲线 ──────────
    # backbone组: 只显示backbone实验，不含Ref_ImgSE本身
    # ablation组: 显示所有消融实验(含Ref_ImgSE)
    for group_name, group_exps in [('backbone', backbone_exps),
                                   ('ablation', ablation_exps)]:
        if not group_exps:
            continue
        fig, ax = plt.subplots(figsize=(9, 8))
        mean_fpr = np.linspace(0, 1, 200)
        colors   = plt.cm.tab10(np.linspace(0, 0.9, len(group_exps)))
        for exp, color in zip(group_exps, colors):
            tprs, exp_aucs = [], []
            for r in all_exp_results[exp]:
                fpr, tpr, _ = roc_curve(r['lbls'], r['probs'][:, 1])
                exp_aucs.append(auc(fpr, tpr))
                tprs.append(np.interp(mean_fpr, fpr, tpr))
            mean_tpr = np.mean(tprs, axis=0)
            # Ref_ImgSE 用黑实线区分
            if exp == 'Ref_ImgSE':
                ax.plot(mean_fpr, mean_tpr,
                        color='black', linewidth=2.5, linestyle='-',
                        label=f'Ref_ImgSE(baseline)  AUC={np.mean(exp_aucs):.4f}+/-{np.std(exp_aucs):.4f}')
            else:
                ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2,
                        label=f'{exp}  AUC={np.mean(exp_aucs):.4f}+/-{np.std(exp_aucs):.4f}')
        # backbone组额外叠加基线曲线(黑色虚线)
        if group_name == 'backbone' and len(ref_tprs) > 0:
            ref_mean_tpr = np.mean(ref_tprs, axis=0)
            ax.plot(mean_fpr_ref, ref_mean_tpr,
                    color='black', linewidth=2.5, linestyle='--',
                    label=f'Ref_ImgSE(baseline)  AUC={np.mean(ref_aucs):.4f}+/-{np.std(ref_aucs):.4f}')
        ax.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle=':')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        title_suffix = '  |  黑线=Img_SE基线' if group_name == 'backbone' else ''
        ax.set_title(f'Mean ROC -- {group_name}{title_suffix}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'roc_{group_name}.png'), dpi=150)
        plt.close()

    print(f"  Summary plots saved: {save_dir}")


def _plot_group(group_exps, all_exp_results, reference,
                metrics, metric_names, title, save_path, new_color):
    if not group_exps:
        return

    n_metrics = len(metrics)
    nrows     = (n_metrics + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(22, 5 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    all_names  = group_exps + list(reference.keys())
    new_colors = [new_color] * len(group_exps)
    ref_colors = ['#bdc3c7'] * len(reference)
    all_colors = new_colors + ref_colors

    for ax, metric, mname in zip(axes_flat, metrics, metric_names):
        # 新实验: 全部有该指标
        new_means, new_stds = [], []
        for e in group_exps:
            vals = [r[metric] for r in all_exp_results[e]]
            new_means.append(np.mean(vals)); new_stds.append(np.std(vals))

        # 参照: macro_f1/macro_prec 原报告未记录，跳过
        ref_names_m, ref_means_m, ref_colors_m = [], [], []
        for rname, rval in reference.items():
            if metric in rval:
                ref_names_m.append(rname)
                ref_means_m.append(rval[metric])
                ref_colors_m.append('#bdc3c7')

        plot_names  = group_exps     + ref_names_m
        plot_means  = new_means      + ref_means_m
        plot_stds   = new_stds       + [0.0] * len(ref_means_m)
        plot_colors = [new_color] * len(group_exps) + ref_colors_m

        order = np.argsort(plot_means)
        ax.barh([plot_names[i]  for i in order],
                [plot_means[i]  for i in order],
                xerr=[plot_stds[i] for i in order],
                color=[plot_colors[i] for i in order],
                alpha=0.85, capsize=4, height=0.6)
        for i_o, i in enumerate(order):
            ax.text(plot_means[i] + plot_stds[i] + 0.002, i_o,
                    f'{plot_means[i]:.4f}', va='center', fontsize=7)
        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.12)
        ax.grid(True, alpha=0.3, axis='x')
        best_i = int(np.argmax([plot_means[i] for i in order]))
        ax.get_yticklabels()[best_i].set_color('red')
        ax.get_yticklabels()[best_i].set_fontweight('bold')

    # 隐藏多余子图
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    plt.suptitle(title + '\n5-Fold CV | Fixed Test Set | seed=42 | 灰=参照',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()




# ==============================================================================
# Grad-CAM 可视化 -- 对比 Ablation_NoSE vs Ablation_NoEMA (有SE)
# ==============================================================================

class GradCAM:
    """
    针对 EfficientNetV2-S backbone 的 Grad-CAM 实现.
    hook 目标层: model.backbone.features[-1]
    """
    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._hooks     = []
        target = model.backbone.features[-1]
        self._hooks.append(
            target.register_forward_hook(self._save_activation))
        self._hooks.append(
            target.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, img_tensor, meta_tensor, class_idx=1):
        self.model.eval()
        img_tensor  = img_tensor.unsqueeze(0).to(DEVICE)
        meta_tensor = meta_tensor.unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_(True)

        logits = self.model(img_tensor, meta_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # GAP over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze()  # (H, W)
        cam     = torch.relu(cam)
        cam     = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def _overlay_cam(orig_img_np, cam, alpha=0.45):
    """将 CAM 热力图叠加到原始图像上"""
    import cv2 as _cv2
    h, w = orig_img_np.shape[:2]
    cam_resized = _cv2.resize(cam, (w, h))
    heatmap = _cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), _cv2.COLORMAP_JET)
    heatmap = _cv2.cvtColor(heatmap, _cv2.COLOR_BGR2RGB)
    overlaid = (alpha * heatmap + (1 - alpha) * orig_img_np).astype(np.uint8)
    return overlaid


def run_gradcam_comparison(all_exp_results, test_data, save_dir, n_samples=8):
    """
    比较 Ablation_NoSE (无SE) 与 Ablation_NoEMA (有SE，结构=Img_SE) 的 Grad-CAM.
    取各自 AUC 最高 fold 的 best_model.pth.
    输出: n_samples 张三列图 [原图 | NoSE CAM | WithSE CAM]
    """
    import cv2 as _cv2

    no_se_results  = all_exp_results.get('Ablation_NoSE')
    with_se_results = all_exp_results.get('Ablation_NoEMA')
    if no_se_results is None or with_se_results is None:
        print("  [GradCAM] Ablation_NoSE 或 Ablation_NoEMA 未完成，跳过")
        return

    # 选 AUC 最高的 fold
    def best_fold(results):
        return max(results, key=lambda r: r['test_auc'])

    no_se_best   = best_fold(no_se_results)
    with_se_best = best_fold(with_se_results)

    no_se_path   = os.path.join(Config.SAVE_ROOT, 'Ablation_NoSE',
                                f"fold_{no_se_best['fold']}", 'best_model.pth')
    with_se_path = os.path.join(Config.SAVE_ROOT, 'Ablation_NoEMA',
                                f"fold_{with_se_best['fold']}", 'best_model.pth')

    for path, label in [(no_se_path, 'NoSE'), (with_se_path, 'WithSE')]:
        if not os.path.exists(path):
            print(f"  [GradCAM] {label} model not found: {path}")
            return

    # 加载模型
    no_se_model   = AblationModel('no_se').to(DEVICE)
    with_se_model = AblationModel('no_ema').to(DEVICE)   # 结构同Img_SE
    no_se_model.load_state_dict(
        torch.load(no_se_path,   weights_only=False)['state'])
    with_se_model.load_state_dict(
        torch.load(with_se_path, weights_only=False)['state'])

    cam_no_se   = GradCAM(no_se_model)
    cam_with_se = GradCAM(with_se_model)

    # 随机挑样本 (各类均匀)
    rng = np.random.default_rng(42)
    benign_idx    = [i for i, d in enumerate(test_data) if d['label'] == 0]
    malignant_idx = [i for i, d in enumerate(test_data) if d['label'] == 1]
    n_each   = n_samples // 2
    selected = (list(rng.choice(benign_idx,    min(n_each, len(benign_idx)),    replace=False)) +
                list(rng.choice(malignant_idx, min(n_each, len(malignant_idx)), replace=False)))
    rng.shuffle(selected)

    eval_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = MultiModalDataset(test_data, eval_tf)

    gradcam_dir = os.path.join(save_dir, 'gradcam')
    Path(gradcam_dir).mkdir(parents=True, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    print(f"\n  [GradCAM] Generating {len(selected)} comparison images...")
    for idx, sample_idx in enumerate(selected):
        img_tensor, meta_tensor, label = ds[sample_idx]
        label_name = Config.CLASS_NAMES[label.item()]

        # 还原原图用于显示
        orig_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        orig_np = np.clip((orig_np * std + mean) * 255, 0, 255).astype(np.uint8)

        cam1 = cam_no_se(img_tensor,   meta_tensor, class_idx=1)
        cam2 = cam_with_se(img_tensor, meta_tensor, class_idx=1)

        overlay1 = _overlay_cam(orig_np, cam1)
        overlay2 = _overlay_cam(orig_np, cam2)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_np);   axes[0].set_title(f'Original [{label_name}]')
        axes[1].imshow(overlay1);  axes[1].set_title('NoSE CAM')
        axes[2].imshow(overlay2);  axes[2].set_title('WithSE CAM')
        for ax in axes:
            ax.axis('off')
        plt.suptitle(
            f'Grad-CAM: NoSE vs WithSE (Img_SE) | Sample {idx+1} [{label_name}]',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(gradcam_dir, f'gradcam_{idx+1:02d}_{label_name}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()

    cam_no_se.remove_hooks()
    cam_with_se.remove_hooks()
    print(f"  [GradCAM] Saved {len(selected)} images to {gradcam_dir}")

    # 合并为一张总览图
    n_cols = 3; n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    ds2 = MultiModalDataset(test_data, eval_tf)
    cam_no_se2   = GradCAM(no_se_model)
    cam_with_se2 = GradCAM(with_se_model)
    for row, sample_idx in enumerate(selected):
        img_tensor, meta_tensor, label = ds2[sample_idx]
        label_name = Config.CLASS_NAMES[label.item()]
        orig_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        orig_np = np.clip((orig_np * std + mean) * 255, 0, 255).astype(np.uint8)
        cam1 = cam_no_se2(img_tensor,   meta_tensor, class_idx=1)
        cam2 = cam_with_se2(img_tensor, meta_tensor, class_idx=1)
        axes[row, 0].imshow(orig_np)
        axes[row, 0].set_ylabel(f'[{label_name}]', fontsize=9)
        axes[row, 1].imshow(_overlay_cam(orig_np, cam1))
        axes[row, 2].imshow(_overlay_cam(orig_np, cam2))
        for ax in axes[row]: ax.axis('off')
    axes[0, 0].set_title('Original',  fontsize=11, fontweight='bold')
    axes[0, 1].set_title('NoSE CAM',  fontsize=11, fontweight='bold')
    axes[0, 2].set_title('WithSE CAM (Img_SE)', fontsize=11, fontweight='bold')
    plt.suptitle('Grad-CAM Overview: NoSE vs WithSE\nRed=high attention, Blue=low',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(gradcam_dir, 'gradcam_overview.png'),
                dpi=120, bbox_inches='tight')
    plt.close()
    cam_no_se2.remove_hooks()
    cam_with_se2.remove_hooks()
    print(f"  [GradCAM] Overview saved.")

# ==============================================================================
# 12. 主流程
# ==============================================================================
METRICS      = ['test_acc', 'test_sens', 'test_spec', 'test_bal',
                'macro_f1', 'macro_prec', 'test_mcc', 'test_auc']
METRIC_NAMES = ['Accuracy', 'Sensitivity', 'Specificity', 'Balanced Acc',
                'Macro F1', 'Macro Prec', 'MCC', 'AUC']


def main():
    set_seed(Config.SEED)

    if os.path.exists(Config.SAVE_ROOT):
        shutil.rmtree(Config.SAVE_ROOT)
    Path(Config.SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    patient_data = load_all_data()
    print(f"\n>>> Preparing fixed test set + {Config.N_FOLDS}-fold CV...")
    test_data, folds = prepare_splits(patient_data)

    all_exp_results = {}

    for exp_name, exp_category, exp_variant in EXPERIMENTS:
        print(f"\n{'='*68}")
        print(f"EXPERIMENT : {exp_name}  [{exp_category}]")
        if exp_category == 'backbone':
            lbl = BACKBONE_LABELS.get(exp_variant, exp_variant)
            print(f"  Backbone : {lbl}")
            print(f"  Img SE   : SEBlock(img_out, reduction=16)")
            print(f"  Meta     : MetaMLP_Baseline(5->128->256)")
            print(f"  Fusion   : Concat(img_se_out, meta_out)")
            print(f"  Head     : build_head_baseline")
        else:
            print(f"  Ablation : {exp_variant}")
            if exp_variant == 'unimodal':
                print(f"  Structure: EffV2S -> SE(1280) -> head(1280d)  [NO meta]")
            elif exp_variant == 'no_se':
                print(f"  Structure: EffV2S -> Concat(1536d) -> head  [NO SE]")
            elif exp_variant == 'no_ema':
                print(f"  Structure: EffV2S -> SE(1280) -> Concat(1536d) -> head  [EMA OFF]")
            elif exp_variant == 'wce':
                print(f"  Structure: EffV2S -> SE(1280) -> Concat(1536d) -> head  [WCE loss, no smoothing]")
        print(f"{'='*68}")

        fold_results = []
        for fold_idx, (train_data, val_data) in enumerate(folds):
            fold_dir = os.path.join(Config.SAVE_ROOT, exp_name, f"fold_{fold_idx+1}")
            try:
                result = train_one_fold(
                    fold_idx, exp_name, exp_category, exp_variant,
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
            print(f"    F{r['fold']}: Acc={r['test_acc']:.4f} "
                  f"MacF1={r['macro_f1']:.4f} MacPrec={r['macro_prec']:.4f} "
                  f"MCC={r['test_mcc']:.4f} AUC={r['test_auc']:.4f} "
                  f"Ep={r['best_epoch']}")
        for m, mn in zip(METRICS, METRIC_NAMES):
            vals = [r[m] for r in fold_results]
            print(f"    {mn:<18}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    if all_exp_results:
        summary_dir = os.path.join(Config.SAVE_ROOT, 'summary')
        plot_summary(all_exp_results, summary_dir)
        # Grad-CAM: NoSE vs WithSE 对比可视化
        run_gradcam_comparison(all_exp_results, test_data, summary_dir, n_samples=8)

    # ── 最终报告 ────────────────────────────────────────────────────────────
    sep  = "=" * 100
    sep2 = "-" * 100

    col_w = 14
    header = (f"{'Experiment':<26}"
              + "".join(f"{mn:>{col_w}}" for mn in METRIC_NAMES))

    lines = [
        sep,
        "对比实验 + 消融实验 -- 5-Fold CV (Fixed Test Set, seed=42)",
        sep,
        f"Time    : {datetime.now()}",
        f"Seed    : {Config.SEED}",
        f"Test set: fixed {Config.TEST_SIZE*100:.0f}% holdout ({len(test_data)} images)",
        f"TTA     : {Config.USE_TTA}",
        "",
        "【对比实验】固定: Img_SE + Concat + build_head_baseline",
        "  Backbone_ResNet50      : ResNet50(2048d)       + SE + MLP + Concat(2304d) + head",
        "  Backbone_ConvNeXt      : ConvNeXt-Tiny(768d)   + SE + MLP + Concat(1024d) + head",
        "  Backbone_EffB3         : EfficientNet-B3(1536d) + SE + MLP + Concat(1792d) + head",
        "  Backbone_MobileNetV3   : MobileNet-V3-L(960d)  + SE + MLP + Concat(1216d) + head",
        "  Backbone_DenseNet121   : DenseNet121(1024d)     + SE + MLP + Concat(1280d) + head",
        "",
        "【消融实验】固定backbone: EfficientNetV2-S(1280d)",
        "  Ablation_Unimodal  : SE(1280) -> head(1280d)            [去掉meta分支]",
        "  Ablation_NoSE      : Concat(1280+256=1536d) -> head     [去掉Img_SE]",
        "  Ablation_NoEMA     : SE -> Concat(1536d) -> head, EMA关闭",
        "  Ablation_WCE       : SE -> Concat(1536d) -> head, loss=WeightedCE(无smoothing)",
        "  参照(已有): Img_SE = SE -> Concat(1536d) -> head + EMA + WLS  Acc=0.8281",
        "",
        sep,
        "",
        "─── 对比实验 (by Accuracy) ─────────────────────────────────────────────────",
        header, sep2,
    ]

    backbone_exps = [(k, v) for k, v in all_exp_results.items()
                     if k.startswith('Backbone')]
    ablation_exps = [(k, v) for k, v in all_exp_results.items()
                     if k.startswith('Ablation')]

    def fmt_row(name, results):
        row = f"{name:<26}"
        for m in METRICS:
            vals = [r[m] for r in results]
            row += f"  {np.mean(vals):.3f}+/-{np.std(vals):.3f}"
        return row

    for name, results in sorted(backbone_exps,
                                 key=lambda x: np.mean([r['test_acc'] for r in x[1]]),
                                 reverse=True):
        lines.append(fmt_row(name, results))

    lines += [
        "",
        "  参照(Img_SE, EfficientNetV2-S, 已有结果):",
        "  Acc=0.8281 Sens=0.8714 Spec=0.7271 Bal=0.7993 "
        "MacF1≈0.82 MCC=0.5944 AUC=0.8712",
        "",
        sep2,
        "",
        "─── 消融实验 (by Accuracy) ─────────────────────────────────────────────────",
        header, sep2,
    ]

    for name, results in sorted(ablation_exps,
                                 key=lambda x: np.mean([r['test_acc'] for r in x[1]]),
                                 reverse=True):
        lines.append(fmt_row(name, results))

    lines += [
        "",
        "  参照(Full: Img_SE+EMA, 已有结果):",
        "  Acc=0.8281 Sens=0.8714 Spec=0.7271 Bal=0.7993 "
        "MacF1≈0.82 MCC=0.5944 AUC=0.8712",
        "",
        sep2,
        "",
        "─── Per-fold detail ─────────────────────────────────────────────────────────",
    ]

    fold_header = (f"  {'Fold':>4} {'Acc':>7} {'Sens':>7} {'Spec':>7} "
                   f"{'Bal':>7} {'MacF1':>7} {'MacPrc':>7} "
                   f"{'MCC':>7} {'AUC':>7} {'Ep':>5} {'Thr':>5}")

    for group_label, group in [("对比实验", backbone_exps),
                                ("消融实验", ablation_exps)]:
        lines.append(f"\n  == {group_label} ==")
        for name, results in group:
            lines.append(f"\n  {name}:")
            lines.append(fold_header)
            for r in results:
                lines.append(
                    f"  {r['fold']:>4} {r['test_acc']:>7.4f} "
                    f"{r['test_sens']:>7.4f} {r['test_spec']:>7.4f} "
                    f"{r['test_bal']:>7.4f} {r['macro_f1']:>7.4f} "
                    f"{r['macro_prec']:>7.4f} {r['test_mcc']:>7.4f} "
                    f"{r['test_auc']:>7.4f} {r['best_epoch']:>5d} "
                    f"{r['threshold']:>5.2f}")
            # 均值行
            lines.append(f"  {'mean':>4} " + " ".join(
                f"{np.mean([r[m] for r in results]):>7.4f}"
                for m in METRICS))
            lines.append(f"  {'std':>4}  " + " ".join(
                f"{np.std([r[m] for r in results]):>7.4f}"
                for m in METRICS))

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