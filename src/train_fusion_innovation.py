"""
融合创新实验脚本 -- 5折交叉验证
================================
在 Img_SE + Concat（当前最优基线）基础上，系统对比以下融合创新方向。
所有实验共享相同的 backbone / 训练超参 / 数据划分，结果可直接横向比较。

实验清单：
  ── 基线参照 ──────────────────────────────────────────────────────────────
  [A] ImgSE_Concat         当前最优基线: SE(img) + Concat → baseline head

  ── 融合门控改造 ──────────────────────────────────────────────────────────
  [B] MetaCondSE_Concat    meta联合条件SE: gate由img+meta共同决定，其余不变
                           gate = sigmoid(FC(img) + FC(meta))

  ── 辅助任务学习 ──────────────────────────────────────────────────────────
  [C] ImgSE_AuxTask        SE(img) + Concat + 辅助头预测吸烟/槟榔/饮酒
                           训练loss = main_loss + 0.3*(aux_smoking+aux_betel+aux_alcohol)
                           推理时辅助头丢弃，结构与[A]完全相同

  [D] MetaCondSE_AuxTask   [B]+[C]的组合: MetaCondSE + 辅助任务

  ── 双向跨模态注意力 ──────────────────────────────────────────────────────
  [E] ImgSE_BidirCrossAttn SE(img) + 双向CrossAttn融合（代替Concat）
                           img(Q)←meta(K,V) + meta(Q)←img(K,V) → FFN → 512d

架构对比总结：
  ┌──────────────────────┬──────────────┬────────────────────┬──────────┐
  │ 实验                 │ SE门控来源   │ 融合方式           │ 训练目标 │
  ├──────────────────────┼──────────────┼────────────────────┼──────────┤
  │ ImgSE_Concat [A]     │ 图像自身     │ Concat→1536d       │ 单任务   │
  │ MetaCondSE   [B]     │ 图像+临床    │ Concat→1536d       │ 单任务   │
  │ ImgSE_AuxTask[C]     │ 图像自身     │ Concat→1536d       │ 多任务   │
  │ MetaCondSE+Aux[D]    │ 图像+临床    │ Concat→1536d       │ 多任务   │
  │ BidirCrossAttn[E]    │ 图像自身     │ CrossAttn→512d     │ 单任务   │
  └──────────────────────┴──────────────┴────────────────────┴──────────┘
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
import os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, matthews_corrcoef, roc_auc_score)
# train_test_split, StratifiedKFold — no longer needed (imported via dataset.py)
import seaborn as sns
import shutil
from datetime import datetime
from collections import deque

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

from models import SEBlock, MetaMLP, build_head, get_efficientnet_v2s
from dataset import load_all_data, prepare_splits
from train_utils import (set_seed, ModelEMA, WeightedLabelSmoothingLoss,
                         find_threshold)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{datetime.now().strftime('%H:%M:%S')}] Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# 1. 实验注册表
# ==============================================================================
EXPERIMENTS = [
    # (exp_name,              arch_type)
    ('ImgSE_Concat',          'imgse_concat'),        # [A] 当前最优基线
    ('MetaCondSE_Concat',     'metacondse_concat'),   # [B] meta联合条件SE
    ('ImgSE_AuxTask',         'imgse_auxtask'),        # [C] SE + 辅助任务
    ('MetaCondSE_AuxTask',    'metacondse_auxtask'),   # [D] MetaCondSE + 辅助任务
    ('ImgSE_BidirCrossAttn',  'imgse_crossattn'),      # [E] SE + 双向CrossAttn
]

CNN_DIM  = 1280
META_DIM = 5
META_OUT = 256


# ==============================================================================
# 2. Config
# ==============================================================================
class Config:
    BASE_DIR    = "/home/wgf_v100/srtp/口腔癌分类识别项目"
    CROP_DIR    = os.path.join(BASE_DIR, "Segmented_Images/Segmented_Images")
    CSV_PATH    = os.path.join(BASE_DIR, "data/Imagewise_Data.csv")
    PATIENT_CSV = os.path.join(BASE_DIR, "data/Patientwise_Data.csv")

    SEED      = 42
    N_FOLDS   = 5
    TEST_SIZE = 0.15
    SAVE_ROOT = os.path.join(BASE_DIR, "results_fusion_innovation_cv5_seed42")

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

    # CrossAttn 超参
    ATTN_PROJ_DIM = 512
    ATTN_HEADS    = 8
    ATTN_DROPOUT  = 0.1

    # 辅助任务权重
    AUX_WEIGHT = 0.3


# set_seed, ModelEMA, WeightedLabelSmoothingLoss — imported from train_utils


# ==============================================================================
# 5. Dataset
#    注意：辅助任务需要从 meta 中额外提取吸烟/槟榔/饮酒标签
# ==============================================================================
class MultiModalDataset(Dataset):
    def __init__(self, data_items, transform=None):
        self.data      = data_items
        self.transform = transform

    def __len__(self): return len(self.data)

    def _encode_meta(self, info):
        """返回 meta_feat(5d) 以及辅助任务标签(smoking, betel, alcohol)"""
        feats = []
        try:
            feats.append(min(100, max(0, float(info.get('Age', 50)))) / 100.0)
        except Exception:
            feats.append(0.5)
        feats.append(1.0 if str(info.get('Gender')).upper().startswith('M') else 0.0)

        smoking = 1.0 if str(info.get('Smoking')).upper()           in ['Y','YES','1'] else 0.0
        betel   = 1.0 if str(info.get('Chewing_Betel_Quid')).upper() in ['Y','YES','1'] else 0.0
        alcohol = 1.0 if str(info.get('Alcohol')).upper()            in ['Y','YES','1'] else 0.0
        feats += [smoking, betel, alcohol]

        return (np.array(feats, dtype=np.float32),
                np.array([smoking, betel, alcohol], dtype=np.float32))

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            img = Image.open(item['path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE))
        if self.transform:
            img = self.transform(img)

        meta_feat, aux_labels = self._encode_meta(item['info'])
        meta      = torch.tensor(meta_feat,  dtype=torch.float32)
        aux       = torch.tensor(aux_labels, dtype=torch.float32)  # (3,) 辅助标签
        label     = torch.tensor(item['label'], dtype=torch.long)
        return img, meta, aux, label


# ==============================================================================
# 6. 数据加载与固定划分（参数与所有前序脚本完全一致）
# ==============================================================================
# load_all_data, prepare_splits — imported from dataset


# ==============================================================================
# 7. 网络架构组件
# ==============================================================================

# get_efficientnet_v2s, MetaMLP (was MetaMLP), SEBlock — imported from models


# ── [B][D] Meta-Conditioned SE ────────────────────────────────────────────────
class MetaConditionedSE(nn.Module):
    """
    联合条件SE：gate 由图像 + 临床元数据共同决定。

    动机：标准SE的gate只看图像内容（content-driven），
         而MetaCondSE让临床先验（吸烟/槟榔/饮酒）参与决定
         激活哪些视觉通道（context-driven）。

    公式：
      h    = ReLU( FC_img(img_feat) + FC_meta(meta_feat) )
      gate = Sigmoid( FC_out(h) )
      out  = img_feat * gate   ← 维度不变，仍 1280d

    与标准SE的唯一区别：FC_img的中间激活额外加了FC_meta(meta_feat)。
    参数增量极小（仅一个 256→mid 的线性层）。
    """
    def __init__(self, img_dim=1280, meta_dim=256, reduction=16):
        super().__init__()
        mid = max(img_dim // reduction, 8)
        self.img_fc  = nn.Linear(img_dim,  mid)
        self.meta_fc = nn.Linear(meta_dim, mid)   # ← 唯一新增参数
        self.out_fc  = nn.Linear(mid, img_dim)

    def forward(self, img_feat, meta_feat):
        h    = F.relu(self.img_fc(img_feat) + self.meta_fc(meta_feat))
        gate = torch.sigmoid(self.out_fc(h))
        return img_feat * gate


# ── [C][D] 辅助任务头 ─────────────────────────────────────────────────────────
class AuxiliaryHeads(nn.Module):
    """
    从图像特征预测临床风险因素（吸烟/槟榔/饮酒），共享特征提取。

    动机：吸烟/嚼槟榔在口腔黏膜上有可见的形态学改变
         （色素沉积、角化异常）。让backbone预测这些标签，
         相当于赋予视觉特征有医学意义的额外监督信号。
         推理时辅助头丢弃，不影响模型结构和速度。

    训练loss：
      loss_total = loss_main + λ*(loss_smoking + loss_betel + loss_alcohol)
      λ = Config.AUX_WEIGHT（默认0.3）
    """
    def __init__(self, img_dim=1280):
        super().__init__()
        mid = 256
        self.shared = nn.Sequential(
            nn.Linear(img_dim, mid),
            nn.BatchNorm1d(mid),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3)
        )
        # 三个二分类头（BCE）
        self.head_smoking = nn.Linear(mid, 1)
        self.head_betel   = nn.Linear(mid, 1)
        self.head_alcohol = nn.Linear(mid, 1)

    def forward(self, img_feat):
        h = self.shared(img_feat)
        return (torch.sigmoid(self.head_smoking(h)).squeeze(1),
                torch.sigmoid(self.head_betel(h)).squeeze(1),
                torch.sigmoid(self.head_alcohol(h)).squeeze(1))


# ── [E] 双向跨模态注意力融合 ─────────────────────────────────────────────────
class BidirectionalCrossModalAttention(nn.Module):
    """
    双向跨模态注意力融合。

    流程：
      1. img(1280d) → proj(512d)，meta(256d) → proj(512d)
      2. 方向①: img(Q) ← meta(K,V)  → img关注与临床相关的视觉维度
         方向②: meta(Q) ← img(K,V)  → meta关注与图像形态匹配的临床语义
      3. 残差 + LayerNorm
      4. concat(512+512=1024d) → FFN → 512d
    """
    def __init__(self, img_dim=1280, meta_dim=256,
                 proj_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.img_proj  = nn.Sequential(nn.Linear(img_dim,  proj_dim),
                                       nn.LayerNorm(proj_dim))
        self.meta_proj = nn.Sequential(nn.Linear(meta_dim, proj_dim),
                                       nn.LayerNorm(proj_dim))
        self.attn_i2m  = nn.MultiheadAttention(proj_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.attn_m2i  = nn.MultiheadAttention(proj_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.ln_img    = nn.LayerNorm(proj_dim)
        self.ln_meta   = nn.LayerNorm(proj_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.out_dim = proj_dim  # 512

    def forward(self, img_feat, meta_feat):
        ip = self.img_proj(img_feat).unsqueeze(1)    # (B,1,512)
        mp = self.meta_proj(meta_feat).unsqueeze(1)  # (B,1,512)
        ia, _ = self.attn_i2m(ip, mp, mp)            # img attend meta
        ma, _ = self.attn_m2i(mp, ip, ip)            # meta attend img
        io = self.ln_img(ip.squeeze(1)  + self.attn_drop(ia.squeeze(1)))
        mo = self.ln_meta(mp.squeeze(1) + self.attn_drop(ma.squeeze(1)))
        return self.ffn(torch.cat([io, mo], dim=1))  # (B, 512)


# build_head (was build_head) — imported from models

def build_head_crossattn(in_dim=512, num_classes=2):
    """CrossAttn输出512d，头结构对应缩小"""
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),  nn.Dropout(0.4),
        nn.Linear(256, 128),    nn.BatchNorm1d(128),
        nn.SiLU(inplace=True),  nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )


# ==============================================================================
# 8. 统一模型容器
# ==============================================================================
class FusionModel(nn.Module):
    """
    根据 arch_type 组装不同的融合创新模型。
    forward 统一返回：
      - 训练时（aux_task实验）: (main_logits, aux_smoking, aux_betel, aux_alcohol)
      - 其他情况            : (main_logits,)
    调用方统一用 out[0] 取主分类logits。
    """
    def __init__(self, arch_type, meta_dim=META_DIM, num_classes=2):
        super().__init__()
        self.arch_type  = arch_type
        self.use_aux    = 'auxtask' in arch_type
        self.use_crossattn = 'crossattn' in arch_type

        # ── backbone ─────────────────────────────────────────────────────────
        self.backbone, img_out = get_efficientnet_v2s()    # 1280d

        # ── SE / MetaCondSE ───────────────────────────────────────────────────
        if 'metacondse' in arch_type:
            self.img_se = MetaConditionedSE(img_out, META_OUT, reduction=16)
        else:
            self.img_se = SEBlock(img_out, reduction=16)

        # ── meta branch ───────────────────────────────────────────────────────
        self.meta_branch = MetaMLP(meta_dim, out_dim=META_OUT)

        # ── 辅助任务头 ────────────────────────────────────────────────────────
        if self.use_aux:
            self.aux_heads = AuxiliaryHeads(img_out)

        # ── 融合 & 分类头 ─────────────────────────────────────────────────────
        if self.use_crossattn:
            self.fusion     = BidirectionalCrossModalAttention(
                img_dim   = img_out,
                meta_dim  = META_OUT,
                proj_dim  = Config.ATTN_PROJ_DIM,
                num_heads = Config.ATTN_HEADS,
                dropout   = Config.ATTN_DROPOUT)
            fusion_out      = self.fusion.out_dim            # 512
            self.classifier = build_head_crossattn(fusion_out, num_classes)

        else:
            # Concat → baseline head
            concat_dim      = img_out + META_OUT             # 1536
            self.classifier = build_head(concat_dim, num_classes)

    def forward(self, img, meta):
        img_feat = self.backbone(img)                        # (B, 1280)

        # SE / MetaCondSE
        if isinstance(self.img_se, MetaConditionedSE):
            # MetaCondSE 需要先跑 meta_branch，让gate利用meta信息
            meta_feat = self.meta_branch(meta)               # (B, 256)
            img_feat  = self.img_se(img_feat, meta_feat)     # (B, 1280)
        else:
            img_feat  = self.img_se(img_feat)                # (B, 1280)
            meta_feat = self.meta_branch(meta)               # (B, 256)

        # 辅助任务预测（训练时）
        aux_out = None
        if self.use_aux:
            aux_out = self.aux_heads(img_feat)               # (smoking, betel, alcohol)

        # 融合
        if self.use_crossattn:
            fused = self.fusion(img_feat, meta_feat)         # (B, 512)
            logits = self.classifier(fused)
        else:
            fused  = torch.cat([img_feat, meta_feat], dim=1) # (B, 1536)
            logits = self.classifier(fused)

        if aux_out is not None:
            return (logits,) + aux_out    # (logits, sm, bt, al)
        return (logits,)


# ==============================================================================
# 9. 阈值搜索 & 评估
# ==============================================================================
# find_threshold — imported from train_utils


def evaluate(model, loader, criterion, threshold=0.5, tta=False):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    with torch.no_grad():
        for img, meta, aux, lbl in loader:
            img, meta, lbl = img.to(DEVICE), meta.to(DEVICE), lbl.to(DEVICE)
            if tta:
                o1 = model(img, meta)[0]
                o2 = model(torch.flip(img, [3]), meta)[0]
                o3 = model(torch.flip(img, [2]), meta)[0]
                out = (o1 + o2 + o3) / 3.0
            else:
                out = model(img, meta)[0]
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
def train_one_fold(fold_idx, exp_name, arch_type,
                   train_data, val_data, test_data, save_dir):

    set_seed(Config.SEED)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    use_aux = 'auxtask' in arch_type

    print(f"\n  {'─'*65}")
    print(f"  Fold {fold_idx+1}/{Config.N_FOLDS} | {exp_name} | arch={arch_type}")
    print(f"  Train={len(train_data)}  Val={len(val_data)}  "
          f"Test={len(test_data)}(fixed)  AuxTask={use_aux}")
    print(f"  {'─'*65}")

    train_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.15, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
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

    model = FusionModel(arch_type, meta_dim=META_DIM).to(DEVICE)
    total   = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: total={total:,}  trainable={n_train:,}")

    ema       = ModelEMA(model, decay=Config.EMA_DECAY)
    w_cls     = torch.tensor([2.0, 1.0]).to(DEVICE)
    criterion = WeightedLabelSmoothingLoss(weight=w_cls, smoothing=0.05)
    aux_bce   = nn.BCELoss()                                 # 辅助任务用BCE

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

        for i, (img, meta, aux_lbl, lbl) in enumerate(
                tqdm(train_dl,
                     desc=f"[{exp_name}|F{fold_idx+1}] Ep{epoch+1}",
                     leave=False)):
            img, meta  = img.to(DEVICE), meta.to(DEVICE)
            lbl        = lbl.to(DEVICE)
            aux_lbl    = aux_lbl.to(DEVICE)                  # (B, 3)

            out = model(img, meta)
            main_logits = out[0]
            loss = criterion(main_logits, lbl)

            # 辅助任务 loss（仅 auxtask 实验）
            if use_aux and len(out) == 4:
                sm_pred, bt_pred, al_pred = out[1], out[2], out[3]
                loss_aux = (aux_bce(sm_pred, aux_lbl[:, 0]) +
                            aux_bce(bt_pred, aux_lbl[:, 1]) +
                            aux_bce(al_pred, aux_lbl[:, 2]))
                loss = loss + Config.AUX_WEIGHT * loss_aux

            (loss / Config.GRAD_ACCUMULATION).backward()
            t_loss += loss.item()

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

        th          = find_threshold(val_lbls, val_probs, Config.THRESH_RANGE)
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

    return {'fold': fold_idx+1, 'exp': exp_name,
            'test_acc': test_acc, 'test_sens': test_sens, 'test_spec': test_spec,
            'test_bal': test_bal, 'test_mcc': test_mcc,  'test_auc': test_auc,
            'threshold': th_final, 'best_epoch': best_epoch+1,
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
    n         = len(exp_names)
    colors    = plt.cm.tab10(np.linspace(0, 0.9, n))

    summary = {exp: {m: (np.mean([r[m] for r in results]),
                          np.std([r[m]  for r in results]))
                     for m in metrics}
               for exp, results in all_exp_results.items()}

    # 横向对比柱状图
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, metric, mname in zip(axes.flatten(), metrics, metric_names):
        means = [summary[e][metric][0] for e in exp_names]
        stds  = [summary[e][metric][1] for e in exp_names]
        order    = np.argsort(means)
        s_names  = [exp_names[i] for i in order]
        s_means  = [means[i]     for i in order]
        s_stds   = [stds[i]      for i in order]
        s_colors = [colors[i]    for i in order]

        ax.barh(s_names, s_means, xerr=s_stds,
                color=s_colors, alpha=0.82, capsize=3, height=0.65)
        for i, (m, s) in enumerate(zip(s_means, s_stds)):
            ax.text(m + s + 0.002, i, f'{m:.4f}', va='center', fontsize=8)
        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.10)
        ax.grid(True, alpha=0.3, axis='x')
        best_i = int(np.argmax(s_means))
        ax.get_yticklabels()[best_i].set_color('red')
        ax.get_yticklabels()[best_i].set_fontweight('bold')

    plt.suptitle(
        'Fusion Innovation Experiments -- EfficientNetV2-S\n'
        '5-Fold CV | Fixed Test Set | seed=42 | mean +/- std | Red = best',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fusion_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ROC 叠加图
    fig, ax = plt.subplots(figsize=(9, 8))
    mean_fpr = np.linspace(0, 1, 200)
    for exp, color in zip(exp_names, colors):
        tprs = []
        for r in all_exp_results[exp]:
            fpr, tpr, _ = roc_curve(r['lbls'], r['probs'][:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean([r['test_auc'] for r in all_exp_results[exp]])
        std_auc  = np.std([r['test_auc']  for r in all_exp_results[exp]])
        ax.plot(mean_fpr, mean_tpr, linewidth=1.8, color=color,
                label=f'{exp}  AUC={mean_auc:.4f}±{std_auc:.4f}')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Mean ROC -- Fusion Innovation Experiments',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fusion_roc.png'), dpi=150)
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

    patient_data = load_all_data(
        Config.CROP_DIR, Config.CSV_PATH, Config.PATIENT_CSV, Config.CLASS_MAP)
    print(f"\n>>> Preparing fixed test set + {Config.N_FOLDS}-fold CV...")
    test_data, folds = prepare_splits(
        patient_data, Config.TEST_SIZE, Config.N_FOLDS, Config.SEED)

    metrics      = ['test_acc', 'test_sens', 'test_spec',
                    'test_bal', 'test_mcc',  'test_auc']
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity',
                    'Balanced Acc', 'MCC', 'AUC']

    print(f"\n{'='*70}")
    print("FUSION INNOVATION EXPERIMENTS")
    print(f"  [A] ImgSE_Concat       : SE(img自身) + Concat  ← 当前最优基线")
    print(f"  [B] MetaCondSE_Concat  : SE(img+meta联合) + Concat")
    print(f"  [C] ImgSE_AuxTask      : SE + Concat + 辅助任务(吸烟/槟榔/饮酒)")
    print(f"  [D] MetaCondSE_AuxTask : [B]+[C]组合")
    print(f"  [E] ImgSE_BidirCrossAttn: SE + 双向跨模态注意力")
    print(f"{'='*70}")

    all_exp_results = {}

    for exp_name, arch_type in EXPERIMENTS:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {exp_name}  arch={arch_type}")
        print(f"{'='*70}")

        fold_results = []
        for fold_idx, (train_data, val_data) in enumerate(folds):
            fold_dir = os.path.join(Config.SAVE_ROOT, exp_name, f"fold_{fold_idx+1}")
            try:
                result = train_one_fold(
                    fold_idx, exp_name, arch_type,
                    train_data, val_data, test_data, fold_dir)
                fold_results.append(result)
            except Exception as e:
                print(f"  [ERROR] Fold {fold_idx+1} | {exp_name}: {e}")
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
        for m in metrics:
            vals = [r[m] for r in fold_results]
            print(f"    {m:<12}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    if all_exp_results:
        plot_summary(all_exp_results,
                     os.path.join(Config.SAVE_ROOT, "summary"))

    # ── 最终汇总报告 ─────────────────────────────────────────────────────────
    sep = "=" * 90
    lines = [
        sep,
        "FUSION INNOVATION EXPERIMENTS -- 5-Fold CV Report",
        sep,
        f"Time      : {datetime.now()}",
        f"Seed      : {Config.SEED}",
        f"Backbone  : EfficientNetV2-S (fixed, 1280d)",
        f"Test set  : fixed {Config.TEST_SIZE*100:.0f}% holdout ({len(test_data)} images)",
        f"TTA       : {Config.USE_TTA}",
        f"AuxWeight : {Config.AUX_WEIGHT}",
        "",
        "实验说明:",
        "  [A] ImgSE_Concat      : SEBlock(img自身) + Concat(1536d) → baseline head",
        "  [B] MetaCondSE_Concat : gate=sigmoid(FC_img(img)+FC_meta(meta)) + Concat",
        "  [C] ImgSE_AuxTask     : [A] + AuxHeads预测吸烟/槟榔/饮酒 (λ=0.3)",
        "  [D] MetaCondSE_AuxTask: [B] + [C]组合",
        "  [E] ImgSE_BidirCrossAttn: [A]的SE + 双向CrossAttn融合(512d) → 小头",
        "",
        "消融链 (逐步叠加):",
        "  [A]基线 → [B]改SE门控 → [C]加辅助任务 → [D]两者组合",
        "  [A]基线 → [E]换融合方式",
        "",
        sep,
        f"{'Experiment':<24} {'Acc':>16} {'Sens':>16} {'Spec':>16} "
        f"{'BalAcc':>16} {'MCC':>16} {'AUC':>16}",
        "-" * 90,
    ]

    sorted_exps = sorted(
        all_exp_results.items(),
        key=lambda x: np.mean([r['test_auc'] for r in x[1]]),
        reverse=True)

    for exp_name, fold_results in sorted_exps:
        row = f"{exp_name:<24}"
        for m in metrics:
            vals = [r[m] for r in fold_results]
            row += f"  {np.mean(vals):.4f}±{np.std(vals):.4f}"
        lines.append(row)

    lines.append(sep)
    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(Config.SAVE_ROOT, "fusion_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")
    return all_exp_results


if __name__ == "__main__":
    main()