import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image, ImageFile
import numpy as np
import cv2
import json
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split  # 新增：用于分层划分测试集

# ================== 1. 配置区域（新增/修改） ==================
BASE_DIR        = "/home/wgf_v100/srtp/口腔癌分类识别项目"
RAW_IMAGE_DIR   = os.path.join(BASE_DIR, "Images")
COCO_JSON       = os.path.join(BASE_DIR, "data/Annotation.json")

SAVE_DIR        = os.path.join(BASE_DIR, "口腔分割结果")
MODEL_SAVE_PATH = str(Path(SAVE_DIR) / "best_oral_unet.pth")

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
VIS_N = 30       # 训练结束后生成的可视化样本数量
TEST_RATIO = 0.1  # 新增：测试集比例（总数据的10%）
VAL_RATIO = 0.1   # 新增：验证集比例（训练集的10%，总数据的9%）
SEED = 42         # 新增：固定随机种子

os.environ['TORCH_HOME'] = os.path.join(BASE_DIR, "hub")
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 新增：固定所有随机种子（保证可复现）
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(SEED)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
ensure_dir(SAVE_DIR)

# ================== 工具函数（无修改） ==================
def imread_unicode(path):
    """cv2.imread 不支持中文路径，用 PIL 替代"""
    try:
        pil_img = Image.open(str(path)).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def compute_dice(pred, gt):
    pred = pred.flatten().astype(np.float32)
    gt   = gt.flatten().astype(np.float32)
    inter = (pred * gt).sum()
    return float((2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6))

# ================== 2. 数据集定义（新增数据增强） ==================
class OralSegmentationDataset(Dataset):
    def __init__(self, img_dir, json_path, transform=True, is_train=False):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_train = is_train  # 新增：标记是否为训练集（用于数据增强）

        print(f"正在解析 JSON: {json_path} ...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.id2name = {img['id']: img['file_name'] for img in data['images']}
        self.samples = {}

        for ann in data['annotations']:
            if ann['category_id'] == 2:
                img_id = ann['image_id']
                if img_id not in self.id2name:
                    continue
                name = Path(self.id2name[img_id]).stem
                seg = ann['segmentation']
                flat = seg[0] if isinstance(seg[0], list) else seg
                poly = np.array(flat).reshape(-1, 2).astype(np.int32)
                if name not in self.samples:
                    self.samples[name] = []
                self.samples[name].append(poly)

        # 建立文件名小写索引（兼容大小写不一致）
        self.file_index = {}
        for f in self.img_dir.rglob('*'):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                self.file_index[f.stem.lower()] = f

        self.image_names = list(self.samples.keys())
        print(f"解析完成: 找到 {len(self.image_names)} 张包含口腔标注的图片")

    def __len__(self):
        return len(self.image_names)

    def _find_image(self, name):
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            p = self.img_dir / f"{name}{ext}"
            if p.exists():
                return p
        return self.file_index.get(name.lower(), None)
    
    # 新增：数据增强（仅训练集使用）
    def _augment(self, img, mask):
        # 随机水平翻转
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        # 随机亮度调整
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            img = np.clip(img * alpha, 0, 255).astype(np.uint8)
        # 随机旋转（0/90/180/270）
        if random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            h, w = img.shape[:2]
            mat = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, mat, (w, h))
            mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST)
        return img, mask

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = self._find_image(name)

        if img_path is None:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)

        bgr = imread_unicode(img_path)
        if bgr is None:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)

        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, self.samples[name], 1)

        # 新增：训练集数据增强
        if self.is_train:
            img, mask = self._augment(img, mask)

        if self.transform:
            img  = cv2.resize(img,  (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            img[0] = (img[0] - 0.485) / 0.229
            img[1] = (img[1] - 0.456) / 0.224
            img[2] = (img[2] - 0.406) / 0.225

            return torch.from_numpy(img).float(), torch.from_numpy(mask).float().unsqueeze(0)

# ================== 3. 模型定义（无修改） ==================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        try:
            resnet = models.resnet18(weights='DEFAULT')
        except Exception:
            resnet = models.resnet18(pretrained=True)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        self.up4  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(512 + 256, 256)
        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(64 + 64, 64)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        d4 = self.dec4(torch.cat([self.up4(x5), x4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), x3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], 1))
        out = nn.functional.interpolate(self.final(d1), scale_factor=2,
                                        mode='bilinear', align_corners=True)
        return out

# ================== 4. 损失函数（无修改） ==================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        return bce + dice_loss

# ================== 5. 训练曲线保存（修改：新增训练/验证Dice、验证Loss） ==================
def save_training_curves(train_losses, val_losses, train_dices, val_dices, save_dir):
    epochs = list(range(1, len(train_losses) + 1))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    # 1. 训练损失曲线
    ax1.plot(epochs, train_losses, color='#E74C3C', lw=2.5,
             marker='o', markersize=4, markerfacecolor='white',
             markeredgewidth=1.5, label='Train Loss')
    best_loss_idx = np.argmin(train_losses)
    ax1.scatter(epochs[best_loss_idx], train_losses[best_loss_idx],
                color='#E74C3C', s=80, zorder=5)
    ax1.annotate(f'Min={train_losses[best_loss_idx]:.4f}',
                 xy=(epochs[best_loss_idx], train_losses[best_loss_idx]),
                 xytext=(epochs[best_loss_idx] + 0.5, train_losses[best_loss_idx] + 0.01),
                 fontsize=9, color='#E74C3C')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. 验证损失曲线
    ax2.plot(epochs, val_losses, color='#3498DB', lw=2.5,
             marker='o', markersize=4, markerfacecolor='white',
             markeredgewidth=1.5, label='Val Loss')
    best_val_loss_idx = np.argmin(val_losses)
    ax2.scatter(epochs[best_val_loss_idx], val_losses[best_val_loss_idx],
                color='#3498DB', s=80, zorder=5)
    ax2.annotate(f'Min={val_losses[best_val_loss_idx]:.4f}',
                 xy=(epochs[best_val_loss_idx], val_losses[best_val_loss_idx]),
                 xytext=(epochs[best_val_loss_idx] + 0.5, val_losses[best_val_loss_idx] + 0.01),
                 fontsize=9, color='#3498DB')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Validation Loss Curve', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. 训练Dice曲线
    ax3.plot(epochs, train_dices, color='#F39C12', lw=2.5,
             marker='s', markersize=4, markerfacecolor='white',
             markeredgewidth=1.5, label='Train Dice')
    best_train_dice_idx = np.argmax(train_dices)
    ax3.scatter(epochs[best_train_dice_idx], train_dices[best_train_dice_idx],
                color='#F39C12', s=80, zorder=5)
    ax3.annotate(f'Max={train_dices[best_train_dice_idx]:.4f}',
                 xy=(epochs[best_train_dice_idx], train_dices[best_train_dice_idx]),
                 xytext=(epochs[best_train_dice_idx] + 0.5, train_dices[best_train_dice_idx] - 0.03),
                 fontsize=9, color='#F39C12')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Dice Score', fontsize=11)
    ax3.set_title('Training Dice Score Curve', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # 4. 验证Dice曲线
    ax4.plot(epochs, val_dices, color='#27AE60', lw=2.5,
             marker='s', markersize=4, markerfacecolor='white',
             markeredgewidth=1.5, label='Val Dice')
    best_val_dice_idx = np.argmax(val_dices)
    ax4.scatter(epochs[best_val_dice_idx], val_dices[best_val_dice_idx],
                color='#27AE60', s=80, zorder=5)
    ax4.annotate(f'Best={val_dices[best_val_dice_idx]:.4f}',
                 xy=(epochs[best_val_dice_idx], val_dices[best_val_dice_idx]),
                 xytext=(epochs[best_val_dice_idx] + 0.5, val_dices[best_val_dice_idx] - 0.03),
                 fontsize=9, color='#27AE60')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Dice Score', fontsize=11)
    ax4.set_title('Validation Dice Score Curve', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.suptitle('Training & Validation Metrics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = Path(save_dir) / 'training_validation_metrics.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  训练/验证指标曲线已保存: {out}")

# ================== 6. 可视化函数（新增测试集可视化） ==================
@torch.no_grad()
def predict_single(model, img_rgb):
    img_r = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    img_f = img_r.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_f[0] = (img_f[0] - 0.485) / 0.229
    img_f[1] = (img_f[1] - 0.456) / 0.224
    img_f[2] = (img_f[2] - 0.406) / 0.225
    t = torch.from_numpy(img_f).unsqueeze(0).to(DEVICE)
    out = model(t)
    prob = torch.sigmoid(out).squeeze().cpu().numpy()
    pred = (prob > 0.5).astype(np.uint8)
    return pred

def save_visualizations(model, dataset, save_dir, n=VIS_N, prefix="train_val"):
    print(f"\n生成{prefix}集可视化图（共 {n} 张）...")
    vis_dir = Path(save_dir) / f'{prefix}_visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    all_names = dataset.image_names.copy()
    random.seed(SEED)
    random.shuffle(all_names)
    candidates = all_names[:min(150, len(all_names))]

    results = []
    for name in candidates:
        img_path = dataset._find_image(name)
        if img_path is None:
            continue
        bgr = imread_unicode(img_path)
        if bgr is None:
            continue
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # 生成GT mask（原始尺寸）
        gt_full = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_full, dataset.samples[name], 1)

        # 推理
        pred = predict_single(model, img_rgb)

        # 把GT resize到预测尺寸做Dice
        gt_256 = cv2.resize(gt_full, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        dice = compute_dice(pred, gt_256)

        results.append({
            'img_rgb': img_rgb,
            'gt_full': gt_full,
            'pred':    pred,
            'gt_256':  gt_256,
            'dice':    dice,
        })

    # 按 Dice 从高到低排，取前 n 张
    results.sort(key=lambda x: x['dice'], reverse=True)
    selected = results[:n]

    col_titles = [
        'Original Image (GT contour in red)',
        'Predicted Oral Region (real pixels, black bg)',
        'GT Mask (white=oral, black=bg)',
        'Predicted Mask (white=oral, black=bg)',
    ]

    for idx, r in enumerate(selected):
        img_256 = cv2.resize(r['img_rgb'], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        gt_256  = r['gt_256']
        pred    = r['pred']
        dice    = r['dice']

        # Col1: 原图 + 红色GT轮廓线
        col1 = img_256.copy()
        contours, _ = cv2.findContours(gt_256, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(col1, contours, -1, (220, 30, 30), 3)

        # Col2: 预测区域保留原始像素，背景置黑
        col2 = np.zeros_like(img_256)
        col2[pred == 1] = img_256[pred == 1]

        # Col3: GT掩码
        col3 = (gt_256 * 255).astype(np.uint8)

        # Col4: 预测掩码
        col4 = (pred * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))

        axes[0].imshow(col1)
        axes[1].imshow(col2)
        axes[2].imshow(col3, cmap='gray', vmin=0, vmax=255)
        axes[3].imshow(col4, cmap='gray', vmin=0, vmax=255)

        for ax, title in zip(axes, col_titles):
            ax.axis('off')
            ax.set_title(title, fontsize=8.5, color='#444444', pad=5)

        fig.suptitle(f'{prefix} set - Dice = {dice:.4f}',
                     fontsize=13, fontweight='bold', color='#C0392B', y=1.04)

        plt.tight_layout(w_pad=0.4)
        out = vis_dir / f'{prefix}_sample_{idx+1:02d}_dice{dice:.4f}.png'
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [{idx+1:02d}/{n}] Dice={dice:.4f}  →  {out.name}")

    print(f"{prefix}集可视化图全部保存至: {vis_dir}")

# ================== 7. 测试集评估函数（新增） ==================
def evaluate_test_set(model, test_dl):
    print("\n开始评估测试集...")
    model.eval()
    test_loss = 0.0
    test_dice = 0.0
    criterion = DiceBCELoss()

    with torch.no_grad():
        for img, mask in tqdm(test_dl, desc="Testing"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            out = model(img)
            # 计算测试损失
            loss = criterion(out, mask)
            test_loss += loss.item()
            # 计算测试Dice
            pred = (torch.sigmoid(out) > 0.5).float()
            inter = (pred * mask).sum()
            union = pred.sum() + mask.sum()
            test_dice += (2 * inter / (union + 1e-6)).item()

    avg_test_loss = test_loss / len(test_dl)
    avg_test_dice = test_dice / len(test_dl)

    # 保存测试集评估结果
    test_metrics = {
        'test_loss': avg_test_loss,
        'test_dice': avg_test_dice,
        'num_samples': len(test_dl.dataset)
    }
    with open(Path(SAVE_DIR) / 'test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)

    print(f"\n测试集评估结果：")
    print(f"  平均损失: {avg_test_loss:.4f}")
    print(f"  平均Dice系数: {avg_test_dice:.4f}")
    print(f"  测试集样本数: {len(test_dl.dataset)}")
    print(f"  评估结果已保存至: {Path(SAVE_DIR) / 'test_metrics.json'}")
    return avg_test_loss, avg_test_dice

# ================== 8. 训练主流程（核心修改） ==================
def train():
    # 1. 加载完整数据集
    full_ds = OralSegmentationDataset(RAW_IMAGE_DIR, COCO_JSON, transform=True)
    
    # 2. 划分训练+验证集 和 测试集（先分测试集，再分训练/验证）
    train_val_names, test_names = train_test_split(
        full_ds.image_names, 
        test_size=TEST_RATIO, 
        random_state=SEED
    )
    # 构建训练+验证集、测试集的子集
    train_val_ds = OralSegmentationDataset(RAW_IMAGE_DIR, COCO_JSON, transform=True)
    train_val_ds.image_names = train_val_names
    test_ds = OralSegmentationDataset(RAW_IMAGE_DIR, COCO_JSON, transform=True)
    test_ds.image_names = test_names

    # 3. 划分训练集和验证集（从训练+验证集中再分）
    train_size = int((1 - VAL_RATIO) * len(train_val_ds))
    val_size = len(train_val_ds) - train_size
    train_ds, val_ds = random_split(train_val_ds, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    # 标记训练集用于数据增强
    train_ds.dataset.is_train = True

    # 4. 创建DataLoader（修改num_workers提升速度）
    num_workers = 4 if torch.cuda.is_available() else 0  # 根据硬件调整
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"\n数据集划分完成：")
    print(f"  训练集: {len(train_ds)} 样本")
    print(f"  验证集: {len(val_ds)} 样本")
    print(f"  测试集: {len(test_ds)} 样本")

    # 5. 初始化模型、优化器、损失函数
    model     = ResNetUNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = DiceBCELoss()
    scaler    = torch.cuda.amp.GradScaler()

    best_val_dice   = 0.0
    # 新增：记录训练/验证的损失、Dice
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []

    print(f"\n开始训练 (Device: {DEVICE}, AMP Enabled)...")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        epoch_train_loss = 0.0
        epoch_train_dice = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for img, mask in pbar:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out  = model(img)
                loss = criterion(out, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 计算训练Dice
            pred = (torch.sigmoid(out) > 0.5).float()
            inter = (pred * mask).sum()
            union = pred.sum() + mask.sum()
            train_dice = (2 * inter / (union + 1e-6)).item()

            epoch_train_loss += loss.item()
            epoch_train_dice += train_dice
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{train_dice:.4f}"})

        avg_train_loss = epoch_train_loss / len(train_dl)
        avg_train_dice = epoch_train_dice / len(train_dl)
        train_losses.append(avg_train_loss)
        train_dices.append(avg_train_dice)

        # ── Validation ──
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_dice = 0.0
        with torch.no_grad():
            for img, mask in val_dl:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                out  = model(img)
                # 计算验证损失
                loss = criterion(out, mask)
                epoch_val_loss += loss.item()
                # 计算验证Dice
                pred = (torch.sigmoid(out) > 0.5).float()
                inter = (pred * mask).sum()
                union = pred.sum() + mask.sum()
                epoch_val_dice += (2 * inter / (union + 1e-6)).item()

        avg_val_loss = epoch_val_loss / len(val_dl)
        avg_val_dice = epoch_val_dice / len(val_dl)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)

        print(f"  Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f}")
        print(f"            | Val Loss:   {avg_val_loss:.4f} | Val Dice:   {avg_val_dice:.4f}")

        # 保存最佳模型（基于验证Dice）
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  模型已保存 (Best Val Dice: {best_val_dice:.4f})")

    # 训练结束后：保存完整指标曲线
    save_training_curves(train_losses, val_losses, train_dices, val_dices, SAVE_DIR)

    # 加载最佳模型
    print("\n加载最佳模型权重...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # 生成训练+验证集可视化
    save_visualizations(model, train_val_ds, SAVE_DIR, n=VIS_N, prefix="train_val")

    # 评估测试集
    test_loss, test_dice = evaluate_test_set(model, test_dl)

    # 生成测试集可视化
    save_visualizations(model, test_ds, SAVE_DIR, n=VIS_N, prefix="test")

    print(f"\n===== 最终结果汇总 =====")
    print(f"最佳验证Dice: {best_val_dice:.4f}")
    print(f"测试集平均Loss: {test_loss:.4f}")
    print(f"测试集平均Dice: {test_dice:.4f}")
    print(f"模型路径: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()