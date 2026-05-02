"""
ImgSE Baseline — 5-Fold Cross-Validation
=========================================
Core experiment: EfficientNetV2-S + SE channel attention + Concat fusion
with EMA and Weighted Label Smoothing.

Derived from ima_SE基线.txt, split into shared modules:
  models.py       — ImgSEModel, SEBlock, MetaMLP, build_head
  dataset.py      — MultiModalDataset, load_all_data, prepare_splits
  train_utils.py  — ModelEMA, WeightedLabelSmoothingLoss, evaluate, find_threshold
"""

import os, sys, shutil
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, matthews_corrcoef, roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import ImgSEModel
from dataset import MultiModalDataset, load_all_data, prepare_splits
from train_utils import (set_seed, ModelEMA, WeightedLabelSmoothingLoss,
                         find_threshold, evaluate)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings('ignore')


class Config:
    BASE_DIR    = "/home/wgf_v100/srtp/口腔癌分类识别项目"
    CROP_DIR    = os.path.join(BASE_DIR, "Segmented_Images/Segmented_Images")
    CSV_PATH    = os.path.join(BASE_DIR, "data/Imagewise_Data.csv")
    PATIENT_CSV = os.path.join(BASE_DIR, "data/Patientwise_Data.csv")

    SEED      = 42
    N_FOLDS   = 5
    TEST_SIZE = 0.15
    SAVE_ROOT = os.path.join(BASE_DIR, "results_img_se_cv5_seed42")

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

    EXP_NAME = 'Img_SE'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_fold(fold_idx, train_data, val_data, test_data, save_dir):
    set_seed(Config.SEED)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n  {'─'*60}")
    print(f"  Fold {fold_idx+1}/{Config.N_FOLDS} | {Config.EXP_NAME} | seed={Config.SEED}")
    print(f"  Train={len(train_data)}  Val={len(val_data)}  "
          f"Test={len(test_data)}(fixed)")
    print(f"  {'─'*60}")

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

    model = ImgSEModel(meta_dim=5).to(DEVICE)

    total   = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: total={total:,}  trainable={n_train:,}")
    print(f"  EfficientNetV2-S(1280d) -> SE(1280d) + MLP(256d) "
          f"-> Concat(1536d) -> FC(1536->1024->512->256->2)")

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
                tqdm(train_dl, desc=f"[{Config.EXP_NAME}|F{fold_idx+1}] Ep{epoch+1}",
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
                model, val_dl, criterion, DEVICE, threshold=0.5, tta=False)
            ema_tag = "raw"
        else:
            with ema.get_context(model):
                _, val_loss, _, val_lbls, val_probs = evaluate(
                    model, val_dl, criterion, DEVICE, threshold=0.5, tta=False)
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
        model, test_dl, criterion, DEVICE, threshold=th_final, tta=Config.USE_TTA)

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
        f'EfficientNetV2-S + SE(img) + Concat + baseline head | seed=42',
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


def main():
    set_seed(Config.SEED)

    if os.path.exists(Config.SAVE_ROOT):
        shutil.rmtree(Config.SAVE_ROOT)
    Path(Config.SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU: {torch.cuda.get_device_name(0)}")

    patient_data = load_all_data(
        Config.CROP_DIR, Config.CSV_PATH, Config.PATIENT_CSV, Config.CLASS_MAP)
    print(f"\n>>> Preparing fixed test set + {Config.N_FOLDS}-fold CV...")
    test_data, folds = prepare_splits(
        patient_data, Config.TEST_SIZE, Config.N_FOLDS, Config.SEED)

    metrics      = ['test_acc', 'test_sens', 'test_spec',
                    'test_bal', 'test_mcc',  'test_auc']
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity',
                    'Balanced Acc', 'MCC', 'AUC']

    print(f"\n{'='*65}")
    print(f"EXPERIMENT: {Config.EXP_NAME}")
    print(f"  Backbone : EfficientNetV2-S (1280d)")
    print(f"  Img SE   : SEBlock(1280, reduction=16)")
    print(f"  Meta     : MetaMLP(5->128->256)")
    print(f"  Fusion   : Concat(1280+256=1536d)")
    print(f"  Head     : FC(1536->1024->512->256->2)")
    print(f"  Loss     : WLS(w=[2,1], s=0.05)")
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
        "",
        "Architecture:",
        "  Backbone   : EfficientNetV2-S  ->  1280d",
        "  Img SE     : SEBlock(dim=1280, reduction=16)",
        "  Meta branch: MetaMLP(5->128->256)  ->  256d",
        "  Fusion     : Concat(img_se_out, meta_out)  ->  1536d",
        "  Classifier : FC(1536->1024->512->256->2) + BN+SiLU+Dropout",
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
