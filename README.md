# Multimodal Recognition of Oral Cancer Integrating SE Attention and Metadata

## Overview

A two-stage multimodal diagnostic pipeline for oral cancer screening:

1. **Stage 1 -- Oral Region Segmentation**: ResNet18-UNet extracts stable ROI from clinical oral images (**Dice: 0.9700**)
2. **Stage 2 -- Multimodal Classification**: EfficientNetV2-S + SE channel attention + clinical metadata fusion with EMA & weighted label smoothing

### Model Architecture

```
Input: Oral ROI Image (448x448) + Clinical Metadata (5-dim)
       |                                    |
       v                                    v
EfficientNetV2-S (1280d)            MetaMLP (5->128->256)
       |                                    |
SE Block (channel recalibration)           |
       |                                    |
       +---------- Concat -----------------+
                       |
                       v
              FC (1536->1024->512->256->2)
                       |
                       v
               Benign / Malignant
```

---

## Results

All tables and figures below correspond to the paper's experimental section.

### Table 1: Training Configuration

| Parameter | Value |
|---|---|
| Training Device | NVIDIA A100 GPU |
| Random Seed | 42 |
| Input Image Size | 448 x 448 |
| Batch Size | 16 |
| Gradient Accumulation | 3 |
| Max Epochs | 100 |
| Early Stop Patience | 20 |
| Optimizer | AdamW |
| Initial Learning Rate | 2x10^-4 |
| Weight Decay | 2x10^-3 |
| LR Schedule | CosineAnnealingLR |
| Min Learning Rate | 5x10^-7 |

### Table 2: Backbone Comparison

| Backbone | Accuracy | Precision | Recall | Macro F1 | MCC | AUC |
|---|---|---|---|---|---|---|
| **EfficientNetV2-S** | **0.8281 +- 0.0153** | **0.8818 +- 0.0112** | **0.8714 +- 0.0173** | **0.7969 +- 0.0175** | **0.5944 +- 0.0346** | **0.8712 +- 0.0168** |
| EfficientNet-B3 | 0.8150 +- 0.0140 | 0.8555 +- 0.0129 | 0.8857 +- 0.0261 | 0.7740 +- 0.0156 | 0.5514 +- 0.0321 | 0.8527 +- 0.0173 |
| DenseNet121 | 0.8012 +- 0.0215 | 0.8745 +- 0.0130 | 0.8366 +- 0.0380 | 0.7698 +- 0.0203 | 0.5435 +- 0.0388 | 0.8563 +- 0.0182 |
| MobileNetV3-Large | 0.7956 +- 0.0100 | 0.8709 +- 0.0248 | 0.8339 +- 0.0453 | 0.7617 +- 0.0106 | 0.5312 +- 0.0184 | 0.8491 +- 0.0102 |
| ConvNeXt-Tiny | 0.7919 +- 0.0305 | 0.8808 +- 0.0152 | 0.8143 +- 0.0660 | 0.7632 +- 0.0236 | 0.5377 +- 0.0352 | 0.8469 +- 0.0136 |
| ResNet50 | 0.7900 +- 0.0280 | 0.8716 +- 0.0175 | 0.8223 +- 0.0577 | 0.7585 +- 0.0235 | 0.5259 +- 0.0488 | 0.8450 +- 0.0182 |

![Fig.4: ROC Backbone Comparison](results/figures/roc_backbone_replot.png)

### Table 3: Fusion Strategy Comparison

| Fusion Method | Accuracy | Precision | Recall | Macro F1 | MCC | AUC |
|---|---|---|---|---|---|---|
| **Concat** | **0.8281 +- 0.0153** | **0.8818 +- 0.0112** | **0.8714 +- 0.0173** | **0.7969 +- 0.0175** | **0.5944 +- 0.0346** | **0.8712 +- 0.0168** |
| Gated Multimodal Fusion | 0.8231 +- 0.0208 | 0.8847 +- 0.0152 | 0.8598 +- 0.0332 | 0.7932 +- 0.0217 | 0.5888 +- 0.0418 | 0.8728 +- 0.0218 |
| Multi-Task Learning | 0.8137 +- 0.0317 | 0.8865 +- 0.0050 | 0.8420 +- 0.0554 | 0.7857 +- 0.0284 | 0.5774 +- 0.0504 | 0.8742 +- 0.0101 |
| Bidirectional Cross-Attention | 0.8044 +- 0.0127 | 0.8859 +- 0.0117 | 0.8277 +- 0.0323 | 0.7762 +- 0.0095 | 0.5585 +- 0.0174 | 0.8629 +- 0.0152 |
| Element-wise Addition | 0.7981 +- 0.0200 | 0.8825 +- 0.0218 | 0.8232 +- 0.0548 | 0.7687 +- 0.0144 | 0.5470 +- 0.0212 | 0.8485 +- 0.0126 |

### Table 4: Ablation Study

| Experiment | Accuracy | Precision | Recall | Macro F1 | MCC | AUC |
|---|---|---|---|---|---|---|
| **Full Model** | **0.8281 +- 0.0153** | **0.8818 +- 0.0112** | **0.8714 +- 0.0173** | **0.7969 +- 0.0175** | **0.5944 +- 0.0346** | **0.8712 +- 0.0168** |
| Weighted CE (no label smoothing) | 0.8125 +- 0.0217 | 0.8856 +- 0.0228 | 0.8429 +- 0.0537 | 0.7827 +- 0.0186 | 0.5741 +- 0.0323 | 0.8710 +- 0.0117 |
| No SE-Block | 0.8056 +- 0.0178 | 0.8807 +- 0.0209 | 0.8366 +- 0.0311 | 0.7752 +- 0.0200 | 0.5551 +- 0.0395 | 0.8641 +- 0.0209 |
| No EMA | 0.7825 +- 0.0183 | 0.8820 +- 0.0161 | 0.7964 +- 0.0313 | 0.7553 +- 0.0182 | 0.5206 +- 0.0345 | 0.8500 +- 0.0198 |
| Unimodal (Image Only) | 0.7531 +- 0.0420 | 0.8626 +- 0.0245 | 0.7723 +- 0.0811 | 0.7226 +- 0.0352 | 0.4627 +- 0.0564 | 0.8273 +- 0.0279 |
| Unimodal (Metadata Only) | 0.7687 +- 0.0551 | 0.8606 +- 0.0133 | 0.8000 +- 0.0970 | 0.7371 +- 0.0474 | 0.4888 +- 0.0837 | 0.8188 +- 0.0115 |

![Fig.5: ROC Ablation Comparison](results/figures/roc_ablation_replot.png)

### Fig.6: Grad-CAM Visualization

SE channel attention concentrates activation on lesion regions, suppressing background noise.

![Fig.6: Grad-CAM Overview](results/figures/gradcam_overview.png)

![Fig.6: Grad-CAM with Ground Truth Mask](results/figures/gradcam_gtmask_overview.png)

---

## Repository Structure

```
├── data/                          # Metadata CSVs (images not included)
├── src/
│   ├── models.py                  # ImgSEModel, SEBlock, MetaMLP, build_head
│   ├── dataset.py                 # MultiModalDataset, data loading & splitting
│   ├── train_utils.py             # ModelEMA, WeightedLabelSmoothingLoss, evaluate
│   ├── segmentation.py            # Stage 1: ResNet18-UNet oral segmentation
│   ├── train_img_se_baseline.py   # Stage 2: ImgSE baseline (MAIN script)
│   ├── train_backbone_ablation.py # Table 2 backbone comparison + Table 4 ablation
│   ├── train_fusion_innovation.py # Table 3 fusion strategy comparison
│   └── train_meta_only.py         # Table 4 metadata-only model
├── results/
│   ├── tables/                    # Per-fold CSV results
│   └── figures/                   # Paper figures (Fig.4-6)
└── requirements.txt
```

### Code Logic

The experiments follow the paper's progressive logic: **backbone → fusion → ablation**.

1. **`segmentation.py`** — Run first. Trains ResNet18-UNet to extract oral ROI from raw clinical images (Dice 0.9700). Outputs segmented images for Stage 2.
2. **`train_img_se_baseline.py`** — The core baseline. Implements the full model: EfficientNetV2-S + SE + MetaMLP + Concat fusion + EMA + Weighted Label Smoothing. This is the paper's main contribution.
3. **`train_backbone_ablation.py`** — Reproduces Table 2 (6 backbone variants) and Table 4 rows 1-5 (NoSE, NoEMA, WCE, Image-only ablations).
4. **`train_fusion_innovation.py`** — Reproduces Table 3 (Concat, Multi-Task, Cross-Attention fusion strategies).
5. **`train_meta_only.py`** — Reproduces Table 4 row 6 (Metadata-only baseline).

Shared modules (`models.py`, `dataset.py`, `train_utils.py`) are extracted from the baseline and imported by all experiment scripts.

---

## Training Strategy

- **Loss**: Weighted Label Smoothing (w=[2,1], smoothing=0.05)
- **EMA**: decay=0.99, warmup=5 epochs
- **Optimizer**: AdamW (lr=2e-4, wd=2e-3) + CosineAnnealingLR
- **Early Stopping**: patience=20, smooth_window=5
- **Data Split**: Patient-level stratified, 15% holdout test set, 5-fold CV
- **Augmentation**: Random H/V flip, rotation (+-20 deg), affine, color jitter, grayscale
- **TTA**: 3-way flip averaging at inference

---

## Dataset

This project uses the **Dataset of Annotated Oral Cavity Images for Oral Cancer Detection** by Piyarathne et al. (2024):

- 3,000 clinical oral cavity images (Benign / OPMD / Oral Cancer)
- Oral region & lesion segmentation annotations
- Patient metadata: Age, Gender, Smoking, Chewing Betel Quid, Alcohol

**Images are not included** in this repository. Obtain the original dataset from:

> Piyarathne N S, Liyanage S N, Rasnayaka R M S G K, et al. A comprehensive dataset of annotated oral cavity images for diagnosis of oral cancer and oral potentially malignant disorders[J]. Oral Oncology, 2024, 156: 106946.

---

## Citation

> 侯宇欣, 徐睿杰, 韩俊杰, 赵奎璋, 查鑫悦, 张琥, 吴贯锋. 融合SE注意力与元数据的口腔癌多模态识别[J].

## Authors

- Hou Yuxin, Xu Ruijie, Han Junjie, Zhao Kuizhang, Zha Xinyue
- School of Mathematics, Southwest Jiaotong University
- The Third People's Hospital of Chengdu
