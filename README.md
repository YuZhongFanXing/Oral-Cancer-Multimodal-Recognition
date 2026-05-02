# Multimodal Recognition of Oral Cancer Integrating SE Attention and Metadata

[中文论文](paper/融合SE注意力与元数据的口腔癌多模态识别(3).docx)

## Overview

A two-stage multimodal diagnostic pipeline for oral cancer screening:

1. **Stage 1 — Oral Region Segmentation**: ResNet18-UNet extracts stable ROI from clinical oral images (Dice: 0.9700)
2. **Stage 2 — Multimodal Classification**: EfficientNetV2-S + SE channel attention + clinical metadata fusion with EMA & weighted label smoothing

## Key Results

| Experiment | Accuracy | Macro F1 | MCC | AUC |
|---|---|---|---|---|
| **Full Model (EfficientNetV2-S + SE + Metadata + EMA + WLS)** | **0.8281 ± 0.0153** | **0.7969 ± 0.0175** | **0.5944 ± 0.0346** | **0.8712 ± 0.0168** |

### Ablation Study

| Experiment | Accuracy | Macro F1 | MCC | AUC |
|---|---|---|---|---|
| Full Model | 0.8281 | 0.7969 | 0.5944 | 0.8712 |
| Weighted CE (no label smoothing) | 0.8125 | 0.7827 | 0.5741 | 0.8710 |
| No SE-Block | 0.8056 | 0.7752 | 0.5551 | 0.8641 |
| No EMA | 0.7825 | 0.7553 | 0.5206 | 0.8500 |
| Unimodal (Image Only) | 0.7531 | 0.7226 | 0.4627 | 0.8273 |
| Unimodal (Metadata Only) | 0.7687 | 0.7371 | 0.4888 | 0.8188 |

### Backbone Comparison

| Backbone | Accuracy | Macro F1 | MCC | AUC |
|---|---|---|---|---|
| **EfficientNetV2-S** | **0.8281** | **0.7969** | **0.5944** | **0.8712** |
| EfficientNet-B3 | 0.8150 | 0.7740 | 0.5514 | 0.8527 |
| DenseNet121 | 0.8012 | 0.7698 | 0.5435 | 0.8563 |
| MobileNetV3-Large | 0.7956 | 0.7617 | 0.5312 | 0.8491 |
| ConvNeXt-Tiny | 0.7919 | 0.7632 | 0.5377 | 0.8469 |
| ResNet50 | 0.7900 | 0.7585 | 0.5259 | 0.8450 |

### Fusion Strategy Comparison

| Fusion Method | Accuracy | Macro F1 | MCC | AUC |
|---|---|---|---|---|
| **Concat** | **0.8281** | **0.7969** | **0.5944** | **0.8712** |
| Gated Multimodal Fusion | 0.8231 | 0.7932 | 0.5888 | 0.8728 |
| Multi-Task Learning | 0.8137 | 0.7857 | 0.5774 | 0.8742 |
| Bidirectional Cross-Attention | 0.8044 | 0.7762 | 0.5585 | 0.8629 |
| Element-wise Addition | 0.7981 | 0.7687 | 0.5470 | 0.8485 |

## Repository Structure

```
├── paper/                          # Paper manuscript
├── data/                           # Metadata CSVs (images not included)
├── src/
│   ├── segmentation.py             # ResNet18-UNet oral segmentation
│   ├── train_img_se_baseline.py    # Baseline: ImgSE + Concat + EMA + WLS
│   ├── train_backbone_ablation.py  # Backbone comparison & ablation study
│   ├── train_meta_only.py          # Metadata-only model
│   ├── generate_meta_npy.py        # Generate test predictions for meta-only
│   ├── train_fusion_innovation.py  # Fusion strategy comparison experiments
│   ├── train_fusion_gated.py       # ImgSE + Gated fusion
│   ├── train_fusion_add.py         # ImgSE + Add fusion
│   └── plot_roc_curves.py          # ROC curve plotting
├── results/
│   ├── tables/                     # Per-fold CSV results
│   └── figures/                    # Comparison charts, ROC curves, Grad-CAM
└── requirements.txt
```

## Model Architecture

```
Input: Oral ROI Image (448×448) + Clinical Metadata (5-dim)
       │                                    │
       ▼                                    ▼
EfficientNetV2-S (1280d)            MetaMLP (5→128→256)
       │                                    │
SE Block (channel recalibration)           │
       │                                    │
       └────────── Concat ─────────────────┘
                       │
                       ▼
              FC (1536→1024→512→256→2)
                       │
                       ▼
               Benign / Malignant
```

**Training Strategy:**
- **Loss**: Weighted Label Smoothing (w=[2,1], smoothing=0.05)
- **EMA**: decay=0.99, warmup=5 epochs
- **Optimizer**: AdamW (lr=2e-4, wd=2e-3) + CosineAnnealingLR
- **Early Stopping**: patience=20, smooth_window=5
- **Data Split**: Patient-level stratified, 15% holdout test, 5-fold CV

## Dataset

This project uses the **Dataset of Annotated Oral Cavity Images for Oral Cancer Detection** by Piyarathne et al. (2024).

The dataset contains:
- 3,000 clinical oral cavity images (Benign, OPMD, Oral Cancer)
- Oral region & lesion segmentation annotations
- Patient metadata: Age, Gender, Smoking, Chewing Betel Quid, Alcohol

**Images are not included** in this repository. Please request the original dataset:

> Piyarathne N S, Liyanage S N, Rasnayaka R M S G K, et al. A comprehensive dataset of annotated oral cavity images for diagnosis of oral cancer and oral potentially malignant disorders[J]. Oral Oncology, 2024, 156: 106946.

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
Pillow
tqdm
```

## Citation

If you use this code or model, please cite our paper:

> 侯宇欣, 徐睿杰, 韩俊杰, 赵奎璋, 查鑫悦, 张琥, 吴贯锋. 融合SE注意力与元数据的口腔癌多模态识别[J].

## Authors

- Hou Yuxin, Xu Ruijie, Han Junjie, Zhao Kuizhang, Zha Xinyue
- School of Mathematics, Southwest Jiaotong University
- The Third People's Hospital of Chengdu
