# Dataset Information

## Source

This project uses the **Dataset of Annotated Oral Cavity Images for Oral Cancer Detection** published by Piyarathne et al. (2024).

## Included Files

- `Imagewise_Data.csv` — Per-image annotations (Image Name, Category, Clinical Diagnosis, Lesion Count)
- `Patientwise_Data.csv` — Per-patient metadata (Patient ID, Age, Gender, Smoking, Chewing Betel Quid, Alcohol, Image Count)

## Image Data

The original dataset contains ~3,000 clinical oral cavity images. Images are **not included** in this repository.

To obtain the full dataset, please refer to:

> Piyarathne N S, Liyanage S N, Rasnayaka R M S G K, et al. A comprehensive dataset of annotated oral cavity images for diagnosis of oral cancer and oral potentially malignant disorders[J]. Oral Oncology, 2024, 156: 106946.

## Data Preparation

1. Download the original dataset (Images + Annotation.json)
2. Run `src/segmentation.py` to generate oral ROI masks and segmented images
3. The segmented images should be placed in a `Segmented_Images/` directory
4. Run classification scripts under `src/` for training/evaluation

## Class Mapping

| Original Category | Binary Label |
|---|---|
| Benign | 0 (Benign) |
| OPMD | 1 (Malignant) |
| OCA (Oral Cancer) | 1 (Malignant) |

## Metadata Encoding

The 5-dimensional metadata vector encodes:
- **Age**: normalized to [0, 1] (capped at 100)
- **Gender**: 1.0 for Male, 0.0 for Female
- **Smoking**: 1.0 for Yes, 0.0 for No
- **Chewing Betel Quid**: 1.0 for Yes, 0.0 for No
- **Alcohol**: 1.0 for Yes, 0.0 for No
