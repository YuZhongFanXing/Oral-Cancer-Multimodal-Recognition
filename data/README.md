# Dataset Information

## Source

This project uses the **Dataset of Annotated Oral Cavity Images for Oral Cancer Detection** published by Piyarathne et al. (2024).

## Included Files

- `Imagewise_Data.csv` — Per-image annotations (Image Name, Category, Clinical Diagnosis, Lesion Count)
- `Patientwise_Data.csv` — Per-patient metadata (Patient ID, Age, Gender, Smoking, Chewing Betel Quid, Alcohol, Image Count)
- `Annotation.json` — mask，   ID: 1, Lesion； ID: 2,  Oral Cavity
## Image Data

The original dataset contains ~3,000 clinical oral cavity images. Images are **not included** in this repository.

To obtain the full dataset, please refer to:

> Piyarathne N S, Liyanage S N, Rasnayaka R M S G K, et al. A comprehensive dataset of annotated oral cavity images for diagnosis of oral cancer and oral potentially malignant disorders[J]. Oral Oncology, 2024, 156: 106946.

## Example Images

The `examples/` directory contains 3 sample images per category (12 total) from the official dataset:

| Category | Binary Label | Example Images |
|---|---|---|
| Healthy | Benign (0) | 3 |
| Benign | Benign (0) | 3 |
| OPMD | Malignant (1) | 3 |
| OCA (Oral Cancer) | Malignant (1) | 3 |

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
  # Dataset Analysis Report

## 1. Image Directory Analysis

- **Total files**: 3000
- **File type distribution**:
  - `.jpg`: 3000 files

## 2. Image CSV File Analysis

- **Number of rows**: 3000
- **Number of columns**: 4
- **Column names**: `['Image Name', 'Category', 'Clinical Diagnosis', 'Lesion Annotation Count']`
- **First 5 rows**:

| Image Name | Category | Clinical Diagnosis | Lesion Annotation Count |
|------------|----------|--------------------|-------------------------|
| R-01-01    | OPMD     | Leukoplakia        | 1                       |
| R-01-02    | OPMD     | Leukoplakia        | 1                       |
| R-01-03    | Benign   | Coated Tongue      | 1                       |
| R-02-01    | Benign   | VBD                | 1                       |
| R-02-02    | Benign   | VBD                | 1                       |

- **Data types**:

## 3. Patient CSV File Analysis

- **Number of rows**: 714
- **Number of columns**: 7
- **Column names**: `['Patient ID', 'Age', 'Gender', 'Smoking', 'Chewing_Betel_Quid', 'Alcohol', 'Image Count']`
- **First 5 rows**:

| Patient ID | Age | Gender | Smoking | Chewing_Betel_Quid | Alcohol | Image Count |
|------------|-----|--------|---------|--------------------|---------|--------------|
| R-01       | 63  | M      | No      | No                 | No      | 3            |
| R-02       | 17  | F      | No      | No                 | No      | 8            |
| R-03       | 70  | M      | No      | No                 | No      | 5            |
| R-04       | 45  | M      | No      | No                 | No      | 5            |
| R-05       | 46  | M      | No      | Yes                | No      | 2            |

- **Data types**:

## 4. COCO Annotation JSON File Analysis

- **Main keys**: `['images', 'annotations', 'categories']`
- **Number of images**: 3000
- **Example image info**:
  ```json
  {"id": 1, "file_name": "R-01-01.jpg"}
  {
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [2574, 1739, 464, 562],
  "segmentation": [2574, 1845, 2848, 1739, 3038, 2088, 2977, 2255, 2908, 2301, 2673, 2058],
  "area": 6990438,
  "iscrowd": 0
}
  ```json
{
  "id": 2,
  "image_id": 1,
  "category_id": 2,
  "bbox": [860, 9, 3735, 3433],
  "segmentation": [2125, 34, 1437, 439, 981, 1050, 860, 2039, 1067, 2925, 1256, 3433, 4595, 3442, 4595, 3338, 4448, 3072, 4216, 2383, 4027, 1686, 3777, 852, 3657, 516, 3373, 172, 3089, 9],
  "area": 15815990,
  "iscrowd": 0
}
