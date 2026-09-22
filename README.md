# Robust Zero-Shot Coreset Selection via Density-Aware Subspace Sampling

## Overview

This project presents a robust extension of ZCore (Iterative Subspace Sampling) for selecting a compact training subset under noisy data conditions.

The proposed approach introduces a density-aware filtering stage before diversity-based selection. Local neighborhood statistics are used to identify isolated samples that are likely to be noisy, after which ZCore performs diversity-based subspace sampling on the filtered feature set.

## Methodology

The pipeline consists of three main stages:

1. **Feature Extraction**
   - Extract image representations using a pretrained DINOv2 encoder.

2. **Density-Aware Filtering**
   - Construct a k-nearest neighbor graph in feature space.
   - Compute average neighborhood distances.
   - Normalize the distances using z-scores.
   - Remove highly isolated samples using a filtering threshold.

3. **Subspace Sampling**
   - Partition the filtered feature representations into subspaces.
   - Iteratively select samples that maximize diversity across subspaces.
   - Construct a compact coreset for downstream training.

## Dataset

**CIFAR-10**
- 50,000 training images
- 10 classes
- Experiments performed under synthetic noisy conditions
- Coreset size fixed to 5% of the full training set

## Results

| Method | Accuracy (%) |
|---|---:|
| Full Dataset | 94.80 |
| ZCore | 49.35 |
| Robust ZCore | **93.44** |

Using only **5% of the training data**, Robust ZCore achieved **93.44% accuracy** under noisy conditions, while standard ZCore achieved 49.35%.

The proposed method showed a performance drop of only **1.36%** relative to full-dataset training, compared with **45.45%** for standard ZCore.

## Key Techniques

- DINOv2 feature representations
- k-Nearest Neighbors
- Density-aware outlier filtering
- Z-score normalization
- Iterative subspace sampling
- Zero-shot coreset selection
- Robust learning under noisy data

## Technologies

- Python
- PyTorch
- NumPy
- Scikit-learn
- DINOv2
- Jupyter Notebook

## Repository Structure

```text
├── CV_PROJECT_FINAL.ipynb
├── Project_Report.pdf
└── README.md
