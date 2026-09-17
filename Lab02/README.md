# Lab 02: Effect of Image Filtering on Skin-Lesion Classification

This repository contains the implementation and experimental results for Lab Task 02, investigating how different spatial filters affect the performance of deep learning models on the HAM10000 skin-lesion dataset.

## Dataset
- **Dataset:** HAM10000 (Downloaded via Kaggle API)
- **Classes:** 7 skin lesion types (nv, mel, bcc, akiec, bkl, df, vasc)

## Models Evaluated
- ResNet50
- DenseNet121
- EfficientNet-B0

## Spatial Filters Applied
1. No Filter (Baseline)
2. Average Filter
3. Gaussian Filter
4. Median Filter
5. Sharpening Filter
6. Sobel Filter

## How to Run the Experiments
1. Open the Google Colab notebook (`.ipynb`) provided in this folder.
2. Enter your Kaggle API token when prompted during dataset download.
3. Run all cells sequentially from top to bottom to replicate data loading, filtering, model evaluation, and visualizations.
