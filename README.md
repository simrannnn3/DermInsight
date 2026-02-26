# 🧬 Skin Lesion Segmentation using MSRF-Net (ISIC 2018)

This project implements a deep learning–based semantic segmentation system for skin lesion analysis using the ISIC 2018 dataset. The core model is based on a Multi-Scale Refinement Network (MSRF-Net) with an edge-aware refinement branch to improve boundary localization and preserve fine-grained details.

In addition to segmentation accuracy, the project focuses on interpretability and reliability using:
- Grad-CAM / Grad-CAM++ for visual explanations
- Monte Carlo Dropout for epistemic uncertainty estimation
- Boundary-focused and error-aware analysis for failure case inspection

---

## Features

- MSRF-Net for multi-scale lesion segmentation  
- Edge-aware refinement for better boundary localization  
- Dice + Cross-Entropy loss for class imbalance handling  
- Evaluation using Dice Coefficient and IoU  
- Explainability with Grad-CAM and Grad-CAM++  
- Uncertainty estimation using Monte Carlo Dropout  
- Boundary-focused and error-aware visual analysis  
- Optional Streamlit UI for interactive visualization  

---

## Dataset

- ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection  
- Contains dermoscopic images and corresponding segmentation masks.

- Official Challenge Page: https://challenge.isic-archive.com/data/#2018  
- Kaggle Mirror: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation  

The dataset includes:
- High-resolution dermoscopic images
- Pixel-wise segmentation masks
- Training, validation, and test splits
- Multiple lesion types with varying contrast and boundary ambiguity

After downloading, place the images and masks in your local dataset directory and update the paths in the training/inference scripts accordingly.

## Model Architecture

The segmentation model is based on a **Multi-Scale Refinement Network (MSRF-Net)** designed to capture both global context and fine-grained boundary details.

Key components:
- Encoder–decoder architecture with skip connections  
- Multi-scale feature extraction to handle lesions of varying sizes  
- Edge-aware refinement branch to improve boundary localization  
- Final sigmoid output producing a pixel-wise probability map  

The design focuses not only on segmentation accuracy, but also on producing sharper boundaries and more stable predictions in challenging, low-contrast regions.

## Running the Application (Streamlit UI)

This project is designed to run using a **Streamlit-based user interface** for inference, visualization, explainability, and uncertainty analysis. No training script is required.

### Steps to run:

1. Make sure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```
2. Ensure the trained model file is available at:
   ```bash
   model/best_model.keras
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
