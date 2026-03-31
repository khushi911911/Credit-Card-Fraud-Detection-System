# Credit Card Fraud Detection System

**ML InnovateX Hackathon** project by **Khushi Donga**  
**Student ID:** 23AIML016

## Project Overview

This repository contains a credit card fraud detection system built with machine learning and deep learning models. The project includes data processing, model training, model artifact export, and a Streamlit dashboard for real-time fraud prediction.

## Included Files

- `credit_card_fraud_detection.ipynb` – complete notebook for data exploration, preprocessing, model training, evaluation, and artifact export
- `app.py` – Streamlit dashboard application for model inference and analytics
- `fraud_detection_model.pkl` – saved trained classifier
- `scaler.pkl` – saved feature scaler used for inference
- `fraud_detection_ann.h5` – saved ANN model artifact (optional)
- `class_distribution.png`, `model_comparison.png`, `roc_curves.png`, `confusion_matrices.png`, `training_history.png`, `overfitting_analysis.png` – training and evaluation visuals
- `creditcard.csv (1)/creditcard.csv` – dataset file

## Requirements

Install required packages before running the app:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
python -m streamlit run app.py
```

## Notes

- The dashboard supports model selection and real-time transaction risk scoring.
- The notebook includes feature engineering, SMOTE balancing, and multiple classifier comparisons.
- This repo is prepared for GitHub with clean structure and project metadata.
