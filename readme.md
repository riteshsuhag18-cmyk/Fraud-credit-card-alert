# 🛡️ Enterprise-Grade Credit Card Fraud Detection System

An end-to-end machine learning pipeline and interactive web application designed to detect fraudulent credit card transactions while minimizing false alarms in highly imbalanced financial datasets.

## 🚀 Project Overview
Credit card fraud detection is a classic machine learning challenge characterized by extreme class imbalance (only ~0.16% fraud cases). This project implements a rigorous data science workflow—from exploratory data analysis and rigorous cleaning to advanced feature engineering, class balancing via SMOTE, and model optimization using XGBoost.

## 📊 Key Pipeline Components
- **Data Preprocessing & Cleaning:** Handled duplicates and outliers using interquartile strategies and `RobustScaler`.
- **Feature Engineering:** Extracted temporal patterns (`Hour` from transaction time) and smoothed skewed distributions (`Log_Amount`).
- **Handling Class Imbalance:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** strictly on the training split to prevent data leakage.
- **Modeling & Evaluation:** Trained Random Forest and **XGBoost Classifier**, achieving an exceptional **ROC-AUC Score of 0.9786**[cite: 11].
- **Probability Threshold Tuning:** Optimized the decision threshold to `0.80` to strike a balance between high recall (catching actual frauds) and high precision (reducing false alarms/false positives)[cite: 13].
- **Deployment:** Built an interactive real-time **Streamlit Web Dashboard** for live transaction monitoring and risk scoring.

## 🛠️ Tech Stack
- **Language:** Python 3.14
- **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn, XGBoost, Plotly, Joblib
- **Web Framework:** Streamlit

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/credit-card-fraud-detection.git](https://github.com/riteshsuhag18-cmyk/credit-card-fraud-detection.git)
   cd credit-card-fraud-detection