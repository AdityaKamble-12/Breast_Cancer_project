# Breast Cancer Diagnosis – Machine Learning Classification

**Quick project you can complete in ~3 hours.**  
**Dataset:** Breast Cancer Wisconsin (Diagnostic) – bundled with scikit-learn  
**Goal:** Predict whether a tumor is malignant or benign.

## Highlights
- Cleaned & analyzed tabular medical data (30 numeric features).
- Trained **Logistic Regression** (baseline) and **Random Forest** (final model).
- Achieved **Test Accuracy (RF): 0.947** and **ROC-AUC (RF): 0.953**.
- Visualized **confusion matrices**, **ROC curves**, and **feature importance**.
- Saved trained model for reuse.

## Tech Stack
Python, pandas, scikit-learn, matplotlib, joblib

## How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib joblib
   ```
2. Run the project script / notebook (this file was executed in a notebook environment).
3. Artifacts:
   - `breast_cancer_rf.pkl` – final Random Forest model
   - `breast_cancer_logreg.pkl` – baseline Logistic Regression
   - `scaler.pkl` – StandardScaler fitted on training data
4. Use the example below to load and predict:
   ```python
   import joblib
   import numpy as np

   rf = joblib.load("breast_cancer_rf.pkl")
   scaler = joblib.load("scaler.pkl")

   # example: single sample from data (replace with your values, shape must be (1, 30))
   x = np.random.rand(1, 30)
   # Random Forest was trained on unscaled features; for it, pass raw features:
   y_pred = rf.predict(x)

   print("Prediction:", y_pred[0])  # 0=malignant, 1=benign
   ```

## Results (Test Set)
- **Random Forest**
  - Accuracy: 0.9474
  - Precision: 0.9583
  - Recall: 0.9583
  - F1-score: 0.9583
  - ROC-AUC: 0.9527

- **Logistic Regression**
  - Accuracy: 0.9825
  - Precision: 0.9861
  - Recall: 0.9861
  - F1-score: 0.9861
  - ROC-AUC: 0.95

## Files
- `breast_cancer_rf.pkl` (Random Forest)
- `breast_cancer_logreg.pkl` (Logistic Regression)
- `scaler.pkl` (StandardScaler for preprocessing)
