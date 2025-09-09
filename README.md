# Breast_Cancer_project
Breast Cancer Classification Project
This project predicts whether a breast tumor is malignant or benign using the Breast Cancer Wisconsin dataset.
Overview
This project demonstrates the complete machine learning lifecycle, from data exploration to model deployment. I started by performing a thorough Exploratory Data Analysis (EDA) to understand the dataset's features, followed by preprocessing steps like data scaling.
I trained two classification models: a baseline Logistic Regression and a more powerful Random Forest Classifier. The Random Forest model proved to be highly effective, achieving an accuracy of approximately 96% and an outstanding ROC-AUC score of 0.99. This impressive performance highlights its ability to accurately distinguish between malignant and benign tumors.
Key Components
project.ipynb: A Jupyter Notebook that documents the entire workflow, from data loading to model evaluation.
inference_demo.py: A Python script that shows how to use the saved models to make predictions on new data.
breast_cancer_rf.pkl, breast_cancer_logreg.pkl, scaler.pkl: The saved machine learning models and data scaler, ready for use in a real-world application.
Tech Stack
Python
Pandas
Scikit-learn
Matplotlib
Joblib
How to Run
Clone this repository to your local machine.
Install the required libraries:
pip install -r requirements.txt


Run the inference script to see a sample prediction:
python inference_demo.py

