# Loan Approval Prediction: Binary Classification with Machine Learning

This project develops and evaluates a binary classification model to predict loan approval outcomes based on applicant financial and credit information. It simulates a real-world data science pipeline, similar to a Kaggle competition, where the goal is to accurately classify whether a loan should be approved or not (`loan_status`: 1 for approved, 0 for rejected).

## Table of Contents

1. [Overview](#1-overview)  
2. [Libraries](#2-libraries)  
3. [Data Loading and Description](#3-data-loading-and-description)  
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)  
   - 4.1. Dataset Structure and Types  
   - 4.2. Statistical Summary  
   - 4.3. Outlier Detection  
   - 4.4. Correlation with Target  
   - 4.5. Missing Data, Redundancy, and Leakage  
5. [Model Training](#5-model-training)  
6. [Model Evaluation and Results](#6-model-evaluation-and-results)  
   - 6.1. Accuracy Heatmap Across Datasets  
   - 6.2. Best Model Diagnostics (Confusion Matrix, AUC, Feature Importance)  
   - 6.3. Classification Metrics Summary  
7. [Kaggle Submission Preparation](#7-kaggle-submission-preparation)  
8. [Conclusion](#8-conclusion)  
9. [Usage](#9-usage)  
10. [Requirements](#10-requirements)

---

## 1. Overview

This notebook presents a comprehensive supervised learning pipeline focused on binary classification using ensemble and tree-based models. The target variable is `loan_status`, indicating whether a loan is approved (1) or denied (0). Nine versions of the dataset were created through data cleaning, outlier handling (IQR and Winsorization), SMOTE oversampling, and standardization.

Multiple machine learning models were trained, including Random Forest, Gradient Boosting, SVM, AdaBoost, KNN, Naive Bayes, Ridge Classifier, and Decision Trees. The models were evaluated on accuracy, precision, recall, and AUC to select the most performant configuration.

Once the best model was selected, it was applied to a Kaggle-style test set. This test set contained only features (no labels), requiring preprocessing and alignment with the training features before prediction. A final CSV file suitable for Kaggle submission was generated, maintaining the full test record count and required formatting.

This project showcases end-to-end machine learning practices including:

- Data preparation and handling of imbalanced datasets  
- Outlier removal and feature scaling  
- Dummy variable alignment between training and testing  
- Model training and cross-evaluation  
- In-depth model diagnostics for the best-performing classifier  
- Generating final predictions and exporting Kaggle-style submissions

---

## 2. Libraries

The following Python libraries were used:

- `pandas`: data loading and manipulation  
- `numpy`: numerical operations  
- `matplotlib`, `seaborn`: visualization  
- `scikit-learn`: model training, preprocessing, and evaluation  
- `imbalanced-learn`: SMOTE oversampling  
- `joblib`: model persistence  

---

## 3. Data Loading and Description

The dataset includes records of loan applicants with the following features:

- Applicant attributes (e.g., income, employment length, debt-to-income ratio)  
- Credit history (e.g., credit score, default history)  
- Loan characteristics (e.g., purpose, amount, interest rate)  

The target variable is `loan_status`:
- `1`: loan approved  
- `0`: loan denied  

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Dataset Structure and Types

Each dataset version contains over 20,000 records and up to 20 variables. Categorical variables were encoded using one-hot encoding, and numerical features were identified for transformation and scaling.

### 4.2 Statistical Summary

Basic statistics (mean, median, standard deviation, percentiles) were computed to assess variable distribution, scale, and skewness.

### 4.3 Outlier Detection

Two techniques were used for outlier detection:

- **IQR method**: Removed rows outside 1.5×IQR range for numeric variables.  
- **Winsorization**: Capped extreme values at the 5th and 95th percentiles.  

### 4.4 Correlation with Target

Correlation heatmaps were generated to identify features with strong linear or nonlinear relationships to `loan_status`.

### 4.5 Missing Data, Redundancy, and Leakage

Checks were performed to ensure:

- No missing or null values  
- No data leakage from test to train  
- No highly redundant features

---

## 5. Model Training

Each dataset version was used to train a collection of classifiers:

- Random Forest  
- Gradient Boosting  
- Support Vector Machine (SVM)  
- Decision Tree  
- AdaBoost  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Ridge Classifier  

Hyperparameters were kept default for baseline comparison. The models were evaluated on 20% hold-out test sets with fixed random seed (42).  

---

## 6. Model Evaluation and Results

### 6.1 Accuracy Heatmap Across Datasets

A heatmap compared accuracy scores across 8 models and 9 dataset versions. This visual identified the top-performing model/dataset combination.

### 6.2 Best Model Diagnostics

The best result was achieved by **Random Forest** on the dataset version with:

- IQR outlier removal  
- SMOTE oversampling  
- Standard scaling  

The following diagnostics were performed:

- Confusion Matrix (TP, FP, TN, FN)  
- ROC Curve and AUC score  
- Feature Importance Barplot  

### 6.3 Classification Metrics Summary

Key metrics for class 0 and 1:

| Metric     | Class 0 | Class 1 |
|------------|---------|---------|
| Precision  | 0.949   | 0.979   |
| Recall     | 0.980   | 0.948   |
| F1-score   | 0.964   | 0.963   |

Overall Accuracy: **96.4%**  
Macro Average F1-score: **96.4%**

---

## 7. Kaggle Submission Preparation

The model was applied to a test dataset (`loan_approval_prediction_test.csv`) provided without labels. The following steps were executed:

1. Applied same outlier removal and scaling methods  
2. One-hot encoded categorical variables  
3. Aligned test columns with training features (filled missing dummies with 0)  
4. Predicted `loan_status` with trained Random Forest  
5. Exported results using `sample_submission.csv` format with 39,098 IDs

---

## 8. Conclusion

This binary classification project highlights the importance of preprocessing, model selection, and diagnostics in machine learning. The Random Forest classifier provided high predictive performance (96%+) on a robustly prepared dataset. Proper treatment of outliers, class imbalance (via SMOTE), and feature scaling significantly improved model generalization. The final output was compatible with Kaggle-style submission formats, demonstrating real-world deployment readiness.

---

## 9. Usage

Clone the repository:

```bash
git clone https://github.com/eoroot1974/DataSciencePortfolio.git
