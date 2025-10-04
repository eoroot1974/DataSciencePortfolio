# Loan Approval Prediction: Binary Classification with Deep Learning

This project develops and evaluates deep learning models to predict loan approval outcomes based on applicant financial and credit information. It reproduces a **real-world AI pipeline** similar to a Kaggle competition, using neural networks implemented in **TensorFlow (Keras)** and **PyTorch**.
The objective is to accurately classify whether a loan should be approved (`loan_status`: 1) or denied (`loan_status`: 0) through robust feature preprocessing, balanced training, and multi-architecture model comparison.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Libraries](#2-libraries)
3. [Data Loading and Description](#3-data-loading-and-description)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)

   * 4.1. Dataset Structure and Feature Types
   * 4.2. Outlier Detection and Treatment
   * 4.3. Class Imbalance Correction (SMOTE)
   * 4.4. Feature Scaling and Normalization
   * 4.5. Correlation and Multivariate Checks
5. [Deep Learning Model Training](#5-deep-learning-model-training)
6. [Model Predictions and Evaluation](#6-model-predictions-and-evaluation)

   * 6.1. Accuracy Heatmap Across Architectures and Datasets
   * 6.2. Best Model Diagnostics (Confusion Matrix, ROC Curve, AUC)
   * 6.3. Precision, Recall, and F1-Score Summary
7. [Kaggle Submission Preparation](#7-kaggle-submission-preparation)
8. [Conclusion](#8-conclusion)
9. [Usage](#9-usage)
10. [Requirements](#10-requirements)

---

## 1. Overview

This notebook implements a complete **deep learning classification pipeline** for binary loan approval prediction.
Three dataset versions were derived using **Winsorization** and **IQR filtering** for outlier control, **SMOTE** for class balancing, and **feature standardization** for optimal neural network convergence.

Each version was trained using **three neural network configurations in both TensorFlow and PyTorch**:

* **Shallow Network:** 2 hidden layers with ReLU activations and dropout regularization
* **Deep Network:** 4 hidden layers with ReLU, dropout, and adaptive learning
* **Batch Normalized Network:** 3–4 layers with batch normalization for faster convergence and stability

This approach allows comparison of performance across frameworks and architectures, offering a clear view of how preprocessing and model depth influence generalization.

---

## 2. Libraries

The following key libraries were used:

* **pandas**, **numpy** – data management and numerical operations
* **matplotlib**, **seaborn** – EDA and performance visualization
* **scikit-learn** – data splitting, scaling, and evaluation metrics
* **imblearn** – SMOTE resampling
* **tensorflow** / **keras** – neural network implementation
* **torch** / **torchvision** – alternative PyTorch network implementation

---

## 3. Data Loading and Description

The dataset contains applicant demographic, financial, and loan-specific features used to determine loan approval outcomes.

* **Numerical features:** income, age, employment length, loan amount, interest rate, debt-to-income ratio, etc.
* **Categorical features:** loan intent, home ownership, credit bureau flags, etc.
* **Target variable:** `loan_status` (1 = approved, 0 = denied)

---

## 4. Exploratory Data Analysis (EDA)

Key insights from the data include:

* **Outliers:** detected and treated using Winsorization and IQR methods
* **Class imbalance:** resolved through SMOTE oversampling for equitable training
* **Feature scaling:** all numerical inputs standardized to mean 0 and std 1
* **Correlation:** pairwise heatmaps revealed moderate positive links between `loan_percent_income`, `loan_int_rate`, and `loan_status`
* **Distribution:** variables like income and loan amount are right-skewed, common in financial data

---

## 5. Deep Learning Model Training

Each of the **three preprocessed datasets** (`loan_data_train_B_scaled`, `loan_data_train_w_B_scaled`, `loan_data_train_r_B_scaled`) was used to train six neural networks (three in TensorFlow and three in PyTorch).
All models were trained using 80/20 train-test splits, Adam optimizer, binary cross-entropy loss, and early stopping on validation accuracy.

Performance across architectures and datasets was summarized via an **accuracy heatmap**, highlighting which preprocessing strategy yielded the most reliable generalization.

---

## 6. Model Predictions and Evaluation

### 6.1 Accuracy Heatmap Across Architectures and Datasets

A comparative heatmap shows accuracy scores for all TensorFlow and PyTorch architectures across datasets, illustrating the performance differences between shallow, deep, and batch-normalized networks.

### 6.2 Best Model Diagnostics

The top model — **PyTorch Deep Network trained on the IQR-filtered, SMOTE-balanced, and standardized dataset (`loan_data_train_r_B_scaled`)** — achieved an accuracy of **95.5%** and **AUC = 0.983**.
It was further evaluated using:

* Confusion matrix
* ROC curve (showing strong separation between classes)
* Precision-Recall trade-off

### 6.3 Precision, Recall, and F1-Score Summary

| Class         | Precision | Recall | F1-Score | Support |
| :------------ | :-------: | :----: | :------: | :-----: |
| Non-Defaulted |    0.93   |  0.98  |   0.96   |   6038  |
| Defaulted     |    0.98   |  0.93  |   0.95   |   6039  |

**Accuracy:** 0.95  **Macro Avg F1:** 0.95  **AUC:** 0.983

These results confirm the deep learning model’s strong balance between sensitivity and specificity.

---

## 7. Kaggle Submission Preparation

The trained best model was used to predict outcomes on an external Kaggle test set (`loan_approval_prediction_test.csv`).
The same preprocessing pipeline (scaling, encoding, feature alignment) was applied before inference.
Predictions were saved in Kaggle format as **`kaggle_submission.csv`**, preserving original IDs and structure.

---

## 8. Conclusion

This deep learning implementation achieved **state-of-the-art performance** with an AUC of 0.983 using a PyTorch Deep Network on the IQR-filtered and SMOTE-balanced dataset.
Compared to traditional ML approaches, the neural networks demonstrated:

* Improved generalization across datasets
* Lower bias and variance trade-off
* High interpretability through gradient-based sensitivity and ROC analysis

The final model is suitable for **real-world credit risk assessment** or **automated loan approval pipelines** where both accuracy and reliability are critical.

---

## 9. Usage

```bash
git clone https://github.com/eoroot1974/DataSciencePortfolio.git
cd DataSciencePortfolio
jupyter notebook Project_CNB-LoanApprovalPrediction_DL.ipynb
```

---

## 10. Requirements

* Python ≥ 3.10
* NumPy, Pandas
* Matplotlib, Seaborn
* scikit-learn
* imbalanced-learn
* TensorFlow ≥ 2.16
* PyTorch ≥ 2.2

---
