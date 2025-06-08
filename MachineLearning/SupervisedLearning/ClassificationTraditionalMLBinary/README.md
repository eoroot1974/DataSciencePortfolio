# Project #2: Project_CTB-LoanApprovalPrediction

## Loan Approval Prediction using Supervised Machine Learning

This project presents an end-to-end machine learning workflow to predict loan approval status based on demographic, credit, and loan-specific attributes. The classification models are trained and evaluated across multiple cleaned, balanced, and standardized versions of the dataset. The goal is to identify a robust model suitable for deployment in real-world credit scoring systems.

---

## 📂 Contents

- Exploratory Data Analysis (EDA)
- Outlier detection and treatment (Winsorization & IQR filtering)
- Class imbalance handling using SMOTE
- Feature encoding and scaling
- Model training across 9 dataset versions and 8 classifiers
- Accuracy benchmarking via heatmap
- Model performance analysis: confusion matrix, AUC, ROC, precision/recall
- Final predictions for Kaggle competition submission

---

## 📊 Dataset Description

The dataset includes the following attributes:
- Demographic: `person_age`, `person_home_ownership`, `person_income`
- Employment: `person_emp_length`
- Loan details: `loan_amnt`, `loan_int_rate`, `loan_intent`, `loan_grade`, `loan_percent_income`
- Credit history: `cb_person_default_on_file`, `cb_person_cred_hist_length`
- Target: `loan_status` (1 = default, 0 = non-default)

---

## 💡 Models Evaluated

Eight classification algorithms were trained on nine versions of the dataset:

- Random Forest
- Gradient Boosting
- Decision Tree
- Support Vector Machine (SVM)
- Naive Bayes
- k-Nearest Neighbors (KNN)
- AdaBoost
- Ridge Classifier

The best performing model was **Random Forest** trained on the **IQR-cleaned, SMOTE-balanced, and standardized dataset** (`loan_data_r_B_scaled`), achieving **96% accuracy**.

---

## 🧪 Performance Analysis

- Precision, Recall, F1-score breakdown
- Confusion Matrix with TP, FP, FN, TN
- Feature Importance Ranking
- ROC Curve and AUC score
- Final model applied to Kaggle test dataset and formatted as a valid competition submission

---

## ⚙️ Usage

Clone the repository:

```bash
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction

