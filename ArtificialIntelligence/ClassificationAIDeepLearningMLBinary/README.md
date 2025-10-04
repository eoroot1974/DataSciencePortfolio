# Deep Learning Binary Classification

## Contents

Project #1: **Project_CNB-LoanApprovalPrediction (Deep Learning Edition)**
This project applies **deep neural network models** to predict loan approval outcomes based on demographic, financial, and credit history data. Multiple dataset variants are prepared using **outlier treatment (Winsorization, IQR filtering)**, **class balancing with SMOTE**, and **feature standardization** to ensure optimal training conditions.

Both **TensorFlow (Keras)** and **PyTorch** implementations are included, featuring shallow, deep, and batch-normalized network architectures.
All models are trained, validated, and compared across the prepared datasets, and the **best-performing deep learning model** is then used to generate predictions for a **Kaggle test set**.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/eoroot1974/DataSciencePortfolio.git
   ```
2. Open the project directory and launch Jupyter Notebook or VS Code.
3. Run the notebook `Project_CNB-LoanApprovalPrediction.ipynb` (Deep Learning version).
4. The pipeline will:

   * Preprocess and balance the data
   * Train and evaluate TensorFlow and PyTorch models
   * Generate performance reports, confusion matrices, ROC curves, and Kaggle-ready predictions
