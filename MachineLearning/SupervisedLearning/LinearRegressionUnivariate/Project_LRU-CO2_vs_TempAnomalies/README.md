# Linear Regression Analysis: CO2 Concentration vs Temperature

This project performs a detailed linear regression analysis to study the relationship between **CO2 concentration (ppm)** and **temperature (°C)**. The analysis involves data exploration, statistical testing for model suitability, training the regression model, and evaluating its performance.

---

## Table of Contents

- [Overview](#overview)
1. [Libraries](#1-libraries)
2. [Data Loading and Data Description](#2-data-loading-and-data-description)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
   - [3.1. Dataset Structure and Data Types](#31-dataset-structure-and-data-types)
   - [3.2. Basic Statistical Information](#32-basic-statistical-information)
   - [3.3. Graphical Dataset Analysis](#33-graphical-dataset-analysis)
     - [3.3.1. Variables Graphical Analysis - Univariate](#331-variables-graphical-analysis---univariate)
     - [3.3.2. Statistical Graphical Analysis - Histograms and Correlation](#332-statistical-graphical-analysis---histograms-and-correlation)
   - [3.4. Model Suitability Statistical Tests](#34-model-suitability-statistical-tests)
4. [Model Training](#4-model-training)
5. [Model Predictions](#5-model-predictions)
6. [Model Results and Evaluation](#6-model-results-and-evaluation)

---

## Overview

This project uses a synthetic dataset of **CO2 concentration** and **temperature** to demonstrate the application of linear regression. The dataset is analyzed for its structure, distribution, and suitability for linear regression. Following the model's training and evaluation, the results are interpreted statistically and visually.

---

## 1. Libraries

The following Python libraries were used in this project:
- `pandas`: For data manipulation and exploration.
- `numpy`: For numerical operations.
- `matplotlib` and `seaborn`: For data visualization.
- `statsmodels`: For detailed statistical analysis and model diagnostics.
- `scikit-learn`: For training the regression model and calculating performance metrics.

---

## 2. Data Loading and Data Description

The dataset consists of two variables:
- **CO2_concentration_ppm**: CO2 concentration in parts per million (ppm) (independent variable).
- **Temperature_Celsius**: Temperature in degrees Celsius (dependent variable).

The data is loaded from a CSV file, and the columns are described in terms of their type, metric units, and relationship (dependent or independent). 

---

## 3. Exploratory Data Analysis (EDA)

### 3.1. Dataset Structure and Data Types
The dataset contains 200 observations and 2 columns. The data types for both variables are numeric, and there are no missing or null values.

### 3.2. Basic Statistical Information
Summary statistics (mean, standard deviation, minimum, maximum, and percentiles) were calculated to understand the dataset's central tendency and spread.

### 3.3. Graphical Dataset Analysis
#### 3.3.1. Variables Graphical Analysis - Univariate
Scatter plots were created to visualize the relationship between **CO2 concentration** and **temperature**.

#### 3.3.2. Statistical Graphical Analysis - Histograms and Correlation
Histograms and correlation matrices were plotted to analyze the distribution of variables and their linear relationship.

### 3.4. Model Suitability Statistical Tests
The dataset was tested for:
- **Linearity**: Checked using residuals vs fitted values plots.
- **Normality of Residuals**: Tested using the Shapiro-Wilk test and Q-Q plots.
- **Homoscedasticity**: Tested using the Breusch-Pagan test.
- **Multicollinearity**: Assessed using the Variance Inflation Factor (VIF).

---

## 4. Model Training

The linear regression model was trained using the **ordinary least squares (OLS)** method from `statsmodels`. The training data (80% of the dataset) was used to estimate the intercept and slope of the regression line.

---

## 5. Model Predictions

Predictions were made on the test dataset (20% of the dataset) using the trained model. The predictions were then compared to the actual values for evaluation.

---

## 6. Model Results and Evaluation

### Summary of Results
- **R-squared**: The model explains 79.2% of the variance in temperature.
- **Adjusted R-squared**: 79.1%, accounting for model simplicity.
- **F-statistic and p-value**: The model is statistically significant with an F-statistic of 753.4 and a p-value of \(2.07e-69\).

### Coefficients
- **Intercept (\( \beta_0 \))**: 9.6946, representing the temperature when CO2 concentration is 0 ppm.
- **Slope (\( \beta_1 \))**: 0.0208, indicating a 0.0208°C increase in temperature for every 1 ppm increase in CO2 concentration.

### Model Diagnostics
- **Normality**: Residuals were found to be approximately normal based on the Omnibus and Jarque-Bera tests.
- **Homoscedasticity**: The Breusch-Pagan test confirmed that residual variance is constant.
- **No Autocorrelation**: Durbin-Watson statistic of 2.105 indicates no significant autocorrelation.

### Performance Metrics
- **Mean Squared Error (MSE)**: 0.193
- **Root Mean Squared Error (RMSE)**: 0.439
- **R-squared**: 0.792

---

## Conclusion

The linear regression model effectively captures the relationship between CO2 concentration and temperature. The diagnostics confirm that the assumptions of linear regression are met, and the model provides reliable predictions. The project highlights the importance of exploratory analysis, statistical testing, and diagnostic evaluation in regression modeling.
