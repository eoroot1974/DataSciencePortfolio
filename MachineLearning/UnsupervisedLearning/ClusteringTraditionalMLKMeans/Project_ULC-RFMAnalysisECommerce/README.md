# Customer Segmentation with KMeans: RFM-Based Clustering in E-Commerce

This project applies **KMeans clustering** to segment customers based on their **Recency**, **Frequency**, and **Monetary (RFM)** behavior. It demonstrates how unsupervised machine learning can uncover distinct customer groups for targeted marketing and CRM strategy, using a complete RFM pipeline from raw transaction data to business interpretation.

## Table of Contents

1. [Overview](#1-overview)  
2. [Libraries](#2-libraries)  
3. [Data Preparation and Transformation](#3-data-preparation-and-transformation)  
   - 3.1. RFM Variable Construction  
   - 3.2. Outlier Treatment and Scaling  
4. [Clustering Process](#4-clustering-process)  
   - 4.1. Normalized Dataset and Elbow Method  
   - 4.2. KMeans with K=3 and K=4  
   - 4.3. Cluster Labeling and Denormalization  
5. [Segment Profiling](#5-segment-profiling)  
   - 5.1. Cluster Centers (K=3 and K=4)  
   - 5.2. Business Interpretation  
6. [Visualization](#6-visualization)  
7. [Conclusion](#7-conclusion)  
8. [Usage](#8-usage)  
9. [Requirements](#9-requirements)

---

## 1. Overview

This unsupervised learning project aims to **identify patterns of customer behavior** through the well-known RFM model. By clustering customers using KMeans, we uncover distinct groups such as high spenders, recent buyers, dormant users, and potential churners. The clustering model is evaluated at **K=3** and **K=4**, and results are interpreted in both technical and business terms.

This project demonstrates how customer analytics can be automated and scaled using Python, allowing for **data-driven marketing, retention strategies, and CRM segmentation**.

---

## 2. Libraries

The following Python libraries were used:

- `pandas`: dataset creation and merging  
- `numpy`: statistical operations  
- `matplotlib`, `seaborn`: exploratory plots and cluster visualizations  
- `sklearn`: KMeans, scaling, silhouette scores, MinMaxScaler  

---

## 3. Data Preparation and Transformation

### 3.1 RFM Variable Construction

From raw transaction data, we computed:

- **Recency**: Days since last purchase  
- **Frequency**: Total number of purchases  
- **Monetary**: Total purchase amount  

### 3.2 Outlier Treatment and Scaling

- Outliers were treated using **bi-weight transformation** to reduce skewness.  
- Data was then scaled using **MinMaxScaler** to prepare for KMeans clustering.

---

## 4. Clustering Process

### 4.1 Normalized Dataset and Elbow Method

We applied the **Elbow Method** and **Silhouette Analysis** to determine optimal cluster numbers. Clustering was then run with both **K=3** and **K=4** to compare segmentation granularity.

### 4.2 KMeans with K=3 and K=4

KMeans was applied on the normalized RFM data, and cluster labels were assigned:
- `rfmdata_bw_S_3C.csv` for K=3 results  
- `rfmdata_bw_S_4C.csv` for K=4 results  

### 4.3 Cluster Labeling and Denormalization

Cluster labels were **merged back** with the **original unscaled RFM data** to compute real-world averages for Recency, Frequency, and Monetary metrics. This allowed for interpretation in business terms.

---

## 5. Segment Profiling

### 5.1 Cluster Centers (K=3 and K=4)

Each cluster was described using:
- Number of customers  
- Mean Recency, Frequency, and Monetary values  
- Behavioral patterns  

### 5.2 Business Interpretation

Clusters were profiled as:

- **High-value loyalists**: Frequent, high-spending, recent buyers  
- **At-risk or dormant users**: Long recency, low spending  
- **Potential loyalists**: Good spenders but not frequent  
- **Recent new users**: Low frequency but high recency

These labels guided recommendations for **targeted campaigns, promotions, and retention strategies**.

---

## 6. Visualization

Scatterplots of clusters in RFM dimensions were generated to validate separation and interpret group characteristics visually. Cluster centroids and boundaries were also plotted.

---

## 7. Conclusion

This clustering project showcases the power of **unsupervised segmentation** in e-commerce using the RFM model. By choosing different `K` values and profiling the results:
- Businesses can identify **high-value** and **at-risk** customers  
- Improve **retention strategies**  
- Prioritize **personalized marketing**

The pipeline is reproducible and extendable to other datasets and industries.

---

## 8. Usage

Clone the repository and run the analysis in a Jupyter Notebook:

```bash
git clone https://github.com/eoroot1974/DataSciencePortfolio.git
