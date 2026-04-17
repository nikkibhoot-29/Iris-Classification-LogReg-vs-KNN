# Iris Dataset Classification – Logistic Regression vs KNN

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikkibhoot-29/Iris-Classification-LogReg-vs-KN/blob/main/IRIS%20dataset%20-%20Analysis.ipynb)

## Overview

This project demonstrates a complete supervised machine learning workflow using the Iris dataset. The objective is to classify iris flower species based on sepal and petal measurements and compare the performance of two classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)

The project emphasizes how **data preparation and model choice influence performance**.

---

## Dataset

The Iris dataset contains 150 observations across three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

Features used:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The dataset is balanced across classes.

---

## Data Preparation

The dataset was prepared to ensure reliability and consistency:

- Converted features to appropriate numeric types  
- Handled missing values:
  - Mean imputation (numerical features)  
  - Mode imputation (categorical feature: *Species*)  
- Removed irrelevant column (`Unnamed: 0`)  

---

## Exploratory Data Analysis

Exploratory analysis was conducted to understand feature distributions and relationships:

- Scatter plot revealed strong class separability using petal features  
- Histograms used for distribution analysis  
- Boxplots highlighted variation across species  
- Correlation heatmap identified relationships between variables  

---

## Methodology

The following pipeline was implemented:

1. Feature-target separation  
2. Stratified train-test split  
3. Feature scaling using `StandardScaler`  
4. Model training:
   - Logistic Regression  
   - KNN (k = 5)  

---

## Evaluation Metrics

Model performance was evaluated using:

- Accuracy Score  
- Confusion Matrix  
- Classification Report  

---

## Results

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | **94.7%** |
| KNN (k = 5)         | **92.1%** |

---

## Key Insights

- Logistic Regression slightly outperformed KNN  
- The dataset exhibits **near-linear separability**  
- Feature scaling had a **significant impact on KNN performance**  
- Petal features are highly discriminative for classification  

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## Usage

You can explore and run the project directly in Google Colab:

👉 Click the **"Open in Colab"** button at the top of this README.

---

## Author

Nikki Bhoot
