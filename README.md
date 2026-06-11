# 💳 Credit Card Fraud Detection System

## Overview

This project uses Machine Learning to detect fraudulent credit card transactions.

The objective is to minimize financial losses by identifying suspicious transactions while maintaining a balance between fraud detection and false alarms.

---

## Business Problem

Credit card fraud causes significant financial losses every year.

A fraud detection system must:

* Detect fraudulent transactions
* Minimize false positives
* Support real-time decision making

---

## Dataset

Source:
Kaggle Credit Card Fraud Detection Dataset

* 284,807 Transactions
* 492 Fraud Cases
* Highly Imbalanced Dataset

Features:

* Time
* V1 – V28
* Amount
* Class (Target)

---

## Machine Learning Pipeline

1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Scaling
4. SMOTE for Class Imbalance
5. Model Training
6. Model Evaluation
7. Streamlit Deployment

---

## Models Evaluated

| Model               | Precision | Recall | F1 Score |
| ------------------- | --------- | ------ | -------- |
| Logistic Regression | 0.06      | 0.92   | 0.11     |
| Random Forest       | 0.42      | 0.85   | 0.56     |
| XGBoost             | 0.78      | 0.84   | 0.81     |

---

## Selected Model

XGBoost

Reasons:

* Highest Precision
* Strong Recall
* Best F1 Score
* Best overall balance

---

## Features

* Batch Fraud Detection
* CSV Upload
* Fraud Probability Prediction
* KPI Dashboard
* Feature Importance Analysis
* Download Results

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* XGBoost
* Streamlit
* Plotly

---

## Author

Atharva Basudkar
