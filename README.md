# 📊 Telco Customer Churn Prediction – IBM Dataset

## 🧠 Project Overview

This project focuses on predicting customer churn for a telecommunications company using structured customer data. The objective is to identify customers who are likely to leave the service and support data-driven retention strategies.

The project is implemented as a complete machine learning pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment using a Streamlit web application.

---

## 🎯 Problem Statement

Customer churn is a major challenge in the telecom industry, directly affecting revenue and customer lifetime value. The goal of this project is to build a predictive model that can classify whether a customer will churn based on behavioral and service-related features.

---

## 📂 Dataset

- **Dataset Name:** WA_Fn-UseC_-Telco-Customer-Churn.csv  
- **Source:** IBM Sample Data (Kaggle)  
- **Total Records:** 7,043 customers  
- **Target Variable:** `Churn` (Yes / No)

### 📌 Selected Features Used in Model

After preprocessing and encoding, the following features were used:

- SeniorCitizen  
- tenure  
- MonthlyCharges  
- TotalCharges  
- gender_Male  
- Partner_Yes  
- Dependents_Yes  
- PhoneService_Yes  
- MultipleLines_No phone service  
- MultipleLines_Yes  

---

## 🧹 Data Preprocessing

- Removed irrelevant column (`customerID`)
- Converted `TotalCharges` to numeric format
- Handled missing values
- Applied One-Hot Encoding for categorical variables
- Converted target variable (`Churn`) into binary format (0/1)
- Feature scaling applied where necessary for model training

---

## 📊 Exploratory Data Analysis (EDA)

Key insights discovered:

- Customers with **month-to-month contracts** have the highest churn rate  
- Higher **monthly charges** increase churn probability  
- Customers with shorter **tenure** are more likely to churn  
- Customers with fewer additional services tend to leave more often  

---

## 🤖 Machine Learning Models Used

The following models were trained and evaluated:

- Logistic Regression  
- Random Forest  
- XGBoost  

---

## 📈 Model Performance Comparison

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.804    | 0.648     | 0.575  | 0.609    |
| Random Forest       | 0.790    | 0.626     | 0.519  | 0.567    |
| XGBoost             | 0.790    | 0.619     | 0.548  | 0.582    |

---

## 🏆 Best Model

**Logistic Regression** was selected as the final model because it achieved:

- Highest Accuracy (80.38%)  
- Best Recall for churn detection (57.5%)  
- Best F1-score among all models  

---

## 🚀 Deployment

A Streamlit web application was developed to:

- Input customer details  
- Predict churn probability in real-time  
- Display risk level (Low / Medium / High)  
- Provide business recommendations  

---

## 💡 Key Business Insights

- Month-to-month contracts have the highest churn rate  
- High monthly charges increase churn risk  
- Longer customer tenure reduces churn probability  
- Customers with fewer additional services are more likely to churn  

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Streamlit  

---

## 📌 Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting customer churn. It combines data analysis, machine learning modeling, and deployment into a single interactive application that can assist businesses in improving customer retention strategies.

---

## ▶️ How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run
```bash
streamlit run app.py
