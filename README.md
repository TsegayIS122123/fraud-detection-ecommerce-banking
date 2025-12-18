# Fraud Detection System for E-commerce and Banking Transactions

## 📋 Project Overview

This project aims to develop advanced fraud detection models for both e-commerce and bank credit transactions. The system leverages machine learning algorithms, geolocation analysis, and transaction pattern recognition to identify fraudulent activities with high accuracy while minimizing false positives.

### **Business Context**
**Adey Innovations Inc.**, a leading financial technology company wants to enhance transaction security by building robust fraud detection models that balance security requirements with user experience.

## 🎯 **Objectives**

### **Primary Goals:**
1. **Improve Detection Accuracy**: Develop models with high precision and recall for fraud detection
2. **Handle Class Imbalance**: Implement techniques to address extreme imbalance in fraud data (typically <1% fraud cases)
3. **Provide Explainability**: Use SHAP to interpret model decisions and provide actionable insights
4. **Real-time Feasibility**: Design models suitable for potential real-time deployment

### **Specific Deliverables:**
- EDA reports for both e-commerce and credit card datasets
- Feature-engineered datasets with derived features
- Trained machine learning models with evaluation metrics
- SHAP analysis and business recommendations

---

## Methodology
The project follows a structured data science workflow:

1. **Exploratory Data Analysis (EDA)**
   - Understand data distributions and class imbalance
   - Identify patterns related to fraudulent behavior

2. **Feature Engineering**
   - Create time-based and behavioral features
   - Integrate geolocation data using IP mapping

3. **Preprocessing**
   - Handle missing values and duplicates
   - Encode categorical variables
   - Scale numerical features
   - Address class imbalance using resampling techniques

4. **Modeling**
   - Train baseline Logistic Regression models
   - Train ensemble models (Random Forest / Gradient Boosting)
   - Evaluate using AUC-PR, F1-score, and confusion matrix

5. **Model Explainability**
   - Use SHAP to interpret model predictions
   - Identify key drivers of fraud
   - Provide actionable business insights

---

## Evaluation Metrics
Due to severe class imbalance, model performance is evaluated using:
- Precision-Recall AUC
- F1-Score
- Confusion Matrix

---

## Tools and Technologies
- Python
- pandas, numpy
- scikit-learn
- imbalanced-learn
- SHAP
- Matplotlib / Seaborn
- Jupyter Notebook

## Deliverables
- Cleaned and feature-engineered datasets
- Trained and evaluated fraud detection models
- Explainable AI visualizations using SHAP



