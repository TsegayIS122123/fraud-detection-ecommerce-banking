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

###  **Task 1: Data Analysis and Preprocessing ** 

#### 1. **Data Cleaning**
- E-commerce dataset: No missing values or duplicates; datetime conversion for `signup_time` and `purchase_time`.
- Credit card dataset: Removed 1,081 duplicate rows; ensured correct numeric types.
- Final cleaned sizes:
  - E-commerce: 151,112 rows
  - Credit card: 283,726 rows

#### 2. **Exploratory Data Analysis (EDA)**
- Comprehensive univariate and bivariate analysis performed.
- **E-commerce**: Fraud rate 9.36% (imbalance ratio ~9.7:1). Weak correlations; age slightly positive with fraud.
- **Credit card**: Fraud rate 0.17% (imbalance ratio ~599:1 – extreme). Strong correlations with several PCA components (e.g., V17: -0.313, V14: -0.293).
- Visualizations and JSON reports saved in `reports/`.

#### 3. **Geolocation Integration (E-commerce only)**
- Converted IP addresses to integers and merged with `IpAddress_to_Country.csv` using range-based lookup.
- Successfully mapped ~87% of transactions to 182 countries.
- Added `country` feature for country-level fraud risk analysis.

#### 4. **Feature Engineering**
- **E-commerce** (21 new features):
  - `time_since_signup` (days/hours between signup and purchase) – captures rapid fraudulent activity.
  - Time-based: `purchase_hour`, `day_of_week`, cyclical sin/cos encodings.
  - Behavioral velocity proxies and device/user patterns.
- **Credit card** (12 new features):
  - Time-to-hour conversion with cyclical encoding.
  - Log-transformed and scaled amount, high-amount flag.
  - Interaction terms between top PCA features and amount.

#### 5. **Data Transformation**
- Numerical features scaled (StandardScaler for e-commerce, RobustScaler for credit card).
- Categorical features prepared for one-hot encoding.
- Stratified train-test split (80/20) preserving original class distribution in test sets.

#### 6. **Class Imbalance Handling**
- **E-commerce**: Strategy documented – SMOTE oversampling planned for training data only (moderate imbalance).
- **Credit card**: Random undersampling applied to training set (balanced to 756 samples) due to extreme imbalance.

#### 🔑 **Key Findings and Insights**

**1. Class Imbalance Severity**
- **E-commerce Dataset**:
  - Fraud rate: **9.36%** (14,151 fraud out of 151,112 transactions)
  - Imbalance ratio: ~9.7:1 (moderate)
  - Feasible to use oversampling techniques like SMOTE
- **Credit Card Dataset**:
  - Fraud rate: **0.17%** (473 fraud out of 283,726 transactions)
  - Imbalance ratio: ~599:1 (extreme)
  - Requires aggressive handling (undersampling or anomaly detection approaches)

**2. Feature Predictiveness**
- **E-commerce**:
  - Weak linear correlations overall (highest: age ~ +0.0066 with fraud)
  - Strong non-linear signals expected in device sharing, rapid signup-to-purchase velocity, and geolocation
- **Credit Card**:
  - Several anonymized PCA features (V1–V28) show strong correlations:
    - V17 (−0.313), V14 (−0.293), V12 (−0.251), V10 (−0.207) – strongest negative correlations with fraud
    - V11 (+0.149), V4 (+0.133) – positive correlations
  - PCA transformation effectively captures underlying fraud patterns despite anonymization

**3. Geolocation Insights (E-commerce only)**
- Successfully mapped ~87% of IP addresses to 182 countries
- Top countries: United States (39%), China (8%), Japan (5%), United Kingdom (3%)
- Enables country-level fraud risk scoring as a powerful feature

**4. Behavioral and Temporal Patterns**
- **E-commerce**:
  - `time_since_signup`: Critical feature – fraudsters often transact very quickly after account creation
  - Hourly/day-of-week patterns extracted with cyclical encoding to preserve temporal continuity
  - Device and user-level transaction velocity show promise for detecting account takeover or bot activity
- **Credit Card**:
  - Transaction amount distribution differs markedly between fraud and legitimate cases
  - Engineered interaction features (e.g., high-risk PCA components × amount) expected to boost detection

**5. Data Quality**
- E-commerce: Clean dataset – no missing values or duplicates
- Credit card: Removed 1,081 duplicate rows; otherwise clean

**6. Preprocessing Strategy**
- Numerical scaling: StandardScaler (e-commerce), RobustScaler (credit card – better for outliers)
- Stratified train-test split preserving original class distribution in test sets
- Imbalance handling:
  - E-commerce: SMOTE planned on training data only
  - Credit card: Random undersampling applied to training set for balanced learning

These insights confirm that the two domains require tailored approaches: richer contextual features in e-commerce vs. strong engineered signals from PCA in credit cards.

**Next Step:** Proceed to **Task 2 – Model Building and Training** (baseline + ensemble models).

---
## 📊 **Task 2: Model Building - COMPLETED RESULTS**

### **🎯 Objectives Achieved:**
1. ✅ **Baseline Model**: Logistic Regression trained and evaluated
2. ✅ **Ensemble Model**: Random Forest with hyperparameter optimization
3. ✅ **Cross-Validation**: 5-fold stratified validation for reliable metrics
4. ✅ **Hyperparameter Tuning**: Demonstrated tuning approach with improvement
5. ✅ **Model Selection**: Best models selected with business justification

### **📈 Key Performance Results:**

#### **E-commerce Fraud Detection:**
| Model | Accuracy | Precision | Recall | F1-Score | PR-AUC | Selection |
|-------|----------|-----------|--------|----------|---------|-----------|
| Logistic Regression | 68.83% | 18.45% | 68.09% | 29.03% | 0.2943 | Baseline |
| Random Forest | 95.65% | 99.93% | 53.60% | 69.78% | 0.6176 | Good |
| **Random Forest (Tuned)** | **95.65%** | **99.93%** | **53.60%** | **69.78%** | **0.6268** | **✅ BEST** |

**Business Insight**: Random Forest makes only **1 false positive** vs 8,518 for Logistic Regression, making it 99.99% better for customer experience.

#### **Credit Card Fraud Detection:**
| Model | Accuracy | Precision | Recall | F1-Score | PR-AUC | Selection |
|-------|----------|-----------|--------|----------|---------|-----------|
| Logistic Regression | 97.30% | 5.18% | 87.37% | 9.79% | 0.5402 | Baseline |
| **Random Forest** | **98.50%** | **9.00%** | **87.37%** | **16.32%** | **0.7163** | **✅ BEST** |

**Business Insight**: Both models catch 87.37% of fraud, but Random Forest has fewer false positives (839 vs 1,518).

### **🔧 Technical Implementation:**

#### **Model Architecture:**
```python
# Random Forest for E-commerce
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)

# Logistic Regression Baseline
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# Task 3: Model Explainability - COMPLETED 

## 📊 Results Summary

### INSTRUCTION 1: Feature Importance Baseline
- **E-commerce Top Features**: time_since_signup (23.2%), time_since_signup_days (17.7%), time_since_signup_hours (15.8%)
- **Credit Card Top Features**: V14 (11.7%), V4 (11.4%), V12 (9.9%)
- **Visualizations**: Feature importance plots saved in `reports/` folder

### INSTRUCTION 2: Prediction Case Analysis
- **True Positives**: Correctly identified fraud cases analyzed
- **False Positives**: Legitimate transactions incorrectly flagged
- **False Negatives**: Missed fraud cases examined
- **Feature contributions**: Analyzed for each case type

### INSTRUCTION 3: Model Interpretation
- **Top 5 Drivers**: Identified for both models
- **Impact Direction**: Analyzed whether features increase/decrease fraud risk
- **Surprising Findings**: Documented counterintuitive patterns

### INSTRUCTION 4: Business Recommendations
- **E-commerce Recommendations**: 
  1. Implement Time-Based Risk Scoring
  2. 24/7 Fraud Monitoring
  3. Continuous Model Improvement
- **Credit Card Recommendations**: 
  1. Focus on Key Indicators (V14)
  2. 24/7 Fraud Monitoring
  3. Continuous Model Improvement





