# E-Commerce Customer Churn Prediction

## Live Demo

Live Application:  https://e-commerce-customer-churn-prediction-pqmy.onrender.com/


## Overview

This project focuses on analyzing and predicting customer churn for an e-commerce business.

The project combines SQL, Python, statistical analysis, exploratory data analysis, machine learning and Power BI to identify customer behavior associated with churn and build a predictive model that can identify customers at higher risk of leaving.

The final machine learning model, a tuned XGBoost pipeline, is deployed as an interactive Streamlit application.

## Objectives

- Analyze customer behavior and identify patterns associated with churn.
- Perform data cleaning and validation using SQL and Python.
- Explore relationships between customer characteristics and churn.
- Use statistical hypothesis testing to validate important observations.
- Build and compare multiple machine learning models.
- Identify the most important factors contributing to churn prediction.
- Develop business recommendations for customer retention.
- Deploy the final churn prediction model as a web application.

## Dataset

The dataset contains 5,630 customer records and 20 variables related to customer behavior, transactions and demographics. About 16.8% of customers (948) churned.

Important features include:

- Customer tenure
- Preferred login device
- City tier
- Warehouse-to-home distance
- Preferred payment mode
- Gender
- App usage
- Number of registered devices
- Preferred order category
- Satisfaction score
- Marital status
- Number of addresses
- Customer complaints
- Order amount increase
- Coupon usage
- Order count
- Days since last order
- Cashback amount

The target variable is `Churn`.

## Project Workflow

- Data understanding
- Data quality analysis
- Missing value analysis
- Missingness and churn analysis
- Data error investigation
- Category standardization
- Duplicate record analysis
- Exploratory data analysis
- Statistical hypothesis testing
- Feature preparation
- Machine learning
- Cross-validated model evaluation
- Hyperparameter tuning with nested cross-validation
- Feature importance analysis
- Customer churn insights
- Business recommendations
- Streamlit deployment

## Data Quality

- **Missing values:** seven columns contained missing values (about 4.5% to 5.5% each): Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount and DaySinceLastOrder. Missingness was significantly associated with churn for six of the seven columns, so it was examined before imputation. Missing values are imputed inside the modelling pipeline (median for numeric, most frequent for categorical), fitted on training data only.
- **Category standardization:** inconsistent payment-mode labels were merged (`CC` to Credit Card, `COD` to Cash on Delivery). The source labels `Phone` and `Mobile Phone` (login device) and `Mobile` and `Mobile Phone` (order category) were kept as separate categories.
- **Data errors:** two implausible WarehouseToHome values (126 and 127) were treated as likely typing errors and corrected to 26 and 27 rather than removing the customers.
- **Duplicates:** no duplicate rows exist when `CustomerID` is included, but 556 rows (9.9%) are identical to another row on all model features, with the same churn label. They were kept in the data. Model performance was re-checked after removing them (see Model Performance).

## Data Analysis

The analysis identified several important patterns.

### Customer Tenure

New customers (tenure of 6 months or less) showed a substantially higher churn rate than existing customers.

- New customers (tenure ≤ 6 months): 32.42% churn rate
- Existing customers (tenure > 6 months): 5.29% churn rate
- Customers with missing tenure (analysed separately): 30.68% churn rate

This indicates that the early stage of the customer lifecycle is an important period for retention.

### Customer Complaints

Customers who reported complaints had a considerably higher observed churn rate.

- Customers with complaints: 31.67%
- Customers without complaints: 10.93%

This highlights the importance of complaint resolution and customer service.

### Warehouse-to-Home Distance

Churn rate increased as warehouse-to-home distance increased.

- Distance ≤10: 13.51%
- Distance 11–20: 15.67%
- Distance 21–30: 20.14%
- Distance >30: 20.90%

### Preferred Order Category

Mobile-related categories showed higher observed churn rates.

- Mobile Phone: 27.54%
- Mobile: 27.19%
- Fashion: 15.50%
- Laptop & Accessory: 10.24%
- Others: 7.58%
- Grocery: 4.88%

## Statistical Analysis

Hypothesis testing was performed to determine whether observed differences were statistically significant.

The analysis found statistically significant relationships between churn and:

- Customer tenure
- Satisfaction score
- Preferred order category
- Preferred payment mode

Chi-square tests were used for categorical variables, while independent sample t-tests (Welch's, which does not assume equal variances) were used for numerical comparisons.

Churned customers had a slightly higher average satisfaction score than retained customers (about 3.39 versus 3.00), a counterintuitive pattern that was not investigated further and was not used as a basis for recommendations. Statistical significance shows an association in this dataset, not causation.

## Machine Learning

Three classification models were developed and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

All models were built as scikit-learn pipelines that contain the preprocessing (imputation and one-hot encoding), so preprocessing is fitted only on training data. Class imbalance was handled with class weighting.

The models were evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (average precision)
- Confusion Matrix

Since churn is the minority class, churn recall, F1-score and PR-AUC were given more importance than accuracy alone.

An initial single 80/20 hold-out split gave very high scores for Random Forest (ROC-AUC 0.9987, precision 1.00). Because a single split with only 190 churned customers in the test set can be optimistic, the models were re-evaluated with stratified 5-fold cross-validation, and those results are reported below.

## Model Performance

Stratified 5-fold cross-validation, churn class (mean ± standard deviation across folds):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|----------:|-------:|---:|--------:|-------:|
| Logistic Regression | 0.461 ± 0.023 | 0.812 ± 0.022 | 0.588 ± 0.023 | 0.890 ± 0.004 | 0.700 ± 0.017 |
| Random Forest | 0.832 ± 0.022 | 0.840 ± 0.042 | 0.835 ± 0.030 | 0.981 ± 0.003 | 0.923 ± 0.016 |
| XGBoost (default settings) | 0.814 ± 0.021 | 0.914 ± 0.024 | 0.861 ± 0.019 | 0.984 ± 0.002 | 0.933 ± 0.013 |
| **XGBoost (tuned, final model)** | 0.946 | 0.930 | 0.938 | 0.9907 | 0.967 ± 0.011 |

Notes:

- The tuned XGBoost row uses out-of-fold predictions at the default 0.5 cutoff for precision, recall, F1 and ROC-AUC, and nested cross-validation for PR-AUC.
- Hyperparameters were tuned with randomized search (40 iterations) inside nested cross-validation, so the tuning did not see the evaluation folds.
- Random Forest and Logistic Regression were evaluated with fixed settings and were not hyperparameter-tuned.
- After removing the 556 duplicate-feature rows, results were very similar: ROC-AUC of 0.890 (Logistic Regression), 0.976 (Random Forest) and 0.982 (XGBoost), and a nested PR-AUC of 0.958 ± 0.014 for tuned XGBoost.

XGBoost was selected as the final model because it had the highest cross-validated PR-AUC and recall, and the tuned version improved on it further. Random Forest and XGBoost were close in ROC-AUC.

Final tuned parameters: `n_estimators=517`, `max_depth=6`, `learning_rate≈0.093`, `subsample≈0.985`, `colsample_bytree≈0.688`, `min_child_weight=1`, `reg_lambda≈4.39`.

### Decision Threshold (tuned XGBoost, out-of-fold)

| Threshold | Recall | Precision |
|----------:|-------:|----------:|
| 0.3 | 0.934 | 0.913 |
| 0.4 | 0.931 | 0.930 |
| 0.5 | 0.930 | 0.946 |
| 0.6 | 0.926 | 0.952 |
| 0.7 | 0.919 | 0.959 |

Recall and precision change little between thresholds because the model scores most customers close to 0 or 1. The best cutoff in practice depends on the cost of a retention offer compared with the value of a retained customer.

## Feature Importance

During model exploration, feature importance was calculated from the Random Forest model (impurity-based importance). The most important features included:

- Tenure
- Cashback amount
- Warehouse-to-home distance
- Customer complaints
- Days since last order
- Number of addresses
- Order amount increase
- Satisfaction score

Tenure was the most important feature in that model. Importance was not recalculated for the final XGBoost model. Feature importance shows how much a feature helps prediction, not that it causes churn.

## Business Recommendations

Based on the analysis, the following actions can be considered:

- Focus on stronger onboarding and engagement for new customers.
- Prioritize complaint resolution and follow-up.
- Monitor customers located farther from warehouses.
- Develop targeted retention strategies for high-churn product categories.
- Monitor customer segments with higher observed churn across payment methods.
- Use the machine learning model to identify customers who may require proactive retention efforts.

These recommendations are based on observed relationships in the dataset and should be further validated before being treated as causal conclusions.

## Streamlit Application

The final tuned XGBoost pipeline is integrated into a Streamlit application.
The application allows users to:
- Enter customer information (quick or detailed mode).
- Generate a churn risk score.
- Identify whether the customer falls in the low, medium or high churn risk band.

Risk bands are based on the model's churn score: below 30% is low risk, 30% to 50% is medium risk and 50% or above is high risk. The score is a model output and is not a calibrated probability.

## Limitations

- The data is a single static snapshot, and models were validated with random cross-validation rather than a time-based split, so performance on live data is likely to be lower.
- The scores are unusually high for churn data. Exact duplicate rows were checked and had little effect, but near-duplicate records were not tested.
- All findings are observational and do not prove that any factor causes churn.
- The churn score is not calibrated, and thresholds were examined on the same cross-validated data used for evaluation.

## Project Structure

```text
ecommerce-customer-churn-prediction/
│
├── README.md
├── app.py
├── requirements.txt
├── .python-version
├── churn_xgboost_pipeline.pkl
│
├── data/
│   ├── ecommerce_churn.csv
│   └── cleaned_churn_data.csv
│
├── notebooks/
│   ├── churn_prediction.ipynb
│   └── ecommerce_churn_insights.ipynb
│
└── SQL analysis and PowerBI dashboard/
    ├── powerbi_dashboard.pdf
    └── ecommerce_customer_churn.sql
```
