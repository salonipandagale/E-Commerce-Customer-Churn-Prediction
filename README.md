# E-Commerce Customer Churn Prediction

## Live Demo

Live Application: 


## Overview

This project focuses on analyzing and predicting customer churn for an e-commerce business.

The project combines SQL, Python, statistical analysis, exploratory data analysis, machine learning and Power BI to identify customer behavior associated with churn and build a predictive model that can identify customers at higher risk of leaving.

The final machine learning model is deployed as an interactive Streamlit application.

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

The dataset contains 5,630 customer records and 20 variables related to customer behavior, transactions and demographics.

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
- Exploratory data analysis
- Statistical hypothesis testing
- Feature preparation
- Machine learning
- Model evaluation
- Feature importance analysis
- Customer churn insights
- Business recommendations
- Streamlit deployment

## Data Analysis

The analysis identified several important patterns.

### Customer Tenure

New customers showed a substantially higher churn rate than existing customers.

- New customers: 32.42% churn rate
- Existing customers: 7.21% churn rate

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

Chi-square tests were used for categorical variables, while independent sample t-tests were used for numerical comparisons.

## Machine Learning

Three classification models were developed and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The models were evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Since churn is the minority class, churn recall and F1-score were given more importance than accuracy alone.

## Model Performance

| Model | Churn Precision | Churn Recall | Churn F1 | ROC-AUC |
|------|----------------:|-------------:|---------:|--------:|
| Logistic Regression | 0.74 | 0.52 | 0.61 | 0.8862 |
| Random Forest | 1.00 | 0.85 | 0.92 | 0.9987 |
| XGBoost | 0.93 | 0.83 | 0.88 | 0.9915 |

Random Forest was selected as the final model based on its overall performance.

The model achieved:

- Churn recall: 85%
- Churn F1-score: 0.92
- ROC-AUC: 0.9987

## Feature Importance

The most important features in the Random Forest model included:

- Tenure
- Cashback amount
- Warehouse-to-home distance
- Customer complaints
- Days since last order
- Number of addresses
- Order amount increase
- Satisfaction score

Tenure was the most important feature in the final model.

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

The final Random Forest pipeline is integrated into a Streamlit application.
The application allows users to:
- Enter customer information.
- Generate a churn prediction.
- View the predicted churn probability.
- Identify whether the customer is classified as high or low churn risk.

## Project Structure

```text
ecommerce-customer-churn-prediction/
│
├── README.md
├── app.py
├── requirements.txt
├── .python-version
├── churn_random_forest_pipeline.pkl
│
├── data/
│   ├── ecommerce_churn.csv
│   └── cleaned_churn_data.csv
│
├── notebooks/
│   ├── churn_prediction.ipynb
│   └── ecommerce_churn_insights.ipynb
│
└── SQL analysis and PowerBI/
    └── powerbi_dashboard.pdf
    ├── ecommerce_customer_churn.sql
