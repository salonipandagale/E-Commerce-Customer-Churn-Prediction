# E-Commerce Customer Churn Prediction

## Live Demo

**Live Application:** https://e-commerce-customer-churn-prediction-pqmy.onrender.com/

---

## Overview

This project focuses on analyzing and predicting customer churn for an e-commerce business.

The project combines **SQL, Python, statistical analysis, exploratory data analysis, machine learning, and Power BI** to identify customer behavior associated with churn and build a predictive model that can identify customers at higher risk of leaving.

The final machine learning model is a **tuned XGBoost pipeline** that includes preprocessing and is deployed as an interactive Streamlit application.

---

## Objectives

* Analyze customer behavior and identify patterns associated with churn.
* Perform data cleaning and validation using SQL and Python.
* Explore relationships between customer characteristics and churn.
* Use statistical hypothesis testing to validate important observations.
* Build and compare multiple machine learning models.
* Tune the final XGBoost model using randomized hyperparameter search.
* Evaluate model performance using stratified cross-validation and nested cross-validation.
* Investigate the effect of duplicate feature rows on model performance.
* Analyze the precision-recall trade-off at different classification thresholds.
* Identify important factors associated with churn prediction.
* Develop business recommendations for customer retention.
* Deploy the final churn prediction model as a web application.

---

## Dataset

The dataset contains **5,630 customer records and 20 variables** related to customer behavior, transactions, and demographics.

The target variable is `Churn`.

* Non-churn customers: **4,682**
* Churned customers: **948**
* Churn rate: **16.8%**

Important features include:

* Customer tenure
* Preferred login device
* City tier
* Warehouse-to-home distance
* Preferred payment mode
* Gender
* App usage
* Number of registered devices
* Preferred order category
* Satisfaction score
* Marital status
* Number of addresses
* Customer complaints
* Order amount increase
* Coupon usage
* Order count
* Days since last order
* Cashback amount

`CustomerID` is used only as an identifier and is excluded from model training.

---

## Project Workflow

1. Data understanding
2. Data quality analysis
3. Missing value analysis
4. Missingness and churn analysis
5. Data error investigation
6. Category standardization
7. Duplicate record analysis
8. Exploratory data analysis
9. Statistical hypothesis testing
10. Feature preparation
11. Machine learning model comparison
12. Cross-validated model evaluation
13. Duplicate robustness analysis
14. XGBoost hyperparameter tuning
15. Nested cross-validation
16. Threshold analysis
17. Feature importance analysis
18. Customer churn insights
19. Business recommendations
20. Streamlit deployment

---

## Data Quality

### Missing Values

Seven columns contained missing values, with approximately 4.5%–5.5% missingness:

* `Tenure`
* `WarehouseToHome`
* `HourSpendOnApp`
* `OrderAmountHikeFromlastYear`
* `CouponUsed`
* `OrderCount`
* `DaySinceLastOrder`

Missingness was also analyzed in relation to churn before imputation.

Missing values are handled inside the machine learning preprocessing pipeline:

* Numerical features → median imputation
* Categorical features → most-frequent imputation
* Categorical features → one-hot encoding

This ensures that preprocessing is fitted within the model pipeline rather than being performed using the entire dataset before cross-validation.

### Category Standardization

Inconsistent payment-mode labels were standardized:

* `CC` → `Credit Card`
* `COD` → `Cash on Delivery`

The source categories `Phone` and `Mobile Phone` for login device, and `Mobile` and `Mobile Phone` for order category, were retained as separate categories because they belong to different variables.

### Data Errors

Two implausible `WarehouseToHome` values, **126 and 127**, were identified as likely data-entry errors and corrected to **26 and 27** respectively.

The records were retained rather than removed.

### Duplicate Feature Rows

No complete duplicate rows exist when `CustomerID` is included.

However, **556 rows (9.9%)** were found to have identical values across all model features and the same churn label.

These rows were retained in the main dataset, and model performance was re-evaluated after removing them as a robustness check.

---

## Data Analysis

The exploratory analysis identified several patterns associated with customer churn.

### Customer Tenure

New customers showed a substantially higher observed churn rate than customers with longer tenure.

* Tenure ≤ 6 months: **32.42% churn**
* Tenure > 6 months: **5.29% churn**
* Missing tenure: **30.68% churn**

This suggests that the early customer lifecycle is an important period to consider when designing retention strategies.

### Customer Complaints

Customers who reported complaints had a higher observed churn rate.

* Customers with complaints: **31.67%**
* Customers without complaints: **10.93%**

This highlights complaint resolution as an area worth monitoring for retention.

### Warehouse-to-Home Distance

Observed churn increased across warehouse-to-home distance groups.

| Warehouse-to-Home Distance | Churn Rate |
| -------------------------- | ---------: |
| ≤ 10                       |     13.51% |
| 11–20                      |     15.67% |
| 21–30                      |     20.14% |
| > 30                       |     20.90% |

### Preferred Order Category

Mobile-related order categories showed higher observed churn rates.

| Preferred Order Category | Churn Rate |
| ------------------------ | ---------: |
| Mobile Phone             |     27.54% |
| Mobile                   |     27.19% |
| Fashion                  |     15.50% |
| Laptop & Accessory       |     10.24% |
| Others                   |      7.58% |
| Grocery                  |      4.88% |

These are observed relationships in the dataset and should not be interpreted as causal effects.

---

## Statistical Analysis

Hypothesis testing was performed to determine whether selected observed differences were statistically significant.

Statistically significant relationships with churn were found for:

* Customer tenure
* Satisfaction score
* Preferred order category
* Preferred payment mode

The following tests were used:

* **Welch's independent-samples t-test** for numerical comparisons
* **Chi-square test of independence** for categorical variables

One counterintuitive result was observed for satisfaction score: churned customers had a slightly higher average satisfaction score than retained customers, approximately **3.39 versus 3.00**.

This result was not used as a basis for business recommendations. Statistical significance indicates an association in this dataset and does not establish causation.

---

## Machine Learning

Three classification models were developed and evaluated:

* Logistic Regression
* Random Forest
* XGBoost

All models were implemented as scikit-learn pipelines containing the required preprocessing steps.

### Preprocessing

Numerical features use:

* Median imputation
* Standard scaling for Logistic Regression

Categorical features use:

* Most-frequent imputation
* One-hot encoding

Unknown categorical values are handled using `handle_unknown="ignore"`.

### Class Imbalance

The dataset contains approximately **16.8% churned customers** and **83.2% non-churned customers**.

Instead of artificially changing the evaluation distribution, class imbalance was handled during model training using class weighting / positive-class weighting.

Model evaluation focuses on:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

PR-AUC and churn recall are particularly useful because churn is the minority class.

---

## Model Performance

The models were evaluated using **stratified 5-fold cross-validation**.

Results are reported as mean ± standard deviation across folds.

| Model               |     Precision |        Recall |            F1 |       ROC-AUC |        PR-AUC |
| ------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Logistic Regression | 0.461 ± 0.023 | 0.812 ± 0.022 | 0.588 ± 0.023 | 0.890 ± 0.004 | 0.700 ± 0.017 |
| Random Forest       | 0.832 ± 0.022 | 0.840 ± 0.042 | 0.835 ± 0.030 | 0.981 ± 0.003 | 0.923 ± 0.016 |
| XGBoost             | 0.814 ± 0.021 | 0.914 ± 0.024 | 0.861 ± 0.019 | 0.984 ± 0.002 | 0.933 ± 0.013 |

XGBoost showed the strongest overall performance among the initial models and was selected for further hyperparameter tuning.

---

## Duplicate Robustness Check

Because 556 rows had identical model features and churn labels, model performance was also evaluated after removing these duplicate feature rows.

| Model               | ROC-AUC | Recall | Precision |    F1 |
| ------------------- | ------: | -----: | --------: | ----: |
| Logistic Regression |   0.890 |  0.810 |     0.457 | 0.584 |
| Random Forest       |   0.976 |  0.829 |     0.831 | 0.830 |
| XGBoost             |   0.982 |  0.916 |     0.836 | 0.874 |

The model performance remained strong after deduplication.

This provides a robustness check against the possibility that identical feature rows were substantially inflating the model's performance.

---

## XGBoost Hyperparameter Tuning

The XGBoost model was tuned using `RandomizedSearchCV` with **40 parameter combinations**.

The search optimized the following parameters:

| Parameter          | Best Value |
| ------------------ | ---------: |
| `n_estimators`     |        517 |
| `max_depth`        |          6 |
| `learning_rate`    |      0.093 |
| `subsample`        |      0.985 |
| `colsample_bytree` |      0.688 |
| `min_child_weight` |          1 |
| `reg_lambda`       |      4.387 |

The tuning objective was **PR-AUC (average precision)**.

---

## Nested Cross-Validation

Nested cross-validation was used to separate hyperparameter tuning from model evaluation.

The inner cross-validation loop was used for hyperparameter selection, while the outer loop was used to estimate generalization performance.

### Results

* **Nested PR-AUC:** `0.967 ± 0.011`
* **Deduplicated Nested PR-AUC:** `0.958 ± 0.014`

The relatively small change after deduplication indicates that duplicate feature rows did not account for the majority of the observed model performance.

---

## Final Model

The final model is a **tuned XGBoost pipeline** containing both preprocessing and the XGBoost classifier.

Using out-of-fold predictions from the nested cross-validation procedure, the final model achieved:

* **ROC-AUC:** `0.9907`
* **Precision at threshold 0.5:** `0.946`
* **Recall at threshold 0.5:** `0.930`
* **F1-score at threshold 0.5:** `0.938`

The ROC-AUC value measures the model's ability to discriminate between churned and non-churned customers across classification thresholds. It should not be interpreted as classification accuracy.

---

## Decision Threshold Analysis

Different classification thresholds were evaluated using out-of-fold predictions from the tuned XGBoost model.

| Threshold | Recall | Precision |
| --------: | -----: | --------: |
|       0.3 |  0.934 |     0.913 |
|       0.4 |  0.931 |     0.930 |
|       0.5 |  0.930 |     0.946 |
|       0.6 |  0.926 |     0.952 |
|       0.7 |  0.919 |     0.959 |

Increasing the threshold slightly decreases recall while increasing precision.

The appropriate threshold depends on the business cost of contacting a customer who may not churn versus the cost of missing a customer who is likely to churn.

---

## Feature Importance

During model exploration, feature importance was calculated from the Random Forest model using impurity-based feature importance.

Important features included:

* Tenure
* Cashback amount
* Warehouse-to-home distance
* Customer complaints
* Days since last order
* Number of addresses
* Order amount increase
* Satisfaction score

Tenure was the most important feature in the Random Forest analysis.

These importance values describe the contribution of features to model predictions and should not be interpreted as evidence that the features cause churn.

Feature importance was not recalculated for the final XGBoost model.

---

## Business Recommendations

Based on the observed relationships in the dataset, the following retention strategies can be considered:

* Strengthen onboarding and engagement for new customers.
* Prioritize complaint resolution and follow-up.
* Monitor customers located farther from warehouses.
* Develop targeted retention strategies for higher-churn product categories.
* Monitor customer segments with higher observed churn across payment methods.
* Use model-generated churn scores to identify customers who may require proactive retention efforts.

These recommendations are based on observational patterns and should be validated further before being treated as causal conclusions.

---

## Streamlit Application

The final tuned XGBoost pipeline is integrated into a Streamlit application.

The application allows users to:

* Enter customer information using quick or detailed input modes.
* Generate a churn risk score.
* Identify whether the customer falls into a low, medium, or high churn-risk band.

The application currently uses the following risk bands:

| Churn Score | Risk Band   |
| ----------: | ----------- |
|       < 30% | Low Risk    |
|     30%–50% | Medium Risk |
|       ≥ 50% | High Risk   |

The churn score is a model output and **is not a calibrated probability**.

---

## Limitations

* The dataset is a single static snapshot rather than a longitudinal customer dataset.
* The models were evaluated primarily using random stratified cross-validation rather than a time-based split, so performance on future production data may differ.
* Model performance is relatively high for this dataset. Exact duplicate feature rows were investigated and had limited effect on performance, but near-duplicate records were not examined.
* The analysis is observational and does not establish causal relationships between customer characteristics and churn.
* The churn score is not calibrated as a probability.
* Threshold performance was evaluated using out-of-fold predictions from the cross-validation procedure and should be validated on a separate untouched dataset before production threshold selection.
* Feature importance was based on the Random Forest model and was not recalculated for the final XGBoost model.

---

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

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **SciPy**
* **Scikit-learn**
* **XGBoost**
* **Joblib**
* **SQL**
* **Power BI**
* **Streamlit**

---

## Conclusion

This project demonstrates an end-to-end customer churn prediction workflow, from data cleaning and exploratory analysis to statistical testing, machine learning, model validation, hyperparameter tuning, and deployment.

The final tuned XGBoost pipeline achieved strong cross-validated performance while additional duplicate and nested-validation checks were used to assess the robustness of the results.

The deployed Streamlit application demonstrates how the trained model can be used to generate churn-risk scores that can support customer-retention analysis.
