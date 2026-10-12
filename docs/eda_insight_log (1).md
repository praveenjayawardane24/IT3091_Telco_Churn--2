# EDA Insight Log

**Project:** IT3091 — Telco Customer Churn Prediction  
**Dataset:** IBM Telco Customer Churn  
**Records:** 7,043  
**Overall churn rate:** 26.54% (1,869 churners; 5,174 non-churners)

This document preserves the exploratory findings already recorded by the group and connects them to modeling and business interpretation. **EDA describes associations, not causes.** The Gradient Boosting notebook was recently revised; final model-specific conclusions should be refreshed from the new results.

## 1. Dataset Understanding and Data Quality

- The dataset contains customer account, service, billing, and churn information.
- `Churn` is the binary target variable.
- `customerID` is an identifier and is excluded from model features.
- There are 19 original predictors after excluding the identifier and target.
- `TotalCharges` contains 11 blanks; all occur for customers with `tenure = 0`. The project converts the column to numeric and fills those blanks with zero.
- No numerical outliers were detected using the group's 1.5×IQR screening rule. This does not mean all observations are error-free, and no rows were removed solely because of IQR screening.

**Modeling implication:** Preserve all customers, clean `TotalCharges`, and fit transformations inside training-only pipelines.

## 2. Churn and Class Imbalance

| Outcome | Customers | Share |
|---|---:|---:|
| No churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |
| Total | 7,043 | 100.00% |

**Insight:** Churn is the minority class. A classifier that predicts no churn for everyone could obtain approximately 73.46% accuracy while failing to identify any churners.

**Modeling implication:** Prioritize churn-class F1 and inspect recall, precision, ROC-AUC, and confusion matrices, not accuracy alone. Use stratified splitting and CV.

## 3. Contract Type

- **Month-to-month:** 42.71% churn.
- **Two-year:** 2.83% churn.

**Insight:** Month-to-month customers show a much higher observed churn rate than customers on two-year contracts.

**Modeling implication:** Keep contract type as a categorical predictor.

**Business implication:** Consider reviewing retention support and contract options for high-risk month-to-month customers. The association does not prove that changing contract type causes lower churn.

## 4. Customer Tenure

- **Tenure 0–12 months:** 47.44% churn.
- **Tenure 49–72 months:** 9.51% churn.

**Insight:** Recently joined customers are more likely to churn in this dataset than long-tenure customers.

**Modeling implication:** Retain numeric `tenure` and investigate the engineered `TenureGroup` feature.

**Business implication:** Evaluate onboarding and early-customer support initiatives.

## 5. Internet Service

- **Fiber optic customers:** 41.89% churn.

**Insight:** Fiber optic customers show an elevated descriptive churn rate in the recorded EDA.

**Modeling implication:** Keep `InternetService` as a categorical predictor.

**Business implication:** Investigate customer experience and pricing concerns before designing an intervention. The observed association alone does not identify the reason for churn.

## 6. Payment Method

- **Electronic check customers:** 45.29% churn.

**Insight:** Electronic-check users show a relatively high observed churn rate.

**Modeling implication:** Keep `PaymentMethod` as a categorical predictor.

**Business implication:** Review payment friction and communication as possible research topics; do not assume the payment method itself causes churn.

## 7. Monthly Charges

- **Average `MonthlyCharges` among churners:** 74.44.
- **Average `MonthlyCharges` among retained customers:** 61.27.

**Insight:** Customers who churned paid more per month on average in this dataset.

**Modeling implication:** Retain `MonthlyCharges` as a numeric predictor. Its relationship with service type and service count should be considered during interpretation.

**Business implication:** Price/value perception is a possible area for further investigation, not a proven cause.

## 8. Technical Support

- **Without technical support:** 41.64% churn.
- **With technical support:** 15.17% churn.

**Insight:** Customers without technical support show higher churn in the group's EDA.

**Modeling implication:** Retain `TechSupport` as a categorical predictor.

**Business implication:** Consider whether support access and awareness should be evaluated as retention interventions.

## 9. Online Security

- **Without online security:** 41.77% churn.
- **With online security:** 14.61% churn.

**Insight:** Customers without online security show higher observed churn.

**Modeling implication:** Retain `OnlineSecurity` as a categorical predictor.

**Business implication:** Explore customer service needs and product fit; do not interpret the association as a causal effect of security subscriptions.

## 10. Correlations and Engineered Features

- **`tenure` vs `TotalCharges` correlation:** 0.826.
- **`ServiceCount` vs `MonthlyCharges` correlation:** 0.852 (previous feature-engineering analysis).

**Insight:** Customers with longer tenure generally accumulate greater total charges; customers with more services generally have higher monthly charges. These variables may share information.

**Modeling implication:** Evaluate `TenureGroup` and `ServiceCount` through CV rather than assuming engineered features always improve performance. Correlation does not automatically require removing predictors.

## 11. Summary of EDA Findings

| Factor | Recorded finding | Practical interpretation |
|---|---|---|
| Contract | Month-to-month 42.71%; two-year 2.83% | Contract type is strongly associated with churn |
| Tenure | 0–12 months 47.44%; 49–72 months 9.51% | Early tenure is associated with higher churn |
| Internet service | Fiber optic 41.89% | Service category is informative |
| Payment | Electronic check 45.29% | Payment method is informative |
| Monthly charges | Churned 74.44; retained 61.27 | Higher charges coincide with churn on average |
| Tech support | No 41.64%; yes 15.17% | Support subscription is associated with churn |
| Online security | No 41.77%; yes 14.61% | Security subscription is associated with churn |

These are descriptive statistics from the group's earlier EDA; they should not be treated as independent causal effects.

## 12. Implications for Machine Learning

1. Exclude `customerID` from predictors and retain the original 19 predictors for baseline modeling.
2. Clean the 11 blank `TotalCharges` records as documented.
3. Use the shared stratified 80/20 train/test split and five-fold stratified CV.
4. Apply imputation, encoding, and any scaling within the pipeline fitted to training folds.
5. Evaluate engineered `TenureGroup` and `ServiceCount` features rather than assuming they help every algorithm.
6. Evaluate imbalance strategies using training data only; apply random oversampling inside the CV pipeline where selected.
7. Compare Logistic Regression, Random Forest, Gradient Boosting, and SVM primarily using churn-class F1, with other metrics for context.

## 13. Confirmed Standard Five-Fold Model Comparison

The final comparison was rerun after the Gradient Boosting notebook revision. All scores below are reported training-set standard stratified five-fold CV means.

| Model | F1 | Recall | Precision | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **0.6359** | 0.8074 | 0.5248 | **0.8490** | 0.7547 |
| Random Forest | 0.6352 | 0.7839 | **0.5342** | 0.8476 | **0.7611** |
| Logistic Regression | 0.6281 | 0.8013 | 0.5167 | 0.8460 | 0.7483 |
| SVM | 0.6062 | **0.8261** | 0.4791 | 0.8415 | 0.7153 |

**Gradient Boosting configuration:** learning rate 0.05, maximum depth 2, 200 estimators, minimum samples per leaf 10, and random oversampling inside the pipeline. The grid-search best F1 and the reported post-tuning CV F1 both equal 0.6359. The latter has standard deviation 0.0164. The revised notebook does **not** run nested CV.

**Interpretation:** Gradient Boosting has the highest observed F1, narrowly ahead of Random Forest by 0.0007. This is not evidence of a statistically significant advantage. Because selected hyperparameters are evaluated again on the same training observations, reported CV metrics may be optimistic.

## 14. Confirmed Final Model and Held-Out Test Results

**Selected model:** Gradient Boosting, chosen using reported training CV F1 rather than held-out test performance.

| Test metric | Result |
|---|---:|
| Accuracy | 0.745919 |
| Precision | 0.513841 |
| Recall | 0.794118 |
| F1 | 0.623950 |
| ROC-AUC | 0.844669 |

**Top-100 historical risk ranking:** 79 actual churners, precision@100 = 0.79, test churn prevalence = 0.265436, lift@100 = 2.97623.

**Customer profile:** Average tenure is 3.33 months in the top 100 versus 31.92 months overall; average monthly charges are 80.98 versus 64.09. Among the top 100, 100% have month-to-month contracts, 93% fiber internet, 99% no online security, 98% no technical support, and 85% electronic-check payments. These are descriptive associations, not causes.

## 15. Responsible Interpretation and Limitations

- EDA patterns and model feature importance show associations, not causation.
- Model ranking and historical top-100 precision do not prove that outreach will prevent churn.
- Predicted probabilities may need calibration, especially with random oversampling.
- This is retrospective test-set analysis, not a deployed or live campaign.
- Protect customer information, assess fairness, and use human review for retention actions.
- Prior evaluations of some other algorithms used the test set; the reported selection here is based on training CV, not those test scores.
- Post-tuning CV may have selection optimism; the 0.0007 F1 lead is too small to claim superiority.
