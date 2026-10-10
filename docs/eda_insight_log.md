
# EDA Insight Log

Dataset: Telco Customer Churn
Records: 7,043
Overall churn rate: 26.54%

## Key Findings

1. Month-to-month contracts: 42.71% churn.
   Two-year contracts: 2.83% churn.

2. Tenure 0-12 months: 47.44% churn.
   Tenure 49-72 months: 9.51% churn.

3. Fiber optic customers: 41.89% churn.

4. Electronic check customers: 45.29% churn.

5. Average monthly charges:
   Churned: 74.44
   Retained: 61.27

6. Customers without technical support: 41.64% churn.
   Customers with technical support: 15.17% churn.

7. Customers without online security: 41.77% churn.
   Customers with online security: 14.61% churn.

8. Tenure and TotalCharges correlation: 0.826.

9. No numerical outliers detected using the 1.5 IQR rule.

## Interpretation

Contract type, tenure, internet service, payment method,
technical support, online security, and monthly charges
show meaningful descriptive associations with churn.

These associations do not establish causation.

## Modeling Implications

- Retain original predictors initially.
- Exclude customerID from model features.
- Investigate optional tenure grouping and service count.
- Do not remove numerical observations based on IQR.
- Use training-only validation for feature selection.
