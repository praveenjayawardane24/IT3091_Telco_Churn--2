# Telco Customer Churn — Data Dictionary

**Dataset:** IBM Telco Customer Churn, 7,043 customer records, 21 original columns. Each row represents one customer account.

| Column | Type | Meaning | Model use |
|---|---|---|---|
| customerID | String | Unique customer identifier | Excluded from predictors; retained for output linkage |
| gender | Category | Customer gender | Predictor |
| SeniorCitizen | Binary integer | Whether customer is a senior citizen (0/1) | Predictor |
| Partner | Category | Whether customer has a partner | Predictor |
| Dependents | Category | Whether customer has dependents | Predictor |
| tenure | Numeric integer | Months with the company | Predictor |
| PhoneService | Category | Whether phone service is subscribed | Predictor |
| MultipleLines | Category | Whether multiple phone lines are subscribed | Predictor |
| InternetService | Category | Internet service type | Predictor |
| OnlineSecurity | Category | Whether online security is subscribed | Predictor |
| OnlineBackup | Category | Whether online backup is subscribed | Predictor |
| DeviceProtection | Category | Whether device protection is subscribed | Predictor |
| TechSupport | Category | Whether technical support is subscribed | Predictor |
| StreamingTV | Category | Whether streaming TV is subscribed | Predictor |
| StreamingMovies | Category | Whether streaming movies are subscribed | Predictor |
| Contract | Category | Contract duration/type | Predictor |
| PaperlessBilling | Category | Whether paperless billing is enabled | Predictor |
| PaymentMethod | Category | Customer payment method | Predictor |
| MonthlyCharges | Numeric | Monthly billed amount | Predictor |
| TotalCharges | Numeric after cleaning | Total billed amount over customer relationship | Predictor |
| Churn | Binary category | Whether customer left the provider (Yes/No) | Target |

## Cleaning and modelling notes
- `TotalCharges` is text in the raw CSV. Convert to numeric; the 11 blank entries correspond to tenure = 0 and were set to zero.
- `customerID` is excluded from model inputs.
- `Churn` is mapped to 0 = No and 1 = Yes for modelling.
- Optional engineered predictors: `TenureGroup` (tenure bands) and `ServiceCount` (count of selected subscribed services). The selected Gradient Boosting pipeline uses engineered features.
- One-hot encoding expands categorical predictors; preprocessing and oversampling are fitted within training folds.
- Demographic features should be reviewed for fairness before real-world use.
