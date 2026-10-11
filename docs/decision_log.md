
# Project Decision Log

## Decision 01 - Data Cleaning

Decision:
Convert TotalCharges to numeric and replace its 11 blank
values with zero when tenure is zero.

Evidence:
All 11 blank records have tenure = 0.
No other invalid TotalCharges values were found.

Reason:
Retain all customers without assigning historical charges
to customers with zero tenure.

Limitation:
Zero TotalCharges is an assumption based on tenure.

## Decision 02 - Validation

Decision:
Use an 80/20 stratified train/test split with random_state=42.

Evidence:
Original churn rate: 26.54%.
Training churn rate: 26.54%.
Testing churn rate: 26.54%.
No overlapping train/test records.

Reason:
Provide a reproducible, comparable split for all four models.

## Decision 03 - Cross-Validation

Decision:
Use StratifiedKFold with 5 folds, shuffle=True,
and random_state=42.

Evidence:
All validation folds contain approximately 26.54% churn.

Reason:
Preserve class distribution during model evaluation.

## Decision 04 - Evaluation

Decision:
Use churn-class F1-score as the provisional primary
selection metric.

Also report:
- Recall
- Precision
- ROC-AUC
- Accuracy
- Confusion matrix

Reason:
Balance the detection of churners with the cost of
incorrectly identifying customers as likely to churn.

Status:
Provisional until the group confirms business priorities.


## Decision 05 - Feature Engineering

Decision:
Retain the 19 original predictors as the baseline.
Evaluate optional TenureGroup and ServiceCount features.

Evidence:
Tenure 0-12 months: 47.44% churn.
Tenure 49-72 months: 9.51% churn.
ServiceCount and MonthlyCharges correlation: 0.852.

Reason:
Test whether customer lifecycle and subscription breadth
provide additional predictive information.

Status:
Final feature selection pending cross-validation.


## Decision 06 - Preprocessing

Decision:
Use median imputation for numerical features,
most-frequent imputation and one-hot encoding for
categorical features, and optional numerical scaling.

Evidence:
Baseline preprocessing produces 45 columns.
Engineered preprocessing produces 50 columns.
Verification found no missing or non-finite values.

Reason:
Support categorical and numerical predictors consistently.
Scale numerical features for scale-sensitive models
such as Logistic Regression and SVM.

Leakage Prevention:
Fit preprocessing within each cross-validation fold.


## Decision 07 - Class Imbalance

Decision:
Compare no balancing, supported class weighting,
and random oversampling.

Evidence:
Training classes before oversampling:
No churn = 4139; Churn = 1495.

After random oversampling:
No churn = 4139; Churn = 4139.

Reason:
Investigate whether imbalance handling improves
minority-class prediction.

Leakage Prevention:
Apply oversampling only to training folds.

Status:
Best strategy not yet determined.


## Decision 08 - Evaluation Verification

Decision:
Use a shared five-fold cross-validation function
for all four classification models.

Evidence:
Dummy classifier results:
Accuracy = 0.7346
F1 = 0.0000
Recall = 0.0000
ROC-AUC = 0.5000

Reason:
Demonstrates that accuracy alone can be misleading
when evaluating imbalanced classification problems.


## Decision 09 - Initial SVM Kernel Comparison

Decision:
Prefer the linear kernel provisionally over the
default RBF kernel for the SVM baseline.

Evidence:
Linear: F1 = 0.5857; Recall = 0.5344;
ROC-AUC = 0.8338.

RBF: F1 = 0.5767; Recall = 0.5010;
ROC-AUC = 0.7977.

Reason:
Linear SVM performed better on the provisional
primary metric (F1) and identified more churners.

Limitation:
The differences have not been tested for
statistical significance.

Status:
Provisional; hyperparameter tuning may change
the preferred kernel.
