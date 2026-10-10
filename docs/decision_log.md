
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
