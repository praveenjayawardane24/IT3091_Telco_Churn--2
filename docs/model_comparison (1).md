# Model and Method Comparison

## Evaluation setup
- Dataset: 7,043 customers; 80/20 stratified split (seed 42): 5,634 train and 1,409 test.
- Training evaluation: stratified five-fold CV, seed 42. F1 for churn class is the primary selection metric; recall, precision, ROC-AUC and accuracy are supporting metrics.
- Preprocessing and any oversampling occur inside model pipelines to avoid fold leakage.

## Selected configuration comparison (training CV)

| Model | F1 | Recall | Precision | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | **0.6359** | 0.8074 | 0.5248 | **0.8490** | 0.7547 |
| Random Forest | 0.6352 | 0.7839 | **0.5342** | 0.8476 | **0.7611** |
| Logistic Regression | 0.6281 | 0.8013 | 0.5167 | 0.8460 | 0.7483 |
| SVM | 0.6062 | **0.8261** | 0.4791 | 0.8415 | 0.7153 |

## Model configurations
- Logistic Regression: baseline original features with `class_weight=balanced`.
- Random Forest: engineered features; `n_estimators=200`, `max_depth=None`, `min_samples_leaf=10`, `class_weight=balanced`.
- Gradient Boosting: engineered features; `learning_rate=0.05`, `max_depth=2`, `n_estimators=200`, `min_samples_leaf=10`, random oversampling.
- Linear SVM: original features, scaled numeric predictors, `C=0.01`, `class_weight=balanced`.

## Selection and methodological limitation
Gradient Boosting was selected by the predeclared highest-F1 rule. Its F1 advantage over Random Forest is only 0.0007, so the models should be considered very close. The revised Gradient Boosting notebook uses standard five-fold `GridSearchCV` tuning followed by standard five-fold CV of the selected pipeline; nested CV is no longer run. Both the grid-search best F1 and the post-tuning CV F1 were 0.6359. Post-tuning CV may be optimistic because hyperparameters were chosen using the same training observations. A fully independent and uniform evaluation of tuning procedures would be stronger.

## Final selected-model held-out test evaluation
Gradient Boosting: F1 = **0.623950**, recall = **0.794118**, precision = **0.513841**, ROC-AUC = **0.844669**, accuracy = **0.745919**. The test set was not used to select the final model, though other candidate notebooks had previously produced test metrics; this is an evaluation limitation to disclose.

## Business-facing result
Top 100 highest-ranked historical test customers contained 79 actual churners: precision@100 = 79%, lift@100 = 2.9762 relative to test churn prevalence (26.54%). These are retrospective demonstration results, not validated intervention outcomes.
