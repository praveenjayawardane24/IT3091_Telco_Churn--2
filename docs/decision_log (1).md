# Project Decision Log

**Project:** IT3091 — Telco Customer Churn Prediction  
**Dataset:** IBM Telco Customer Churn  
**Status:** Model development completed; final comparison must be refreshed after the Gradient Boosting notebook revision.

This log records decisions, supporting evidence, and limitations. **Cross-validation (CV) results are training-set results; held-out test results must not be used to select the winning algorithm.**

## Decision 01 — Data Cleaning

**Decision:** Convert `TotalCharges` to numeric and replace 11 blank values with zero where `tenure = 0`.

**Evidence:** All 11 affected customers have zero tenure; no other invalid `TotalCharges` values were identified.

**Reason:** Retain all 7,043 customers without inventing past charges for customers who have not completed a month of service.

**Limitation:** Zero charges is an assumption supported by the observed zero tenure.

## Decision 02 — Validation Split

**Decision:** Use a shared, stratified 80/20 train/test split with `random_state=42`.

**Evidence:** 5,634 training records and 1,409 held-out test records; the original, training, and test churn proportions are approximately 26.54%. Train/test indices do not overlap.

**Reason:** Ensure that the four models are developed on comparable data while preserving a separate set for final evaluation.

## Decision 03 — Cross-Validation Strategy

**Decision:** Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for model development and reported training CV results.

**Evidence:** All five folds preserve approximately the original churn distribution.

**Reason:** Make comparisons reproducible and avoid folds with strongly different class proportions.

**Updated Gradient Boosting implementation:** The revised notebook uses `GridSearchCV` with the shared five-fold splitter to select hyperparameters, followed by `evaluate_cv(best_pipeline, X_train, y_train)` for a standard five-fold report. **It no longer runs nested CV.** These are two uses of five-fold CV on the same training data, not two independent validation datasets.

**Limitation:** Because the best configuration is chosen using the training data and subsequently cross-validated on that same training data, the post-selection CV estimate may be optimistic. Using the same folds across algorithms does not remove this limitation.

## Decision 04 — Primary Evaluation Metric

**Decision:** Use churn-class **F1-score** as the primary model-comparison metric, and also report recall, precision, ROC-AUC, accuracy, and confusion matrices.

**Reason:** F1 balances identifying actual churners with avoiding too many false churn alerts. Accuracy alone can hide weak churn detection in an imbalanced dataset.

**Limitation:** F1 does not directly represent the financial cost of outreach or missed churners. A business deployment should assess those costs separately.

## Decision 05 — Feature Engineering

**Decision:** Retain the 19 original predictors as the baseline and evaluate optional `TenureGroup` and `ServiceCount` features within the modeling pipeline.

**Evidence:** Customers with tenure 0–12 months have 47.44% churn versus 9.51% for tenure 49–72 months. `ServiceCount` and `MonthlyCharges` have a reported correlation of 0.852.

**Reason:** Test whether tenure grouping and subscription breadth improve predictive performance.

**Implementation:** The revised Gradient Boosting pipeline enables both engineered features. Other models use their own CV-selected feature configurations.

## Decision 06 — Preprocessing

**Decision:** Use median imputation for numeric predictors, most-frequent imputation and one-hot encoding for categorical predictors, and scaling where appropriate.

**Evidence:** Previously verified preprocessing generated 45 baseline columns or 50 with engineered features, with no missing or non-finite transformed values.

**Reason:** Consistently prepare numeric and categorical features. Scaling is important for Logistic Regression and SVM, but not required for tree-based Gradient Boosting.

**Leakage prevention:** Fit preprocessing inside the cross-validation pipeline rather than on the entire dataset before validation.

## Decision 07 — Class Imbalance

**Decision:** Compare no balancing, class weighting where supported, and random oversampling where appropriate.

**Evidence:** The training split contains 4,139 non-churners and 1,495 churners. Random oversampling can equalize these class counts within its training input.

**Implementation:** Gradient Boosting searches `sampler = passthrough` versus `RandomOverSampler(random_state=42)` as a hyperparameter. The previously selected Gradient Boosting configuration used random oversampling. Logistic Regression, Random Forest, and SVM use the imbalance-handling choices established in their respective notebooks.

**Leakage prevention:** Oversampling is placed inside the `imblearn` pipeline, so it is applied to each fold's training portion, not its validation portion.

## Decision 08 — Baseline Verification

**Decision:** Compare model performance with a dummy majority-class classifier.

**Evidence:** Dummy accuracy ≈ 0.7346, churn-class F1 = 0, recall = 0, and ROC-AUC = 0.5000.

**Reason:** Show that a model can achieve apparently reasonable accuracy while detecting no churners.

## Decision 09 — Initial SVM Kernel Comparison

**Decision:** Investigate a linear SVM rather than relying only on the default RBF kernel.

**Evidence:** Earlier baseline results: linear F1 = 0.5857, recall = 0.5344, ROC-AUC = 0.8338; RBF F1 = 0.5767, recall = 0.5010, ROC-AUC = 0.7977.

**Reason:** The linear baseline performed better on the primary metric and recall.

**Status:** Superseded by later SVM tuning and model selection.

## Decision 10 — Logistic Regression Final Configuration

**Decision:** Retain the balanced-class-weight Logistic Regression configuration selected using training-data CV.

**Previously recorded standard CV results:** F1 = 0.6281; recall = 0.8013; precision = 0.5167; ROC-AUC = 0.8460; accuracy = 0.7483.

**Reason:** Provides an interpretable linear classification benchmark with strong recall.

## Decision 11 — Random Forest Final Configuration

**Decision:** Retain the CV-selected Random Forest configuration with 200 trees, `min_samples_leaf=10`, balanced class weights, and engineered features.

**Previously recorded standard CV results:** F1 = 0.6352; recall = 0.7839; precision = 0.5342; ROC-AUC = 0.8476; accuracy = 0.7611.

**Reason:** Captures nonlinear relationships and provides a strong tree-based comparison.

**Limitation:** Its reported CV score follows training-data configuration selection and is not an independently nested estimate.

## Decision 12 — SVM Final Configuration

**Decision:** Retain a linear SVM with `C=0.01`, balanced class weights, numeric scaling, and no optional engineered features or random oversampling.

**Previously recorded standard CV results:** F1 = 0.6062; recall = 0.8261; precision = 0.4791; ROC-AUC = 0.8415; accuracy = 0.7153.

**Reason:** Prioritizes churn detection while allowing direct comparison against the other classifiers.

## Decision 13 — Gradient Boosting Configuration Search

**Decision:** Use `GradientBoostingClassifier(random_state=42, min_samples_leaf=10)` with engineered features, no numeric scaling, and a five-fold `GridSearchCV` search scored by F1.

**Search space:** `n_estimators ∈ {100, 200}`; `learning_rate ∈ {0.05, 0.10}`; `max_depth ∈ {2, 3}`; sampler ∈ {passthrough, random oversampling}. This gives 16 candidate combinations.

**Reason:** Tune model complexity, learning rate, and imbalance handling on training data only.

**Previously selected configuration:** 200 estimators, learning rate 0.05, maximum depth 2, and random oversampling. **Reconfirm the selected configuration from the revised notebook output before treating it as current.**

## Decision 14 — Gradient Boosting Validation Revision

**Decision:** Remove the earlier nested-CV evaluation and use the team's requested standard five-fold CV workflow.

**Current notebook sequence:** (1) `tuned_search.fit(X_train, y_train)`; (2) retrieve `best_pipeline`; (3) run `evaluate_cv(best_pipeline, X_train, y_train)`; (4) save `results/tables/gradient_boosting_standard_cv.csv`; (5) save `gradient_boosting.joblib` and diagnostic figures.

**Important distinction:** `tuned_search.best_score_` is the grid search's best five-fold tuning F1. `evaluate_cv(best_pipeline, ...)` reports several CV metrics for the selected fixed configuration. These scores are related but serve different purposes and may not be numerically identical.

**Limitation:** This is **post-tuning CV** on the same training dataset. It is not nested CV and can be optimistic because hyperparameters were selected using those observations. It does **not**, by itself, indicate leakage from the held-out test set.

**Results status:** The revised notebook's output values have not yet been supplied. Do not reuse the earlier 0.6359 figure as a newly verified result without checking the regenerated CSV.

## Decision 15 — Cross-Validated Gradient Boosting Diagnostics

**Decision:** Use `cross_val_predict(best_pipeline, X_train, y_train, cv=get_cross_validation(), method="predict_proba")` to create out-of-fold training predictions for a confusion matrix and precision–recall curve.

**Reason:** Inspect recall, false alerts, and the precision–recall trade-off without using the held-out test set.

**Limitation:** These are out-of-fold predictions for a configuration already selected on the full training split; they are not independent nested-CV predictions. Previously written confusion-matrix counts and average precision should be refreshed from the revised notebook output.

## Decision 16 — Model Comparison and Winner Selection

**Decision:** Compare Logistic Regression, Random Forest, Gradient Boosting, and SVM using the same named primary metric, churn-class F1, from their training-data standard five-fold CV summaries.

**Previously recorded results:** Logistic Regression 0.6281; Random Forest 0.6352; SVM 0.6062. Gradient Boosting's **new** standard CV F1 must be read from the regenerated `gradient_boosting_standard_cv.csv`.

**Status:** **Pending rerun of the final comparison notebook.** Do not assume Gradient Boosting or Random Forest is the current winner until the refreshed results are checked.

**Interpretation:** Small observed CV differences do not establish statistically significant superiority, particularly when configurations were selected using the same training observations.

## Decision 17 — Held-Out Test Set

**Decision:** Select the preferred model from training CV results and use the 1,409-record held-out test set to report its final performance.

**Historical note:** The earlier Gradient Boosting final pipeline reported test F1 = 0.6239, accuracy = 0.7459, precision = 0.5138, recall = 0.7941, and ROC-AUC = 0.8447. These are **historical outputs**, not automatically the results of the revised final comparison.

**Important:** Logistic Regression and Random Forest notebooks also contained test-set evaluations, whereas the Gradient Boosting and SVM individual notebooks did not use the test set for their own evaluations. Earlier test results must not be used to change or justify the new model-selection decision.

## Decision 18 — Retention Prioritization

**Decision:** After selecting the final model, rank held-out customers by predicted churn risk and examine the highest-ranked 100 customers.

**Previously reported Gradient Boosting example:** 79 of the top 100 customers were actual churners (precision@100 = 0.79; lift ≈ 2.98 against a 26.54% churn baseline).

**Status:** **Historical illustration only.** Regenerate the shortlist and associated metrics if the final selected model or saved predictions change.

**Limitation:** Retrospective ranking performance is not evidence that contacting these customers will prevent churn. Oversampling can affect probability calibration.

## Decision 19 — Business Recommendations and Responsible Use

**Decision:** Use model results to propose targeted retention review, especially for short-tenure, month-to-month customers and customers without support/security services where descriptive evidence supports prioritization.

**Reason:** Focus attention on customers with observed elevated churn risk rather than sending identical offers to everyone.

**Safeguards:** Predictions are associations, not causal explanations. Avoid automated adverse decisions, protect customer information, review potential fairness concerns, and test actual retention interventions before claiming business impact.

## Decision 20 — Documentation and Reproducibility

**Decision:** Keep all final reports, README figures, decision logs, and model-comparison tables synchronized with regenerated CSVs and notebook outputs.

**Required refresh after the Gradient Boosting change:** Re-run the revised Gradient Boosting notebook; confirm its best parameters and CV metrics; update the final comparison to read `gradient_boosting_standard_cv.csv`; reselect the model using training CV; then regenerate final-model test, ranking, and business outputs as needed.

**Current status:** Awaiting updated Gradient Boosting output and refreshed final comparison. Historical values in this log are labeled explicitly so they are not mistaken for new results.
