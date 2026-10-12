# IT3091 — Telco Customer Churn Prediction

**Machine Learning Group Project | Sri Lanka Institute of Information Technology (SLIIT)**

**Repository:** [IT3091_Telco_Churn--2](https://github.com/praveenjayawardane24/IT3091_Telco_Churn--2)

## 1. Project Overview

Customer churn occurs when a customer stops using a company's services. This project uses machine learning to predict churn among telecommunications customers and identify customer segments that may benefit from retention outreach.

The project compares four supervised classification algorithms using a consistent dataset, preprocessing pipeline, train/test split, and evaluation framework:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

**Primary selection metric:** F1-score for the churn class. Recall, precision, ROC-AUC, and accuracy are also considered.

The selected model is **Gradient Boosting**, based on the highest observed standard five-fold cross-validation F1-score. The difference from Random Forest is very small and should not be interpreted as proof of statistically significant superiority.

## 2. Objectives

1. Understand and clean the customer churn dataset.
2. Explore relationships between customer characteristics and churn.
3. Build reproducible preprocessing and validation pipelines.
4. Compare feature engineering and class-imbalance strategies.
5. Train and tune four classification algorithms.
6. Select a model using training-data cross-validation.
7. Evaluate the selected model on a held-out test set.
8. Demonstrate risk-based customer prioritization and propose responsible retention actions.

## 3. Dataset

**Dataset:** IBM Telco Customer Churn  
**Raw file:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

| Property | Value |
|---|---:|
| Customer records | 7,043 |
| Original columns | 21 |
| Original predictors | 19 (excluding `customerID` and `Churn`) |
| Churned customers | 1,869 (26.54%) |
| Non-churned customers | 5,174 (73.46%) |
| Training records | 5,634 (80%) |
| Test records | 1,409 (20%) |

The target is **`Churn`**, encoded as a binary classification outcome. The unique `customerID` field is excluded from predictive features and may be retained separately to associate predictions with customer records.

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for variable descriptions.

## 4. Data Preparation and Exploratory Analysis

- Converted `TotalCharges` to numeric.
- Replaced 11 blank `TotalCharges` values with zero; all affected customers had `tenure = 0`.
- Investigated class balance, tenure, contracts, internet service, payment methods, support services, and monthly charges.
- Examined optional engineered features: `TenureGroup` and `ServiceCount`.
- Used numerical imputation, categorical imputation, and one-hot encoding in reusable pipelines.
- Applied numerical scaling where needed, especially for Logistic Regression and SVM.

Notable exploratory associations included higher observed churn among customers with month-to-month contracts, shorter tenure, and no technical support or online security. These are **associations, not established causes**.

See [`docs/eda_insight_log.md`](docs/eda_insight_log.md).

## 5. Validation and Leakage Prevention

- **Train/test split:** Stratified 80/20, `random_state=42`.
- **Model validation:** Stratified five-fold cross-validation with shuffling and `random_state=42`.
- **Preprocessing:** Fitted within model pipelines, using training folds only during cross-validation.
- **Imbalance strategies:** Compared unbalanced training, class weights, and random oversampling where supported.
- **Random oversampling:** Applied only within training folds, never to validation folds or the held-out test set.
- **Model selection:** Based on training-data cross-validation, not test-set metrics.

The Logistic Regression and Random Forest notebooks additionally report test-set evaluations of their selected configurations. Those test results were **not used to choose the overall winning algorithm**. The Gradient Boosting and SVM individual notebooks use training-data validation only; the selected Gradient Boosting model is evaluated on the held-out test set in the final comparison notebook.

## 6. Model Development

| Model | Final reported configuration |
|---|---|
| Logistic Regression | Balanced class weights; scaled numerical features |
| Random Forest | 200 trees; `max_depth=None`; `min_samples_leaf=10`; balanced class weights; engineered features |
| Gradient Boosting | 200 estimators; learning rate `0.05`; depth `2`; `min_samples_leaf=10`; random oversampling; engineered features |
| SVM | Linear kernel; `C=0.01`; balanced class weights; scaled numerical features; original predictors |

All models use the shared project data split and preprocessing framework. Each algorithm's experiments, configuration comparisons, and saved outputs are documented in its own notebook.

## 7. Cross-Validation Model Comparison

The following **standard five-fold CV results** were used in the final comparison:

| Model | F1 | Recall | Precision | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **0.6359** | 0.8074 | 0.5248 | **0.8490** | 0.7547 |
| Random Forest | 0.6352 | 0.7839 | **0.5342** | 0.8476 | **0.7611** |
| Logistic Regression | 0.6281 | 0.8013 | 0.5167 | 0.8460 | 0.7483 |
| SVM | 0.6062 | **0.8261** | 0.4791 | 0.8415 | 0.7153 |

**Selection:** Gradient Boosting had the highest observed CV F1-score (0.6359). Random Forest was very close (0.6352), a difference of only 0.0007. The result does not establish statistically significant superiority.

**Tuning caveat:** The revised Gradient Boosting notebook uses standard five-fold `GridSearchCV` tuning and then standard five-fold evaluation of the selected configuration; it does not run nested CV. Both reported F1 values were 0.6359. The post-tuning CV score may be optimistic because hyperparameters were selected using the same training observations. The 0.0007 F1 margin over Random Forest does not prove a meaningful advantage.

## 8. Final Held-Out Test Evaluation

After selection, the Gradient Boosting pipeline was evaluated on **1,409 held-out test customers**.

| Metric | Test result |
|---|---:|
| Accuracy | 0.745919 |
| Precision | 0.513841 |
| Recall | 0.794118 |
| F1 | 0.623950 |
| ROC-AUC | 0.844669 |

The model identified approximately **79.41% of actual churners** in the held-out test set. Its precision of **51.38%** indicates that some predicted churn alerts are false positives.

## 9. Customer Risk Ranking and Business Insights

For a retrospective demonstration, the final model ranked the held-out test customers by estimated churn probability and selected the **100 highest-risk customers**.

| Shortlist measure | Result |
|---|---:|
| Customers selected | 100 |
| Actual churners in shortlist | 79 |
| Precision@100 | 0.7900 |
| Test-set churn prevalence | 0.2654 |
| Lift@100 | 2.9762 |

The top-100 group had an average tenure of **3.33 months**, compared with **31.92 months** across the test set, and average monthly charges of **80.98**, compared with **64.09**. Among the top 100, **100%** had month-to-month contracts, **93%** used fiber-optic internet, **99%** lacked online security, and **98%** lacked technical support.

These historical results illustrate **ranking usefulness**, not verified future churn or guaranteed retention outcomes. The estimated probabilities have not been independently calibrated.

Potential retention experiments include early onboarding support, contract incentives, pricing/value reviews, proactive technical support, relevant security offers, and payment-experience improvements. These actions require testing before claims of effectiveness.

See [`docs/business_recommendations.md`](docs/business_recommendations.md).

## 10. Project Structure

```text
IT3091_Telco_Churn--2/
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── docs/
│   ├── data_dictionary.md
│   ├── decision_log.md
│   ├── eda_insight_log.md
│   ├── model_comparison.md
│   └── business_recommendations.md
├── notebooks/
│   ├── 08_logistic_regression.ipynb
│   ├── 09_random_forest.ipynb
│   ├── 10_gradient_boosting.ipynb
│   ├── 11_svm.ipynb
│   └── 12_final_model_comparison.ipynb
├── results/
│   ├── figures/
│   ├── models/
│   └── tables/
├── src/
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── validation.py
│   ├── imbalance.py
│   └── evaluation.py
└── README.md
```

*The tree highlights the principal project files; additional notebooks, source files, or dependency files may also be present in the repository.*

## 11. How to Run

### Prerequisites

- Python 3.10+ recommended
- Jupyter Notebook or JupyterLab
- Python libraries used in the project, including `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`, and `joblib`

### Setup

Clone the repository:

```bash
git clone https://github.com/praveenjayawardane24/IT3091_Telco_Churn--2.git
cd IT3091_Telco_Churn--2
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies using the repository's dependency file, **if present**:

```bash
pip install -r requirements.txt
```

If no `requirements.txt` is present, install the core libraries:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Run the notebooks from the project root or the `notebooks/` directory. The notebooks locate the project root automatically. Run the individual model notebooks first if the required saved outputs are not yet available; then run `12_final_model_comparison.ipynb` to reproduce the model comparison and business-analysis outputs.

**Note:** Reproducibility also depends on compatible package versions. Use a pinned environment file if the repository provides one.

## 12. Outputs

The project writes reproducible outputs to `results/`:

**Model comparison inputs:**

- `results/tables/logistic_regression_cv.csv`
- `results/tables/random_forest_final_cv.csv`
- `results/tables/gradient_boosting_standard_cv.csv`
- `results/tables/svm_final_cv.csv`

**Selected model:**

- `results/models/gradient_boosting.joblib`

**Final analysis:**

- `results/tables/final_model_comparison.csv`
- Test-set evaluation tables and figures
- Ranked churn-risk scores and top-100 shortlist
- High-risk customer profiles
- Feature importance results
- Retention recommendations and project summary

The final comparison notebook also saves figures under `results/figures/final_comparison/`.

## 13. Responsible AI and Limitations

- The dataset is historical and may not represent present-day customer behavior.
- Classifier performance can change under distribution shift or changing customer behavior.
- False positives may lead to unnecessary outreach costs; false negatives may miss customers who churn.
- High-risk characteristics and model feature importance are **not causal evidence**.
- Oversampling may affect probability calibration; calibrated risk probabilities have not been verified.
- The top-100 analysis is retrospective, not a live deployment or a proven retention intervention.
- Fairness across relevant customer groups, privacy protection, and human oversight should be assessed before deployment.
- Retention recommendations should be validated with appropriate experiments or business evaluation.

This project is an **academic decision-support prototype**, not a production-ready system.

## 14. Documentation

| Document | Purpose |
|---|---|
| [`data_dictionary.md`](docs/data_dictionary.md) | Dataset columns, meanings, and engineered features |
| [`decision_log.md`](docs/decision_log.md) | Technical choices, evidence, and limitations |
| [`eda_insight_log.md`](docs/eda_insight_log.md) | Exploratory findings and business interpretation |
| [`model_comparison.md`](docs/model_comparison.md) | Four-model comparison and evaluation caveats |
| [`business_recommendations.md`](docs/business_recommendations.md) | Retention ideas, risk profiles, and limitations |

## 15. Team Contributions

This is a collaborative IT3091 group project. Individual model notebooks document their respective contributors:

| Workstream | Notebook / contribution |
|---|---|
| Logistic Regression | `08_logistic_regression.ipynb` — IT24101780 |
| Random Forest | `09_random_forest.ipynb` — IT24102178 |
| Gradient Boosting | `10_gradient_boosting.ipynb` — IT24102310 |
| SVM | `11_svm.ipynb` — IT24101808 |
| Shared foundation and final analysis | Team collaboration; see source modules, notebooks, and decision log |

## 16. Academic Context

Developed for **IT3091 — Machine Learning** at **SLIIT**. Results are intended for coursework, learning, reproducible experimentation, and illustrative business analysis.
