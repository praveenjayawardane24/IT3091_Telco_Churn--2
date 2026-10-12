# IT3091 — Telco Customer Churn Prediction

## Project Overview

This project is developed for the IT3091 Machine Learning group assignment at SLIIT.

**Assignment Track:** Guided Data Track  
**Domain:** Telecommunications  
**Primary Decision Lens:** Customer Churn Risk Prediction  
**Secondary Decision Lens:** High-Risk Customer Profiling  
**Task:** Binary Classification

The objective is to predict customer churn using historical telecommunications customer information and compare four supervised machine learning algorithms.

## Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- Records: 7,043
- Original columns: 21
- Target: `Churn` (0 = No, 1 = Yes)
- Churn rate: 26.54%

Dataset location:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Team Responsibilities

| Member ID | Responsibility |
|---|---|
| IT24101780 | Logistic Regression |
| IT24102178 | Random Forest |
| IT24102310 | Gradient Boosting |
| IT24101808 | Shared foundation, validation, evaluation, and SVM |

## Project Structure

- `data/raw/` — Original dataset
- `notebooks/` — Data analysis, verification, model experiments, and final comparison
- `src/` — Reusable Python functions and machine learning pipelines
- `results/tables/` — EDA and model evaluation tables
- `results/figures/` — EDA and model visualizations
- `results/models/` — Locally saved trained models
- `docs/` — Decision logs and project documentation

## Installation and Setup

1. Clone the GitHub repository using GitHub Desktop.
2. Open the project folder in Visual Studio Code.
3. Install the Python and Jupyter VS Code extensions.
4. Open a terminal in the project root.

Create a Python virtual environment:

```powershell
py -3 -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. Open a notebook and select the `.venv` Python kernel.

## Notebook Execution Order

### Shared Foundation

1. `01_data_understanding.ipynb`
2. `02_eda.ipynb`
3. `03_validation.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_preprocessing.ipynb`
6. `06_imbalance.ipynb`
7. `07_evaluation.ipynb`

These notebooks document and verify the shared machine learning workflow.

For a quick setup verification, run `07_evaluation.ipynb`.

### Individual Model Development

- `08_logistic_regression.ipynb`
- `09_random_forest.ipynb`
- `10_gradient_boosting.ipynb`
- `11_svm.ipynb`

### Final Comparison

- `12_final_model_comparison.ipynb`

Run notebooks 01 to 12 in order to reproduce the complete set of results.
The preprocessing steps are fitted inside each cross-validation fold through the shared model pipeline.

## Shared Python Modules

| Module | Purpose |
|---|---|
| `data_preparation.py` | Data loading, cleaning, predictors and target |
| `validation.py` | Stratified train/test split and cross-validation |
| `feature_engineering.py` | Optional engineered features |
| `preprocessing.py` | Imputation, encoding, and scaling |
| `imbalance.py` | Model pipelines and optional oversampling |
| `evaluation.py` | Common cross-validation evaluation |

**Important:** Execute notebook cells using the Jupyter kernel. Do not run individual `src/*.py` files directly using VS Code's Run Python File button.

## Model Evaluation Rules

- Random state: 42
- Training/testing split: 80/20 stratified
- Model selection: Stratified five-fold cross-validation
- Primary metric: F1-score
- Supporting metrics: Recall, precision, ROC-AUC, accuracy
- Fit preprocessing and oversampling inside training folds
- Do not use the held-out test set for tuning

## Current Progress

All 12 notebooks have been completed and executed. The generated tables and figures are available in the `results/` folder.

## Final Model Comparison

| Model | CV F1 | Recall | Precision | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | **0.6359** | 0.8074 | 0.5248 | **0.8490** | 0.7547 |
| Random Forest | 0.6352 | 0.7839 | **0.5342** | 0.8476 | **0.7611** |
| Logistic Regression | 0.6281 | 0.8013 | 0.5167 | 0.8460 | 0.7483 |
| SVM | 0.6062 | **0.8261** | 0.4791 | 0.8415 | 0.7153 |

Gradient Boosting was selected because it achieved the highest mean F1-score using standard stratified five-fold cross-validation.

### Final Held-Out Test Results

- Accuracy: 0.7459
- Precision: 0.5138
- Recall: 0.7941
- F1-score: 0.6239
- ROC-AUC: 0.8447

The detailed comparison is saved in `results/tables/final_model_comparison.csv`. The selected model and final test results are recorded in `final_model_selection.csv` and `final_test_metrics.csv`.

Trained `.joblib` files are generated locally in `results/models/`. This folder is excluded from Git because the model files can be recreated by running the notebooks.

## Reproducibility and AI Use

The project uses reusable Python modules, fixed random states, consistent cross-validation, and documented experiment results.

AI assistance has been used for explanations, implementation guidance, troubleshooting, and documentation. The team is responsible for verifying code execution, results, interpretations, and final conclusions.
