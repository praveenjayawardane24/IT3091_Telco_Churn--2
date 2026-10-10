
# ==========================================================
# IT3091 - Telco Customer Churn
# Shared Model Evaluation
#
# Responsibility: IT24101808 - Validation
# Model comparison: All group members
# ==========================================================

import pandas as pd

from sklearn.model_selection import cross_validate

from src.validation import get_cross_validation


SCORING = {
    "F1": "f1",
    "Recall": "recall",
    "Precision": "precision",
    "ROC_AUC": "roc_auc",
    "Accuracy": "accuracy"
}


def evaluate_cv(pipeline, X_train, y_train):
    """
    Evaluate a complete model pipeline using
    stratified 5-fold cross-validation.

    Uses training data only.
    Does not access the held-out test set.
    """

    cv = get_cross_validation()

    scores = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise"
    )

    results = []

    for metric in SCORING:
        values = scores[f"test_{metric}"]

        results.append({
            "Metric": metric,
            "Mean": round(values.mean(), 4),
            "Std": round(values.std(), 4)
        })

    return pd.DataFrame(results)
