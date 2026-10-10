
# ==========================================================
# IT3091 - Telco Customer Churn
# Shared Validation Strategy
#
#Responsibility: IT24101808
# ==========================================================

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

PRIMARY_METRIC = "f1"

SECONDARY_METRICS = [
    "recall",
    "precision",
    "roc_auc",
    "accuracy"
]


def create_train_test_split(X, y):
    """
    Create the common stratified 80/20 split.

    All four models must use this function.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def get_cross_validation():
    """
    Return the common stratified 5-fold
    cross-validation strategy.
    """

    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
