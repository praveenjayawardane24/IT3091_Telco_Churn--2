
# ==========================================================
# IT3091 - Telco Customer Churn
# Shared Feature Engineering
#
# Responsibility: IT24102178 - Feature Engineering
# Integration: IT24101808
# ==========================================================

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Add optional engineered features.

    Features:
    - TenureGroup
    - ServiceCount

    Compatible with scikit-learn pipelines.
    """

    def __init__(self, add_engineered_features=True):
        self.add_engineered_features = add_engineered_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if not self.add_engineered_features:
            return X

        # Responsibility: IT24102178
        # Tenure grouping
        X["TenureGroup"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=[
                "0-12",
                "13-24",
                "25-48",
                "49-72"
            ]
        ).astype(str)

        # Responsibility: IT24102178
        # Count subscribed services
        service_count = pd.Series(
            0,
            index=X.index,
            dtype=int
        )

        for col in SERVICE_COLUMNS:
            if col == "InternetService":
                service_count += (X[col] != "No").astype(int)
            else:
                service_count += (X[col] == "Yes").astype(int)

        X["ServiceCount"] = service_count

        return X
