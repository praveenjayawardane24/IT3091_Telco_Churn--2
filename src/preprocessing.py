
# ==========================================================
# IT3091 - Telco Customer Churn
# Shared Preprocessing
#
# Responsibility: IT24101780 - Common Preprocessing
# Responsibility: IT24102310 - Scaling
# Integration: IT24101808
# ==========================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

BINARY_NUMERIC_FEATURES = [
    "SeniorCitizen"
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


def build_preprocessor(
    scale_numeric=True,
    add_engineered_features=False
):
    """
    Create shared preprocessing.

    Parameters
    ----------
    scale_numeric : bool
        Whether to standardize continuous numeric features.

    add_engineered_features : bool
        Whether to include TenureGroup and ServiceCount.
    """

    numeric_features = NUMERIC_FEATURES.copy()
    categorical_features = CATEGORICAL_FEATURES.copy()

    if add_engineered_features:
        numeric_features.append("ServiceCount")
        categorical_features.append("TenureGroup")

    # Continuous numerical preprocessing
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "binary_numeric",
                "passthrough",
                BINARY_NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor
