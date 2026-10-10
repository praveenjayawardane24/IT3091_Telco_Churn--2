
# ==========================================================
# IT3091 - Telco Customer Churn
# Shared Data Preparation
#
# Data quality responsibility: IT24102310
# Shared preprocessing responsibility: IT24101780
# ==========================================================

from pathlib import Path
import pandas as pd


def load_and_clean_data(project_root):
    """
    Load the raw Telco Churn dataset and apply the
    agreed deterministic data quality corrections.

    Returns a cleaned DataFrame.
    """

    dataset_path = (
        Path(project_root)
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(dataset_path)

    # Responsibility: IT24102310
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Responsibility: IT24102310
    # Replace blank TotalCharges only for zero-tenure customers
    zero_tenure_mask = (
        df["tenure"].eq(0)
        & df["TotalCharges"].isna()
    )

    df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    # Quality checks
    if df["TotalCharges"].isna().any():
        raise ValueError(
            "Unexpected missing TotalCharges values remain."
        )

    if df["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs detected."
        )

    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError(
            "Unexpected target labels detected."
        )

    return df


def separate_features_target(df):
    """
    Separate predictor features and target.

    customerID is excluded because it is an identifier.
    Churn is excluded from predictors to prevent leakage.
    """

    X = df.drop(columns=["customerID", "Churn"])

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y
