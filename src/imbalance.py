
# ==========================================================
# IT3091 - Telco Customer Churn
# Shared Class Imbalance Utilities
#
# Responsibility: IT24101808
# ==========================================================

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from src.feature_engineering import TelcoFeatureEngineer
from src.preprocessing import build_preprocessor

RANDOM_STATE = 42


def get_sampler(strategy="none"):
    """
    Return an optional training-only sampler.

    Supported strategies:
    - none
    - random_oversampling
    """

    if strategy == "none":
        return None

    if strategy == "random_oversampling":
        return RandomOverSampler(
            random_state=RANDOM_STATE
        )

    raise ValueError(
        f"Unknown imbalance strategy: {strategy}"
    )


def build_model_pipeline(
    model,
    scale_numeric=True,
    add_engineered_features=False,
    imbalance_strategy="none"
):
    """
    Build a reusable ML pipeline.

    Feature engineering -> preprocessing ->
    optional resampling -> model.
    """

    steps = [
        (
            "features",
            TelcoFeatureEngineer(
                add_engineered_features=add_engineered_features
            )
        ),
        (
            "preprocessor",
            build_preprocessor(
                scale_numeric=scale_numeric,
                add_engineered_features=add_engineered_features
            )
        )
    ]

    sampler = get_sampler(imbalance_strategy)

    if sampler is not None:
        steps.append(("sampler", sampler))

    steps.append(("model", model))

    return ImbPipeline(steps)
