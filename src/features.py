"""Feature matrix and preprocessing pipelines for the mid-semester scenario."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Mid-semester feature frame and binary at-risk target.

    The frame carries the demographic columns alongside the model inputs so the
    ablation and subgroup audit can see them; the preprocessor only feeds the
    model-input columns to the classifier.
    """
    cols = (config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
            + config.DEMOGRAPHIC_NUMERIC + config.DEMOGRAPHIC_CATEGORICAL)
    X = df[cols].copy()
    y = df["at_risk"]
    return X, y


def build_preprocessor(
    scale_numeric: bool,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> ColumnTransformer:
    """Shared preprocessing.

    - Parent_Education_Level nulls become an explicit 'Unknown' category:
      missingness may itself be informative, and inventing a real education
      level would be a stronger assumption than admitting we don't know.
    - Numeric columns have no nulls in either file, but a median imputer is
      included so the pipeline survives future extracts that do.
    - scale_numeric=True for logistic regression; trees don't need it.
    - numeric/categorical default to the model-input lists; the ablation passes
      explicit subsets. Columns not listed are dropped, so demographic columns
      in the frame never reach the classifier.
    """
    if numeric is None:
        numeric = config.NUMERIC_FEATURES
    if categorical is None:
        categorical = config.CATEGORICAL_FEATURES
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric),
        ("cat", categorical_pipe, categorical),
    ])


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable names for the transformed matrix (fitted preprocessor)."""
    return [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]
