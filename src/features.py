"""Shared feature building, used by both training and the live API.

Keeping this in one place guarantees that training and inference build the same
feature matrix with the same column order and the same encoder.
"""

from __future__ import annotations

import pandas as pd

from src import config


def encode_features(df: pd.DataFrame, encoder, fit: bool,
                    categorical=None, numeric=None) -> pd.DataFrame:
    """Builds the feature matrix X: encoded categoricals + numeric features.

    Args:
        df: DataFrame containing all required feature columns.
        encoder: an (Ordinal)Encoder.
        fit: True during training (fit_transform), False at inference (transform).
        categorical: List of categorical feature columns (defaults to config.CATEGORICAL_FEATURES).
        numeric: List of numeric feature columns (defaults to config.NUMERIC_FEATURES).
    """
    categorical = categorical if categorical is not None else config.CATEGORICAL_FEATURES
    numeric = numeric if numeric is not None else config.NUMERIC_FEATURES

    cat = df[categorical]
    if fit:
        encoded = encoder.fit_transform(cat)
    else:
        encoded = encoder.transform(cat)

    encoded_df = pd.DataFrame(
        encoded, columns=categorical
    ).reset_index(drop=True)
    num_df = df[numeric].reset_index(drop=True)

    return pd.concat([encoded_df, num_df], axis=1)
