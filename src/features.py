"""Shared feature building, used by both training and the live API.

Keeping this in one place guarantees that training and inference build the same
feature matrix with the same column order and the same encoder.
"""

from __future__ import annotations

import pandas as pd

from src import config


def encode_features(df: pd.DataFrame, encoder, fit: bool) -> pd.DataFrame:
    """Builds the feature matrix X: encoded categoricals + numeric features.

    Args:
        df: DataFrame containing all CATEGORICAL_FEATURES and NUMERIC_FEATURES.
        encoder: an (Ordinal)Encoder.
        fit: True during training (fit_transform), False at inference (transform).
    """
    cat = df[config.CATEGORICAL_FEATURES]
    if fit:
        encoded = encoder.fit_transform(cat)
    else:
        encoded = encoder.transform(cat)

    encoded_df = pd.DataFrame(
        encoded, columns=config.CATEGORICAL_FEATURES
    ).reset_index(drop=True)
    num_df = df[config.NUMERIC_FEATURES].reset_index(drop=True)

    return pd.concat([encoded_df, num_df], axis=1)
