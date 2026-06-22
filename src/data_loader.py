"""Loading, cleaning, target derivation and feature engineering."""

from __future__ import annotations

import pandas as pd

from src import config
from src.enrichment import event_calendar


def load_raw() -> pd.DataFrame:
    """Reads the raw CSV."""
    return pd.read_csv(config.RAW_DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parses dates and removes invalid rows."""
    df = df.copy()
    df[config.COL_DATE] = pd.to_datetime(df[config.COL_DATE], errors="coerce")
    df = df.dropna(subset=[config.COL_DATE, config.COL_AMOUNT])
    return df.reset_index(drop=True)


def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """Derives the collection priority from TrashAmount via binning."""
    df = df.copy()
    df[config.TARGET] = pd.cut(
        df[config.COL_AMOUNT],
        bins=config.PRIORITY_BIN_EDGES,
        labels=config.PRIORITY_BIN_LABELS,
    ).astype(str)
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives hour, day-of-week and month from the date."""
    df = df.copy()
    df[config.COL_DATE] = pd.to_datetime(df[config.COL_DATE])
    df[config.COL_HOUR] = df[config.COL_DATE].dt.hour
    df[config.COL_DOW] = df[config.COL_DATE].dt.dayofweek
    df[config.COL_MONTH] = df[config.COL_DATE].dt.month
    return df


def add_event_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds event_size based on the events calendar (local, no API)."""
    df = df.copy()
    df[config.COL_DATE] = pd.to_datetime(df[config.COL_DATE])
    df[config.COL_EVENT] = [
        event_calendar.size_for(
            row[config.COL_LAT], row[config.COL_LON], row[config.COL_DATE]
        )
        for _, row in df.iterrows()
    ]
    return df


def fill_missing_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the location type is always present (otherwise "other")."""
    df = df.copy()
    if config.COL_LOCATION_TYPE not in df.columns:
        df[config.COL_LOCATION_TYPE] = "other"
    else:
        df[config.COL_LOCATION_TYPE] = df[config.COL_LOCATION_TYPE].fillna("other")
    return df
