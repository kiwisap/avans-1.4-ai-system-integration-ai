"""Trains and evaluates a Decision Tree and Random Forest on the Breda trash data,
and saves the models + metadata.

The target (collection priority) is derived from TrashAmount. Temperature and weather
type are already in the dataset; the coordinates are enriched with a location type via
reverse geocoding, and an event size via the events calendar.

Usage:
    python -m src.train                # with location enrichment (Nominatim)
    python -m src.train --no-enrich    # without (faster / offline testing)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from src import config, data_loader, features
from src.enrichment import location


def evaluate(name: str, model, X_test, y_test) -> dict:
    """Computes and prints evaluation metrics for one model."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"\n=== {name} ===")
    print(f"Nauwkeurigheid  : {acc:.3f}")
    print(f"F1 (macro)      : {f1:.3f}")
    print("Classificatie rapport:")
    print(classification_report(y_test, preds, zero_division=0))
    print("Verwarringsmatrix (rijen=werkelijk, kolommen=voorspeld):")
    labels = sorted(pd.unique(y_test))
    print(pd.DataFrame(
        confusion_matrix(y_test, preds, labels=labels),
        index=labels, columns=labels,
    ))

    importances = dict(
        sorted(
            zip(X_test.columns, model.feature_importances_),
            key=lambda kv: kv[1], reverse=True,
        )
    )
    print("Feature importantie:")
    for feat, imp in importances.items():
        print(f"  {feat:22s} {imp:.3f}")

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "feature_importance": {k: round(v, 4) for k, v in importances.items()},
    }


def train_trash_type_classifier(df: pd.DataFrame) -> dict:
    """Trains and evaluates trash-type classifiers (Decision Tree + Random Forest)."""
    from sklearn.metrics import accuracy_score, f1_score

    print("\n--- Training Trash Type Classifier ---")
    y = df[config.TRASH_TYPE_TARGET]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    X_train = features.encode_features(
        df_train, encoder, fit=True,
        categorical=config.TRASH_TYPE_CATEGORICAL_FEATURES,
        numeric=config.TRASH_TYPE_NUMERIC_FEATURES
    )
    X_test = features.encode_features(
        df_test, encoder, fit=False,
        categorical=config.TRASH_TYPE_CATEGORICAL_FEATURES,
        numeric=config.TRASH_TYPE_NUMERIC_FEATURES
    )

    dt = DecisionTreeClassifier(max_depth=8, random_state=config.RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    dt_preds = dt.predict(X_test)
    rf_preds = rf.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    rf_acc = accuracy_score(y_test, rf_preds)
    dt_f1 = f1_score(y_test, dt_preds, average="macro")
    rf_f1 = f1_score(y_test, rf_preds, average="macro")

    print(f"  Decision Tree -> Accuracy: {dt_acc:.3f}, F1: {dt_f1:.3f}")
    print(f"  Random Forest -> Accuracy: {rf_acc:.3f}, F1: {rf_f1:.3f}")

    joblib.dump(dt, config.TRASH_TYPE_DT_PATH)
    joblib.dump(rf, config.TRASH_TYPE_RF_PATH)
    joblib.dump(encoder, config.TRASH_TYPE_ENCODER_PATH)

    return {
        "decision_tree": {"accuracy": round(dt_acc, 4), "f1_macro": round(dt_f1, 4)},
        "random_forest": {"accuracy": round(rf_acc, 4), "f1_macro": round(rf_f1, 4)},
        "features": list(X_train.columns),
    }


def train_amount_regressor(df: pd.DataFrame, encoder) -> dict:
    """Trains amount regressor (reuses priority encoder and features)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    print("\n--- Training Amount Regressor ---")
    y = df[config.COL_AMOUNT]
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    X_train = features.encode_features(
        df_train, encoder, fit=False,
        categorical=config.CATEGORICAL_FEATURES,
        numeric=config.NUMERIC_FEATURES
    )
    X_test = features.encode_features(
        df_test, encoder, fit=False,
        categorical=config.CATEGORICAL_FEATURES,
        numeric=config.NUMERIC_FEATURES
    )

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"  Amount Regressor -> MAE: {mae:.2f}, R2: {r2:.3f}")

    joblib.dump(rf, config.AMOUNT_RF_PATH)

    return {
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }


def main(enrich: bool = True) -> None:
    print("1) Data laden, opschonen en doel afleiden...")
    df = data_loader.clean(data_loader.load_raw())
    df = data_loader.derive_target(df)
    print(f"   {len(df)} rijen. Prioriteit verdeling:")
    print(df[config.TARGET].value_counts().to_string())

    if enrich:
        print("2) Locatietype afleiden via reverse geocoding (Nominatim)...")
        df = location.enrich(df, config.COL_LAT, config.COL_LON, config.COL_LOCATION_TYPE)
    else:
        print("2) Verrijking overgeslagen (--no-enrich) -> location_type = 'other'")

    print("3) Feature engineering...")
    df = data_loader.add_date_features(df)
    df = data_loader.add_event_feature(df)
    df = data_loader.fill_missing_enrichment(df)

    y = df[config.TARGET]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    print("4) Train/test split...")
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    X_train = features.encode_features(df_train, encoder, fit=True)
    X_test = features.encode_features(df_test, encoder, fit=False)

    print("5) Modellen trainen...")
    dt = DecisionTreeClassifier(max_depth=6, random_state=config.RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    print("6) Evalueren...")
    dt_metrics = evaluate("Decision Tree", dt, X_test, y_test)
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

    print("\n7) Modellen opslaan...")
    joblib.dump(dt, config.DECISION_TREE_PATH)
    joblib.dump(rf, config.RANDOM_FOREST_PATH)
    joblib.dump(encoder, config.ENCODER_PATH)

    # Train trash-type classifier
    trash_type_metrics = train_trash_type_classifier(df)

    # Train amount regressor (reuses the priority encoder)
    amount_metrics = train_amount_regressor(df, encoder)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "target": config.TARGET,
        "target_derived_from": config.COL_AMOUNT,
        "classes": config.TARGET_CLASSES,
        "features": list(X_train.columns),
        "location_enriched": enrich,
        "priority_task": {
            "decision_tree": dt_metrics,
            "random_forest": rf_metrics,
            "production_model": "random_forest",
        },
        "trash_type_task": trash_type_metrics,
        "amount_task": amount_metrics,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"\n   Modellen opgeslagen in {config.MODELS_DIR}")
    print(f"   Metadata: {config.METADATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-enrich", action="store_true",
        help="skip the reverse-geocoding enrichment (faster / offline)",
    )
    args = parser.parse_args()
    main(enrich=not args.no_enrich)
