"""Traint en evalueert een Decision Tree en Random Forest op de Breda afvaldata,
en slaat de modellen + metadata op.

Het doel (inzamelprioriteit) wordt afgeleid van WasteAmount. Temperatuur en
weertype zijn al in de dataset; de coördinaten worden verrijkt met een
locatietype via reverse geocoding, en een evenementgrootte via de evenementen-
kalender.

Gebruik:
    python -m src.train                # met locatie verrijking (Nominatim)
    python -m src.train --no-enrich    # zonder (sneller / offline testen)
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
    """Berekent en print de evaluatie metrics voor één model."""
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

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "target": config.TARGET,
        "target_derived_from": config.COL_AMOUNT,
        "classes": config.TARGET_CLASSES,
        "features": list(X_train.columns),
        "location_enriched": enrich,
        "decision_tree": dt_metrics,
        "random_forest": rf_metrics,
        "production_model": "random_forest",
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"   Modellen opgeslagen in {config.MODELS_DIR}")
    print(f"   Metadata: {config.METADATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-enrich", action="store_true",
        help="sla de reverse-geocoding verrijking over (sneller / offline)",
    )
    args = parser.parse_args()
    main(enrich=not args.no_enrich)
