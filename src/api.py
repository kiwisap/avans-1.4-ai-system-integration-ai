"""FastAPI service for zwerfafvalherkenning priority predictions.

Endpoints:
    POST /predict        location + environment parameters -> collection priority
    GET  /health         service, model and database status
    GET  /model/info     model version + metrics
    GET  /predictions    recent predictions

The service recognizes location type via reverse geocoding, predicts the base
priority with the model, adjusts it live for nearby Ticketmaster events,
and logs every prediction to the database.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src import config, db, features
from src.enrichment import event_calendar, events, location

_state: dict = {
    "models": {},
    "encoder": None,
    "metadata": None,
    "trash_type_models": {},
    "trash_type_encoder": None,
    "amount_model": None,
}

# --- Authorization ---------------------------------------------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key() -> str:
    """Get the expected API key from environment variable."""
    return os.getenv("API_KEY", "default-insecure-key")


def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Dependency to verify the API key."""
    expected_key = _get_api_key()
    if api_key is None or api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Ongeldige of ontbrekende API key. Gebruik header 'X-API-Key'.",
        )
    return api_key


def _load_models() -> None:
    try:
        # Load priority prediction models
        _state["models"]["random_forest"] = joblib.load(config.RANDOM_FOREST_PATH)
        _state["models"]["decision_tree"] = joblib.load(config.DECISION_TREE_PATH)
        _state["encoder"] = joblib.load(config.ENCODER_PATH)
        _state["metadata"] = json.loads(config.METADATA_PATH.read_text())

        # Load trash type prediction models
        _state["trash_type_models"]["random_forest"] = joblib.load(config.TRASH_TYPE_RF_PATH)
        _state["trash_type_models"]["decision_tree"] = joblib.load(config.TRASH_TYPE_DT_PATH)
        _state["trash_type_encoder"] = joblib.load(config.TRASH_TYPE_ENCODER_PATH)

        # Load amount prediction model
        _state["amount_model"] = joblib.load(config.AMOUNT_RF_PATH)

        print("API: alle modellen geladen")
    except Exception as exc:
        print(f"API: modellen nog niet geladen ({exc}) -- train eerst met src.train")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    db.init_db()
    yield


app = FastAPI(
    title="Zwerfafvalherkenning voorspelling API",
    description="Voorspelt de ophaalprioriteit van afval op basis van "
    "locatie, weer en tijd, verrijkt met locatietype en live evenementdata.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic models -----------------------------------------------------
class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, examples=[51.5887], description="Breedtegraad")
    longitude: float = Field(..., ge=-180, le=180, examples=[4.7750], description="Lengtegraad")
    trash_type: Literal[
        "Residual", "Bulky", "Paper/Cardboard", "Plastic",
        "Electronics", "Glass", "Cans",
    ] = Field(None, examples=["Plastic"], description="Type afval (optioneel; wordt voorspeld indien niet opgegeven)")
    temperature: float = Field(..., examples=[18.5], description="Temperatuur in °C")
    weather_type: Literal["Storm", "Cloudy", "Fog", "Rain", "Sunny"] = Field(
        ..., examples=["Sunny"], description="Weertype"
    )
    date_time: Optional[datetime] = Field(None, description="Optioneel; standaard is nu")
    model: Literal["random_forest", "decision_tree"] = Field(
        default="random_forest", description="Model type"
    )


class ClassProbability(BaseModel):
    label: str = Field(..., description="Klasselabel")
    probability: float = Field(..., description="Waarschijnlijkheid")


class EventOut(BaseModel):
    name: str = Field(..., description="Naam van het evenement")
    date: str = Field(..., description="Datum van het evenement")
    type: str = Field(..., description="Type evenement")


class LocationOut(BaseModel):
    name: str = Field(..., description="Naam van de locatie")
    type: str = Field(..., description="Type locatie")


class PredictionResponse(BaseModel):
    base_priority: str = Field(..., description="Basis prioriteit van het model")
    final_priority: str = Field(..., description="Eindprioriteit na aanpassing")
    model: str = Field(..., description="Gebruikt model")
    location: LocationOut = Field(..., description="Locatie informatie")
    known_event: Optional[str] = Field(None, description="Bekend evenement uit de kalender")
    nearby_events: list[EventOut] = Field(..., description="Nabijgelegen live evenementen")
    explanation: str = Field(..., description="Uitleg van de voorspelling")
    # New fields
    priority_probabilities: list[ClassProbability] = Field(..., description="Prioriteit waarschijnlijkheden per klasse")
    estimated_amount: Optional[float] = Field(None, description="Voorspelde afvalhoeveelheid")
    trash_type: str = Field(..., description="Gebruikt/voorspeld afvaltype")
    trash_type_provided: bool = Field(..., description="Of afvaltype was opgegeven")
    trash_type_probabilities: list[ClassProbability] = Field(..., description="Afvaltype waarschijnlijkheden per klasse")


# --- Helper --------------------------------------------------------------
def _predict_trash_type(lat: float, lon: float, weather: str, temp: float, when: datetime, loc_type: str, event_size: int) -> tuple[str, list[tuple[str, float]]]:
    """Predicts trash type and returns (top_label, [(label, prob), ...] sorted desc)."""
    model = _state["trash_type_models"].get("random_forest")
    encoder = _state["trash_type_encoder"]

    if model is None or encoder is None:
        # Fallback to most common type if model not loaded
        return ("Plastic", [("Plastic", 1.0)])

    row = {
        config.COL_WEATHER: weather,
        config.COL_LOCATION_TYPE: loc_type,
        config.COL_TEMP: temp,
        config.COL_HOUR: when.hour,
        config.COL_DOW: when.weekday(),
        config.COL_MONTH: when.month,
        config.COL_EVENT: event_size,
    }
    df = pd.DataFrame([row])
    X = features.encode_features(
        df, encoder, fit=False,
        categorical=config.TRASH_TYPE_CATEGORICAL_FEATURES,
        numeric=config.TRASH_TYPE_NUMERIC_FEATURES
    )

    predicted_type = str(model.predict(X)[0])
    probas = model.predict_proba(X)[0]

    # Pair labels with probabilities and sort descending
    label_prob_pairs = list(zip(model.classes_, probas))
    label_prob_pairs.sort(key=lambda x: x[1], reverse=True)

    return predicted_type, label_prob_pairs


def _build_request_dataframe(trash_type: str, weather_type: str, loc_type: str,
                            temp: float, when: datetime, event_size: int) -> pd.DataFrame:
    row = {
        config.COL_TRASH_CATEGORY: trash_type,
        config.COL_WEATHER: weather_type,
        config.COL_LOCATION_TYPE: loc_type,
        config.COL_TEMP: temp,
        config.COL_HOUR: when.hour,
        config.COL_DOW: when.weekday(),
        config.COL_MONTH: when.month,
        config.COL_EVENT: event_size,
    }
    return pd.DataFrame([row])


# --- Endpoints -----------------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint (no authorization required)."""
    return {
        "status": "ok",
        "models_loaded": bool(_state["models"]),
        "database": "aan" if db.is_enabled() else "uit",
    }


@app.get("/model/info")
def model_info(_: str = Security(verify_api_key)):
    """Model information (requires API key)."""
    if not _state["metadata"]:
        raise HTTPException(503, "Geen model metadata; train de modellen eerst.")
    return _state["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, _: str = Security(verify_api_key)):
    """Predict collection priority (requires API key)."""
    model = _state["models"].get(req.model)
    encoder = _state["encoder"]
    if model is None or encoder is None:
        raise HTTPException(
            503, "Modellen niet geladen. Voer eerst uit: python -m src.train"
        )

    # 1. Recognize location and get event info
    loc = location.lookup(req.latitude, req.longitude)
    when = req.date_time or datetime.now()
    calendar_event = event_calendar.event_for(req.latitude, req.longitude, when)
    event_size = event_calendar.size_for(req.latitude, req.longitude, when)

    # 2. Predict Trash type if not provided
    trash_type_provided = req.trash_type is not None
    if trash_type_provided:
        effective_type = req.trash_type
        # Still get probabilities for provided type
        _, trash_type_probs = _predict_trash_type(
            req.latitude, req.longitude, req.weather_type, req.temperature,
            when, loc["type"], event_size
        )
    else:
        effective_type, trash_type_probs = _predict_trash_type(
            req.latitude, req.longitude, req.weather_type, req.temperature,
            when, loc["type"], event_size
        )

    # 3. Build features and predict priority with probabilities
    df = _build_request_dataframe(
        effective_type, req.weather_type, loc["type"],
        req.temperature, when, event_size
    )
    X = features.encode_features(df, encoder, fit=False)
    base_priority = str(model.predict(X)[0])

    # Get priority probabilities
    priority_probas = model.predict_proba(X)[0]
    priority_probs = [
        (label, float(prob))
        for label, prob in zip(model.classes_, priority_probas)
    ]
    priority_probs.sort(key=lambda x: x[1], reverse=True)

    # 4. Predict estimated amount
    estimated_amount = None
    if _state["amount_model"] is not None:
        try:
            estimated_amount = float(_state["amount_model"].predict(X)[0])
        except Exception:
            pass  # Silently fallback to None

    # 5. Get live events (Ticketmaster) and adjust priority if needed
    nearby = events.events_near(req.latitude, req.longitude)
    final_priority = events.adjust_priority(base_priority, nearby)

    # 6. Build explanation
    event_name = calendar_event["name"] if calendar_event else None
    parts = []
    if not trash_type_provided:
        parts.append(f"afvaltype '{effective_type}' voorspeld")
    if calendar_event:
        parts.append(
            f"bekend evenement '{event_name}' vindt hier plaats -> "
            "meegenomen in het model"
        )
    if nearby and final_priority != base_priority:
        parts.append(
            f"{len(nearby)} live evenement(en) in de buurt -> prioriteit verhoogd naar "
            f"'{final_priority}'"
        )
    elif nearby:
        parts.append(f"{len(nearby)} live evenement(en) in de buurt")
    if not parts:
        parts.append(
            f"geen evenementen bij '{loc['name']}' ({loc['type']}); "
            "prioriteit volgt het model"
        )
    explanation = "; ".join(parts).capitalize() + "."

    # 7. Log to SQL Server (no-op if DB is disabled)
    db.save_prediction(
        {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "trash_type": effective_type,
            "temperature": req.temperature,
            "weather_type": req.weather_type,
            "location_type": loc["type"],
            "base_priority": base_priority,
            "final_priority": final_priority,
            "nearby_events_count": len(nearby),
            "model": req.model,
        }
    )

    return PredictionResponse(
        base_priority=base_priority,
        final_priority=final_priority,
        model=req.model,
        location=LocationOut(name=loc["name"], type=loc["type"]),
        known_event=event_name,
        nearby_events=[EventOut(**e) for e in nearby],
        explanation=explanation,
        priority_probabilities=[
            ClassProbability(label=label, probability=prob)
            for label, prob in priority_probs
        ],
        estimated_amount=estimated_amount,
        trash_type=effective_type,
        trash_type_provided=trash_type_provided,
        trash_type_probabilities=[
            ClassProbability(label=label, probability=prob)
            for label, prob in trash_type_probs
        ],
    )


@app.get("/predictions")
def recent_predictions(limit: int = 20, _: str = Security(verify_api_key)):
    """Recent predictions from database (requires API key)."""
    if not db.is_enabled():
        return {"database": "uit", "predictions": []}
    return {"database": "aan", "predictions": db.get_recent_predictions(limit)}


class TrashTypeRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, examples=[51.5887])
    longitude: float = Field(..., ge=-180, le=180, examples=[4.7750])
    temperature: float = Field(..., examples=[18.5])
    weather_type: Literal["Storm", "Cloudy", "Fog", "Rain", "Sunny"] = Field(..., examples=["Sunny"])
    date_time: Optional[datetime] = Field(None, description="Optioneel; standaard is nu")


class TrashTypeResponse(BaseModel):
    predicted_type: str = Field(..., description="Voorspeld afvaltype")
    probabilities: list[ClassProbability] = Field(..., description="Top waarschijnlijkheden")
    location: LocationOut = Field(..., description="Locatie informatie")


@app.post("/predict/trash-type", response_model=TrashTypeResponse)
def predict_trash_type_only(req: TrashTypeRequest, _: str = Security(verify_api_key)):
    """Predict only trash type based on location, weather, and time (requires API key)."""
    if _state["trash_type_models"].get("random_forest") is None:
        raise HTTPException(503, "Trash-type model niet geladen")
    
    loc = location.lookup(req.latitude, req.longitude)
    when = req.date_time or datetime.now()
    event_size = event_calendar.size_for(req.latitude, req.longitude, when)
    
    predicted_type, probs = _predict_trash_type(
        req.latitude, req.longitude, req.weather_type, req.temperature,
        when, loc["type"], event_size
    )
    
    return TrashTypeResponse(
        predicted_type=predicted_type,
        probabilities=[
            ClassProbability(label=label, probability=prob)
            for label, prob in probs[:5]  # Top 5
        ],
        location=LocationOut(name=loc["name"], type=loc["type"]),
    )


