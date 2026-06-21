"""FastAPI service voor afvalinzameling prioriteit.

Endpoints:
    POST /predict        locatie + omgevingsparameters -> inzamelprioriteit
    GET  /health         status van de service, modellen en database
    GET  /model/info     modelversie + metrics (vereist API key)
    GET  /predictions    recente voorspellingen van SQL Server (vereist API key)

De service herkent het locatietype via reverse geocoding, voorspelt de basis
prioriteit met het model, verhoogt deze live voor nabijgelegen Ticketmaster events,
en logt elke voorspelling naar SQL Server (indien geconfigureerd).
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

_state: dict = {"models": {}, "encoder": None, "metadata": None}

# --- Autorisatie -----------------------------------------------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key() -> str:
    """Haalt de verwachte API key op uit de omgevingsvariabele."""
    return os.getenv("API_KEY", "default-insecure-key")


def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Dependency om de API key te verifiëren."""
    expected_key = _get_api_key()
    if api_key is None or api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Ongeldige of ontbrekende API key. Gebruik header 'X-API-Key'.",
        )
    return api_key


def _load_models() -> None:
    try:
        _state["models"]["random_forest"] = joblib.load(config.RANDOM_FOREST_PATH)
        _state["models"]["decision_tree"] = joblib.load(config.DECISION_TREE_PATH)
        _state["encoder"] = joblib.load(config.ENCODER_PATH)
        _state["metadata"] = json.loads(config.METADATA_PATH.read_text())
        print("API: modellen geladen")
    except Exception as exc:
        print(f"API: modellen nog niet geladen ({exc}) -- train eerst met src.train")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    db.init_db()
    yield


app = FastAPI(
    title="Afvalinzameling Prioriteit API",
    description="Voorspelt de inzamelprioriteit van (straat)afval op basis van "
    "locatie, weer en tijd, verrijkt met locatietype en live evenementdata.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic models -----------------------------------------------------
class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, examples=[51.5887], description="Breedtegraad")
    longitude: float = Field(..., ge=-180, le=180, examples=[4.7750], description="Lengtegraad")
    waste_type: Literal[
        "Residual", "Bulky", "Paper/Cardboard", "Plastic",
        "Electronics", "Glass", "Cans",
    ] = Field(..., examples=["Plastic"], description="Type afval")
    temperature: float = Field(..., examples=[18.5], description="Temperatuur in °C")
    weather_type: Literal["Storm", "Cloudy", "Fog", "Rain", "Sunny"] = Field(
        ..., examples=["Sunny"], description="Weertype"
    )
    date_time: Optional[datetime] = Field(None, description="Optioneel; standaard is nu")
    model: Literal["random_forest", "decision_tree"] = Field(
        default="random_forest", description="Model type"
    )


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


# --- Helper --------------------------------------------------------------
def _build_request_dataframe(req: PredictionRequest, loc_type: str) -> pd.DataFrame:
    when = req.date_time or datetime.now()
    row = {
        config.COL_WASTE_CATEGORY: req.waste_type,
        config.COL_WEATHER: req.weather_type,
        config.COL_LOCATION_TYPE: loc_type,
        config.COL_TEMP: req.temperature,
        config.COL_HOUR: when.hour,
        config.COL_DOW: when.weekday(),
        config.COL_MONTH: when.month,
        config.COL_EVENT: event_calendar.size_for(
            req.latitude, req.longitude, when
        ),
    }
    return pd.DataFrame([row])


# --- Endpoints -----------------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint (geen autorisatie vereist)."""
    return {
        "status": "ok",
        "models_loaded": bool(_state["models"]),
        "database": "aan" if db.is_enabled() else "uit",
    }


@app.get("/model/info")
def model_info(_: str = Security(verify_api_key)):
    """Model informatie (vereist API key)."""
    if not _state["metadata"]:
        raise HTTPException(503, "Geen model metadata; train de modellen eerst.")
    return _state["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, _: str = Security(verify_api_key)):
    """Voorspel de inzamelprioriteit (vereist API key)."""
    model = _state["models"].get(req.model)
    encoder = _state["encoder"]
    if model is None or encoder is None:
        raise HTTPException(
            503, "Modellen niet geladen. Voer eerst uit: python -m src.train"
        )

    # 1. Herken de locatie (park/plein/...) en basis voorspelling
    loc = location.lookup(req.latitude, req.longitude)
    when = req.date_time or datetime.now()
    calendar_event = event_calendar.event_for(req.latitude, req.longitude, when)
    df = _build_request_dataframe(req, loc["type"])
    X = features.encode_features(df, encoder, fit=False)
    base_priority = str(model.predict(X)[0])

    # 2. Haal live evenementen op (Ticketmaster) en verhoog prioriteit indien nodig
    nearby = events.events_near(req.latitude, req.longitude)
    final_priority = events.adjust_priority(base_priority, nearby)

    event_name = calendar_event["name"] if calendar_event else None
    parts = []
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

    # 3. Log naar SQL Server (no-op als DB is uitgeschakeld)
    db.save_prediction(
        {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "waste_type": req.waste_type,
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
    )


@app.get("/predictions")
def recent_predictions(limit: int = 20, _: str = Security(verify_api_key)):
    """Recente voorspellingen uit de database (vereist API key)."""
    if not db.is_enabled():
        return {"database": "uit", "predictions": []}
    return {"database": "aan", "predictions": db.get_recent_predictions(limit)}
