"""FastAPI service for waste collection priority.

Endpoints:
    POST /predict        location + environment parameters -> collection priority
    GET  /health         status of the service, models and database
    GET  /model/info     model version + metrics
    GET  /predictions    recent predictions from SQL Server

The service recognizes the location type via reverse geocoding, predicts the base
priority with the model, raises it live for nearby Ticketmaster events, and logs
every prediction to SQL Server (if configured).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config, db, features
from src.enrichment import event_calendar, events, location

_state: dict = {"models": {}, "encoder": None, "metadata": None}


def _load_models() -> None:
    try:
        _state["models"]["random_forest"] = joblib.load(config.RANDOM_FOREST_PATH)
        _state["models"]["decision_tree"] = joblib.load(config.DECISION_TREE_PATH)
        _state["encoder"] = joblib.load(config.ENCODER_PATH)
        _state["metadata"] = json.loads(config.METADATA_PATH.read_text())
        print("API: models loaded")
    except Exception as exc:
        print(f"API: models not loaded yet ({exc}) -- train first with src.train")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    db.init_db()
    yield


app = FastAPI(
    title="Waste Collection Priority API",
    description="Predicts the collection priority of (street) waste based on "
    "location, weather and time, enriched with location type and live event data.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic models -----------------------------------------------------
class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, examples=[51.5887])
    longitude: float = Field(..., ge=-180, le=180, examples=[4.7750])
    waste_type: Literal[
        "Residual", "Bulky", "Paper/Cardboard", "Plastic",
        "Electronics", "Glass", "Cans",
    ] = Field(..., examples=["Plastic"])
    temperature: float = Field(..., examples=[18.5])
    weather_type: Literal["Storm", "Cloudy", "Fog", "Rain", "Sunny"] = Field(
        ..., examples=["Sunny"]
    )
    date_time: Optional[datetime] = Field(None, description="optional; defaults to now")
    model: Literal["random_forest", "decision_tree"] = "random_forest"


class EventOut(BaseModel):
    name: str
    date: str
    type: str


class LocationOut(BaseModel):
    name: str
    type: str


class PredictionResponse(BaseModel):
    base_priority: str
    final_priority: str
    model: str
    location: LocationOut
    known_event: Optional[str]
    nearby_events: list[EventOut]
    explanation: str


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
    return {
        "status": "ok",
        "models_loaded": bool(_state["models"]),
        "database": "on" if db.is_enabled() else "off",
    }


@app.get("/model/info")
def model_info():
    if not _state["metadata"]:
        raise HTTPException(503, "No model metadata; train the models first.")
    return _state["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    model = _state["models"].get(req.model)
    encoder = _state["encoder"]
    if model is None or encoder is None:
        raise HTTPException(
            503, "Models not loaded. Run first: python -m src.train"
        )

    # 1. Recognize the location (park/square/...) and base prediction
    loc = location.lookup(req.latitude, req.longitude)
    when = req.date_time or datetime.now()
    calendar_event = event_calendar.event_for(req.latitude, req.longitude, when)
    df = _build_request_dataframe(req, loc["type"])
    X = features.encode_features(df, encoder, fit=False)
    base_priority = str(model.predict(X)[0])

    # 2. Fetch live events (Ticketmaster) and raise the priority if needed
    nearby = events.events_near(req.latitude, req.longitude)
    final_priority = events.adjust_priority(base_priority, nearby)

    event_name = calendar_event["name"] if calendar_event else None
    parts = []
    if calendar_event:
        parts.append(
            f"known event '{event_name}' is taking place here -> "
            "factored into the model"
        )
    if nearby and final_priority != base_priority:
        parts.append(
            f"{len(nearby)} live event(s) nearby -> priority raised to "
            f"'{final_priority}'"
        )
    elif nearby:
        parts.append(f"{len(nearby)} live event(s) nearby")
    if not parts:
        parts.append(
            f"no events at '{loc['name']}' ({loc['type']}); "
            "priority follows the model"
        )
    explanation = "; ".join(parts).capitalize() + "."

    # 3. Log to SQL Server (no-op if the DB is disabled)
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
def recent_predictions(limit: int = 20):
    if not db.is_enabled():
        return {"database": "off", "predictions": []}
    return {"database": "on", "predictions": db.get_recent_predictions(limit)}
