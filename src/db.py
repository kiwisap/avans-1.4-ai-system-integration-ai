"""SQL Server layer for logging and reading back predictions.

The connection is made via SQLAlchemy + pyodbc (ODBC Driver 18), which also works
directly with Azure SQL. Everything degrades gracefully: if there is no database
configuration or the connection fails, the API keeps serving predictions; they
are simply not logged.

Required environment variables:
    DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD   (optional DB_PORT, default 1433)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional

_engine = None
_enabled = False


def _build_connection_url() -> Optional[str]:
    server = os.getenv("DB_SERVER")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT", "1433")

    if not all([server, name, user, password]):
        return None

    return (
        f"mssql+pyodbc://{user}:{password}@{server}:{port}/{name}"
        "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )


def init_db() -> bool:
    """Creates the engine and the predictions table. Returns True on success."""
    global _engine, _enabled

    url = _build_connection_url()
    if url is None:
        print("DB: no configuration found -> logging disabled")
        return False

    try:
        from sqlalchemy import create_engine, text

        _engine = create_engine(url, pool_pre_ping=True)
        with _engine.begin() as conn:
            conn.execute(
                text(
                    """
                    IF NOT EXISTS (
                        SELECT * FROM sysobjects WHERE name='predictions' AND xtype='U'
                    )
                    CREATE TABLE predictions (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        timestamp DATETIME2 NOT NULL,
                        latitude FLOAT NOT NULL,
                        longitude FLOAT NOT NULL,
                        waste_type NVARCHAR(50) NOT NULL,
                        temperature FLOAT NOT NULL,
                        weather_type NVARCHAR(50) NOT NULL,
                        location_type NVARCHAR(50) NOT NULL,
                        base_priority NVARCHAR(20) NOT NULL,
                        final_priority NVARCHAR(20) NOT NULL,
                        nearby_events_count INT NOT NULL,
                        model NVARCHAR(50) NOT NULL
                    )
                    """
                )
            )
        _enabled = True
        print("DB: connected and table ready -> logging enabled")
        return True
    except Exception as exc:
        print(f"DB: connection failed ({exc}) -> logging disabled")
        _enabled = False
        return False


def is_enabled() -> bool:
    return _enabled


def save_prediction(record: dict) -> None:
    """Logs a single prediction. Does nothing if the DB is disabled."""
    if not _enabled or _engine is None:
        return
    try:
        from sqlalchemy import text

        with _engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO predictions
                        (timestamp, latitude, longitude, waste_type,
                         temperature, weather_type, location_type,
                         base_priority, final_priority, nearby_events_count, model)
                    VALUES
                        (:timestamp, :latitude, :longitude, :waste_type,
                         :temperature, :weather_type, :location_type,
                         :base_priority, :final_priority, :nearby_events_count, :model)
                    """
                ),
                {"timestamp": datetime.now(timezone.utc), **record},
            )
    except Exception as exc:
        print(f"DB: saving failed: {exc}")


def get_recent_predictions(limit: int = 20) -> List[dict]:
    """Returns the most recent predictions. Empty list if the DB is disabled."""
    if not _enabled or _engine is None:
        return []
    try:
        from sqlalchemy import text

        with _engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT TOP (:limit) * FROM predictions ORDER BY timestamp DESC"
                ),
                {"limit": limit},
            ).mappings().all()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"DB: reading failed: {exc}")
        return []
