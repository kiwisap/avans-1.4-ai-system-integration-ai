"""Curated events calendar for Breda.

Unlike Ticketmaster (only commercially ticketed events, no history), a calendar
is known for both the past and the future. That lets "event" become a real model
feature: derived per date + location during training, and in exactly the same way
at inference.

Events recur annually; we match on (month, day) within a window and on distance
to the event location. "size" (1-3) scales the effect.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

# name, lat, lon, radius_km, start (month, day), end (month, day), size
EVENTS = [
    {"name": "Carnival", "lat": 51.5887, "lon": 4.7750, "radius_km": 2.0,
     "start": (2, 14), "end": (2, 18), "size": 3},
    {"name": "King's Day", "lat": 51.5887, "lon": 4.7750, "radius_km": 2.0,
     "start": (4, 27), "end": (4, 27), "size": 3},
    {"name": "Liberation Festival", "lat": 51.5862, "lon": 4.7805, "radius_km": 1.2,
     "start": (5, 5), "end": (5, 5), "size": 2},
    {"name": "Breda Jazz Festival", "lat": 51.5890, "lon": 4.7760, "radius_km": 2.0,
     "start": (5, 14), "end": (5, 17), "size": 3},
    {"name": "Breda Live", "lat": 51.5862, "lon": 4.7805, "radius_km": 1.2,
     "start": (6, 26), "end": (6, 28), "size": 3},
    {"name": "Valkenberg Summer Festival", "lat": 51.5886, "lon": 4.7766, "radius_km": 0.8,
     "start": (7, 10), "end": (7, 20), "size": 2},
    {"name": "National Tattoo", "lat": 51.5887, "lon": 4.7750, "radius_km": 1.2,
     "start": (9, 12), "end": (9, 14), "size": 2},
    {"name": "Breda Singelloop", "lat": 51.5880, "lon": 4.7780, "radius_km": 2.5,
     "start": (10, 4), "end": (10, 5), "size": 2},
    {"name": "Ginneken Christmas Market", "lat": 51.5660, "lon": 4.7880, "radius_km": 1.0,
     "start": (12, 14), "end": (12, 23), "size": 2},
]


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _doy(month: int, day: int) -> int:
    return datetime(2001, month, day).timetuple().tm_yday  # fixed non-leap year


def event_for(lat: float, lon: float, when) -> Optional[dict]:
    """Returns the (largest) event taking place at this location on this date."""
    if not isinstance(when, datetime):
        when = datetime.fromisoformat(str(when))
    wd = when.timetuple().tm_yday
    best = None
    for ev in EVENTS:
        s, e = _doy(*ev["start"]), _doy(*ev["end"])
        in_window = (s <= wd <= e) if s <= e else (wd >= s or wd <= e)
        if in_window and _haversine(lat, lon, ev["lat"], ev["lon"]) <= ev["radius_km"]:
            if best is None or ev["size"] > best["size"]:
                best = ev
    return best


def size_for(lat: float, lon: float, when) -> int:
    """Returns 0 if no event is taking place, otherwise the size (1-3)."""
    ev = event_for(lat, lon, when)
    return ev["size"] if ev else 0
