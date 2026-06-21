"""Live event enrichment via the Ticketmaster Discovery API.

Used at inference (in the /predict call) to check whether there are upcoming
events near the given location. If so, the predicted collection priority is
raised. This complements the events calendar, which already covers known
recurring events; Ticketmaster adds live, commercial events.

Requires the environment variable TICKETMASTER_API_KEY. If it is missing or the
call fails, the function degrades gracefully: no events, no priority increase.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, TypedDict

import requests

DISCOVERY_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

# Geohash encoder: Ticketmaster's geoPoint expects a geohash instead of the
# deprecated latlong parameter. Precision 7 (~150 m) keeps the radius search
# from being too narrow.
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def _geohash(lat: float, lon: float, precision: int = 7) -> str:
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    bits = [16, 8, 4, 2, 1]
    out, bit, ch, even = [], 0, 0, True
    while len(out) < precision:
        if even:
            mid = sum(lon_range) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            mid = sum(lat_range) / 2
            if lat > mid:
                ch |= bits[bit]
                lat_range[0] = mid
            else:
                lat_range[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_BASE32[ch])
            bit, ch = 0, 0
    return "".join(out)


class Event(TypedDict):
    name: str
    date: str
    type: str


def _api_key() -> str | None:
    return os.getenv("TICKETMASTER_API_KEY")


def events_near(
    lat: float,
    lon: float,
    days_ahead: int = 7,
    radius_km: int = 10,
    max_events: int = 20,
) -> List[Event]:
    """Fetches upcoming events within a radius around the location.

    Returns an empty list if there is no API key or the call fails, so the
    prediction can always continue.
    """
    key = _api_key()
    if not key:
        return []

    now = datetime.now(timezone.utc)
    params = {
        "apikey": key,
        "geoPoint": _geohash(lat, lon),
        "radius": radius_km,
        "unit": "km",
        "startDateTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDateTime": (now + timedelta(days=days_ahead)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "countryCode": "NL",
        "size": max_events,
        "sort": "date,asc",
    }

    try:
        resp = requests.get(DISCOVERY_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # network/timeout/HTTP error -> no events
        print(f"  ! Ticketmaster call failed: {exc}")
        return []

    raw_events = data.get("_embedded", {}).get("events", [])

    events: List[Event] = []
    for ev in raw_events:
        classifications = ev.get("classifications", [])
        segment = "unknown"
        if classifications:
            segment = classifications[0].get("segment", {}).get("name", "unknown")
        events.append(
            {
                "name": ev.get("name", "unknown"),
                "date": ev.get("dates", {}).get("start", {}).get("localDate", ""),
                "type": segment,
            }
        )
    return events


_PRIORITY_ORDER = ["low", "medium", "high"]


def adjust_priority(base_priority: str, events: List[Event]) -> str:
    """Raises the priority based on the number of nearby events.

    No events  -> unchanged
    1-2 events -> one level up
    3+ events  -> two levels up (capped at "high")
    """
    if base_priority not in _PRIORITY_ORDER:
        return base_priority

    n = len(events)
    if n == 0:
        return base_priority

    steps = 1 if n < 3 else 2
    idx = _PRIORITY_ORDER.index(base_priority)
    new_idx = min(idx + steps, len(_PRIORITY_ORDER) - 1)
    return _PRIORITY_ORDER[new_idx]
