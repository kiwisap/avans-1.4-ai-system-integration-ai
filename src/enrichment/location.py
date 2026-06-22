"""Reverse geocoding via Nominatim (OpenStreetMap).

Converts coordinates into a simplified location type (park / square /
residential / industrial / other) plus a readable name. Used offline during
training (to derive the "location_type" feature) and live in the API.

Respects the Nominatim usage policy: a custom User-Agent and at most ~1 request
per second. Results are cached per rounded coordinate, so repeated locations
(such as fixed bin spots) are only looked up once.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "zwerfafvalherkenning/1.0"

_CACHE: Dict[Tuple[float, float], dict] = {}


def _classify(category: str, osm_type: str, addr: dict) -> str:
    """Maps the raw OSM category/type to a simplified location type."""
    category = (category or "").lower()
    osm_type = (osm_type or "").lower()

    if category == "leisure" and osm_type in {"park", "garden", "recreation_ground"}:
        return "park"
    if osm_type in {"square"} or (category == "highway" and osm_type == "pedestrian"):
        return "square"
    if category == "highway" and osm_type in {"residential", "living_street"}:
        return "residential"
    if "industrial" in osm_type or "retail" in osm_type or "commercial" in osm_type:
        return "industrial"
    # fall back on the address: a street with no clear type = residential
    if addr.get("road"):
        return "residential"
    return "other"


def lookup(lat: float, lon: float) -> dict:
    """Returns {"name": str, "type": str} for the coordinates.

    On error or without a network connection it degrades to type "other", so
    training and the API can always continue.
    """
    key = (round(lat, 3), round(lon, 3))
    if key in _CACHE:
        return _CACHE[key]

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 17,
                "addressdetails": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        addr = data.get("address", {})
        loc_type = _classify(data.get("category", ""), data.get("type", ""), addr)
        name = (
            addr.get("leisure")
            or addr.get("road")
            or addr.get("suburb")
            or data.get("name")
            or "unknown"
        )
        result = {"name": name, "type": loc_type}
    except Exception as exc:
        print(f"  ! Reverse geocoding failed for {lat},{lon}: {exc}")
        result = {"name": "unknown", "type": "other"}

    _CACHE[key] = result
    time.sleep(1.0)  # Nominatim: at most ~1 request per second
    return result


def enrich(df, col_lat: str, col_lon: str, out_col: str):
    """Adds a column with the location type based on lat/lon (cached)."""
    df = df.copy()
    df[out_col] = [
        lookup(row[col_lat], row[col_lon])["type"] for _, row in df.iterrows()
    ]
    return df
