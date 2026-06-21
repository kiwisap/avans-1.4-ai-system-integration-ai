"""Central configuration for the waste-collection-priority project.

Tailored to the dataset litter_breda.csv. All paths, column names, features and
the target definition live here in one place, so you can plug in your own data
by adjusting only this file.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

RAW_DATA_PATH = DATA_DIR / "litter_breda.csv"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# --- Dataset schema (columns in litter_breda.csv) ------------------------
COL_DATE = "DateTime"
COL_LAT = "Latitude"
COL_LON = "Longitude"
COL_WASTE_CATEGORY = "WasteType"
COL_WEATHER = "WeatherType"
COL_TEMP = "Temperature"
COL_AMOUNT = "WasteAmount"            # basis for the target; NOT used as a feature
COL_LOCATION_TYPE = "location_type"   # derived via reverse geocoding

# Derived time features
COL_HOUR = "hour"
COL_DOW = "day_of_week"
COL_MONTH = "month"
COL_EVENT = "event_size"              # derived from the events calendar (0-3)

# Allowed values (used for API validation)
WASTE_TYPES = [
    "Residual", "Bulky", "Paper/Cardboard", "Plastic",
    "Electronics", "Glass", "Cans",
]
WEATHER_TYPES = ["Storm", "Cloudy", "Fog", "Rain", "Sunny"]
LOCATION_TYPES = ["park", "square", "residential", "industrial", "other"]

# --- Target: collection priority, derived from WasteAmount ---------------
TARGET = "collection_priority"
TARGET_CLASSES = ["low", "medium", "high"]
# Binning of WasteAmount (0-20) -> priority. Edges give balanced classes.
PRIORITY_BIN_EDGES = [-1, 3, 6, float("inf")]   # (0-3], (3-6], (6+)
PRIORITY_BIN_LABELS = ["low", "medium", "high"]

# --- Features ------------------------------------------------------------
CATEGORICAL_FEATURES = [COL_WASTE_CATEGORY, COL_WEATHER, COL_LOCATION_TYPE]
NUMERIC_FEATURES = [COL_TEMP, COL_HOUR, COL_DOW, COL_MONTH, COL_EVENT]

# --- Model files ---------------------------------------------------------
DECISION_TREE_PATH = MODELS_DIR / "decision_tree.pkl"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42
