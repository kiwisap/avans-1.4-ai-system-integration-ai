"""Central configuration for the trash-collection-priority project.

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
COL_TRASH_CATEGORY = "TrashType"
COL_WEATHER = "WeatherType"
COL_TEMP = "Temperature"
COL_AMOUNT = "TrashAmount"            # basis for the target; NOT used as a feature
COL_LOCATION_TYPE = "location_type"   # derived via reverse geocoding

# Derived time features
COL_HOUR = "hour"
COL_DOW = "day_of_week"
COL_MONTH = "month"
COL_EVENT = "event_size"              # derived from the events calendar (0-3)

# Allowed values (used for API validation)
TRASH_TYPES = [
    "Residual", "Bulky", "Paper/Cardboard", "Plastic",
    "Electronics", "Glass", "Cans",
]
WEATHER_TYPES = ["Storm", "Cloudy", "Fog", "Rain", "Sunny"]
LOCATION_TYPES = ["park", "square", "residential", "industrial", "other"]

# --- Target: collection priority, derived from TrashAmount ---------------
TARGET = "collection_priority"
TARGET_CLASSES = ["low", "medium", "high"]
# Binning of TrashAmount (0-20) -> priority. Edges give balanced classes.
PRIORITY_BIN_EDGES = [-1, 3, 6, float("inf")]   # (0-3], (3-6], (6+)
PRIORITY_BIN_LABELS = ["low", "medium", "high"]

# --- Features ------------------------------------------------------------
CATEGORICAL_FEATURES = [COL_TRASH_CATEGORY, COL_WEATHER, COL_LOCATION_TYPE]
NUMERIC_FEATURES = [COL_TEMP, COL_HOUR, COL_DOW, COL_MONTH, COL_EVENT]

# --- Model files ---------------------------------------------------------
DECISION_TREE_PATH = MODELS_DIR / "decision_tree.pkl"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42

# --- Second task: predicting the trash type ------------------------------
# TrashType is the target here, so it is NOT a feature. TrashAmount stays
# excluded as well (it is only measured after the fact).
TRASH_TYPE_TARGET = COL_TRASH_CATEGORY
TRASH_TYPE_CLASSES = TRASH_TYPES
TRASH_TYPE_CATEGORICAL_FEATURES = [COL_WEATHER, COL_LOCATION_TYPE]
TRASH_TYPE_NUMERIC_FEATURES = [COL_TEMP, COL_HOUR, COL_DOW, COL_MONTH, COL_EVENT]

TRASH_TYPE_DT_PATH = MODELS_DIR / "trash_type_decision_tree.pkl"
TRASH_TYPE_RF_PATH = MODELS_DIR / "trash_type_random_forest.pkl"
TRASH_TYPE_ENCODER_PATH = MODELS_DIR / "trash_type_encoder.pkl"