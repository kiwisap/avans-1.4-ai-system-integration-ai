#!/bin/sh
set -e

# Make sure a trained model exists. If not, train on the available dataset.
if [ ! -f models/random_forest.pkl ]; then
  echo "Training model..."
  python -m src.train --no-enrich
fi

exec uvicorn src.api:app --host 0.0.0.0 --port 8000
