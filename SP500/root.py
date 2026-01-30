from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DATA"
RAW_DIR = DATA_DIR
PROCESSED_DIR = DATA_DIR / "processed"

PRICE_PARQUET = PROCESSED_DIR / "price.parquet"
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_PRICE_PARQUET = FEATURES_DIR / "features_price.parquet"
