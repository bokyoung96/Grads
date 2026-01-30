from __future__ import annotations

import logging
import warnings
import sys
from pathlib import Path

PREPROCESS_DIR = Path(__file__).resolve().parent
SP500_DIR = PREPROCESS_DIR.parent
REPO_ROOT = SP500_DIR.parent

for cand in (REPO_ROOT, SP500_DIR):
    if str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

from SP500.preprocess.base.align import Align
from SP500.preprocess.base.load import Load
from SP500.preprocess.base.duck import DB
from SP500.preprocess.factory.data import Data
from SP500.preprocess.entry import register


PARQUET_CHUNK_ROWS = 200_000


def main():
    warnings.filterwarnings(
        "ignore",
        message="Dtype inference on a pandas object.*",
        category=FutureWarning,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    register()
    loader = Load()
    aligner = Align()
    db = DB()
    data = Data(loader, aligner, db)
    data.make(parquet_chunk_rows=PARQUET_CHUNK_ROWS)


if __name__ == "__main__":
    main()
