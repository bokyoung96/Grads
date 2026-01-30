from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from SP500.root import PRICE_PARQUET


class DB:
    def __init__(self, path: str | Path = PRICE_PARQUET):
        self.default_path = Path(path)

    def save_parquet(self, df: Any, path: str | Path | None = None, *, chunk_rows: Optional[int] = None):
        dest = Path(path) if path is not None else self.default_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        df = df.copy(deep=False)
        df.index = pd.to_datetime(df.index)
        idx_name = df.index.name or "date"
        df.index.name = idx_name
        if not all(isinstance(c, str) for c in df.columns):
            df.columns = [str(c) for c in df.columns]

        if chunk_rows is None:
            env = os.getenv("GRADS_PARQUET_CHUNK_ROWS", "").strip()
            if env:
                try:
                    chunk_rows = int(env)
                except ValueError:
                    chunk_rows = None

        if not chunk_rows or chunk_rows <= 0:
            df.to_parquet(dest, index=True)
            return

        if len(df) == 0:
            df.to_parquet(dest, index=True)
            return

        writer = None
        schema = None
        try:
            for start in range(0, len(df), int(chunk_rows)):
                batch = df.iloc[start : start + int(chunk_rows)]
                obj_cols = list(batch.select_dtypes(include=["object"]).columns)
                if obj_cols:
                    batch = batch.copy()
                    for col in obj_cols:
                        batch[col] = batch[col].astype("string")
                table = pa.Table.from_pandas(batch, preserve_index=True)
                if writer is None:
                    schema = pa.schema([field.with_nullable(True) for field in table.schema])
                    schema = schema.with_metadata(table.schema.metadata)
                    writer = pq.ParquetWriter(str(dest), schema=schema)
                if table.schema != schema:
                    table = table.cast(schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
