from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

PREPROCESS_DIR = Path(__file__).resolve().parent
SP500_DIR = PREPROCESS_DIR.parent
REPO_ROOT = SP500_DIR.parent
for cand in (REPO_ROOT, SP500_DIR):
    if str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

from SP500.root import FEATURES_PRICE_PARQUET, PRICE_PARQUET, DATA_DIR
from SP500.preprocess.base.util import apply_scaling


def _maybe_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [c for c in df.columns if str(c).lower() in {"date", "index", "level_0"} or str(c).startswith("Unnamed")]
    if date_cols:
        date_col = date_cols[0]
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df.index.name = "date"
        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).lower() in {"index", "level_0"}]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    parsed = []
    tuple_found = False
    for c in df.columns:
        if isinstance(c, tuple):
            parsed.append(c)
            tuple_found = True
            continue
        if isinstance(c, str) and c.startswith("(") and c.endswith(")"):
            try:
                val = ast.literal_eval(c)
                parsed.append(val)
                tuple_found = tuple_found or isinstance(val, tuple)
                continue
            except Exception:
                pass
        parsed.append(c)
    if tuple_found:
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples(parsed)
    return df


def _select(path: Path, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _maybe_multiindex(df)
    if cols:
        first_level = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        keep_mask = first_level.isin(cols) if hasattr(first_level, "isin") else [c in cols for c in first_level]
        df = df.loc[:, keep_mask]
    return df


class Loader:
    def price(self, cols: Optional[Sequence[str]] = None, *, scale: Optional[str] = None, scale_kwargs: Optional[dict] = None):
        df = _select(PRICE_PARQUET, cols)
        if scale:
            df = apply_scaling(df, mode=scale, **(scale_kwargs or {}))
        return df

    def features_price(
        self,
        cols: Optional[Sequence[str]] = None,
        *,
        scale: Optional[str] = None,
        scale_kwargs: Optional[dict] = None,
    ):
        df = _select(FEATURES_PRICE_PARQUET, cols)
        if scale:
            df = apply_scaling(df, mode=scale, **(scale_kwargs or {}))
        return df

    def raw_subset(self, cols: Optional[Sequence[str]] = None):
        return self.price(cols)

    def feature_subset(self, cols: Optional[Sequence[str]] = None):
        return self.features_price(cols)

    def univ_sedol(self, path: Optional[Path] = None, cols: Optional[Sequence[str]] = None):
        target = path or (DATA_DIR / "univ" / "sedol_daily.parquet")
        df = _select(Path(target), cols)
        return df

    def univ(self, name: str = "sedol", *, path: Optional[Path] = None, cols: Optional[Sequence[str]] = None):
        if name == "sedol":
            return self.univ_sedol(path, cols)
        raise ValueError(f"Unsupported universe name: {name}")


if __name__ == "__main__":
    ldr = Loader()
    print(ldr.price().head())
