from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

import pandas as pd

from SP500.root import RAW_DIR, DATA_DIR
from SP500.preprocess.base.util import ensure_datetime_index


def _resolve_latest(stem: str, base: Path, prefer_date: str | None = None) -> Path:
    base_path = base / f"{stem}.csv"
    if base_path.exists():
        return base_path

    cands = list(base.glob(f"{stem}_*.csv"))
    ordered = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)

    if prefer_date:
        tagged = [p for p in ordered if p.stem[len(stem) + 1 :].startswith(prefer_date)]
        if tagged:
            return tagged[0]
    if ordered:
        return ordered[0]

    raise FileNotFoundError(f"CSV not found for stem '{stem}' under {base}")


def load_panel_csv(path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(path, header=None, low_memory=False)

    ticker_row_idx = df_raw.index[df_raw[0] == "FSYM_TICKER_REGION_LOCAL"]
    if len(ticker_row_idx) == 0:
        tick_idx = 0
        raw_tickers = df_raw.iloc[0].tolist()[1:]
        data = df_raw.iloc[1:].copy()
    else:
        tick_idx = ticker_row_idx[0]
        raw_tickers = df_raw.loc[tick_idx].tolist()[1:]
        data = df_raw.iloc[tick_idx + 1 :].copy()

    tickers = []
    for t in raw_tickers:
        if pd.isna(t):
            continue
        s = str(t).strip()
        if not s or s == "#VALUE!":
            continue
        tickers.append(s)

    date_num = pd.to_numeric(data.iloc[:, 0], errors="coerce")
    mask = ~pd.isna(date_num)
    data = data.loc[mask].copy()
    date_vals = date_num.loc[mask].astype("int64", copy=False)
    date_dt = pd.to_datetime(date_vals, unit="D", origin="1899-12-30")
    data.iloc[:, 0] = date_dt.values

    cols = ["date"] + tickers[: data.shape[1] - 1]
    data = data.iloc[:, : len(cols)]
    data.columns = pd.Index(cols, dtype="object")
    data = data.set_index("date")
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.loc[:, [c for c in data.columns if c and c != "#VALUE!"]]
    data = data.dropna(axis=1, how="all")
    return ensure_datetime_index(data)


class Load:
    def __init__(self, path: str | Path | None = None, *, prefer_date: str | None = None):
        self.p = Path(path) if path is not None else RAW_DIR
        self.prefer_date = prefer_date.lstrip("_") if prefer_date else None

    def _resolve_csv_path(self, stem: str) -> Path:
        return _resolve_latest(stem, self.p, self.prefer_date)

    def csv(self, name: str) -> pd.DataFrame:
        return load_panel_csv(self._resolve_csv_path(name))

    def price(self):
        out = {
            "close": self.csv("Daily_close"),
            "open": self.csv("Daily_open"),
            "high": self.csv("Daily_high"),
            "low": self.csv("Daily_low"),
            "vol": self.csv("Daily_volume"),
            "tr": self.csv("Daily_tr_price"),
            "vwap": self.csv("Daily_VWAP"),
            "mcap": self.csv("Daily_MktCap"),
        }
        out["mcap"] = out["mcap"] * 1_000_000
        return out

    def univ_sedol(self, path: Path | None = None):
        target = path or (DATA_DIR / "univ" / "sedol_daily.parquet")
        return pd.read_parquet(target)
