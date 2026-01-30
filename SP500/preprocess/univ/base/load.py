from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from SP500.root import DATA_DIR


def resolve_latest(pattern: str, base: Path = DATA_DIR) -> Path:
    files = sorted(base.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern {pattern} under {base}")
    return files[-1]


def read_mapping(path: Path) -> dict[str, str]:
    df = pd.read_csv(
        path,
        skiprows=4,
        names=["item", "sedol", "fsym_ticker", "fg_company_name"],
    )
    df = df.dropna(subset=["sedol", "fsym_ticker"])
    df["sedol"] = df["sedol"].astype(str).str.strip()
    df["fsym_ticker"] = df["fsym_ticker"].astype(str).str.strip()
    mapping = dict(zip(df["sedol"], df["fsym_ticker"]))
    if not mapping:
        raise ValueError("Empty sedol→ticker mapping from whole file")
    return mapping


def read_monthly_rows(path: Path) -> list[tuple[pd.Timestamp, list[str]]]:
    df = pd.read_csv(path, skiprows=5, header=None)
    df = df.dropna(axis=1, how="all")
    df = df[df[0].ne("SEDOL")].copy()
    df.rename(columns={0: "date"}, inplace=True)  # type: ignore[arg-type]
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # type: ignore[call-overload]
    df["date"] = pd.to_datetime(df["date"], unit="D", origin="1899-12-30")

    rows: list[tuple[pd.Timestamp, list[str]]] = []
    for _, row in df.iterrows():
        date_val = row["date"]
        date_ts = date_val if isinstance(date_val, pd.Timestamp) else pd.Timestamp(date_val)  # type: ignore[arg-type]
        if pd.isna(date_ts):
            continue
        sedols = [
            str(x).strip()
            for x in row.iloc[1:]
            if pd.notna(x) and str(x).strip() != ""
        ]
        rows.append((date_ts, sedols))
    if not rows:
        raise ValueError("No monthly rows parsed from sedol monthly file")
    return rows
