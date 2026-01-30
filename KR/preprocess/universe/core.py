from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from preprocess.base.util import ensure_datetime_index


def _to_bool_mask(df: pd.DataFrame) -> pd.DataFrame:
    if df.dtypes.nunique() == 1 and df.dtypes.iloc[0] == bool:
        return df
    if df.dtypes.nunique() == 1 and str(df.dtypes.iloc[0]).startswith("int"):
        return df != 0
    if df.dtypes.nunique() == 1 and str(df.dtypes.iloc[0]).startswith("float"):
        return df.fillna(0.0) != 0.0
    return df.astype(bool)


def master(df: pd.DataFrame) -> list[str]:
    mask = _to_bool_mask(df.fillna(False))
    return mask.any(axis=0)[lambda s: s].index.tolist()


def active(df: pd.DataFrame, universe: Optional[Iterable[str]] = None) -> pd.DataFrame:
    mask = _to_bool_mask(df.fillna(False))
    if universe is None:
        return mask.astype(bool)
    sub = mask.loc[:, [c for c in mask.columns if c in set(universe)]]
    return sub.astype(bool)


@dataclass(frozen=True)
class UniverseFrames:
    raw: pd.DataFrame
    daily: pd.DataFrame
    active: pd.DataFrame
    master: list[str]


def build_universe(load, align, *, name: str = "k200", idx: Optional[pd.DatetimeIndex] = None) -> UniverseFrames:
    raw = ensure_datetime_index(load.universe(name))
    daily = align.daily(raw, idx) if idx is not None else raw
    daily = ensure_datetime_index(daily)
    daily = _to_bool_mask(daily)
    master_list = master(daily)
    active_mask = active(daily, None)
    return UniverseFrames(raw=raw, daily=daily, active=active_mask, master=master_list)


def apply_pool(df: pd.DataFrame, *, uni: UniverseFrames, pool: str = "all") -> pd.DataFrame:
    mode = str(pool).strip().lower()
    if mode in {"all", "none"}:
        return df
    if mode in {"ever", "master"}:
        keep = [c for c in df.columns if c in set(uni.master)]
        return df.loc[:, keep]
    if mode in {"active", "mask"}:
        mask = uni.active.reindex(index=df.index, columns=df.columns).fillna(False)
        return df.where(mask)
    raise ValueError(f"Unknown universe pool: {pool!r}. Use 'all', 'ever', or 'active'.")
