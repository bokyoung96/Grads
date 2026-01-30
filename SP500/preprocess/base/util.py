from __future__ import annotations

import numpy as np
import pandas as pd

try:  # optional dependency
    from scipy.stats import norm as _norm
except Exception:  # pragma: no cover - scipy may be unavailable
    _norm = None


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out
    return df


def rolling_zscore(df: pd.DataFrame, window: int, *, min_periods: int | None = None, clip: float | None = 5.0) -> pd.DataFrame:
    mp = window if min_periods is None else min_periods
    mean = df.rolling(window, min_periods=mp).mean()
    std = df.rolling(window, min_periods=mp).std()
    out = (df - mean) / std
    if clip is not None:
        out = out.clip(-clip, clip)
    return out


def expanding_zscore(df: pd.DataFrame, *, min_periods: int = 20, clip: float | None = 5.0) -> pd.DataFrame:
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std()
    out = (df - mean) / std
    if clip is not None:
        out = out.clip(-clip, clip)
    return out


def cross_sectional_zscore(df: pd.DataFrame, *, clip: float | None = 5.0) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    out = df.sub(mean, axis=0).div(std, axis=0)
    if clip is not None:
        out = out.clip(-clip, clip)
    return out


def _rank_to_uniform(row: pd.Series) -> pd.Series:
    n = row.count()
    if n == 0:
        return row
    ranks = row.rank(method="average", na_option="keep")
    return (ranks - 0.5) / n


def cross_sectional_ranknorm(df: pd.DataFrame, *, gaussian: bool = True) -> pd.DataFrame:
    ranks = df.rank(axis=1, method="average", na_option="keep")
    counts = df.notna().sum(axis=1)
    u = ranks.sub(0.5).div(counts.replace(0, np.nan), axis=0)

    if gaussian and _norm is not None:
        eps = 1e-6
        u = u.clip(eps, 1 - eps)
        return pd.DataFrame(_norm.ppf(u), index=df.index, columns=df.columns)

    return pd.DataFrame(u, index=df.index, columns=df.columns)


def apply_scaling(df: pd.DataFrame, *, mode: str, **kwargs) -> pd.DataFrame:
    if mode == "ts_roll_z":
        return rolling_zscore(df, **kwargs)
    if mode == "ts_expand_z":
        return expanding_zscore(df, **kwargs)
    if mode == "cs_z":
        return cross_sectional_zscore(df, **kwargs)
    if mode == "cs_ranknorm":
        return cross_sectional_ranknorm(df, **kwargs)
    raise ValueError(f"Unsupported scaling mode: {mode}")
