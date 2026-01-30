from __future__ import annotations

from typing import Any

import pandas as pd


class F:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


def add_feature(df: pd.DataFrame, name: str, values: Any) -> pd.DataFrame:
    out = df.copy()
    if isinstance(values, pd.DataFrame):
        cols = pd.MultiIndex.from_product([[name], values.columns])
        vals = values.copy()
        vals.columns = cols
        out = pd.concat([out, vals], axis=1)
    elif isinstance(values, pd.Series):
        out[(name,)] = values
    else:
        vals = pd.DataFrame(values, index=df.index)
        cols = pd.MultiIndex.from_product([[name], vals.columns])
        vals.columns = cols
        out = pd.concat([out, vals], axis=1)
    return out
