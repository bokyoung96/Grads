from __future__ import annotations

from typing import Iterable

import pandas as pd

from SP500.preprocess.base.feature import F


class Build:
    def __init__(self, feats: Iterable[F]):
        self.feats = list(feats)

    def run(self, df: pd.DataFrame, *, feats: Iterable[F] | None = None) -> pd.DataFrame:
        todo = list(feats) if feats is not None else self.feats
        out = df.copy()
        for feat in todo:
            out = feat.run(out)
        return out
