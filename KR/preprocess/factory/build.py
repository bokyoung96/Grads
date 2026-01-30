from __future__ import annotations

from typing import Iterable

from tqdm import tqdm

from preprocess.base.reg import REG


class Build:
    def __init__(self, feats: Iterable):
        self.feats = feats

    def run(self, df, *, feats: Iterable | None = None):
        use = list(self.feats if feats is None else feats)
        for f in tqdm(use, desc="features"):
            df = REG[f]().run(df)
        return df
