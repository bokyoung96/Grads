from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from SP500.root import FEATURES_PRICE_PARQUET, PRICE_PARQUET
from SP500.preprocess.base.reg import FE, get_features
from SP500.preprocess.factory.build import Build


DEFAULT_FEATURES: list[FE] = [
    FE.RET1,
    FE.RET5,
    FE.RET10,
    FE.M1,
    FE.M3,
    FE.M6,
    FE.M12,
    FE.M1_VA,
    FE.M12_VA,
    FE.REV5,
    FE.VOL5,
    FE.VOL10,
    FE.VOL20,
    FE.VOL60,
    FE.VOL120,
    FE.HLR,
    FE.IDV,
    FE.TRANGE,
    FE.VOLZ,
    FE.VOLMA20,
    FE.VOLMA_R,
    FE.AMIHUD,
    FE.SPRD,
    FE.PIMPACT,
    FE.TURNOVER,
    FE.VOLSHOCK,
    FE.MA5,
    FE.MA20,
    FE.MA60,
    FE.MA120,
    FE.MACD,
    FE.MACDS,
    FE.RSI14,
    FE.STO_K,
    FE.STO_D,
    FE.BOLL_UP,
    FE.BOLL_LOW,
    FE.BOLL_W,
    FE.HIGH52,
    FE.LOW52,
    FE.PRICE_Z,
    FE.DIST_MA20,
    FE.BREAKOUT,
]


class Data:
    def __init__(self, load, align, db):
        self.load = load
        self.align = align
        self.db = db
        self.log = logging.getLogger(__name__)

    def _drop_all_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=1, how="all")

    def _common_cols(self, frames: dict[str, pd.DataFrame]) -> list[str]:
        cols = None
        for df in frames.values():
            df_cols = set(df.columns)
            cols = df_cols if cols is None else cols & df_cols
        return sorted(cols) if cols else []

    def _full_index(self, frames: dict[str, pd.DataFrame]):
        indexes = [pd.DatetimeIndex(df.index) for df in frames.values()]
        if not indexes:
            return pd.DatetimeIndex([])
        idx = pd.DatetimeIndex(pd.to_datetime(indexes[0]))
        for other in indexes[1:]:
            idx = pd.DatetimeIndex(pd.to_datetime(idx.union(other)))
        return idx.sort_values()

    def make(
        self,
        *,
        parquet_chunk_rows: int | None = None,
        out_path: str | Path = PRICE_PARQUET,
        make_features: bool = True,
        features_out: str | Path = FEATURES_PRICE_PARQUET,
        feature_names: list[FE] | None = None,
    ):
        price_raw = {k: self._drop_all_nan(v) for k, v in self.load.price().items()}

        cols = self._common_cols(price_raw)
        if not cols:
            raise ValueError("No overlapping tickers across price files")

        self.log.info("Common tickers after drop-all-NaN: %d", len(cols))

        idx = self._full_index(price_raw)

        aligned = {}
        for name, df in price_raw.items():
            use = df.loc[:, cols]
            use = use.apply(pd.to_numeric, errors="coerce")
            use = self._drop_all_nan(use)
            self.log.info("Field %s: rows=%d cols=%d", name, len(use), use.shape[1])
            aligned[name] = self.align.daily(use, idx)

        price_df = pd.concat(aligned, axis=1)

        self.log.info("Price matrix shape after align: %s", price_df.shape)
        with tqdm(total=1, desc="save_price", unit="file") as bar:
            self.db.save_parquet(price_df, out_path, chunk_rows=parquet_chunk_rows)
            bar.update(1)

        if make_features:
            feats = feature_names or DEFAULT_FEATURES
            builder = Build(get_features(feats))
            feat_df = builder.run(price_df)
            want = {f.value for f in feats}
            feat_only = feat_df.loc[:, feat_df.columns.get_level_values(0).isin(want)]
            feat_only = feat_only.dropna(axis=1, how="all")
            self.log.info("Features matrix shape: %s", feat_only.shape)
            with tqdm(total=1, desc="save_features", unit="file") as bar:
                self.db.save_parquet(feat_only, features_out, chunk_rows=parquet_chunk_rows)
                bar.update(1)
        return price_df
