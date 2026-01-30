from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess.base.duck import DB
from preprocess.base.feature import add_feature
from preprocess.base.reg import FE
from preprocess.market.bm import BM, BMAdj, BM_TICKER
from preprocess.universe import apply_pool, build_universe
from root import CONS_PARQUET, FUND_PARQUET, PRICE_PARQUET, SECTOR_PARQUET, UNIVERSE_PARQUET
from root import FEATURES_PRICE_PARQUET, FEATURES_FUND_PARQUET, FEATURES_CONS_PARQUET, FEATURES_SECTOR_PARQUET, FEATURES_DIR


class Data:
    def __init__(self, load, align, build):
        self.load = load
        self.align = align
        self.build = build
        self.db = DB()
        self.log = logging.getLogger(__name__)

    def _feature_groups(self, *, out_dir: Path | None = None):
        feature_groups = {
            "price": {
                "names": {
                    "ret1",
                    "ret5",
                    "ret10",
                    "m1",
                    "m3",
                    "m6",
                    "m12",
                    "m1_va",
                    "m12_va",
                    "rev5",
                    "vol5",
                    "vol10",
                    "vol20",
                    "vol60",
                    "vol120",
                    "hlr",
                    "idv",
                    "trange",
                    "volz",
                    "volma20",
                    "volma_r",
                    "turnover",
                    "amihud",
                    "sprd",
                    "pimpact",
                    "volshock",
                    "ma5",
                    "ma20",
                    "ma60",
                    "ma120",
                    "macd",
                    "macds",
                    "rsi14",
                    "sto_k",
                    "sto_d",
                    "boll_up",
                    "boll_low",
                    "boll_w",
                    "high52",
                    "low52",
                    "price_z",
                    "dist_ma20",
                    "breakout",
                    "bm_ret1",
                    "bm_vol20",
                    "bm_dd",
                },
                "path": FEATURES_PRICE_PARQUET,
            },
            "fundamentals": {
                "names": {
                    "bm",
                    "ep",
                    "roe",
                    "gp_a",
                    "acc",
                    "opm",
                    "sg",
                    "ag",
                    "lev",
                    "turn",
                },
                "path": FEATURES_FUND_PARQUET,
            },
            "consensus": {
                "names": {
                    "op_fq_sprd",
                    "eps_fq_sprd",
                    "op_fq_turn",
                    "eps_fq_turn",
                    "rev_op_fq1",
                    "rev_op_fq2",
                    "rev_op_fy1",
                    "rev_eps_fq1",
                    "rev_eps_fq2",
                },
                "path": FEATURES_CONS_PARQUET,
            },
            "sector": {
                "names": {
                    "sector_oh",
                    "sector_id",
                },
                "path": FEATURES_SECTOR_PARQUET,
            },
        }

        if out_dir is not None:
            out_dir = Path(out_dir)
            feature_groups["price"]["path"] = out_dir / "features_price.parquet"
            feature_groups["fundamentals"]["path"] = out_dir / "features_fundamentals.parquet"
            feature_groups["consensus"]["path"] = out_dir / "features_consensus.parquet"
            feature_groups["sector"]["path"] = out_dir / "features_sector.parquet"
        return feature_groups

    def _feats_for_names(self, names: set[str]):
        wanted = set(names)
        return [f for f in FE if f.value in wanted]

    def _quarter_labels(self, idx: pd.DatetimeIndex) -> pd.Index:
        months = idx.month
        years = idx.year
        q_year = []
        q_num = []
        for y, m in zip(years, months):
            if m in (4, 5):
                q_year.append(y - 1)
                q_num.append(4)
            elif m in (6, 7, 8):
                q_year.append(y)
                q_num.append(1)
            elif m in (9, 10, 11):
                q_year.append(y)
                q_num.append(2)
            else:
                q_year.append(y - 1)
                q_num.append(3)
        return pd.Index([f"{yy}Q{qq}" for yy, qq in zip(q_year, q_num)], name="quarter")

    def _quarter_sort_key(self, idx: pd.Index) -> pd.Series:
        s = idx.to_series().astype(str)
        y = pd.to_numeric(s.str.slice(0, 4), errors="coerce")
        q = pd.to_numeric(s.str.extract(r"Q(?P<q>\d)", expand=False), errors="coerce")
        return y.fillna(-1) * 4 + q.fillna(-1)

    def _to_quarter(self, df: pd.DataFrame, q_labels: pd.Index) -> pd.DataFrame:
        temp = df.copy()
        temp["quarter"] = q_labels
        return temp.groupby("quarter").last()

    def _flow_ttm(self, qdf: pd.DataFrame) -> pd.DataFrame:
        qdf = qdf.sort_index(key=self._quarter_sort_key)
        return qdf.rolling(4, min_periods=4).sum()

    def _stock_avg(self, qdf: pd.DataFrame) -> pd.DataFrame:
        qdf = qdf.sort_index(key=self._quarter_sort_key)
        return (qdf + qdf.shift(1)) / 2

    def _monthly_to_daily(self, df: pd.DataFrame, daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
        period_df = df.copy()
        period_df["period"] = period_df.index.to_period("M")
        period_df = period_df.set_index("period")
        daily_period = pd.Index(daily_idx.to_period("M"), name="period")
        out = period_df.reindex(daily_period).ffill()
        out.index = daily_idx
        return out

    def _to_monthly(self, qdf: pd.DataFrame, q_labels: pd.Index, base_idx: pd.DatetimeIndex) -> pd.DataFrame:
        out = qdf.reindex(q_labels)
        out.index = base_idx
        return out

    def _bm_expand(self, s: pd.Series, cols: list[str]) -> pd.DataFrame:
        arr = np.repeat(s.to_numpy()[:, None], len(cols), axis=1)
        return pd.DataFrame(arr, index=s.index, columns=cols)

    def make(
        self,
        table: str = "features",
        universe: str = "k200",
        *,
        bm_mode: str = "none",
        bm_tkr: str | None = None,
        univ_pool: str = "all",
        parquet_chunk_rows: int | None = None,
    ):
        price_raw = self.load.price()
        idx = price_raw["c"].index

        univ = build_universe(self.load, self.align, name=universe, idx=idx)

        def filter_cols(df: pd.DataFrame) -> pd.DataFrame:
            return apply_pool(df, uni=univ, pool=univ_pool)

        price_raw = {k: filter_cols(v) for k, v in price_raw.items()}
        bm_on = str(bm_mode).lower() not in {"", "none", "off", "false", "0"}
        price_use = price_raw
        bm_frames = None
        if bm_on:
            tkr = bm_tkr or BM_TICKER
            bm = BM(ticker=tkr)
            adj = BMAdj(bm=bm.series())
            price_use = adj.apply(price_raw)
            bm_df = bm.feats().reindex(idx)
            cols = list(price_use["c"].columns)
            bm_frames = {name: self._bm_expand(bm_df[name], cols) for name in bm_df.columns}

        fund_src = self.load.fundamentals()
        fund_src = {k: filter_cols(v) for k, v in fund_src.items()}
        base_idx = next(iter(fund_src.values())).index
        if base_idx.hasnans:
            self.log.warning("Found NaT in fundamentals index; dropping NaT rows")
            fund_src = {k: v.loc[~v.index.isna()] for k, v in fund_src.items()}
            base_idx = next(iter(fund_src.values())).index
        q_labels = self._quarter_labels(base_idx)

        base_q = {
            "assets": self._to_quarter(fund_src["a"], q_labels),
            "liab": self._to_quarter(fund_src["l"], q_labels),
            "equity": self._to_quarter(fund_src["e"], q_labels),
            "gp": self._to_quarter(fund_src["gp"], q_labels),
            "op": self._to_quarter(fund_src["op"], q_labels),
            "ni": self._to_quarter(fund_src["ni"], q_labels),
            "ocf": self._to_quarter(fund_src["ocf"], q_labels),
        }
        flows = ("ni", "gp", "op", "ocf")
        ttm_q = {f"{name}_ttm": self._flow_ttm(base_q[name]) for name in flows}
        stock_avg_q = {
            "equity_avg": self._stock_avg(base_q["equity"]),
            "assets_avg": self._stock_avg(base_q["assets"]),
        }
        growth_q = {"assets_g": base_q["assets"].pct_change(1, fill_method=None)}

        fund_monthly = {}
        for name, qdf in {**base_q, **ttm_q, **stock_avg_q, **growth_q}.items():
            fund_monthly[name] = self._to_monthly(qdf, q_labels, base_idx)

        fundamentals = {k: self._monthly_to_daily(v, idx) for k, v in fund_monthly.items()}
        consensus = {k: filter_cols(self.align.daily(v, idx)) for k, v in self.load.consensus().items()}
        sectors = filter_cols(self.align.daily(self.load.sector(), idx))

        price_df_raw = pd.concat(
            [
                price_raw["c"],
                price_raw["o"],
                price_raw["h"],
                price_raw["l"],
                price_raw["v"],
                price_raw["mc"],
            ],
            axis=1,
            keys=["close", "open", "high", "low", "vol", "mcap"],
        )
        price_df = pd.concat(
            [
                price_use["c"],
                price_use["o"],
                price_use["h"],
                price_use["l"],
                price_use["v"],
                price_use["mc"],
            ],
            axis=1,
            keys=["close", "open", "high", "low", "vol", "mcap"],
        )
        fund_keys = [
            "assets",
            "liab",
            "equity",
            "gp_ttm",
            "op_ttm",
            "ni_ttm",
            "ocf_ttm",
            "equity_avg",
            "assets_avg",
            "assets_g",
        ]
        existing = [k for k in fund_keys if k in fundamentals]
        fund_df = pd.concat([fundamentals[k] for k in existing], axis=1, keys=existing)
        cons_df = pd.concat(
            [
                consensus["op_fq1_raw"],
                consensus["op_fq2_raw"],
                consensus["op_fy1_raw"],
                consensus["eps_fq1_raw"],
                consensus["eps_fq2_raw"],
                consensus["eps_fy1_raw"],
            ],
            axis=1,
            keys=[
                "op_fq1_raw",
                "op_fq2_raw",
                "op_fy1_raw",
                "eps_fq1_raw",
                "eps_fq2_raw",
                "eps_fy1_raw",
            ],
        )
        sector_df = pd.concat([sectors], axis=1, keys=["sector"])
        active_df = univ.active.reindex(idx).ffill()

        self.log.info("Saving raw price to %s", PRICE_PARQUET)
        self.db.save_parquet(price_df_raw, PRICE_PARQUET, chunk_rows=parquet_chunk_rows)
        self.log.info("Saving raw fundamentals to %s", FUND_PARQUET)
        self.db.save_parquet(fund_df, FUND_PARQUET, chunk_rows=parquet_chunk_rows)
        self.log.info("Saving raw consensus to %s", CONS_PARQUET)
        self.db.save_parquet(cons_df, CONS_PARQUET, chunk_rows=parquet_chunk_rows)
        self.log.info("Saving raw sector to %s", SECTOR_PARQUET)
        self.db.save_parquet(sector_df, SECTOR_PARQUET, chunk_rows=parquet_chunk_rows)
        self.log.info(
            "Saving universe '%s' (active mask, pool=%s) to %s",
            universe,
            univ_pool,
            UNIVERSE_PARQUET,
        )
        self.db.save_parquet(active_df, UNIVERSE_PARQUET, chunk_rows=parquet_chunk_rows)

        feat_dir = FEATURES_DIR if not bm_on else (FEATURES_DIR.parent / "features_bm")
        groups = self._feature_groups()

        price_feats = self._feats_for_names(groups["price"]["names"])
        fund_feats = self._feats_for_names(groups["fundamentals"]["names"])
        cons_feats = self._feats_for_names(groups["consensus"]["names"])
        sector_feats = self._feats_for_names(groups["sector"]["names"])

        self.log.info("Applying %d price features", len(price_feats))
        price_out = self.build.run(price_df, feats=price_feats)
        if bm_frames:
            for name, vals in bm_frames.items():
                price_out = add_feature(price_out, name, vals)
        self.save_features(price_out, out_dir=feat_dir, parquet_chunk_rows=parquet_chunk_rows)

        self.log.info("Applying %d fundamentals features", len(fund_feats))
        mcap_df = pd.concat([price_use["mc"]], axis=1, keys=["mcap"])
        fund_input = pd.concat([mcap_df, fund_df], axis=1)
        fund_out = self.build.run(fund_input, feats=fund_feats)
        self.save_features(fund_out, out_dir=feat_dir, parquet_chunk_rows=parquet_chunk_rows)

        self.log.info("Applying %d consensus features", len(cons_feats))
        cons_out = self.build.run(cons_df, feats=cons_feats)
        self.save_features(cons_out, out_dir=feat_dir, parquet_chunk_rows=parquet_chunk_rows)

        self.log.info("Applying %d sector features", len(sector_feats))
        sector_out = self.build.run(sector_df, feats=sector_feats)
        self.save_features(sector_out, out_dir=feat_dir, parquet_chunk_rows=parquet_chunk_rows)

        self.log.info("Saved features to %s", feat_dir)

    def save_features(self, df: pd.DataFrame, *, out_dir: Path | None = None, parquet_chunk_rows: int | None = None):
        feature_groups = self._feature_groups(out_dir=out_dir)
        for group_name, cfg in tqdm(feature_groups.items(), desc="Saving features"):
            names = cfg["names"]
            cols = []
            for c in df.columns:
                if isinstance(c, tuple):
                    if c[0] in names:
                        cols.append(c)
                elif c in names:
                    cols.append(c)
            if not cols:
                continue
            subset = df.loc[:, cols]
            if subset.columns.duplicated().any():
                self.log.warning("Dropping duplicated columns in %s features", group_name)
                subset = subset.loc[:, ~subset.columns.duplicated()]
            self.db.save_parquet(subset, cfg["path"], chunk_rows=parquet_chunk_rows)
