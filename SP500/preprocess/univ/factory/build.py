from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from SP500.preprocess.univ.base.load import read_mapping, read_monthly_rows, resolve_latest
from SP500.root import DATA_DIR


class UnivBuild:
    def __init__(
        self,
        *,
        monthly_path: Path | None = None,
        whole_path: Path | None = None,
        out_path: Path | None = None,
    ):
        self.monthly_path = monthly_path
        self.whole_path = whole_path
        self.out_path = out_path or (DATA_DIR / "univ" / "sedol_daily.parquet")

    def _build_matrix(self, rows: Iterable[tuple[pd.Timestamp, list[str]]], mapping: dict[str, str]) -> pd.DataFrame:
        monthly = list(rows)
        tickers: set[str] = set()
        for _, sedols in monthly:
            tickers.update(mapping[s] for s in sedols if s in mapping)
        if not tickers:
            raise ValueError("No mapped tickers found in monthly data")

        dates = [d for d, _ in monthly]
        idx = pd.DatetimeIndex(dates).sort_values()
        cols = sorted(tickers)

        base = pd.DataFrame(0, index=pd.Index(idx.unique()), columns=pd.Index(cols), dtype="int8")
        for date, sedols in monthly:
            tickers_here = [mapping[s] for s in sedols if s in mapping]
            if tickers_here:
                base.loc[date, tickers_here] = 1

        full_index = pd.date_range(start=idx.min(), end=idx.max(), freq="D")
        daily = base.reindex(full_index).ffill().fillna(0).astype("int8")
        daily.index.name = "date"
        return daily

    def run(self) -> Path:
        monthly_path = self.monthly_path or resolve_latest("SPY_sedol_monthly_*.csv")
        whole_path = self.whole_path or resolve_latest("SPY_sedol_whole_*.csv")
        out_path = self.out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mapping = read_mapping(whole_path)
        monthly_rows = read_monthly_rows(monthly_path)
        daily = self._build_matrix(monthly_rows, mapping)
        daily.to_parquet(out_path)
        return out_path
