from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PREPROCESS_DIR = Path(__file__).resolve().parents[1]
GRADS_DIR = PREPROCESS_DIR.parent
if str(GRADS_DIR) not in sys.path:
    sys.path.insert(0, str(GRADS_DIR))

import pandas as pd

from preprocess.base.util import ensure_datetime_index
from root import RAW_DIR


BM_TICKER = "IKS200"


@dataclass(frozen=True)
class BM:
    xlsx: Path = RAW_DIR / "qw_BM.xlsx"
    pq: Path = RAW_DIR / "qw_BM.parquet"
    ticker: str = BM_TICKER
    row_tickers: int = 7
    row_data: int = 14

    def read(self) -> pd.DataFrame:
        df = pd.read_excel(self.xlsx, header=None)
        tickers = df.iloc[self.row_tickers, 1:]
        tickers = tickers[tickers.notna()].astype(str).str.strip().tolist()
        if not tickers:
            raise ValueError("No tickers found in BM sheet.")

        data = df.iloc[self.row_data:, : len(tickers) + 1].copy()
        data.columns = ["date"] + tickers
        data = data.dropna(subset=["date"])
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        for c in tickers:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        data = data.set_index("date")
        data.index.name = "date"
        return data.sort_index()

    def load(self) -> pd.DataFrame:
        if self.pq.exists():
            return ensure_datetime_index(pd.read_parquet(self.pq))
        return self.read()

    def series(self) -> pd.Series:
        df = self.load()
        if self.ticker not in df.columns:
            raise KeyError(f"Unknown BM ticker: {self.ticker}")
        return df[self.ticker]

    def feats(self) -> pd.DataFrame:
        s = self.series()
        ret1 = s.pct_change(1)
        vol20 = ret1.rolling(20).std()
        dd = s / s.cummax() - 1.0
        out = pd.DataFrame({"bm_ret1": ret1, "bm_vol20": vol20, "bm_dd": dd})
        out.index.name = "date"
        return out

    def save(self) -> Path:
        df = self.read()
        self.pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.pq)
        return self.pq


class BMAdj:
    def __init__(self, *, bm: pd.Series):
        s = bm.copy()
        s.index = pd.to_datetime(s.index)
        self.bm = s.sort_index()

    def _ret(self, idx: pd.DatetimeIndex) -> pd.Series:
        s = self.bm.reindex(idx)
        return s.pct_change(1)

    def _p(self, ret: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype="float64")
        for c in ret.columns:
            r = ret[c]
            ok = r.notna()
            if not ok.any():
                continue
            seg = (~ok).cumsum()
            vals = (1 + r[ok]).groupby(seg[ok]).cumprod()
            out.loc[ok, c] = vals
        return out

    def apply(self, price: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        c = price["c"]
        bm_ret = self._ret(c.index)
        ret_ex = c.pct_change(1, fill_method=None).sub(bm_ret, axis=0)
        c_ex = self._p(ret_ex)
        scale = c_ex.divide(c)
        out = dict(price)
        out["c"] = c_ex
        out["o"] = price["o"].multiply(scale)
        out["h"] = price["h"].multiply(scale)
        out["l"] = price["l"].multiply(scale)
        return out


if __name__ == "__main__":
    path = BM().save()
    print(f"saved {path}")
