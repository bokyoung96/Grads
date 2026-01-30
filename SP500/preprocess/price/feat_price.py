from __future__ import annotations

import numpy as np
import pandas as pd

from SP500.preprocess.base.feature import F, add_feature
from SP500.preprocess.base.reg import FE, FeatureSpec


def _safe_div(num, denom):
    out = num.divide(denom)
    return out.replace([np.inf, -np.inf], np.nan)


def _ema(x, span: int):
    return x.ewm(span=span, adjust=False).mean()


def _zscore(x, window: int, *, clip: float | None = 5.0):
    mean = x.rolling(window, min_periods=5).mean()
    std = x.rolling(window, min_periods=5).std()
    out = (x - mean) / std
    out = out.replace([np.inf, -np.inf], np.nan)
    if clip is not None:
        out = out.clip(-clip, clip)
    return out


def _rolling_cap(x, *, window: int = 252, q: float = 0.99, min_periods: int | None = None):
    mp = window if min_periods is None else min_periods
    cap = x.rolling(window=window, min_periods=mp).quantile(q)
    return x.clip(upper=cap)


def _trading_value(df: pd.DataFrame):
    vwap = df.get(("vwap"))
    price = vwap.fillna(df[("close")]) if vwap is not None else df[("close")]
    return price * df[("vol")]


class TURNOVER(F):
    def run(self, df):
        trading_value = _trading_value(df)
        turnover = _safe_div(trading_value, df[("mcap")] * 1e6)
        capped = _rolling_cap(turnover, window=252, q=0.99, min_periods=60)
        return add_feature(df, FE.TURNOVER.value, capped)


class RET1(F):
    def run(self, df):
        ret = df[("close")].pct_change(1, fill_method=None)
        return add_feature(df, FE.RET1.value, ret)


class RET5(F):
    def run(self, df):
        ret = df[("close")].pct_change(5, fill_method=None)
        return add_feature(df, FE.RET5.value, ret)


class RET10(F):
    def run(self, df):
        ret = df[("close")].pct_change(10, fill_method=None)
        return add_feature(df, FE.RET10.value, ret)


class M1(F):
    def run(self, df):
        ret = df[("close")].pct_change(21, fill_method=None)
        return add_feature(df, FE.M1.value, ret)


class M3(F):
    def run(self, df):
        ret = df[("close")].pct_change(63, fill_method=None)
        return add_feature(df, FE.M3.value, ret)


class M6(F):
    def run(self, df):
        ret = df[("close")].pct_change(126, fill_method=None)
        return add_feature(df, FE.M6.value, ret)


class M12(F):
    def run(self, df):
        ret = df[("close")].pct_change(252, fill_method=None)
        return add_feature(df, FE.M12.value, ret)


class M1_VA(F):
    def run(self, df):
        m1 = df[("close")].pct_change(21, fill_method=None)
        vol20 = df[("close")].pct_change(fill_method=None).rolling(20).std()
        return add_feature(df, FE.M1_VA.value, _safe_div(m1, vol20))


class M12_VA(F):
    def run(self, df):
        m12 = df[("close")].pct_change(252, fill_method=None)
        vol20 = df[("close")].pct_change(fill_method=None).rolling(20).std()
        return add_feature(df, FE.M12_VA.value, _safe_div(m12, vol20))


class REV5(F):
    def run(self, df):
        ret = -df[("close")].pct_change(5, fill_method=None)
        return add_feature(df, FE.REV5.value, ret)


class VOL5(F):
    def run(self, df):
        vol = df[("close")].pct_change(fill_method=None).rolling(5).std()
        return add_feature(df, FE.VOL5.value, vol)


class VOL10(F):
    def run(self, df):
        vol = df[("close")].pct_change(fill_method=None).rolling(10).std()
        return add_feature(df, FE.VOL10.value, vol)


class VOL20(F):
    def run(self, df):
        vol = df[("close")].pct_change(fill_method=None).rolling(20).std()
        return add_feature(df, FE.VOL20.value, vol)


class VOL60(F):
    def run(self, df):
        vol = df[("close")].pct_change(fill_method=None).rolling(60).std()
        return add_feature(df, FE.VOL60.value, vol)


class VOL120(F):
    def run(self, df):
        vol = df[("close")].pct_change(fill_method=None).rolling(120).std()
        return add_feature(df, FE.VOL120.value, vol)


class HLR(F):
    def run(self, df):
        hlr = _safe_div(df[("high")] - df[("low")], df[("close")])
        return add_feature(df, FE.HLR.value, hlr)


class IDV(F):
    def run(self, df):
        idv = _safe_div(df[("high")] - df[("low")], df[("open")])
        return add_feature(df, FE.IDV.value, idv)


class TRANGE(F):
    def run(self, df):
        prev_close = df[("close")].shift(1)
        range1 = df[("high")] - df[("low")]
        range2 = (df[("high")] - prev_close).abs()
        range3 = (df[("low")] - prev_close).abs()
        tr = np.maximum(range1, np.maximum(range2, range3))
        tr_norm = _safe_div(tr, df[("close")])
        return add_feature(df, FE.TRANGE.value, tr_norm)


class VOLZ(F):
    def run(self, df):
        volz = _zscore(df[("vol")], 20)
        return add_feature(df, FE.VOLZ.value, volz)


class VOLMA20(F):
    def run(self, df):
        volma = df[("vol")].rolling(20).mean()
        volma = np.log1p(volma)
        return add_feature(df, FE.VOLMA20.value, volma)


class VOLMA_R(F):
    def run(self, df):
        volma20 = df[("vol")].rolling(20).mean()
        volr = _safe_div(df[("vol")], volma20)
        return add_feature(df, FE.VOLMA_R.value, volr)


class AMIHUD(F):
    def run(self, df):
        ret = df[("close")].pct_change(fill_method=None)
        trading_value = _trading_value(df)
        # scale to per-$1mm to avoid vanishing magnitudes
        amihud = _safe_div(ret.abs() * 1e6, trading_value)
        capped = _rolling_cap(amihud, window=252, q=0.99, min_periods=60)
        return add_feature(df, FE.AMIHUD.value, capped)


class SPRD(F):
    def run(self, df):
        sprd = _safe_div(df[("high")] - df[("low")], df[("close")])
        sprd = sprd.abs()
        return add_feature(df, FE.SPRD.value, sprd)


class PIMPACT(F):
    def run(self, df):
        prev_close = df[("close")].shift(1)
        range1 = df[("high")] - df[("low")]
        range2 = (df[("high")] - prev_close).abs()
        range3 = (df[("low")] - prev_close).abs()
        tr = np.maximum(range1, np.maximum(range2, range3))
        trading_value = _trading_value(df)
        pimpact = _safe_div(tr * 1e6, trading_value)
        capped = _rolling_cap(pimpact, window=252, q=0.99, min_periods=60)
        return add_feature(df, FE.PIMPACT.value, capped)


class VOLSHOCK(F):
    def run(self, df):
        volma20 = df[("vol")].rolling(20).mean()
        volstd20 = df[("vol")].rolling(20).std()
        volshock = _safe_div(df[("vol")] - volma20, volstd20)
        return add_feature(df, FE.VOLSHOCK.value, volshock.clip(-5, 5))


class MA5(F):
    def run(self, df):
        ma = df[("close")].rolling(5).mean()
        dist = _safe_div(df[("close")], ma) - 1
        return add_feature(df, FE.MA5.value, dist)


class MA20(F):
    def run(self, df):
        ma = df[("close")].rolling(20).mean()
        dist = _safe_div(df[("close")], ma) - 1
        return add_feature(df, FE.MA20.value, dist)


class MA60(F):
    def run(self, df):
        ma = df[("close")].rolling(60).mean()
        dist = _safe_div(df[("close")], ma) - 1
        return add_feature(df, FE.MA60.value, dist)


class MA120(F):
    def run(self, df):
        ma = df[("close")].rolling(120).mean()
        dist = _safe_div(df[("close")], ma) - 1
        return add_feature(df, FE.MA120.value, dist)


class MACD(F):
    def run(self, df):
        ema12 = _ema(df[("close")], 12)
        ema26 = _ema(df[("close")], 26)
        macd = ema12 - ema26
        macd_ratio = _safe_div(macd, df[("close")])
        return add_feature(df, FE.MACD.value, macd_ratio)


class MACDS(F):
    def run(self, df):
        ema12 = _ema(df[("close")], 12)
        ema26 = _ema(df[("close")], 26)
        macd = ema12 - ema26
        macd_ratio = _safe_div(macd, df[("close")])
        macds = _ema(macd_ratio, 9)
        return add_feature(df, FE.MACDS.value, macds)


class RSI14(F):
    def run(self, df):
        delta = df[("close")].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = _safe_div(avg_gain, avg_loss)
        rsi = (100 - (100 / (1 + rs))) / 100.0
        return add_feature(df, FE.RSI14.value, rsi)


class STO_K(F):
    def run(self, df):
        low_k = df[("low")].rolling(14).min()
        high_k = df[("high")].rolling(14).max()
        sto_k = _safe_div(df[("close")] - low_k, (high_k - low_k))
        return add_feature(df, FE.STO_K.value, sto_k)


class STO_D(F):
    def run(self, df):
        low_k = df[("low")].rolling(14).min()
        high_k = df[("high")].rolling(14).max()
        sto_k = _safe_div(df[("close")] - low_k, (high_k - low_k))
        sto_d = sto_k.rolling(3).mean()
        return add_feature(df, FE.STO_D.value, sto_d)


class BOLL_UP(F):
    def run(self, df):
        mean = df[("close")].rolling(20).mean()
        std = df[("close")].rolling(20).std()
        boll_up = mean + 2 * std
        up_dist = _safe_div(boll_up - df[("close")], df[("close")])
        return add_feature(df, FE.BOLL_UP.value, up_dist)


class BOLL_LOW(F):
    def run(self, df):
        mean = df[("close")].rolling(20).mean()
        std = df[("close")].rolling(20).std()
        boll_low = mean - 2 * std
        low_dist = _safe_div(df[("close")] - boll_low, df[("close")])
        return add_feature(df, FE.BOLL_LOW.value, low_dist)


class BOLL_W(F):
    def run(self, df):
        mean = df[("close")].rolling(20).mean()
        std = df[("close")].rolling(20).std()
        upper = mean + 2 * std
        lower = mean - 2 * std
        bw = _safe_div((upper - lower), mean)
        return add_feature(df, FE.BOLL_W.value, bw)


class BOLL_PCT(F):
    def run(self, df):
        mean = df[("close")].rolling(20).mean()
        std = df[("close")].rolling(20).std()
        upper = mean + 2 * std
        lower = mean - 2 * std
        pct = _safe_div(df[("close")] - lower, upper - lower)
        pct = pct.clip(0, 1)
        return add_feature(df, FE.BOLL_PCT.value, pct)


class HIGH52(F):
    def run(self, df):
        high52 = _safe_div(df[("close")], df[("close")].rolling(252).max())
        return add_feature(df, FE.HIGH52.value, high52)


class LOW52(F):
    def run(self, df):
        low52 = _safe_div(df[("close")], df[("close")].rolling(252).min())
        return add_feature(df, FE.LOW52.value, low52)


class PRICE_Z(F):
    def run(self, df):
        pz = _zscore(np.log(df[("close")]), 252)
        return add_feature(df, FE.PRICE_Z.value, pz)


class DIST_MA20(F):
    def run(self, df):
        ma20 = df[("close")].rolling(20).mean()
        dist = _safe_div(df[("close")], ma20) - 1
        return add_feature(df, FE.DIST_MA20.value, dist)


class BREAKOUT(F):
    def run(self, df):
        rolling_max = df[("close")].rolling(20).max().shift(1)
        br = (df[("close")] > rolling_max).astype(int)
        return add_feature(df, FE.BREAKOUT.value, br)


PRICE_FEATURES = [
    FeatureSpec(FE.RET1, RET1),
    FeatureSpec(FE.RET5, RET5),
    FeatureSpec(FE.RET10, RET10),
    FeatureSpec(FE.M1, M1),
    FeatureSpec(FE.M3, M3),
    FeatureSpec(FE.M6, M6),
    FeatureSpec(FE.M12, M12),
    FeatureSpec(FE.M1_VA, M1_VA),
    FeatureSpec(FE.M12_VA, M12_VA),
    FeatureSpec(FE.REV5, REV5),
    FeatureSpec(FE.VOL5, VOL5),
    FeatureSpec(FE.VOL10, VOL10),
    FeatureSpec(FE.VOL20, VOL20),
    FeatureSpec(FE.VOL60, VOL60),
    FeatureSpec(FE.VOL120, VOL120),
    FeatureSpec(FE.HLR, HLR),
    FeatureSpec(FE.IDV, IDV),
    FeatureSpec(FE.TRANGE, TRANGE),
    FeatureSpec(FE.VOLZ, VOLZ),
    FeatureSpec(FE.VOLMA20, VOLMA20),
    FeatureSpec(FE.VOLMA_R, VOLMA_R),
    FeatureSpec(FE.AMIHUD, AMIHUD),
    FeatureSpec(FE.SPRD, SPRD),
    FeatureSpec(FE.PIMPACT, PIMPACT),
    FeatureSpec(FE.TURNOVER, TURNOVER),
    FeatureSpec(FE.VOLSHOCK, VOLSHOCK),
    FeatureSpec(FE.MA5, MA5),
    FeatureSpec(FE.MA20, MA20),
    FeatureSpec(FE.MA60, MA60),
    FeatureSpec(FE.MA120, MA120),
    FeatureSpec(FE.MACD, MACD),
    FeatureSpec(FE.MACDS, MACDS),
    FeatureSpec(FE.RSI14, RSI14),
    FeatureSpec(FE.STO_K, STO_K),
    FeatureSpec(FE.STO_D, STO_D),
    FeatureSpec(FE.BOLL_UP, BOLL_UP),
    FeatureSpec(FE.BOLL_LOW, BOLL_LOW),
    FeatureSpec(FE.BOLL_W, BOLL_W),
    FeatureSpec(FE.BOLL_PCT, BOLL_PCT),
    FeatureSpec(FE.HIGH52, HIGH52),
    FeatureSpec(FE.LOW52, LOW52),
    FeatureSpec(FE.PRICE_Z, PRICE_Z),
    FeatureSpec(FE.DIST_MA20, DIST_MA20),
    FeatureSpec(FE.BREAKOUT, BREAKOUT),
]
