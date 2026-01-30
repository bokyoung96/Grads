from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, Type

from SP500.preprocess.base.feature import F


class FE(str, Enum):
    RET1 = "ret1"
    RET5 = "ret5"
    RET10 = "ret10"
    M1 = "m1"
    M3 = "m3"
    M6 = "m6"
    M12 = "m12"
    M1_VA = "m1_va"
    M12_VA = "m12_va"
    REV5 = "rev5"
    VOL5 = "vol5"
    VOL10 = "vol10"
    VOL20 = "vol20"
    VOL60 = "vol60"
    VOL120 = "vol120"
    HLR = "hlr"
    IDV = "idv"
    TRANGE = "trange"
    VOLZ = "volz"
    VOLMA20 = "volma20"
    VOLMA_R = "volma_r"
    AMIHUD = "amihud"
    SPRD = "sprd"
    PIMPACT = "pimpact"
    TURNOVER = "turnover"
    VOLSHOCK = "volshock"
    MA5 = "ma5"
    MA20 = "ma20"
    MA60 = "ma60"
    MA120 = "ma120"
    MACD = "macd"
    MACDS = "macds"
    RSI14 = "rsi14"
    STO_K = "sto_k"
    STO_D = "sto_d"
    BOLL_UP = "boll_up"
    BOLL_LOW = "boll_low"
    BOLL_W = "boll_w"
    BOLL_PCT = "boll_pct"
    HIGH52 = "high52"
    LOW52 = "low52"
    PRICE_Z = "price_z"
    DIST_MA20 = "dist_ma20"
    BREAKOUT = "breakout"


_REGISTRY: Dict[FE, Type[F]] = {}


@dataclass(frozen=True)
class FeatureSpec:
    name: FE
    cls: Type[F]


def register(name: FE) -> Callable[[Type[F]], Type[F]]:
    def deco(cls: Type[F]) -> Type[F]:
        _REGISTRY[name] = cls
        return cls

    return deco


def get_features(names: list[FE]) -> list[F]:
    return [(_REGISTRY[n])() for n in names if n in _REGISTRY]


def register_all(items: Iterable[FeatureSpec]) -> None:
    for spec in items:
        _REGISTRY[spec.name] = spec.cls
