## Price Features

`feat_price.py`에 정의된 가격·거래량·유동성 특성 목록입니다. 괄호에 기간/공식, 정규화 여부, 해석을 기재했습니다. 거래대금은 `vwap*vol` (vwap이 NaN이면 close 대체)로 계산합니다. 일부 지표는 이상치를 99% 분위수 클리핑(또는 ±5σ)으로 완화합니다.

### 정규화/클리핑 요약
- **z-score/스케일 포함**: `m1_va`, `m12_va`(σ로 나눔), `volz`(20d z, ±5σ), `price_z`(log 252d z, ±5σ), `volshock`((vol-avg)/σ, ±5σ), `boll_up/low`(밴드 대비 가격 괴리율 비율화), `boll_pct`(0~1 밴드 위치 비율).
- **비율/범위 정규화**: `hlr`, `idv`, `trange`, `high52`, `low52`, `dist_ma20`, `volma_r`, `sto_k`, `sto_d`, `boll_w`, `turnover`, `amihud`, `pimpact`, `sprd`, `macd`류(무차원화), `ma`류(괴리율)는 가격/거래량/거래대금의 비율·차이·범위로 스케일됨.
- **클리핑**: `volz`, `price_z`, `volshock`는 ±5σ; `turnover`, `amihud`, `pimpact`는 252일 롤링 99% 상단 클립.
- **비정규화(원시 스케일)**: `ret1/5/10`, `m1/3/6/12`, `rev5`, `vol5/10/20/60/120`, `ma5/20/60/120`, `macd`, `macds`, `breakout`(0/1) 등은 추가 스케일 없이 계산된 값.

### 가격/수익률 기반
- `ret1`, `ret5`, `ret10`, `m1(21d)`, `m3(63d)`, `m6(126d)`, `m12(252d)`: 단순 누적 수익률(비정규화)로 추세 강도 측정.
- `rev5`: -5d 수익률. 단기 과열/침체 반전 신호.

### 변동성 및 스케일 조정
- `m1_va`, `m12_va`: 기간 수익률 / 20d σ (정규화). 변동성 대비 모멘텀.
- `vol5/10/20/60/120`: 수익률 롤링 σ. 리스크 레짐 파악.

### 가격 범위/레벨 (가격 정규화)
- `hlr`((H-L)/C), `idv`((H-L)/O), `trange`(True Range/C). 일중 변동폭과 슬리피지 위험.
- `high52`, `low52`: 52주 고/저 대비 비율. 위치 정규화.
- `price_z`: log(close) 252d z-score. 장기 위치 정규화.
- `dist_ma20`: 종가/20d MA - 1. 단기 괴리율.

### 거래량/수급
- `volz`: 20d z-score (정규화, ±5σ 클립). 거래량 이탈 탐지.
- `volma20`: 20d 평균 거래량을 log1p 변환. 스케일 안정.
- `volma_r`: 거래량 / 20d 평균 (비단위 비율).

### 유동성·충격 (거래대금 사용)
- `turnover`: 거래대금 / (시가총액 * 1e6) (정규화, 252d 롤링 99% 클립). 시총 대비 회전율(시총이 백만달러 단위일 때 보정).
- `amihud`: |ret| / 거래대금 * 1e6 (252d 롤링 99% 클립). 가격충격 민감도(달러 백만 단위 스케일).
- `pimpact`: True Range * 1e6 / 거래대금 (252d 롤링 99% 클립). 체결 충격 대용(달러 백만 단위 스케일).
- `sprd`: |H-L|/C. 스프레드 대용.
- `volshock`: (vol - 20d 평균) / 20d σ (정규화, ±5σ 클립). 거래량 쇼크.

### 추세/모멘텀 필터
- `ma5/20/60/120`: (종가/MA - 1) 괴리율. 가격 레벨 제거.
- `macd`(EMA12-EMA26), `macds`(macd의 EMA9): 종가 대비 비율로 무차원화.

### 오실레이터·밴드
- `rsi14`: 14d 평균 상승/하락 비율을 0~1로 스케일.
- `sto_k`(14d), `sto_d`(sto_k 3d 평균): 위치 기반 오실레이터.
- `boll_up/low`: 밴드 상·하단 대비 **가격 괴리율**. `((상단-종가)/종가)`, `((종가-하단)/종가)` 형태라 가격 스케일 제거.
- `boll_w`: (상단-하단)/중심. 밴드폭 정규화.
- `boll_pct`: 밴드 내 위치 비율 `(close - lower)/(upper - lower)`; 0~1 범위, 밴드 상대 위치.

### 돌파 신호
- `breakout`: 전일 20d 고점 상향 돌파 여부(0/1).

### 정규화/바이어스 방지 메모
- z-score 류(`volz`, `price_z`, `volshock`)는 과거 롤링 창(예: 20/252d)으로 계산하며 미래 데이터를 보지 않습니다. 창 내 최소 5개 값이 있을 때만 산출하고, 극단치는 ±5σ로 클리핑합니다.
- 거래대금 기반 지표(`turnover`, `amihud`, `pimpact`)는 **252일 롤링** 99% 분위수를 사용해 시점별 상단을 클리핑합니다(미래 데이터 미사용). 기본 `min_periods=60`으로 워밍업 전 구간은 NaN 유지.
- 추가 스케일링이 필요하면 `preprocess/base/util.py`의 헬퍼를 활용하세요:
  - 시계열: `rolling_zscore(df, window, clip=5)` 또는 `expanding_zscore(df, min_periods, clip=5)`
  - 단면: `cross_sectional_zscore(df, clip=5)` 또는 `cross_sectional_ranknorm(df, gaussian=True)`
  - 로더에서 즉시 적용: `Loader().features_price(scale="ts_roll_z", scale_kwargs={"window":252})`
- `pct_change`는 `fill_method=None`으로 미래 결측 전파를 막습니다. 정렬 단계의 ffill은 동일 시점 단면 정합(날짜 갭 채우기)만을 위해 사용되며 미래 정보는 사용하지 않습니다.
