from __future__ import annotations

from pathlib import Path
from typing import Optional

from KR.root import DATA_DIR, PROCESSED_DIR, RAW_DIR, ROOT
from KR.transformer.core.params import TransformerParams, build_name


DEFAULT_REGION = "KR"


def region_processed_dir(region: str = DEFAULT_REGION) -> Path:
    base = DATA_DIR / region / "processed"
    if base.exists():
        return base
    return PROCESSED_DIR


def region_scores_dir(region: str = DEFAULT_REGION) -> Path:
    cand = ROOT / "scores" / region
    return cand if cand.exists() else (ROOT / "scores")


def region_default_paths(region: str = DEFAULT_REGION) -> dict[str, Path]:
    processed = region_processed_dir(region)
    raw_bench = RAW_DIR / "qw_BM.parquet"
    return {
        "processed": processed,
        "close": processed / "price.parquet",
        "universe": processed / "universe_k200.parquet",
        "sector": processed / "sector.parquet",
        "benchmark": (processed / "benchmark.parquet") if (processed / "benchmark.parquet").exists() else raw_bench,
        "scores_root": region_scores_dir(region),
    }


def transformer_score_path(
    *,
    mode: str = "TEST",
    timeframe: str = "MEDIUM",
    config_path: Optional[Path] = None,
    use_bm: bool = False,
    view: str = "score_full.parquet",
) -> Path:
    params = TransformerParams(config_path=config_path)
    cfg = params.get_config(mode=mode, timeframe=timeframe)
    params.validate_features(cfg.features, use_bm=cfg.use_bm)
    run_name = build_name(cfg.mode, cfg.rolling_train_years, cfg.rolling_test_years)
    out_root = Path(cfg.output_dir) / run_name
    return out_root / view
