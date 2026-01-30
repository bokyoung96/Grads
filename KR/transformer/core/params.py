import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


TRANSFORMER_DIR = Path(__file__).resolve().parents[1]
GRADS_DIR = TRANSFORMER_DIR.parent

DEFAULT_CONFIG_PATH = TRANSFORMER_DIR / "config" / "config.json"
BM_CONFIG_PATH = TRANSFORMER_DIR / "config" / "config_bm.json"

from .model.groups import feature_order


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_config_path(config_path: Optional[Path], use_bm: bool = False) -> Path:
    if config_path is not None:
        return Path(config_path)
    return BM_CONFIG_PATH if use_bm else DEFAULT_CONFIG_PATH


@dataclass(frozen=True)
class TransformerConfig:
    mode: str
    batch_size: int
    max_epoch: int
    lr: float
    lookback: int
    stride: int
    horizon: int
    min_assets: int
    min_assets_floor: int
    min_assets_ratio: float
    min_valid: float
    rolling_train_years: int
    rolling_test_years: int
    rolling_step_years: int
    d_model: int
    nhead: int
    n_layers: int
    d_ff: int
    drop: float
    features: Tuple[str, ...]
    use_bm: bool
    use_univ: bool
    universe_path: Path
    norm: str
    norm_scope: str
    label_type: str
    threshold: float
    cache_dir: Path
    features_dir: Path
    output_dir: Path
    checkpoint_dir: Path



class TransformerParams:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config: dict[str, Any] = json.load(f)

    def get_config(self, mode: str = "TEST", timeframe: str = "MEDIUM") -> TransformerConfig:
        mode_cfg = self.config["mode_configs"][mode]
        tf_cfg = self.config["timeframe_configs"][timeframe]
        model_cfg = tf_cfg["model"]
        rolling_cfg = self.config.get("rolling", {})
        cache_dir = _resolve(GRADS_DIR, self.config.get("cache_dir", "DATA/transformer"))
        features_dir = _resolve(GRADS_DIR, self.config.get("features_dir", "DATA/processed/features"))
        universe_path = _resolve(GRADS_DIR, self.config.get("universe_path", "DATA/processed/universe_k200.parquet"))
        output_dir = _resolve(TRANSFORMER_DIR, self.config.get("output_dir", "artifacts/out"))
        checkpoint_dir = _resolve(TRANSFORMER_DIR, self.config.get("checkpoint_dir", "artifacts/models"))
        min_assets_floor = int(self.config.get("min_assets_floor", 190))
        min_assets_ratio = float(self.config.get("min_assets_ratio", 0.25))
        return TransformerConfig(
            mode=f"{mode_cfg['mode']}_{timeframe.lower()}",
            batch_size=int(mode_cfg["batch_size"]),
            max_epoch=int(mode_cfg["max_epoch"]),
            lr=float(mode_cfg["lr"]),
            lookback=int(tf_cfg["lookback"]),
            stride=int(tf_cfg["stride"]),
            horizon=int(tf_cfg["horizon"]),
            min_assets=int(tf_cfg["min_assets"]),
            min_assets_floor=min_assets_floor,
            min_assets_ratio=min_assets_ratio,
            min_valid=float(tf_cfg.get("min_valid", 0.95)),
            rolling_train_years=int(rolling_cfg.get("train_years", 5)),
            rolling_test_years=int(rolling_cfg.get("test_years", 1)),
            rolling_step_years=int(rolling_cfg.get("step_years", 1)),
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            n_layers=int(model_cfg["n_layers"]),
            d_ff=int(model_cfg["d_ff"]),
            drop=float(model_cfg["drop"]),
            features=tuple(self.config["features"]),
            use_bm=bool(self.config.get("use_bm", False)),
            use_univ=bool(self.config.get("use_univ", False)),
            universe_path=universe_path,
            norm=str(self.config.get("norm", "none")),
            norm_scope=str(self.config.get("norm_scope", "full")),
            label_type=str(self.config.get("label_type", "classification")),
            threshold=float(self.config.get("threshold", 0.0)),
            cache_dir=cache_dir,
            features_dir=features_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )

    def validate_features(self, features: Tuple[str, ...], *, use_bm: bool = False) -> None:
        feats = tuple(features)
        expected = tuple(feature_order(use_bm))
        if feats != expected:
            raise ValueError(
                "mfd feature order mismatch: config features must exactly match FEATURE_ORDER "
                "(grouped hard-coded order)."
            )
        has_bm = any(str(f).startswith("bm_") for f in feats)
        if use_bm and not has_bm:
            raise ValueError("use_bm=True requires bm_* features present in config features list.")
        if (not use_bm) and has_bm:
            raise ValueError("use_bm=False cannot include bm_* features; use the BM config or disable bm features.")

    @property
    def modes(self) -> List[str]:
        return list(self.config["mode_configs"].keys())

    @property
    def timeframes(self) -> List[str]:
        return list(self.config["timeframe_configs"].keys())


def build_name(mode: str, train_years: int, test_years: int, model_type: str = "transformer", base: str = "mfd") -> str:
    return f"{base}_{model_type.lower()}_{mode.lower()}_{int(train_years)}_{int(test_years)}"
