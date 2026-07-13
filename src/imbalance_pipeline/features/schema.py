import hashlib
import json
from dataclasses import dataclass
from typing import Literal

FeatureGroup = Literal["local", "context", "static"]


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    group: FeatureGroup
    dtype: str
    scaling_kind: str
    max_source_age_minutes: int | None = None


@dataclass(frozen=True, slots=True)
class FeatureRegistry:
    features: tuple[FeatureDefinition, ...]
    deadband_mw: float = 10.0
    local_window_minutes: int = 180
    context_steps: int = 96
    context_resolution_minutes: int = 15

    def __post_init__(self) -> None:
        if self.deadband_mw <= 0:
            raise ValueError("deadband_mw must be positive")
        if self.local_window_minutes <= 0:
            raise ValueError("local_window_minutes must be positive")
        if self.context_steps <= 0 or self.context_resolution_minutes <= 0:
            raise ValueError("context window dimensions must be positive")
        for group in ("local", "context", "static"):
            names = tuple(feature.name for feature in self.features if feature.group == group)
            if len(names) != len(set(names)):
                raise ValueError(f"{group} feature names must be unique")
        if not all(feature.group in {"local", "context", "static"} for feature in self.features):
            raise ValueError("feature groups must be local, context, or static")

    @property
    def local_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features if feature.group == "local")

    @property
    def context_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features if feature.group == "context")

    @property
    def static_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features if feature.group == "static")

    @property
    def fingerprint(self) -> str:
        canonical = {
            "context_resolution_minutes": self.context_resolution_minutes,
            "context_steps": self.context_steps,
            "deadband_mw": self.deadband_mw,
            "features": [
                {
                    "dtype": feature.dtype,
                    "group": feature.group,
                    "max_source_age_minutes": feature.max_source_age_minutes,
                    "name": feature.name,
                    "scaling_kind": feature.scaling_kind,
                }
                for feature in self.features
            ],
            "local_window_minutes": self.local_window_minutes,
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def default(cls, *, deadband_mw: float = 10.0) -> "FeatureRegistry":
        return cls(features=_default_features(), deadband_mw=deadband_mw)


def _feature(
    name: str,
    group: FeatureGroup,
    scaling_kind: str,
    *,
    max_source_age_minutes: int | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        group=group,
        dtype="float32",
        scaling_kind=scaling_kind,
        max_source_age_minutes=max_source_age_minutes,
    )


def _default_features() -> tuple[FeatureDefinition, ...]:
    local = (
        _feature("system_imbalance_mw", "local", "robust"),
        _feature("ace_mw", "local", "robust"),
        _feature("imbalance_price_eur_mwh", "local", "robust"),
        _feature("marginal_incremental_price_eur_mwh", "local", "robust"),
        _feature("marginal_decremental_price_eur_mwh", "local", "robust"),
        _feature("quality_validated", "local", "none"),
        _feature("delta_1_mw", "local", "robust"),
        _feature("delta_2_mw", "local", "robust"),
        _feature("acceleration_mw", "local", "robust"),
        _feature("ewm_5_mw", "local", "robust"),
        _feature("ewm_15_mw", "local", "robust"),
        _feature("ewm_60_mw", "local", "robust"),
        _feature("rolling_median_5_mw", "local", "robust"),
        _feature("rolling_median_15_mw", "local", "robust"),
        _feature("robust_scale_15_mw", "local", "robust"),
        _feature("robust_scale_60_mw", "local", "robust"),
        _feature("slope_5_mw_per_minute", "local", "robust"),
        _feature("slope_15_mw_per_minute", "local", "robust"),
        _feature("rolling_min_15_mw", "local", "robust"),
        _feature("rolling_max_15_mw", "local", "robust"),
        _feature("boundary_distance_mw", "local", "robust"),
        _feature("confirmed_state_sign", "local", "none"),
        _feature("state_duration_minutes", "local", "log1p"),
        _feature("imbalance_age_minutes", "local", "log1p", max_source_age_minutes=5),
    )
    context = (
        _feature("imbalance_mean_mw", "context", "robust"),
        _feature("imbalance_min_mw", "context", "robust"),
        _feature("imbalance_max_mw", "context", "robust"),
        _feature("imbalance_std_mw", "context", "robust"),
        _feature("imbalance_age_minutes", "context", "log1p", max_source_age_minutes=15),
        _feature("load_actual_mw", "context", "robust", max_source_age_minutes=30),
        _feature("load_forecast_mw", "context", "robust", max_source_age_minutes=60),
        _feature("load_error_mw", "context", "robust", max_source_age_minutes=30),
        _feature("load_actual_age_minutes", "context", "log1p", max_source_age_minutes=30),
        _feature("wind_actual_mw", "context", "robust", max_source_age_minutes=30),
        _feature("wind_forecast_mw", "context", "robust", max_source_age_minutes=60),
        _feature("wind_error_mw", "context", "robust", max_source_age_minutes=30),
        _feature("wind_forecast_spread_mw", "context", "robust", max_source_age_minutes=60),
        _feature("wind_actual_age_minutes", "context", "log1p", max_source_age_minutes=30),
        _feature("solar_actual_mw", "context", "robust", max_source_age_minutes=30),
        _feature("solar_forecast_mw", "context", "robust", max_source_age_minutes=60),
        _feature("solar_error_mw", "context", "robust", max_source_age_minutes=30),
        _feature("solar_forecast_spread_mw", "context", "robust", max_source_age_minutes=60),
        _feature("solar_actual_age_minutes", "context", "log1p", max_source_age_minutes=30),
    )
    static = (
        _feature("quarter_hour_phase_sin", "static", "none"),
        _feature("quarter_hour_phase_cos", "static", "none"),
        _feature("minute_phase_sin", "static", "none"),
        _feature("minute_phase_cos", "static", "none"),
        _feature("hour_phase_sin", "static", "none"),
        _feature("hour_phase_cos", "static", "none"),
        _feature("weekday_phase_sin", "static", "none"),
        _feature("weekday_phase_cos", "static", "none"),
        _feature("day_of_year_phase_sin", "static", "none"),
        _feature("day_of_year_phase_cos", "static", "none"),
        _feature("is_weekend", "static", "none"),
        _feature("is_belgian_holiday", "static", "none"),
        _feature("is_dst_transition", "static", "none"),
    )
    return local + context + static


DEFAULT_FEATURE_REGISTRY = FeatureRegistry.default()
