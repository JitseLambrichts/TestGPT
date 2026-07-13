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
    source_profile: str = "imbalance-only-v1"
    transform_version: str = "causal-v1"
    ewm_windows: tuple[int, ...] = (5, 15, 60)
    median_windows: tuple[int, ...] = (5, 15)
    robust_scale_windows: tuple[int, ...] = (15, 60)
    slope_windows: tuple[int, ...] = (5, 15)
    missingness_policy: str = "zero-mask-causal-age-v1"
    context_aggregation: str = "right-closed-15-minute-v1"
    calendar_timezone: str = "Europe/Brussels"

    def __post_init__(self) -> None:
        if self.deadband_mw <= 0:
            raise ValueError("deadband_mw must be positive")
        if (
            self.local_window_minutes,
            self.context_steps,
            self.context_resolution_minutes,
        ) != (180, 96, 15):
            raise ValueError("fixed feature contract requires 180 local minutes and 96x15 context")
        if not self.source_profile:
            raise ValueError("source_profile cannot be empty")
        if not self.transform_version:
            raise ValueError("transform_version cannot be empty")
        if not all(
            (
                self.ewm_windows,
                self.median_windows,
                self.robust_scale_windows,
                self.slope_windows,
            )
        ):
            raise ValueError("transform window groups cannot be empty")
        if not all(
            window > 0
            for window in (
                self.ewm_windows
                + self.median_windows
                + self.robust_scale_windows
                + self.slope_windows
            )
        ):
            raise ValueError("transform windows must be positive")
        for group in ("local", "context", "static"):
            names = tuple(feature.name for feature in self.features if feature.group == group)
            if len(names) != len(set(names)):
                raise ValueError(f"{group} feature names must be unique")
        if not all(feature.group in {"local", "context", "static"} for feature in self.features):
            raise ValueError("feature groups must be local, context, or static")
        local_names = {feature.name for feature in self.features if feature.group == "local"}
        required_transform_names = {
            *(f"ewm_{window}_mw" for window in self.ewm_windows),
            *(f"rolling_median_{window}_mw" for window in self.median_windows),
            *(f"robust_scale_{window}_mw" for window in self.robust_scale_windows),
            *(f"slope_{window}_mw_per_minute" for window in self.slope_windows),
            f"rolling_min_{max(self.median_windows)}_mw",
            f"rolling_max_{max(self.median_windows)}_mw",
        }
        if not required_transform_names <= local_names:
            raise ValueError("transform windows must match the local feature contract")

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
            "context_aggregation": self.context_aggregation,
            "calendar_timezone": self.calendar_timezone,
            "deadband_mw": self.deadband_mw,
            "ewm_windows": self.ewm_windows,
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
            "median_windows": self.median_windows,
            "missingness_policy": self.missingness_policy,
            "robust_scale_windows": self.robust_scale_windows,
            "slope_windows": self.slope_windows,
            "source_profile": self.source_profile,
            "transform_version": self.transform_version,
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def default(
        cls,
        *,
        deadband_mw: float = 10.0,
        source_profile: str = "imbalance-only-v1",
        transform_version: str = "causal-v1",
    ) -> "FeatureRegistry":
        return cls(
            features=_default_features(),
            deadband_mw=deadband_mw,
            source_profile=source_profile,
            transform_version=transform_version,
        )


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
