import pytest

from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY, FeatureRegistry


def test_default_registry_has_stable_ordered_sha256_fingerprint() -> None:
    registry = DEFAULT_FEATURE_REGISTRY

    assert registry.local_names[:6] == (
        "system_imbalance_mw",
        "ace_mw",
        "imbalance_price_eur_mwh",
        "marginal_incremental_price_eur_mwh",
        "marginal_decremental_price_eur_mwh",
        "quality_validated",
    )
    assert registry.context_names[:4] == (
        "imbalance_mean_mw",
        "imbalance_min_mw",
        "imbalance_max_mw",
        "imbalance_std_mw",
    )
    assert registry.static_names[:4] == (
        "quarter_hour_phase_sin",
        "quarter_hour_phase_cos",
        "minute_phase_sin",
        "minute_phase_cos",
    )
    assert len(registry.fingerprint) == 64
    assert registry.fingerprint == FeatureRegistry.default().fingerprint


def test_registry_fingerprint_changes_with_hysteresis_or_ordered_schema() -> None:
    baseline = FeatureRegistry.default()

    assert FeatureRegistry.default(deadband_mw=12.0).fingerprint != baseline.fingerprint
    assert (
        FeatureRegistry.default(source_profile="imbalance-load-wind-solar-v1").fingerprint
        != baseline.fingerprint
    )
    assert (
        FeatureRegistry.default(transform_version="causal-v2").fingerprint != baseline.fingerprint
    )
    assert baseline.local_names != tuple(reversed(baseline.local_names))
    assert {feature.group for feature in baseline.features} == {
        "local",
        "context",
        "static",
    }


def test_registry_rejects_dimensions_the_feature_engine_cannot_preserve() -> None:
    baseline = FeatureRegistry.default()

    with pytest.raises(ValueError, match="fixed feature contract"):
        FeatureRegistry(
            features=baseline.features,
            local_window_minutes=181,
        )
