from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from conftest import MemoryFeatureSource, VersionedObservation, minute_rows, observation

from imbalance_pipeline.domain.imbalance import ConfirmedState, ConfirmedStateSeed
from imbalance_pipeline.features.engine import FeatureEngine
from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY

CUTOFF = datetime(2026, 7, 13, 10, 7, tzinfo=UTC)
KNOWLEDGE_CUTOFF = CUTOFF + timedelta(seconds=10)


@pytest.mark.asyncio
async def test_builds_exact_local_context_and_static_shapes_at_next_minute() -> None:
    source = MemoryFeatureSource(minute_rows(CUTOFF, 24 * 60, value=20.0))
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    snapshot = await engine.build(CUTOFF, knowledge_cutoff=KNOWLEDGE_CUTOFF)

    assert snapshot.local_values.shape == (180, len(DEFAULT_FEATURE_REGISTRY.local_names))
    assert snapshot.local_masks.shape == snapshot.local_values.shape
    assert snapshot.context_values.shape == (96, len(DEFAULT_FEATURE_REGISTRY.context_names))
    assert snapshot.context_masks.shape == snapshot.context_values.shape
    assert snapshot.static_values.shape == (len(DEFAULT_FEATURE_REGISTRY.static_names),)
    assert snapshot.static_masks.shape == snapshot.static_values.shape
    assert snapshot.target_time == CUTOFF + timedelta(minutes=1)
    assert snapshot.cutoff == CUTOFF
    assert snapshot.knowledge_cutoff == KNOWLEDGE_CUTOFF
    assert snapshot.current_state is ConfirmedState.POSITIVE
    assert snapshot.model_eligible is True
    assert source.calls == [(CUTOFF, 24 * 60 + 7, KNOWLEDGE_CUTOFF)]


@pytest.mark.asyncio
async def test_hysteresis_keeps_neutral_state_until_a_boundary_is_crossed() -> None:
    start = CUTOFF - timedelta(minutes=179)
    values = [20.0] * 30 + [0.0] * 149 + [-15.0]
    source = MemoryFeatureSource(
        [
            VersionedObservation(
                observation(start + timedelta(minutes=index), value),
                start + timedelta(minutes=index, seconds=2),
            )
            for index, value in enumerate(values)
        ]
    )
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    snapshot = await engine.build(CUTOFF, knowledge_cutoff=KNOWLEDGE_CUTOFF)
    state_index = DEFAULT_FEATURE_REGISTRY.local_names.index("confirmed_state_sign")

    assert snapshot.current_state is ConfirmedState.NEGATIVE
    assert snapshot.local_values[30, state_index] == 1.0
    assert snapshot.local_values[-1, state_index] == -1.0


@pytest.mark.asyncio
async def test_seeded_state_persists_through_a_neutral_history_and_counts_elapsed_minutes() -> None:
    rows = [
        VersionedObservation(
            observation(CUTOFF - timedelta(minutes=index), 0.0),
            CUTOFF - timedelta(minutes=index) + timedelta(seconds=2),
        )
        for index in range(180)
    ]
    seed = ConfirmedStateSeed(
        ConfirmedState.POSITIVE,
        CUTOFF - timedelta(minutes=2_000),
        CUTOFF - timedelta(minutes=1_447),
    )
    source = MemoryFeatureSource(rows, state_seed=seed)
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    snapshot = await engine.build(CUTOFF, knowledge_cutoff=KNOWLEDGE_CUTOFF)
    sign_index = DEFAULT_FEATURE_REGISTRY.local_names.index("confirmed_state_sign")
    duration_index = DEFAULT_FEATURE_REGISTRY.local_names.index("state_duration_minutes")

    assert snapshot.current_state is ConfirmedState.POSITIVE
    assert snapshot.local_values[-1, sign_index] == 1.0
    assert snapshot.local_masks[-1, sign_index] == 1
    assert snapshot.local_values[-1, duration_index] == 2_000.0
    assert snapshot.local_masks[-1, duration_index] == 1


@pytest.mark.asyncio
async def test_state_features_remain_visible_when_the_current_minute_is_missing() -> None:
    first = CUTOFF - timedelta(minutes=179)
    rows = [
        VersionedObservation(
            observation(first + timedelta(minutes=index), 0.0),
            first + timedelta(minutes=index, seconds=2),
        )
        for index in range(179)
    ]
    seed = ConfirmedStateSeed(
        ConfirmedState.NEGATIVE,
        CUTOFF - timedelta(minutes=2_000),
        CUTOFF - timedelta(minutes=1_447),
    )
    source = MemoryFeatureSource(rows, state_seed=seed)
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    snapshot = await engine.build(CUTOFF, knowledge_cutoff=KNOWLEDGE_CUTOFF)
    sign_index = DEFAULT_FEATURE_REGISTRY.local_names.index("confirmed_state_sign")
    duration_index = DEFAULT_FEATURE_REGISTRY.local_names.index("state_duration_minutes")

    assert snapshot.local_values[-1, sign_index] == -1.0
    assert snapshot.local_masks[-1, sign_index] == 1
    assert snapshot.local_values[-1, duration_index] == 2_000.0
    assert snapshot.local_masks[-1, duration_index] == 1


@pytest.mark.asyncio
async def test_missing_history_is_zero_masked_and_forces_fallback_eligibility() -> None:
    rows = minute_rows(CUTOFF, 20, value=-25.0)
    source = MemoryFeatureSource(rows)
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    snapshot = await engine.build(CUTOFF, knowledge_cutoff=KNOWLEDGE_CUTOFF)
    imbalance_index = DEFAULT_FEATURE_REGISTRY.local_names.index("system_imbalance_mw")
    load_index = DEFAULT_FEATURE_REGISTRY.context_names.index("load_actual_mw")
    load_age_index = DEFAULT_FEATURE_REGISTRY.context_names.index("load_actual_age_minutes")

    assert snapshot.model_eligible is False
    assert snapshot.observed_imbalance_minutes == 20
    assert np.all(snapshot.local_values[:160, imbalance_index] == 0.0)
    assert np.all(snapshot.local_masks[:160, imbalance_index] == 0)
    assert np.all(snapshot.context_values[:, load_index] == 0.0)
    assert np.all(snapshot.context_masks[:, load_index] == 0)
    assert np.all(snapshot.context_values[:, load_age_index] == 0.0)
    assert np.all(snapshot.context_masks[:, load_age_index] == 0)


@pytest.mark.asyncio
async def test_model_eligibility_rejects_a_stale_last_observation() -> None:
    source = MemoryFeatureSource(
        minute_rows(CUTOFF - timedelta(minutes=150), 30, value=25.0),
        state_seed=ConfirmedStateSeed(
            ConfirmedState.POSITIVE,
            CUTOFF - timedelta(minutes=2_000),
            CUTOFF - timedelta(minutes=1_447),
        ),
    )
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    snapshot = await engine.build(CUTOFF, knowledge_cutoff=KNOWLEDGE_CUTOFF)

    assert snapshot.observed_imbalance_minutes >= 30
    assert snapshot.current_state is ConfirmedState.POSITIVE
    assert snapshot.model_eligible is False


@pytest.mark.asyncio
async def test_context_age_tracks_the_last_causal_observation_across_empty_bins() -> None:
    first_context_timestamp = CUTOFF - timedelta(minutes=24 * 60 + 6)
    source = MemoryFeatureSource(
        [
            VersionedObservation(
                observation(first_context_timestamp, 25.0),
                first_context_timestamp + timedelta(seconds=2),
            )
        ]
    )
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    snapshot = await engine.build(CUTOFF, knowledge_cutoff=KNOWLEDGE_CUTOFF)
    age_index = DEFAULT_FEATURE_REGISTRY.context_names.index("imbalance_age_minutes")

    assert snapshot.context_values[0, age_index] == 14.0
    assert snapshot.context_masks[0, age_index] == 1
    assert snapshot.context_values[1, age_index] == 29.0
    assert snapshot.context_masks[1, age_index] == 1


@pytest.mark.asyncio
async def test_context_fetch_includes_the_complete_oldest_quarter_hour_bin() -> None:
    earliest = CUTOFF - timedelta(minutes=24 * 60 + 6)
    rows = [
        VersionedObservation(
            observation(
                earliest + timedelta(minutes=index),
                -100.0 if index < 7 else 100.0,
            ),
            earliest + timedelta(minutes=index, seconds=2),
        )
        for index in range(24 * 60 + 7)
    ]
    source = MemoryFeatureSource(rows)
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    snapshot = await engine.build(CUTOFF, knowledge_cutoff=KNOWLEDGE_CUTOFF)
    mean_index = DEFAULT_FEATURE_REGISTRY.context_names.index("imbalance_mean_mw")

    assert snapshot.context_values[0, mean_index] == pytest.approx(100.0 / 15.0)
    assert source.calls == [(CUTOFF, 24 * 60 + 7, KNOWLEDGE_CUTOFF)]


@pytest.mark.asyncio
async def test_build_many_reuses_online_transform_and_is_byte_identical() -> None:
    cutoffs = [CUTOFF - timedelta(minutes=offset) for offset in range(10)]
    source = MemoryFeatureSource(minute_rows(CUTOFF, 25 * 60, value=15.0))
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    offline = await engine.build_many(
        cutoffs,
        knowledge_cutoffs=[cutoff + timedelta(seconds=10) for cutoff in cutoffs],
    )

    for cutoff, expected in zip(cutoffs, offline, strict=True):
        online = await engine.build(cutoff, knowledge_cutoff=cutoff + timedelta(seconds=10))
        np.testing.assert_array_equal(online.local_values, expected.local_values)
        np.testing.assert_array_equal(online.local_masks, expected.local_masks)
        np.testing.assert_array_equal(online.context_values, expected.context_values)
        np.testing.assert_array_equal(online.context_masks, expected.context_masks)
        np.testing.assert_array_equal(online.static_values, expected.static_values)
        assert online.feature_schema_hash == expected.feature_schema_hash
    assert len(source.version_calls) == 1
    assert len(source.seed_batch_calls) == 1
    assert len(source.seed_calls) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cutoff",
    [
        datetime(2026, 3, 29, 0, 30, tzinfo=UTC),
        datetime(2026, 10, 25, 0, 30, tzinfo=UTC),
    ],
)
async def test_static_features_mark_both_brussels_dst_transitions(cutoff: datetime) -> None:
    source = MemoryFeatureSource([])
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)

    snapshot = await engine.build(cutoff, knowledge_cutoff=cutoff + timedelta(seconds=1))
    transition_index = DEFAULT_FEATURE_REGISTRY.static_names.index("is_dst_transition")

    assert snapshot.static_values[transition_index] == 1.0
    assert snapshot.static_masks[transition_index] == 1
