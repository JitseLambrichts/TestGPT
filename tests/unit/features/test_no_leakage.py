from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from conftest import MemoryFeatureSource, VersionedObservation, minute_rows, observation

from imbalance_pipeline.features.engine import FeatureEngine
from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY

CUTOFF = datetime(2026, 7, 13, 10, 7, tzinfo=UTC)


@pytest.mark.asyncio
async def test_future_event_time_cannot_change_an_existing_snapshot() -> None:
    source = MemoryFeatureSource(minute_rows(CUTOFF, 24 * 60, value=15.0))
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)
    knowledge_cutoff = CUTOFF + timedelta(seconds=10)

    before = await engine.build(CUTOFF, knowledge_cutoff=knowledge_cutoff)
    source.add(
        VersionedObservation(
            observation(CUTOFF + timedelta(minutes=1), 9_999.0),
            CUTOFF + timedelta(minutes=1, seconds=5),
        )
    )
    after = await engine.build(CUTOFF, knowledge_cutoff=knowledge_cutoff)

    np.testing.assert_array_equal(before.local_values, after.local_values)
    np.testing.assert_array_equal(before.context_values, after.context_values)
    assert before.feature_schema_hash == after.feature_schema_hash


@pytest.mark.asyncio
async def test_late_correction_is_hidden_until_its_knowledge_cutoff() -> None:
    original_rows = minute_rows(CUTOFF, 24 * 60, value=15.0)
    source = MemoryFeatureSource(original_rows)
    engine = FeatureEngine(source, DEFAULT_FEATURE_REGISTRY)
    before_correction = CUTOFF + timedelta(seconds=10)

    baseline = await engine.build(CUTOFF, knowledge_cutoff=before_correction)
    source.add(
        VersionedObservation(
            observation(CUTOFF - timedelta(minutes=5), -2_500.0),
            CUTOFF + timedelta(minutes=2),
        )
    )
    same_vintage = await engine.build(CUTOFF, knowledge_cutoff=before_correction)
    revised = await engine.build(
        CUTOFF,
        knowledge_cutoff=CUTOFF + timedelta(minutes=3),
    )

    np.testing.assert_array_equal(baseline.local_values, same_vintage.local_values)
    assert not np.array_equal(baseline.local_values, revised.local_values)
