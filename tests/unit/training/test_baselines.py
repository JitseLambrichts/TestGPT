from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.training.baselines import (
    _compact_example,
    classical_baseline,
    clipped_linear_drift_baseline,
    persistence_baseline,
    rolling_median_baseline,
)
from imbalance_pipeline.training.data import TrainingExample


def test_deterministic_regression_baselines_use_only_the_local_history() -> None:
    examples = [
        _example(history=[1.0, 2.0, 3.0], target=4.0),
        _example(history=[-10.0, 50.0, 0.0], target=0.0),
    ]

    persistence = persistence_baseline(examples)
    drift = clipped_linear_drift_baseline(examples)
    median = rolling_median_baseline(examples)

    np.testing.assert_allclose(persistence, [3.0, 0.0])
    assert 3.0 <= drift[0] <= 5.0
    assert -10.0 <= drift[1] <= 50.0
    np.testing.assert_allclose(median, [2.0, 0.0])


def test_classical_baseline_returns_finite_point_and_flip_probability() -> None:
    train = [
        _example(
            history=[float(index), float(index + 1), float(index + 2)],
            target=float(index + 3),
        )
        for index in range(24)
    ]
    evaluation = [
        _example(
            history=[float(index), float(index + 1), float(index + 2)],
            target=float(index + 3),
        )
        for index in range(24, 28)
    ]

    result = classical_baseline(train, evaluation)

    assert result.predicted_mw.shape == (4,)
    assert result.flip_probability.shape == (4,)
    assert np.isfinite(result.predicted_mw).all()
    assert np.all((0.0 <= result.flip_probability) & (result.flip_probability <= 1.0))


def test_classical_baseline_accepts_streams_and_bounds_its_training_reservoir() -> None:
    training = (
        _example(
            history=[float(index), float(index + 1), float(index + 2)],
            target=float(index + 3),
        )
        for index in range(12)
    )
    evaluation = (
        _example(
            history=[float(index), float(index + 1), float(index + 2)],
            target=float(index + 3),
        )
        for index in range(12, 16)
    )

    result = classical_baseline(training, evaluation, max_training_examples=3)

    assert result.predicted_mw.shape == (4,)
    assert result.flip_probability.shape == (4,)


def test_classical_baseline_preserves_each_feature_channel_in_its_compact_representation() -> None:
    template = _example(history=[0.0, 1.0], target=2.0)
    first = replace(
        template,
        local_values=np.asarray([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32),
    )
    second = replace(
        template,
        local_values=np.asarray([[0.0, 20.0], [10.0, 30.0]], dtype=np.float32),
    )

    first_compact = _compact_example(first)
    second_compact = _compact_example(second)

    assert first_compact.shape == (20,)
    assert second_compact.shape == (20,)
    assert not np.array_equal(first_compact, second_compact)


def _example(*, history: list[float], target: float) -> TrainingExample:
    cutoff = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    values = np.asarray([[value, value * 2] for value in history], dtype=np.float32)
    masks = np.ones(values.shape, dtype=np.uint8)
    return TrainingExample(
        event_id=f"example-{history[-1]}",
        cutoff=cutoff,
        feature_schema_hash="schema-v1",
        local_values=values,
        local_masks=masks,
        context_values=np.asarray([[1.0]], dtype=np.float32),
        context_masks=np.asarray([[1]], dtype=np.uint8),
        static_values=np.asarray([0.0], dtype=np.float32),
        static_masks=np.asarray([1], dtype=np.uint8),
        target_next=target,
        target_delta=target - history[-1],
        delta_mask=1.0,
        auxiliary_target=np.zeros(3, dtype=np.float32),
        auxiliary_mask=np.zeros(3, dtype=np.float32),
        current_state=ConfirmedState.POSITIVE,
        flip_target=float(int(history[-1] < 0)),
        flip_mask=1.0,
    )
