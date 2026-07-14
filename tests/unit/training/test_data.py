from datetime import UTC, datetime, timedelta

import numpy as np
import torch

from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.features.engine import FeatureSnapshot
from imbalance_pipeline.training.data import (
    RobustPreprocessor,
    TrainingBatch,
    build_training_examples,
    preprocess_training_examples,
)

NOW = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)


def snapshot(
    *,
    cutoff: datetime = NOW,
    current_state: ConfirmedState | None,
    local_values: np.ndarray | None = None,
    local_masks: np.ndarray | None = None,
) -> FeatureSnapshot:
    values = (
        local_values
        if local_values is not None
        else np.asarray([[5.0, 1.0], [10.0, 2.0]], dtype=np.float32)
    )
    masks = (
        local_masks
        if local_masks is not None
        else np.ones(values.shape, dtype=np.uint8)
    )
    return FeatureSnapshot(
        event_id=f"feature-{cutoff.isoformat()}",
        cutoff=cutoff,
        knowledge_cutoff=cutoff,
        target_time=cutoff + timedelta(minutes=1),
        feature_schema_hash="schema-v1",
        local_values=values,
        local_masks=masks,
        context_values=np.asarray([[3.0]], dtype=np.float32),
        context_masks=np.asarray([[1]], dtype=np.uint8),
        static_values=np.asarray([4.0], dtype=np.float32),
        static_masks=np.asarray([1], dtype=np.uint8),
        current_state=current_state,
        model_eligible=True,
        observed_imbalance_minutes=int(masks[:, 0].sum()),
        created_at=cutoff,
    )


def targets(*values: float | None) -> dict[datetime, float]:
    return {
        NOW + timedelta(minutes=index + 1): value
        for index, value in enumerate(values)
        if value is not None
    }


def test_labels_target_next_minute_and_hysteresis_flip_semantics() -> None:
    unknown = build_training_examples([snapshot(current_state=None)], targets(0.0))[0]
    neutral = build_training_examples(
        [snapshot(current_state=ConfirmedState.POSITIVE)],
        targets(5.0),
    )[0]
    negative = build_training_examples(
        [snapshot(current_state=ConfirmedState.POSITIVE)],
        targets(-11.0, -12.0, None, None, -14.0, None, None, None, None, -15.0),
    )[0]

    assert unknown.flip_mask == 0.0
    assert neutral.flip_mask == 1.0
    assert neutral.flip_target == 0.0
    assert negative.flip_target == 1.0
    assert negative.target_next == -11.0
    np.testing.assert_array_equal(negative.auxiliary_target, [-12.0, -14.0, -15.0])
    np.testing.assert_array_equal(negative.auxiliary_mask, [1.0, 1.0, 1.0])


def test_preprocessing_is_train_only_and_preserves_missing_zeros() -> None:
    train = snapshot(
        current_state=ConfirmedState.POSITIVE,
        local_values=np.asarray([[1.0, 2.0], [3.0, 99.0]], dtype=np.float32),
        local_masks=np.asarray([[1, 1], [1, 0]], dtype=np.uint8),
    )
    held_out = snapshot(
        cutoff=NOW + timedelta(minutes=1),
        current_state=ConfirmedState.POSITIVE,
        local_values=np.asarray([[10_000.0, -4.0], [20.0, 7.0]], dtype=np.float32),
        local_masks=np.asarray([[1, 1], [0, 0]], dtype=np.uint8),
    )
    train_example = build_training_examples([train], targets(12.0))[0]
    preprocessor = RobustPreprocessor.fit([train_example])

    transformed = preprocessor.transform(held_out)
    restored = RobustPreprocessor.from_json(preprocessor.to_json())

    assert preprocessor.local_location[0] == 2.0
    assert transformed.local_values[0, 0] == 12.0
    assert transformed.local_values[1, 0] == 0.0
    assert transformed.local_values[1, 1] == 0.0
    np.testing.assert_array_equal(transformed.local_masks, held_out.local_masks)
    np.testing.assert_allclose(restored.transform(held_out).local_values, transformed.local_values)


def test_preprocess_training_examples_uses_fitted_statistics_without_changing_labels() -> None:
    train_example = build_training_examples(
        [snapshot(current_state=ConfirmedState.POSITIVE)],
        targets(12.0),
    )[0]
    held_out_example = build_training_examples(
        [
            snapshot(
                cutoff=NOW + timedelta(minutes=1),
                current_state=ConfirmedState.POSITIVE,
                local_values=np.asarray([[100.0, 2.0], [300.0, 4.0]], dtype=np.float32),
            )
        ],
        {NOW + timedelta(minutes=2): 400.0},
    )[0]

    transformed = preprocess_training_examples(
        [held_out_example],
        RobustPreprocessor.fit([train_example]),
    )[0]

    assert transformed.target_next == held_out_example.target_next
    assert transformed.flip_target == held_out_example.flip_target
    assert transformed.local_values[-1, 0] == 12.0


def test_streaming_preprocessor_matches_full_fit_when_its_reservoir_covers_input() -> None:
    examples = [
        build_training_examples(
            [
                snapshot(
                    cutoff=NOW + timedelta(minutes=index),
                    current_state=ConfirmedState.POSITIVE,
                )
            ],
            {NOW + timedelta(minutes=index + 1): float(index)},
        )[0]
        for index in range(3)
    ]

    full = RobustPreprocessor.fit(examples)
    streamed = RobustPreprocessor.fit_stream(iter(examples), max_examples=3, seed=17)

    np.testing.assert_allclose(streamed.local_location, full.local_location)
    np.testing.assert_allclose(streamed.context_scale, full.context_scale)


def test_batch_preprocessing_transforms_only_features_and_keeps_labels_on_cpu() -> None:
    example = build_training_examples(
        [snapshot(current_state=ConfirmedState.POSITIVE)],
        targets(12.0),
    )[0]
    batch = TrainingBatch.from_examples([example], batch_id="streaming")

    transformed = RobustPreprocessor.fit([example]).transform_batch(batch)

    assert transformed.local.shape == batch.local.shape
    assert transformed.target_next.item() == batch.target_next.item()
    assert transformed.local.device.type == "cpu"
    assert torch.equal(transformed.flip_mask, batch.flip_mask)
