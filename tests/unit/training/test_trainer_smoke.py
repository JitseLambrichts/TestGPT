import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.model.bundle import validate_bundle
from imbalance_pipeline.training.data import TrainingExample
from imbalance_pipeline.training.export_data import write_training_dataset
from imbalance_pipeline.training.splits import IndexRange, TimeSplit
from imbalance_pipeline.training.train import (
    TrainingConfig,
    _cohort_masks,
    _EnsemblePredictions,
    _resolve_split,
    _validation_objective,
    train_ensemble,
)


def test_train_ensemble_writes_three_seeded_onnx_members_and_evaluation(tmp_path: Path) -> None:
    dataset = write_training_dataset(_examples(), tmp_path / "dataset")
    config = TrainingConfig(
        batch_size=8,
        d_model=8,
        epochs=2,
        seeds=(17, 29, 43),
        tcn_blocks=1,
        transformer_heads=2,
        transformer_layers=1,
        split=TimeSplit(
            train=IndexRange(0, 18),
            validation=IndexRange(18, 22),
            calibration=IndexRange(22, 26),
            test=IndexRange(26, 32),
        ),
    )

    candidate = train_ensemble(dataset, tmp_path / "candidate", config)
    evaluation = json.loads((candidate / "evaluation.json").read_text())
    baselines = json.loads((candidate / "baseline_evaluation.json").read_text())
    run_config = json.loads((candidate / "run_config.json").read_text())
    validation = validate_bundle(candidate, expected_schema_hash="synthetic-schema")

    assert candidate == tmp_path / "candidate"
    assert validation.valid is True
    assert [member["seed"] for member in evaluation["members"]] == [17, 29, 43]
    assert all(np.isfinite(member["best_validation_loss"]) for member in evaluation["members"])
    assert all((candidate / f"member-{index}.pt").is_file() for index in range(3))
    assert {"persistence", "clipped_linear_drift", "rolling_median", "classical"} == set(baselines)
    assert "current_positive" in evaluation["candidate"]["cohort_mae"]
    assert run_config["device"] == "cpu"
    assert run_config["split"]["test"] == {"start": 26, "stop": 32}
    assert (
        run_config["dataset"]["digest"]
        == json.loads((dataset / "metadata.json").read_text())["dataset_digest"]
    )


def test_cohorts_cover_state_volatility_phase_and_source_quality() -> None:
    prediction = _EnsemblePredictions(
        predicted_mw=np.zeros(4),
        p10_mw=np.zeros(4),
        p90_mw=np.zeros(4),
        raw_flip_probability=np.zeros(4),
        actual_mw=np.zeros(4),
        flip_target=np.zeros(4, dtype=np.int64),
        flip_mask=np.ones(4, dtype=bool),
        current_state=np.asarray([1, -1, 0, 1], dtype=np.int64),
        volatility=np.asarray([1.0, 2.0, 3.0, 4.0]),
        quarter_hour_phase=np.asarray([0, 5, 10, 14], dtype=np.int8),
        source_quality=np.asarray([-1, 0, 1, 1], dtype=np.int8),
    )

    cohorts = _cohort_masks(prediction)

    assert {"current_positive", "current_negative", "current_unknown"} <= set(cohorts)
    assert {"low_volatility", "medium_volatility", "high_volatility"} <= set(cohorts)
    phase_masks = [
        cohorts["quarter_hour_phase_start"],
        cohorts["quarter_hour_phase_middle"],
        cohorts["quarter_hour_phase_end"],
    ]
    assert np.array_equal(np.sum(phase_masks, axis=0), np.ones(4, dtype=np.int64))
    assert {
        "source_quality_unknown",
        "source_quality_unvalidated",
        "source_quality_validated",
    } <= set(cohorts)


def test_trainer_resolves_default_at_the_latest_dataset_period_and_rejects_leakage() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=index) for index in range(365 * 24 * 60)]
    invalid = TimeSplit(
        train=IndexRange(0, 20),
        validation=IndexRange(10, 24),
        calibration=IndexRange(25, 30),
        test=IndexRange(31, 36),
    )

    resolved = _resolve_split(timestamps, TrainingConfig())

    assert resolved.test.stop == len(timestamps)
    with pytest.raises(ValueError, match="strictly chronological and disjoint"):
        _resolve_split(timestamps, TrainingConfig(split=invalid))


def test_validation_objective_weights_examples_and_known_flip_labels_not_batches() -> None:
    value = _validation_objective(
        nll_sum=10.0 * 100 + 1_000.0,
        nll_count=101,
        brier_sum=0.5,
        brier_count=1,
    )

    assert value == pytest.approx((2_000.0 / 101) + 0.125)


def _examples() -> list[TrainingExample]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    examples: list[TrainingExample] = []
    for index in range(32):
        value = float((index % 11) - 5)
        target = value + (1.0 if index % 2 else -1.0)
        current_state = ConfirmedState.POSITIVE if index % 2 else ConfirmedState.NEGATIVE
        examples.append(
            TrainingExample(
                event_id=f"synthetic-{index}",
                cutoff=start + timedelta(days=index),
                feature_schema_hash="synthetic-schema",
                local_values=np.full((12, 4), value, dtype=np.float32),
                local_masks=np.ones((12, 4), dtype=np.uint8),
                context_values=np.full((8, 3), value / 2, dtype=np.float32),
                context_masks=np.ones((8, 3), dtype=np.uint8),
                static_values=np.asarray([float(index % 4), 1.0], dtype=np.float32),
                static_masks=np.ones(2, dtype=np.uint8),
                target_next=target,
                target_delta=target - value,
                delta_mask=1.0,
                auxiliary_target=np.asarray([target, target, target], dtype=np.float32),
                auxiliary_mask=np.ones(3, dtype=np.float32),
                current_state=current_state,
                flip_target=float(index % 2),
                flip_mask=1.0,
            )
        )
    return examples
