import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.model.bundle import validate_bundle
from imbalance_pipeline.training.data import TrainingExample
from imbalance_pipeline.training.export_data import write_training_dataset
from imbalance_pipeline.training.splits import IndexRange, TimeSplit
from imbalance_pipeline.training.train import TrainingConfig, train_ensemble


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
    validation = validate_bundle(candidate, expected_schema_hash="synthetic-schema")

    assert candidate == tmp_path / "candidate"
    assert validation.valid is True
    assert [member["seed"] for member in evaluation["members"]] == [17, 29, 43]
    assert all(np.isfinite(member["best_validation_loss"]) for member in evaluation["members"])
    assert all((candidate / f"member-{index}.pt").is_file() for index in range(3))


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
