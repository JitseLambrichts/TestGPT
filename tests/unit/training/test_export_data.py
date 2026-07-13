from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.training.data import TrainingExample
from imbalance_pipeline.training.export_data import load_training_dataset, write_training_dataset


def test_training_dataset_round_trips_checksums_schema_and_binary_masks(tmp_path: Path) -> None:
    examples = [_example("one", 1.0), _example("two", -2.0)]

    output = write_training_dataset(examples, tmp_path / "dataset")
    loaded = load_training_dataset(output)

    assert output == tmp_path / "dataset"
    assert [example.event_id for example in loaded] == ["one", "two"]
    assert loaded[0].current_state is ConfirmedState.POSITIVE
    assert loaded[1].current_state is ConfirmedState.NEGATIVE
    np.testing.assert_allclose(loaded[1].local_values, examples[1].local_values)
    np.testing.assert_array_equal(loaded[0].local_masks, examples[0].local_masks)
    assert (output / "metadata.json").is_file()


def test_training_dataset_rejects_a_tampered_shard_and_overwrite_without_force(
    tmp_path: Path,
) -> None:
    output = write_training_dataset([_example("one", 1.0)], tmp_path / "dataset")
    (output / "shard-00000.npz").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        load_training_dataset(output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_training_dataset([_example("two", 2.0)], output)


def _example(event_id: str, value: float) -> TrainingExample:
    cutoff = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    return TrainingExample(
        event_id=event_id,
        cutoff=cutoff,
        feature_schema_hash="schema-v1",
        local_values=np.asarray([[value, 2.0], [value + 1.0, 3.0]], dtype=np.float32),
        local_masks=np.asarray([[1, 1], [1, 0]], dtype=np.uint8),
        context_values=np.asarray([[3.0]], dtype=np.float32),
        context_masks=np.asarray([[1]], dtype=np.uint8),
        static_values=np.asarray([4.0], dtype=np.float32),
        static_masks=np.asarray([1], dtype=np.uint8),
        target_next=value + 2.0,
        target_delta=1.0,
        delta_mask=1.0,
        auxiliary_target=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        auxiliary_mask=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        current_state=ConfirmedState.POSITIVE if value >= 0 else ConfirmedState.NEGATIVE,
        flip_target=float(value < 0),
        flip_mask=1.0,
    )
