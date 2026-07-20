import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from imbalance_pipeline.features.schema import DEFAULT_FEATURE_REGISTRY
from imbalance_pipeline.training.export_csv import (
    export_ods133_training_dataset,
    read_ods133_csv,
    select_training_cutoffs,
)
from imbalance_pipeline.training.export_data import open_training_dataset


def test_read_ods133_csv_normalizes_bom_semicolon_rows_and_reverse_order(
    tmp_path: Path,
) -> None:
    newest = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    path = _write_csv(tmp_path / "ods133.csv", [newest, newest - timedelta(minutes=1)])

    observations = read_ods133_csv(path)

    assert [row.timestamp for row in observations] == [
        newest - timedelta(minutes=1),
        newest,
    ]
    assert observations[1].system_imbalance_mw == -25.0
    assert observations[0].ace_mw == -5.0
    assert observations[0].imbalance_price_eur_mwh == 100.0


def test_read_ods133_csv_rejects_duplicate_minutes(tmp_path: Path) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    path = _write_csv(tmp_path / "ods133.csv", [timestamp, timestamp])

    with pytest.raises(ValueError, match="duplicate ODS133 minute"):
        read_ods133_csv(path)


def test_select_training_cutoffs_requires_complete_history_and_next_minute() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=offset) for offset in range(12)]
    timestamps.remove(start + timedelta(minutes=5))

    cutoffs = select_training_cutoffs(timestamps, history_minutes=4, stride_minutes=2)

    assert cutoffs == [start + timedelta(minutes=10)]


@pytest.mark.asyncio
async def test_export_ods133_csv_writes_flip_training_examples(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=offset) for offset in range(1443)]
    source = _write_csv(tmp_path / "ods133.csv", timestamps)

    output = await export_ods133_training_dataset(
        source,
        tmp_path / "training",
        stride_minutes=1,
        batch_size=2,
    )

    dataset = open_training_dataset(output)
    examples = list(dataset.iter_range(section=_all(dataset.count)))
    assert dataset.count == 3
    assert dataset.feature_schema_hash == DEFAULT_FEATURE_REGISTRY.fingerprint
    assert [example.cutoff for example in examples] == timestamps[1439:1442]
    assert all(example.flip_mask == 1.0 for example in examples)


def _all(count: int):
    from imbalance_pipeline.training.splits import IndexRange

    return IndexRange(0, count)


def _write_csv(path: Path, timestamps: list[datetime]) -> Path:
    fieldnames = [
        "Datetime",
        "Resolution code",
        "Quarter hour",
        "Quality status",
        "ACE",
        "System imbalance",
        "Alpha",
        "Alpha'",
        "Marginal incremental price",
        "Marginal decremental price",
        "Imbalance Price",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            writer.writerow(
                {
                    "Datetime": timestamp.isoformat(),
                    "Resolution code": "PT1M",
                    "Quarter hour": timestamp.replace(
                        minute=(timestamp.minute // 15) * 15
                    ).isoformat(),
                    "Quality status": "NotValidated",
                    "ACE": "-5.0",
                    "System imbalance": "-25.0" if index == 0 else "25.0",
                    "Alpha": "0.0",
                    "Alpha'": "0.0",
                    "Marginal incremental price": "120.0",
                    "Marginal decremental price": "100.0",
                    "Imbalance Price": "100.0",
                }
            )
    return path
