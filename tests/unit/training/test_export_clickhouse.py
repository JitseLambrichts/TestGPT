from datetime import UTC, datetime

import pytest

from imbalance_pipeline.training.export_clickhouse import export_clickhouse_training_dataset


def test_rejects_naive_boundaries(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        import asyncio

        asyncio.run(
            export_clickhouse_training_dataset(
                object(), tmp_path / "dataset", datetime(2025, 1, 1), datetime(2025, 1, 2)
            )
        )


def test_rejects_reversed_range(tmp_path):
    import asyncio

    start = datetime(2025, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError, match="end must be after start"):
        asyncio.run(
            export_clickhouse_training_dataset(object(), tmp_path / "dataset", start, start)
        )


@pytest.mark.asyncio
async def test_empty_input_raises_and_does_not_create_output(tmp_path):
    class Repository:
        async def fetch_imbalance_versions(self, *args, **kwargs):
            return []

    with pytest.raises(ValueError, match="no usable"):
        await export_clickhouse_training_dataset(
            Repository(),
            tmp_path / "dataset",
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        )
    assert not (tmp_path / "dataset").exists()
