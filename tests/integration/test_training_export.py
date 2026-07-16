import hashlib
import json
import os
from datetime import UTC, datetime, timedelta

import pytest
from test_clickhouse import event, migrated_client

from imbalance_pipeline.storage.clickhouse import ClickHouseRepository
from imbalance_pipeline.training.export_clickhouse import export_clickhouse_training_dataset
from imbalance_pipeline.training.export_data import open_training_dataset
from imbalance_pipeline.training.splits import IndexRange

CLICKHOUSE_URL = os.getenv("IMBALANCE_CLICKHOUSE_URL")
CLICKHOUSE_ADMIN_USER = os.getenv("IMBALANCE_CLICKHOUSE_ADMIN_USER")
CLICKHOUSE_ADMIN_PASSWORD = os.getenv("IMBALANCE_CLICKHOUSE_ADMIN_PASSWORD")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        CLICKHOUSE_URL is None,
        reason="set IMBALANCE_CLICKHOUSE_URL to run the real ClickHouse integration tests",
    ),
    pytest.mark.skipif(
        CLICKHOUSE_URL is not None
        and (CLICKHOUSE_ADMIN_USER is None or CLICKHOUSE_ADMIN_PASSWORD is None),
        reason="set ClickHouse admin credentials for schema migrations",
    ),
]


@pytest.mark.asyncio
async def test_clickhouse_export_writes_verified_causal_dataset(tmp_path) -> None:
    client = await migrated_client()
    repository = ClickHouseRepository(client, database="imbalance")
    start = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    values: dict[datetime, float] = {}
    try:
        # 180 minutes of local history, the two requested cutoffs, and the
        # final next-minute target required to build both labels.
        for offset in range(-180, 3):
            timestamp = start + timedelta(minutes=offset)
            value = float(offset * 2 + 25)
            values[timestamp] = value
            ingested_at = timestamp + timedelta(seconds=5)
            if timestamp == start + timedelta(minutes=2):
                ingested_at = end - timedelta(seconds=1)
            await repository.insert_event(
                event(
                    event_id=f"training-export-{offset}",
                    event_time=timestamp,
                    ingested_at=ingested_at,
                    value=value,
                )
            )

        output = await export_clickhouse_training_dataset(
            repository, tmp_path / "dataset", start, end
        )
        opened = open_training_dataset(output)
        assert opened.count > 0
        assert opened.feature_schema_hash
        cutoffs = opened.cutoffs()
        assert cutoffs == sorted(cutoffs)
        assert cutoffs == [start, start + timedelta(minutes=1)]

        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["count"] == opened.count
        assert metadata["feature_schema_hash"] == opened.feature_schema_hash
        assert metadata["dataset_digest"] == opened.dataset_digest
        for shard in metadata["shards"]:
            digest = hashlib.sha256((output / shard["name"]).read_bytes()).hexdigest()
            assert digest == shard["checksum"]

        examples = list(opened.iter_range(IndexRange(0, opened.count)))
        assert [example.target_next for example in examples] == [
            values[start + timedelta(minutes=1)],
            values[start + timedelta(minutes=2)],
        ]
    finally:
        await repository.aclose()
