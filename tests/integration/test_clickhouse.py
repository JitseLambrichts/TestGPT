import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import clickhouse_connect
import pytest
from clickhouse_connect.driver.asyncclient import AsyncClient

from imbalance_pipeline.domain.events import EventEnvelope
from imbalance_pipeline.storage.clickhouse import ClickHouseRepository

CLICKHOUSE_URL = os.getenv("IMBALANCE_CLICKHOUSE_URL")
CLICKHOUSE_USER = os.getenv("IMBALANCE_CLICKHOUSE_USER", "imbalance")
CLICKHOUSE_PASSWORD = os.getenv("IMBALANCE_CLICKHOUSE_PASSWORD", "imbalance")
CLICKHOUSE_ADMIN_USER = os.getenv("IMBALANCE_CLICKHOUSE_ADMIN_USER", CLICKHOUSE_USER)
CLICKHOUSE_ADMIN_PASSWORD = os.getenv(
    "IMBALANCE_CLICKHOUSE_ADMIN_PASSWORD",
    CLICKHOUSE_PASSWORD,
)
DATABASE = os.getenv("IMBALANCE_CLICKHOUSE_DATABASE", "imbalance")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:The current async client is a thread-pool wrapper"),
    pytest.mark.skipif(
        CLICKHOUSE_URL is None,
        reason="set IMBALANCE_CLICKHOUSE_URL to run the real ClickHouse integration tests",
    ),
]


def event(
    *,
    event_id: str = "integration-event-001",
    ingested_at: datetime = datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC),
    value: float = 325.224,
) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        event_type="elia.imbalance.observed",
        schema_version="1",
        source="elia",
        dataset="ods161",
        event_time=datetime(2026, 7, 13, 10, 1, tzinfo=UTC),
        observed_at=datetime(2026, 7, 13, 10, 1, 4, tzinfo=UTC),
        ingested_at=ingested_at,
        correlation_id="integration-correlation",
        causation_id="integration-source-request",
        quality_status="Validated",
        payload={
            "timestamp": "2026-07-13T10:01:00Z",
            "quarter_hour": "2026-07-13T10:00:00Z",
            "resolution_code": "PT1M",
            "quality_status": "Validated",
            "ace_mw": -12.5,
            "system_imbalance_mw": value,
            "alpha_eur_mwh": None,
            "alpha_prime_eur_mwh": None,
            "marginal_incremental_price_eur_mwh": 120.13,
            "marginal_decremental_price_eur_mwh": 99.97,
            "imbalance_price_eur_mwh": 99.97,
        },
    )


async def migrated_client() -> AsyncClient:
    assert CLICKHOUSE_URL is not None
    admin = await clickhouse_connect.get_async_client(
        dsn=CLICKHOUSE_URL,
        username=CLICKHOUSE_ADMIN_USER,
        password=CLICKHOUSE_ADMIN_PASSWORD,
        database="default",
    )
    try:
        schema_path = Path(__file__).parents[2] / "infra" / "clickhouse" / "001_schema.sql"
        statements = [statement.strip() for statement in schema_path.read_text().split(";")]
        for statement in statements:
            if statement:
                await admin.command(statement)
        for table in ("raw_events", "imbalance_observations"):
            await admin.command(f"TRUNCATE TABLE {DATABASE}.{table}")
    finally:
        await admin.close()
    return await clickhouse_connect.get_async_client(
        dsn=CLICKHOUSE_URL,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=DATABASE,
    )


@pytest.mark.asyncio
async def test_duplicate_insert_is_canonical_and_preserves_raw_json() -> None:
    client = await migrated_client()
    repository = ClickHouseRepository(client, database=DATABASE)
    source = event()
    try:
        await repository.insert_event(source)
        await repository.insert_event(source)

        observations = await repository.fetch_imbalance_window(
            source.event_time,
            minutes=5,
        )
        raw = await client.query(
            f"""
            SELECT argMax(envelope_json, row_version), count()
            FROM {DATABASE}.raw_events
            WHERE event_id = {{event_id:String}}
            """,
            parameters={"event_id": source.event_id},
        )

        assert len(observations) == 1
        assert observations[0].system_imbalance_mw == 325.224
        raw_json, physical_count = raw.first_row
        assert json.loads(raw_json) == source.model_dump(mode="json")
        assert physical_count == 2
    finally:
        await repository.aclose()


@pytest.mark.asyncio
async def test_canonical_query_returns_newest_version_for_one_natural_key() -> None:
    client = await migrated_client()
    repository = ClickHouseRepository(client, database=DATABASE)
    older = event(event_id="correction-key", value=100.0)
    newer = event(
        event_id="correction-key",
        ingested_at=older.ingested_at + timedelta(seconds=1),
        value=-250.0,
    )
    try:
        await repository.insert_event(older)
        await repository.insert_event(newer)

        observations = await repository.fetch_imbalance_window(
            newer.event_time,
            minutes=5,
        )

        assert len(observations) == 1
        assert observations[0].timestamp == newer.event_time
        assert observations[0].system_imbalance_mw == -250.0
    finally:
        await repository.aclose()
