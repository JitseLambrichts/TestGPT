import json
import os
from datetime import UTC, datetime, timedelta

import clickhouse_connect
import pytest
from clickhouse_connect.driver.asyncclient import AsyncClient

from imbalance_pipeline.domain.events import EventEnvelope
from imbalance_pipeline.storage.clickhouse import ClickHouseRepository
from imbalance_pipeline.storage.migrate import apply_migrations

CLICKHOUSE_URL = os.getenv("IMBALANCE_CLICKHOUSE_URL")
CLICKHOUSE_USER = os.getenv("IMBALANCE_CLICKHOUSE_USER", "imbalance")
CLICKHOUSE_PASSWORD = os.getenv("IMBALANCE_CLICKHOUSE_PASSWORD", "imbalance")
CLICKHOUSE_ADMIN_USER = os.getenv("IMBALANCE_CLICKHOUSE_ADMIN_USER")
CLICKHOUSE_ADMIN_PASSWORD = os.getenv("IMBALANCE_CLICKHOUSE_ADMIN_PASSWORD")
DATABASE = os.getenv("IMBALANCE_CLICKHOUSE_DATABASE", "imbalance")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore:The current async client is a thread-pool wrapper"),
    pytest.mark.skipif(
        CLICKHOUSE_URL is None,
        reason="set IMBALANCE_CLICKHOUSE_URL to run the real ClickHouse integration tests",
    ),
    pytest.mark.skipif(
        CLICKHOUSE_URL is not None
        and (CLICKHOUSE_ADMIN_USER is None or CLICKHOUSE_ADMIN_PASSWORD is None),
        reason=(
            "set IMBALANCE_CLICKHOUSE_ADMIN_USER and "
            "IMBALANCE_CLICKHOUSE_ADMIN_PASSWORD for schema migrations"
        ),
    ),
]


def event(
    *,
    event_id: str = "integration-event-001",
    ingested_at: datetime = datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC),
    event_time: datetime = datetime(2026, 7, 13, 10, 1, tzinfo=UTC),
    value: float = 325.224,
) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        event_type="elia.imbalance.observed",
        schema_version="1",
        source="elia",
        dataset="ods161",
        event_time=event_time,
        observed_at=event_time + timedelta(seconds=4),
        ingested_at=ingested_at,
        correlation_id="integration-correlation",
        causation_id="integration-source-request",
        quality_status="Validated",
        payload={
            "timestamp": event_time.isoformat().replace("+00:00", "Z"),
            "quarter_hour": event_time.replace(
                minute=(event_time.minute // 15) * 15,
                second=0,
                microsecond=0,
            )
            .isoformat()
            .replace("+00:00", "Z"),
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


def prediction_event(
    *,
    event_id: str,
    target_time: datetime,
    generated_at: datetime | None = None,
    current_state: str | None = "positive",
) -> EventEnvelope:
    generated_at = generated_at or target_time - timedelta(minutes=1) + timedelta(seconds=10)
    return EventEnvelope(
        event_id=event_id,
        event_type="imbalance.prediction.generated",
        source="predictor",
        dataset="system-imbalance",
        event_time=generated_at,
        observed_at=generated_at,
        ingested_at=generated_at,
        correlation_id=event_id,
        quality_status="model",
        payload={
            "cutoff": (target_time - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
            "target_time": target_time.isoformat().replace("+00:00", "Z"),
            "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
            "system_imbalance_mw": 100.0,
            "p10_mw": 80.0,
            "p90_mw": 120.0,
            "flip_probability": 0.2,
            "will_flip": False,
            "current_state": current_state,
            "predicted_state": "positive",
            "prediction_quality": "model",
            "model_version": "model-v1",
            "feature_schema_hash": "feature-schema-001",
        },
    )


async def migrated_client() -> AsyncClient:
    assert CLICKHOUSE_URL is not None
    assert CLICKHOUSE_ADMIN_USER is not None
    assert CLICKHOUSE_ADMIN_PASSWORD is not None
    admin = await clickhouse_connect.get_async_client(
        dsn=CLICKHOUSE_URL,
        username=CLICKHOUSE_ADMIN_USER,
        password=CLICKHOUSE_ADMIN_PASSWORD,
        database="default",
    )
    try:
        await admin.command(f"DROP DATABASE IF EXISTS {DATABASE} SYNC")
        await apply_migrations(admin, database=DATABASE)
        grants = await admin.query("SHOW GRANTS FOR imbalance")
        assert {str(row[0]) for row in grants.result_rows} == {
            f"GRANT SELECT, INSERT ON {DATABASE}.* TO imbalance"
        }
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


async def optimize_table(table: str) -> None:
    assert CLICKHOUSE_URL is not None
    assert CLICKHOUSE_ADMIN_USER is not None
    assert CLICKHOUSE_ADMIN_PASSWORD is not None
    admin = await clickhouse_connect.get_async_client(
        dsn=CLICKHOUSE_URL,
        username=CLICKHOUSE_ADMIN_USER,
        password=CLICKHOUSE_ADMIN_PASSWORD,
        database=DATABASE,
    )
    try:
        await admin.command(f"OPTIMIZE TABLE {DATABASE}.{table} FINAL")
    finally:
        await admin.close()


async def rerun_migrations() -> None:
    assert CLICKHOUSE_URL is not None
    assert CLICKHOUSE_ADMIN_USER is not None
    assert CLICKHOUSE_ADMIN_PASSWORD is not None
    admin = await clickhouse_connect.get_async_client(
        dsn=CLICKHOUSE_URL,
        username=CLICKHOUSE_ADMIN_USER,
        password=CLICKHOUSE_ADMIN_PASSWORD,
        database="default",
    )
    try:
        await apply_migrations(admin, database=DATABASE)
    finally:
        await admin.close()


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
            knowledge_cutoff=source.ingested_at,
        )
        raw = await client.query(
            f"""
            SELECT argMax(envelope_json, row_version)
            FROM {DATABASE}.raw_events
            WHERE event_id = {{event_id:String}}
            """,
            parameters={"event_id": source.event_id},
        )

        assert len(observations) == 1
        assert observations[0].system_imbalance_mw == 325.224
        (raw_json,) = raw.first_row
        assert json.loads(raw_json) == source.model_dump(mode="json")
    finally:
        await repository.aclose()


@pytest.mark.asyncio
async def test_point_in_time_query_excludes_a_late_correction_until_it_was_known() -> None:
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

        before_correction = await repository.fetch_imbalance_window(
            older.event_time,
            minutes=5,
            knowledge_cutoff=older.ingested_at,
        )
        after_correction = await repository.fetch_imbalance_window(
            newer.event_time,
            minutes=5,
            knowledge_cutoff=newer.ingested_at,
        )

        assert len(before_correction) == 1
        assert before_correction[0].timestamp == older.event_time
        assert before_correction[0].system_imbalance_mw == 100.0
        assert len(after_correction) == 1
        assert after_correction[0].timestamp == newer.event_time
        assert after_correction[0].system_imbalance_mw == -250.0
    finally:
        await repository.aclose()


@pytest.mark.asyncio
async def test_state_seed_duration_starts_at_the_last_confirmed_sign_change() -> None:
    client = await migrated_client()
    repository = ClickHouseRepository(client, database=DATABASE)
    start = datetime(2026, 7, 13, 9, 0, tzinfo=UTC)
    observations = [
        event(
            event_id=f"positive-{offset}",
            event_time=start + timedelta(minutes=offset),
            ingested_at=start + timedelta(minutes=offset, seconds=5),
            value=20.0,
        )
        for offset in (0, 1, 60)
    ]
    try:
        for source in observations:
            await repository.insert_event(source)

        seed = await repository.fetch_imbalance_state_seed(
            observations[-1].event_time,
            knowledge_cutoff=observations[-1].ingested_at,
            deadband_mw=10.0,
        )

        assert seed.state is not None and seed.state.value == "positive"
        assert seed.state_since == start
        assert seed.last_observed_at == observations[-1].event_time
    finally:
        await repository.aclose()


@pytest.mark.asyncio
async def test_source_version_migration_is_safe_to_rerun() -> None:
    client = await migrated_client()
    repository = ClickHouseRepository(client, database=DATABASE)
    source = event(event_id="survives-managed-migration", value=123.0)
    try:
        await repository.insert_event(source)
        await rerun_migrations()
        created = await client.query(f"SHOW CREATE TABLE {DATABASE}.imbalance_observations")
        observations = await repository.fetch_imbalance_window(
            source.event_time,
            minutes=5,
            knowledge_cutoff=source.ingested_at,
        )

        assert "ORDER BY (timestamp, event_id, row_version)" in str(created.first_row[0])
        assert [observation.system_imbalance_mw for observation in observations] == [123.0]
    finally:
        await repository.aclose()


@pytest.mark.asyncio
async def test_predictions_for_a_realized_target_are_read_without_waiting_for_merges() -> None:
    client = await migrated_client()
    repository = ClickHouseRepository(client, database=DATABASE)
    target = datetime(2026, 7, 13, 10, 2, tzinfo=UTC)
    try:
        await repository.insert_event(prediction_event(event_id="prediction-a", target_time=target))
        await repository.insert_event(prediction_event(event_id="prediction-b", target_time=target))

        predictions = await repository.fetch_predictions_for_target(target)

        assert [prediction.event_id for prediction in predictions] == [
            "prediction-a",
            "prediction-b",
        ]
        assert all(prediction.target_time == target for prediction in predictions)
    finally:
        await repository.aclose()


@pytest.mark.asyncio
async def test_prediction_revision_reads_a_single_tuple_when_latest_state_is_null() -> None:
    client = await migrated_client()
    repository = ClickHouseRepository(client, database=DATABASE)
    target = datetime(2026, 7, 13, 10, 2, tzinfo=UTC)
    generated_at = target - timedelta(minutes=1) + timedelta(seconds=10)
    try:
        await repository.insert_event(
            prediction_event(
                event_id="prediction-revision",
                target_time=target,
                generated_at=generated_at,
            )
        )
        await repository.insert_event(
            prediction_event(
                event_id="prediction-revision",
                target_time=target,
                generated_at=generated_at + timedelta(seconds=1),
                current_state=None,
            )
        )

        predictions = await repository.fetch_predictions_for_target(target)

        assert len(predictions) == 1
        assert predictions[0].event_id == "prediction-revision"
        assert predictions[0].current_state is None
    finally:
        await repository.aclose()


@pytest.mark.asyncio
async def test_ineligible_prediction_revision_does_not_resurrect_an_older_revision() -> None:
    client = await migrated_client()
    repository = ClickHouseRepository(client, database=DATABASE)
    target = datetime(2026, 7, 13, 10, 2, tzinfo=UTC)
    generated_at = target - timedelta(minutes=1) + timedelta(seconds=10)
    try:
        await repository.insert_event(
            prediction_event(
                event_id="prediction-ineligible-revision",
                target_time=target,
                generated_at=generated_at,
            )
        )
        await repository.insert_event(
            prediction_event(
                event_id="prediction-ineligible-revision",
                target_time=target,
                generated_at=target + timedelta(seconds=1),
            )
        )

        predictions = await repository.fetch_predictions_for_target(target)

        assert predictions == []
    finally:
        await repository.aclose()


@pytest.mark.asyncio
async def test_versioned_reads_survive_clickhouse_merges_and_preserve_knowledge_cutoffs() -> None:
    client = await migrated_client()
    repository = ClickHouseRepository(client, database=DATABASE)
    original = event(event_id="history-key", value=100.0)
    correction = event(
        event_id="history-key",
        ingested_at=original.ingested_at + timedelta(seconds=1),
        value=-250.0,
    )
    try:
        await repository.insert_event(original)
        await repository.insert_event(correction)
        await optimize_table("imbalance_observations")

        versions = await repository.fetch_imbalance_versions(
            original.event_time,
            original.event_time,
            knowledge_cutoff=correction.ingested_at,
        )
        before = await repository.fetch_imbalance_state_seed(
            original.event_time,
            knowledge_cutoff=original.ingested_at,
            deadband_mw=10.0,
        )
        after = await repository.fetch_imbalance_state_seed(
            correction.event_time,
            knowledge_cutoff=correction.ingested_at,
            deadband_mw=10.0,
        )

        assert [row.observation.system_imbalance_mw for row in versions] == [100.0, -250.0]
        assert before.state is not None and before.state.value == "positive"
        assert after.state is not None and after.state.value == "negative"
    finally:
        await repository.aclose()
