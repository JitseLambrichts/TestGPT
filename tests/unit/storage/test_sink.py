import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from clickhouse_connect.driver.asyncclient import AsyncClient
from clickhouse_connect.driver.exceptions import OperationalError

from imbalance_pipeline.domain.events import EventEnvelope
from imbalance_pipeline.messaging.base import Message
from imbalance_pipeline.services.sink import (
    CLICKHOUSE_DLQ_SUBJECT,
    Sink,
)
from imbalance_pipeline.storage.clickhouse import (
    ClickHouseRepository,
    DeadLetterReason,
    PermanentEventError,
    Prediction,
    TransientStorageError,
)


def imbalance_event() -> EventEnvelope:
    return EventEnvelope(
        event_id="source-event-001",
        event_type="elia.imbalance.observed",
        schema_version="1",
        source="elia",
        dataset="ods161",
        event_time=datetime(2026, 7, 13, 10, 1, tzinfo=UTC),
        observed_at=datetime(2026, 7, 13, 10, 1, 4, tzinfo=UTC),
        ingested_at=datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC),
        correlation_id="source-correlation-001",
        causation_id="source-request-001",
        quality_status="Validated",
        payload={
            "timestamp": "2026-07-13T10:01:00Z",
            "quarter_hour": "2026-07-13T10:00:00Z",
            "resolution_code": "PT1M",
            "quality_status": "Validated",
            "ace_mw": -12.5,
            "system_imbalance_mw": 325.224,
            "alpha_eur_mwh": None,
            "alpha_prime_eur_mwh": None,
            "marginal_incremental_price_eur_mwh": 120.13,
            "marginal_decremental_price_eur_mwh": 99.97,
            "imbalance_price_eur_mwh": 99.97,
        },
    )


class RecordingRepository:
    def __init__(
        self,
        trace: list[tuple[str, object]],
        failure: Exception | None = None,
    ) -> None:
        self.trace = trace
        self.failure = failure

    async def insert_event(self, event: EventEnvelope) -> None:
        self.trace.append(("insert", event.event_id))
        if self.failure is not None:
            raise self.failure


class RecordingBus:
    def __init__(
        self,
        trace: list[tuple[str, object]],
        publish_failure: Exception | None = None,
    ) -> None:
        self.trace = trace
        self.publish_failure = publish_failure
        self.published: list[tuple[str, EventEnvelope]] = []

    async def publish(self, subject: str, event: EventEnvelope) -> None:
        self.trace.append(("publish", subject))
        if self.publish_failure is not None:
            raise self.publish_failure
        self.published.append((subject, event))

    async def messages(self, subject: str, durable: str) -> AsyncIterator[Message]:
        del subject, durable
        if False:
            yield RecordingMessage(imbalance_event(), self.trace)


class RecordingMessage:
    def __init__(
        self,
        event: EventEnvelope,
        trace: list[tuple[str, object]],
        *,
        delivery_count: int = 1,
    ) -> None:
        self.event = event
        self.delivery_count = delivery_count
        self.trace = trace
        self.acked = False
        self.nak_delay: float | None = None

    async def ack(self) -> None:
        self.trace.append(("ack", self.event.event_id))
        self.acked = True

    async def nak(self, delay_seconds: float) -> None:
        self.trace.append(("nak", delay_seconds))
        self.nak_delay = delay_seconds


class RecordingClickHouseClient:
    def __init__(self) -> None:
        self.inserts: list[tuple[str, list[list[object]], tuple[str, ...]]] = []
        self.queries: list[tuple[str, dict[str, object], dict[str, object]]] = []
        self.query_rows: list[tuple[object, ...]] = []
        self.insert_failure: Exception | None = None

    async def insert(
        self,
        table: str,
        data: list[list[object]],
        column_names: tuple[str, ...],
    ) -> None:
        if self.insert_failure is not None:
            raise self.insert_failure
        self.inserts.append((table, data, column_names))

    async def query(
        self,
        query: str,
        parameters: dict[str, object],
        **settings: object,
    ) -> SimpleNamespace:
        self.queries.append((query, parameters, settings))
        return SimpleNamespace(
            result_rows=self.query_rows,
            first_row=self.query_rows[0] if self.query_rows else None,
        )

    async def close(self) -> None:
        return None


def repository_with(client: RecordingClickHouseClient) -> ClickHouseRepository:
    return ClickHouseRepository(cast(AsyncClient, client), database="imbalance")


@dataclass(frozen=True)
class Snapshot:
    event_id: str
    cutoff: datetime
    target_time: datetime
    feature_schema_hash: str
    local_values: tuple[tuple[float, ...], ...]
    local_masks: tuple[tuple[int, ...], ...]
    context_values: tuple[tuple[float, ...], ...]
    context_masks: tuple[tuple[int, ...], ...]
    static_values: tuple[float, ...]
    static_masks: tuple[int, ...]
    current_state: str | None
    created_at: datetime


def snapshot() -> Snapshot:
    return Snapshot(
        event_id="feature-event-001",
        cutoff=datetime(2026, 7, 13, 10, 1, tzinfo=UTC),
        target_time=datetime(2026, 7, 13, 10, 2, tzinfo=UTC),
        feature_schema_hash="feature-schema-001",
        local_values=((1.0, 2.0), (3.0, 4.0)),
        local_masks=((1, 1), (1, 0)),
        context_values=((5.0, 6.0),),
        context_masks=((1, 0),),
        static_values=(7.0, 8.0),
        static_masks=(1, 1),
        current_state="positive",
        created_at=datetime(2026, 7, 13, 10, 1, 6, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_feature_snapshot_structural_input_is_validated_and_flattened() -> None:
    client = RecordingClickHouseClient()

    await repository_with(client).insert_feature_snapshot(snapshot())

    table, rows, columns = client.inserts[0]
    inserted = dict(zip(columns, rows[0], strict=True))
    assert table == "imbalance.feature_snapshots"
    assert inserted["local_values"] == [1.0, 2.0, 3.0, 4.0]
    assert inserted["local_masks"] == [1, 1, 1, 0]
    assert inserted["context_values"] == [5.0, 6.0]
    assert inserted["static_masks"] == [1, 1]
    assert inserted["created_at"] == snapshot().created_at


@pytest.mark.asyncio
async def test_insert_feature_snapshot_rejects_mismatched_masks_before_insert() -> None:
    client = RecordingClickHouseClient()
    invalid = replace(snapshot(), local_masks=((1,),))

    with pytest.raises(PermanentEventError) as raised:
        await repository_with(client).insert_feature_snapshot(invalid)

    assert raised.value.reason is DeadLetterReason.INVALID_PAYLOAD
    assert client.inserts == []


@pytest.mark.asyncio
async def test_latest_prediction_returns_typed_latest_row_without_final() -> None:
    client = RecordingClickHouseClient()
    client.query_rows = [
        (
            "prediction-event-001",
            datetime(2026, 7, 13, 10, 1),
            datetime(2026, 7, 13, 10, 2),
            datetime(2026, 7, 13, 10, 1, 7),
            120.0,
            80.0,
            160.0,
            0.25,
            0,
            "positive",
            "positive",
            "model",
            "model-v1",
            "feature-schema-001",
        )
    ]

    latest = await repository_with(client).latest_prediction()

    assert isinstance(latest, Prediction)
    assert latest.target_time == datetime(2026, 7, 13, 10, 2, tzinfo=UTC)
    assert latest.will_flip is False
    query, parameters, settings = client.queries[0]
    assert "FINAL" not in query.upper()
    assert "ORDER BY target_time DESC, row_version DESC" in query
    assert parameters == {}
    assert settings["tz_mode"] == "aware"


@pytest.mark.asyncio
async def test_latest_prediction_returns_none_for_an_empty_table() -> None:
    client = RecordingClickHouseClient()

    assert await repository_with(client).latest_prediction() is None


@pytest.mark.asyncio
async def test_repository_validates_then_inserts_exact_raw_envelope_before_normalized_row() -> None:
    client = RecordingClickHouseClient()
    repository = repository_with(client)
    event = imbalance_event()

    await repository.insert_event(event)

    assert [call[0] for call in client.inserts] == [
        "imbalance.raw_events",
        "imbalance.imbalance_observations",
    ]
    _, raw_rows, raw_columns = client.inserts[0]
    raw = dict(zip(raw_columns, raw_rows[0], strict=True))
    _, normalized_rows, normalized_columns = client.inserts[1]
    normalized = dict(zip(normalized_columns, normalized_rows[0], strict=True))
    assert json.loads(cast(str, raw["payload_json"])) == event.payload
    assert json.loads(cast(str, raw["envelope_json"])) == event.model_dump(mode="json")
    assert raw["event_time"] == event.event_time
    assert raw["observed_at"] == event.observed_at
    assert raw["ingested_at"] == event.ingested_at
    assert raw["row_version"] == normalized["row_version"]
    assert normalized["event_id"] == event.event_id
    assert normalized["timestamp"] == event.event_time
    assert normalized["system_imbalance_mw"] == 325.224


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "reason"),
    [
        (
            imbalance_event().model_copy(update={"schema_version": "2"}),
            DeadLetterReason.UNSUPPORTED_SCHEMA_VERSION,
        ),
        (
            imbalance_event().model_copy(update={"event_type": "unknown.event"}),
            DeadLetterReason.UNSUPPORTED_EVENT_TYPE,
        ),
        (
            imbalance_event().model_copy(update={"payload": {"system_imbalance_mw": "bad"}}),
            DeadLetterReason.INVALID_PAYLOAD,
        ),
    ],
)
async def test_repository_rejects_permanent_contract_errors_before_any_insert(
    event: EventEnvelope,
    reason: DeadLetterReason,
) -> None:
    client = RecordingClickHouseClient()

    with pytest.raises(PermanentEventError) as raised:
        await repository_with(client).insert_event(event)

    assert raised.value.reason is reason
    assert client.inserts == []


@pytest.mark.asyncio
async def test_repository_sanitizes_clickhouse_errors_as_transient() -> None:
    client = RecordingClickHouseClient()
    secret = "clickhouse password=do-not-expose"
    client.insert_failure = OperationalError(secret)

    with pytest.raises(TransientStorageError) as raised:
        await repository_with(client).insert_event(imbalance_event())

    assert secret not in str(raised.value)


@pytest.mark.asyncio
async def test_repository_rejects_non_json_numbers_before_any_insert() -> None:
    client = RecordingClickHouseClient()
    payload = {**imbalance_event().payload, "system_imbalance_mw": float("nan")}
    invalid = imbalance_event().model_copy(update={"payload": payload})

    with pytest.raises(PermanentEventError) as raised:
        await repository_with(client).insert_event(invalid)

    assert raised.value.reason is DeadLetterReason.INVALID_PAYLOAD
    assert client.inserts == []


@pytest.mark.asyncio
async def test_fetch_imbalance_window_uses_atomic_latest_aggregation_without_final() -> None:
    client = RecordingClickHouseClient()
    client.query_rows = [
        (
            datetime(2026, 7, 13, 10, 1),
            (
                datetime(2026, 7, 13, 10, 0),
                "PT1M",
                "Validated",
                -12.5,
                325.224,
                None,
                None,
                120.13,
                99.97,
                99.97,
            ),
        )
    ]
    repository = repository_with(client)
    cutoff = datetime(2026, 7, 13, 10, 1, tzinfo=UTC)

    observations = await repository.fetch_imbalance_window(cutoff, 180)

    assert len(observations) == 1
    assert observations[0].timestamp == cutoff
    assert observations[0].system_imbalance_mw == 325.224
    query, parameters, settings = client.queries[0]
    assert "argMax(tuple(" in query
    assert "FINAL" not in query.upper()
    assert parameters == {"start": cutoff - timedelta(minutes=180), "cutoff": cutoff}
    assert settings["tz_mode"] == "aware"


@pytest.mark.asyncio
async def test_fetch_imbalance_window_rejects_invalid_window_before_query() -> None:
    client = RecordingClickHouseClient()
    repository = repository_with(client)

    with pytest.raises(ValueError, match="minutes must be positive"):
        await repository.fetch_imbalance_window(datetime.now(UTC), 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        await repository.fetch_imbalance_window(datetime(2026, 7, 13), 5)

    assert client.queries == []


def test_clickhouse_schema_covers_all_tables_utc_versions_partitions_ttl_and_grant() -> None:
    schema_path = Path(__file__).parents[3] / "infra" / "clickhouse" / "001_schema.sql"
    schema = schema_path.read_text(encoding="utf-8")

    for table in (
        "schema_migrations",
        "raw_events",
        "imbalance_observations",
        "load_observations",
        "wind_observations",
        "solar_observations",
        "feature_snapshots",
        "predictions",
        "prediction_outcomes",
        "model_versions",
    ):
        assert f"CREATE TABLE IF NOT EXISTS imbalance.{table}" in schema
    assert schema.count("DateTime64(3, 'UTC')") >= 20
    assert schema.count("ReplacingMergeTree(row_version)") >= 7
    assert schema.count("PARTITION BY toYYYYMM(") >= 8
    assert "TTL toDateTime(event_time, 'UTC') + INTERVAL 90 DAY DELETE" in schema
    assert "GRANT SELECT, INSERT ON imbalance.* TO imbalance" in schema


@pytest.mark.asyncio
async def test_sink_inserts_then_publishes_deterministic_stored_trigger_then_acks() -> None:
    trace: list[tuple[str, object]] = []
    source = imbalance_event()
    repository = RecordingRepository(trace)
    bus = RecordingBus(trace)
    message = RecordingMessage(source, trace)

    await Sink(repository, bus).handle(message)

    assert trace == [
        ("insert", source.event_id),
        ("publish", "grid.stored.elia.imbalance.v1"),
        ("ack", source.event_id),
    ]
    subject, stored = bus.published[0]
    assert subject == "grid.stored.elia.imbalance.v1"
    assert stored.correlation_id == source.correlation_id
    assert stored.causation_id == source.event_id
    assert stored.payload == {
        "source_event_id": source.event_id,
        "timestamp": "2026-07-13T10:01:00Z",
    }

    replay = RecordingMessage(source, trace)
    await Sink(repository, bus).handle(replay)

    assert bus.published[1][1].event_id == stored.event_id
    assert bus.published[1][1] == stored
    assert replay.acked is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delivery_count", "expected_delay"),
    [(1, 1.0), (2, 5.0), (3, 30.0), (4, 120.0)],
)
async def test_transient_insert_failure_naks_with_delivery_delay_without_publishing(
    delivery_count: int,
    expected_delay: float,
) -> None:
    trace: list[tuple[str, object]] = []
    repository = RecordingRepository(trace, TransientStorageError("safe storage failure"))
    bus = RecordingBus(trace)
    message = RecordingMessage(imbalance_event(), trace, delivery_count=delivery_count)

    await Sink(repository, bus).handle(message)

    assert trace == [("insert", "source-event-001"), ("nak", expected_delay)]
    assert bus.published == []
    assert message.acked is False
    assert message.nak_delay == expected_delay


@pytest.mark.asyncio
async def test_fifth_transient_delivery_publishes_safe_dlq_then_acks() -> None:
    trace: list[tuple[str, object]] = []
    secret = "password=do-not-publish"
    repository = RecordingRepository(trace, TransientStorageError(secret))
    bus = RecordingBus(trace)
    message = RecordingMessage(imbalance_event(), trace, delivery_count=5)

    await Sink(repository, bus).handle(message)

    assert trace == [
        ("insert", "source-event-001"),
        ("publish", CLICKHOUSE_DLQ_SUBJECT),
        ("ack", "source-event-001"),
    ]
    dlq = bus.published[0][1]
    assert dlq.payload["reason"] == DeadLetterReason.DELIVERY_EXHAUSTED
    assert dlq.payload["delivery_count"] == 5
    assert secret not in dlq.model_dump_json()
    assert message.nak_delay is None


@pytest.mark.asyncio
async def test_permanent_schema_error_publishes_typed_safe_dlq_before_ack() -> None:
    trace: list[tuple[str, object]] = []
    secret = "unexpected payload contained token=secret"
    repository = RecordingRepository(
        trace,
        PermanentEventError(DeadLetterReason.INVALID_PAYLOAD, secret),
    )
    bus = RecordingBus(trace)
    message = RecordingMessage(imbalance_event(), trace)

    await Sink(repository, bus).handle(message)

    assert trace == [
        ("insert", "source-event-001"),
        ("publish", CLICKHOUSE_DLQ_SUBJECT),
        ("ack", "source-event-001"),
    ]
    dlq = bus.published[0][1]
    assert dlq.event_type == "clickhouse.event.rejected"
    assert dlq.correlation_id == message.event.correlation_id
    assert dlq.causation_id == message.event.event_id
    assert dlq.payload == {
        "source_event_id": message.event.event_id,
        "event_type": message.event.event_type,
        "schema_version": message.event.schema_version,
        "source": message.event.source,
        "dataset": message.event.dataset,
        "reason": DeadLetterReason.INVALID_PAYLOAD,
        "delivery_count": 1,
    }
    assert secret not in dlq.model_dump_json()


@pytest.mark.asyncio
async def test_publish_failure_leaves_inserted_source_unacked_for_safe_replay() -> None:
    trace: list[tuple[str, object]] = []
    repository = RecordingRepository(trace)
    bus = RecordingBus(trace, RuntimeError("broker unavailable"))
    message = RecordingMessage(imbalance_event(), trace)

    with pytest.raises(RuntimeError, match="broker unavailable"):
        await Sink(repository, bus).handle(message)

    assert trace == [
        ("insert", "source-event-001"),
        ("publish", "grid.stored.elia.imbalance.v1"),
    ]
    assert message.acked is False
    assert message.nak_delay is None


@pytest.mark.asyncio
async def test_dlq_publish_failure_does_not_ack_the_rejected_message() -> None:
    trace: list[tuple[str, object]] = []
    repository = RecordingRepository(
        trace,
        PermanentEventError(DeadLetterReason.UNSUPPORTED_SCHEMA_VERSION),
    )
    bus = RecordingBus(trace, RuntimeError("broker unavailable"))
    message = RecordingMessage(imbalance_event(), trace)

    with pytest.raises(RuntimeError, match="broker unavailable"):
        await Sink(repository, bus).handle(message)

    assert message.acked is False
    assert message.nak_delay is None
