import asyncio
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
from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.messaging.base import Message
from imbalance_pipeline.services.sink import (
    CLICKHOUSE_DLQ_SUBJECT,
    DeadLetterPayload,
    Sink,
)
from imbalance_pipeline.storage.clickhouse import (
    ClickHouseRepository,
    DeadLetterReason,
    PermanentEventError,
    Prediction,
    TransientStorageError,
)
from imbalance_pipeline.storage.migrations import SOURCE_VERSION_MIGRATIONS


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


def prediction_source_event() -> EventEnvelope:
    generated_at = datetime(2026, 7, 13, 10, 1, 7, tzinfo=UTC)
    target_time = generated_at + timedelta(seconds=53)
    return EventEnvelope(
        event_id="prediction-source-001",
        event_type="imbalance.prediction.generated",
        schema_version="1",
        source="predictor",
        dataset="system-imbalance",
        event_time=generated_at,
        observed_at=generated_at,
        ingested_at=generated_at,
        correlation_id="feature-event-001",
        causation_id="feature-event-001",
        quality_status="model",
        payload={
            "cutoff": "2026-07-13T10:01:00Z",
            "target_time": target_time.isoformat().replace("+00:00", "Z"),
            "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
            "system_imbalance_mw": 100.0,
            "p10_mw": 80.0,
            "p90_mw": 120.0,
            "flip_probability": 0.2,
            "will_flip": False,
            "current_state": "positive",
            "predicted_state": "positive",
            "prediction_quality": "model",
            "model_version": "model-v1",
            "feature_schema_hash": "feature-schema-001",
        },
    )


def routed_events() -> list[tuple[str, dict[str, object], str | None]]:
    timestamp = "2026-07-13T10:01:00Z"
    return [
        (
            "elia.load.observed",
            {
                "timestamp": timestamp,
                "resolution_code": "PT15M",
                "measured_mw": 9_100.0,
                "most_recent_forecast_mw": 9_050.0,
                "most_recent_confidence_10_mw": 8_900.0,
                "most_recent_confidence_90_mw": 9_200.0,
                "day_ahead_forecast_mw": 9_000.0,
                "day_ahead_confidence_10_mw": 8_800.0,
                "day_ahead_confidence_90_mw": 9_250.0,
                "week_ahead_forecast_mw": 8_950.0,
            },
            "load_observations",
        ),
        (
            "elia.wind.observed",
            {
                "timestamp": timestamp,
                "resolution_code": "PT15M",
                "offshore_onshore": "Offshore",
                "region": "Belgium",
                "grid_connection_type": "Grid connected",
                "real_time_mw": 1_500.0,
                "most_recent_forecast_mw": 1_450.0,
                "most_recent_confidence_10_mw": 1_300.0,
                "most_recent_confidence_90_mw": 1_600.0,
                "day_ahead_11h_forecast_mw": 1_400.0,
                "day_ahead_11h_confidence_10_mw": 1_200.0,
                "day_ahead_11h_confidence_90_mw": 1_650.0,
                "day_ahead_forecast_mw": 1_420.0,
                "day_ahead_confidence_10_mw": 1_250.0,
                "day_ahead_confidence_90_mw": 1_620.0,
                "week_ahead_forecast_mw": 1_350.0,
                "week_ahead_confidence_10_mw": 1_100.0,
                "week_ahead_confidence_90_mw": 1_700.0,
                "monitored_capacity_mw": 2_300.0,
                "load_factor": 0.65,
                "decremental_bid_id": 42,
            },
            "wind_observations",
        ),
        (
            "elia.solar.observed",
            {
                "timestamp": timestamp,
                "resolution_code": "PT15M",
                "region": "Belgium",
                "real_time_mw": 2_100.0,
                "most_recent_forecast_mw": 2_000.0,
                "most_recent_confidence_10_mw": 1_800.0,
                "most_recent_confidence_90_mw": 2_200.0,
                "day_ahead_11h_forecast_mw": 1_950.0,
                "day_ahead_11h_confidence_10_mw": 1_700.0,
                "day_ahead_11h_confidence_90_mw": 2_250.0,
                "day_ahead_forecast_mw": 1_980.0,
                "day_ahead_confidence_10_mw": 1_750.0,
                "day_ahead_confidence_90_mw": 2_230.0,
                "week_ahead_forecast_mw": 1_900.0,
                "week_ahead_confidence_10_mw": 1_600.0,
                "week_ahead_confidence_90_mw": 2_300.0,
                "load_factor": 0.42,
                "monitored_capacity_mw": 5_000.0,
            },
            "solar_observations",
        ),
        (
            "weather.forecast.observed",
            {
                "valid_time": timestamp,
                "available_at": "2026-07-13T10:01:04Z",
                "locations": ["Brussels"],
                "variables": {},
            },
            None,
        ),
        (
            "imbalance.feature.snapshot",
            {
                "cutoff": timestamp,
                "target_time": "2026-07-13T10:02:00Z",
                "feature_schema_hash": "feature-schema-001",
                "local_values": [[1.0, 2.0]],
                "local_masks": [[1, 1]],
                "context_values": [[3.0]],
                "context_masks": [[1]],
                "static_values": [4.0],
                "static_masks": [1],
                "current_state": "positive",
                "created_at": "2026-07-13T10:01:06Z",
            },
            "feature_snapshots",
        ),
        (
            "imbalance.prediction.generated",
            {
                "cutoff": timestamp,
                "target_time": "2026-07-13T10:02:00Z",
                "generated_at": "2026-07-13T10:01:07Z",
                "system_imbalance_mw": 120.0,
                "p10_mw": 80.0,
                "p90_mw": 160.0,
                "flip_probability": 0.25,
                "will_flip": False,
                "current_state": "positive",
                "predicted_state": "positive",
                "prediction_quality": "model",
                "model_version": "model-v1",
                "feature_schema_hash": "feature-schema-001",
            },
            "predictions",
        ),
        (
            "imbalance.prediction.evaluated",
            {
                "prediction_event_id": "prediction-event-001",
                "target_time": "2026-07-13T10:02:00Z",
                "realized_event_id": "realized-event-001",
                "realized_system_imbalance_mw": -90.0,
                "realized_state": "negative",
                "flip_actual": True,
                "evaluated_at": "2026-07-13T10:02:06Z",
            },
            "prediction_outcomes",
        ),
        (
            "model.version.promoted",
            {
                "model_version": "model-v1",
                "feature_schema_hash": "feature-schema-001",
                "manifest_json": {"format": "onnx"},
                "metrics_json": {"mae": 12.0},
                "promoted_at": "2026-07-13T10:01:08Z",
                "created_at": "2026-07-13T10:01:07Z",
            },
            "model_versions",
        ),
    ]


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


class SequencedBus(RecordingBus):
    def __init__(
        self,
        trace: list[tuple[str, object]],
        messages: list["RecordingMessage"],
    ) -> None:
        super().__init__(trace)
        self._messages = messages
        self._publish_failures = 1

    async def publish(self, subject: str, event: EventEnvelope) -> None:
        self.trace.append(("publish", subject))
        if self._publish_failures:
            self._publish_failures -= 1
            raise RuntimeError("broker unavailable")
        self.published.append((subject, event))

    async def messages(self, subject: str, durable: str) -> AsyncIterator[Message]:
        del durable
        if subject == "grid.raw.elia.imbalance.v1":
            for message in self._messages:
                yield message


class RecordingMessage:
    def __init__(
        self,
        event: EventEnvelope,
        trace: list[tuple[str, object]],
        *,
        delivery_count: int = 1,
        ack_failure: Exception | None = None,
        nak_failure: Exception | None = None,
    ) -> None:
        self.event = event
        self.delivery_count = delivery_count
        self.trace = trace
        self.acked = False
        self.nak_delay: float | None = None
        self.ack_failure = ack_failure
        self.nak_failure = nak_failure

    async def ack(self) -> None:
        self.trace.append(("ack", self.event.event_id))
        if self.ack_failure is not None:
            raise self.ack_failure
        self.acked = True

    async def nak(self, delay_seconds: float) -> None:
        self.trace.append(("nak", delay_seconds))
        if self.nak_failure is not None:
            raise self.nak_failure
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
async def test_fetch_predictions_for_target_returns_each_canonical_prediction() -> None:
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
        ),
        (
            "prediction-event-002",
            datetime(2026, 7, 13, 10, 1),
            datetime(2026, 7, 13, 10, 2),
            datetime(2026, 7, 13, 10, 1, 8),
            -120.0,
            -160.0,
            -80.0,
            0.75,
            1,
            "negative",
            "positive",
            "model",
            "model-v2",
            "feature-schema-001",
        ),
    ]
    target = datetime(2026, 7, 13, 10, 2, tzinfo=UTC)

    predictions = await repository_with(client).fetch_predictions_for_target(target)

    assert [prediction.event_id for prediction in predictions] == [
        "prediction-event-001",
        "prediction-event-002",
    ]
    assert predictions[1].will_flip is True
    query, parameters, settings = client.queries[0]
    assert "GROUP BY event_id" in query
    assert "argMax(tuple(" in "".join(query.split())
    assert "FINAL" not in query.upper()
    assert parameters == {"target_time": target}
    assert settings["tz_mode"] == "aware"


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
@pytest.mark.parametrize(("event_type", "payload", "normalized_table"), routed_events())
async def test_repository_routes_every_supported_event_type(
    event_type: str,
    payload: dict[str, object],
    normalized_table: str | None,
) -> None:
    client = RecordingClickHouseClient()
    source = imbalance_event().model_copy(
        update={
            "event_id": f"routed-{event_type}",
            "event_type": event_type,
            "payload": payload,
        }
    )

    await repository_with(client).insert_event(source)

    expected = ["imbalance.raw_events"]
    if normalized_table is not None:
        expected.append(f"imbalance.{normalized_table}")
    assert [insert[0] for insert in client.inserts] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "payload_update"),
    [
        ("imbalance.feature.snapshot", {"local_masks": [[1]]}),
        ("imbalance.prediction.generated", {"p10_mw": 200.0, "p90_mw": 100.0}),
        ("model.version.promoted", {"manifest_json": "not-json"}),
    ],
)
async def test_repository_rejects_invalid_future_payloads_before_raw_insert(
    event_type: str,
    payload_update: dict[str, object],
) -> None:
    client = RecordingClickHouseClient()
    _, valid_payload, _ = next(item for item in routed_events() if item[0] == event_type)
    source = imbalance_event().model_copy(
        update={
            "event_type": event_type,
            "payload": {**valid_payload, **payload_update},
        }
    )

    with pytest.raises(PermanentEventError) as raised:
        await repository_with(client).insert_event(source)

    assert raised.value.reason is DeadLetterReason.INVALID_PAYLOAD
    assert client.inserts == []


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
    event_cutoff = datetime(2026, 7, 13, 10, 1, tzinfo=UTC)
    knowledge_cutoff = datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC)

    observations = await repository.fetch_imbalance_window(
        event_cutoff,
        180,
        knowledge_cutoff=knowledge_cutoff,
    )

    assert len(observations) == 1
    assert observations[0].timestamp == event_cutoff
    assert observations[0].system_imbalance_mw == 325.224
    query, parameters, settings = client.queries[0]
    assert "argMax(tuple(" in query
    assert "FINAL" not in query.upper()
    assert "ingested_at <= {knowledge_cutoff:DateTime64(3, 'UTC')}" in query
    assert parameters == {
        "start": event_cutoff - timedelta(minutes=180),
        "event_cutoff": event_cutoff,
        "knowledge_cutoff": knowledge_cutoff,
    }
    assert settings["tz_mode"] == "aware"


@pytest.mark.asyncio
async def test_fetch_imbalance_window_rejects_invalid_window_before_query() -> None:
    client = RecordingClickHouseClient()
    repository = repository_with(client)

    with pytest.raises(ValueError, match="minutes must be positive"):
        await repository.fetch_imbalance_window(
            datetime.now(UTC),
            0,
            knowledge_cutoff=datetime.now(UTC),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        await repository.fetch_imbalance_window(
            datetime(2026, 7, 13),
            5,
            knowledge_cutoff=datetime.now(UTC),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        await repository.fetch_imbalance_window(
            datetime.now(UTC),
            5,
            knowledge_cutoff=datetime(2026, 7, 13),
        )

    assert client.queries == []


@pytest.mark.asyncio
async def test_fetch_imbalance_versions_keeps_all_known_corrections_for_pit_reads() -> None:
    client = RecordingClickHouseClient()
    timestamp = datetime(2026, 7, 13, 10, 1)
    client.query_rows = [
        (
            "original",
            timestamp,
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
            datetime(2026, 7, 13, 10, 1, 5),
            1,
        ),
        (
            "correction",
            timestamp,
            datetime(2026, 7, 13, 10, 0),
            "PT1M",
            "Validated",
            -11.0,
            300.0,
            None,
            None,
            120.13,
            99.97,
            99.97,
            datetime(2026, 7, 13, 10, 1, 7),
            2,
        ),
    ]
    repository = repository_with(client)
    start = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    end = datetime(2026, 7, 13, 10, 2, tzinfo=UTC)
    known_at = datetime(2026, 7, 13, 10, 1, 10, tzinfo=UTC)

    versions = await repository.fetch_imbalance_versions(
        start,
        end,
        knowledge_cutoff=known_at,
    )

    assert [row.event_id for row in versions] == ["original", "correction"]
    assert [row.row_version for row in versions] == [1, 2]
    query, parameters, settings = client.queries[0]
    assert "FINAL" not in query.upper()
    assert "row_version" in query
    assert "ingested_at <= {knowledge_cutoff:DateTime64(3, 'UTC')}" in query
    assert parameters == {"start": start, "end": end, "knowledge_cutoff": known_at}
    assert settings["tz_mode"] == "aware"


@pytest.mark.asyncio
async def test_fetch_imbalance_state_seed_is_point_in_time_safe() -> None:
    client = RecordingClickHouseClient()
    client.query_rows = [
        (0, 25.0, datetime(2026, 7, 13, 9, 1), datetime(2026, 7, 13, 10, 0)),
    ]
    repository = repository_with(client)
    before = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    known_at = datetime(2026, 7, 13, 11, 0, 5, tzinfo=UTC)

    seed = await repository.fetch_imbalance_state_seed(
        before,
        knowledge_cutoff=known_at,
        deadband_mw=10.0,
    )

    assert seed.state is ConfirmedState.POSITIVE
    assert seed.state_since == datetime(2026, 7, 13, 9, 1, tzinfo=UTC)
    assert seed.last_observed_at == datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    assert len(client.queries) == 1
    query, parameters, settings = client.queries[0]
    assert "WITH requests AS" in query
    assert "argMaxIf" in query
    assert "lagInFrame" in query
    assert "FINAL" not in query.upper()
    assert parameters["request_0_before"] == before
    assert parameters["deadband_mw"] == 10.0
    assert settings["tz_mode"] == "aware"


def test_clickhouse_schema_covers_all_tables_utc_versions_partitions_ttl_and_grant() -> None:
    schema_path = Path(__file__).parents[3] / "infra" / "clickhouse" / "001_schema.sql"
    schema = schema_path.read_text(encoding="utf-8")
    version_retention = (
        schema_path.parent / "002_preserve_source_versions.sql"
    ).read_text(encoding="utf-8")

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
    assert "migrate_source_version_retention" in version_retention
    assert {definition.table for definition in SOURCE_VERSION_MIGRATIONS} == {
        "raw_events",
        "imbalance_observations",
        "load_observations",
        "wind_observations",
        "solar_observations",
    }
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
        "source_ingested_at": "2026-07-13T10:01:05Z",
    }

    replay = RecordingMessage(source, trace)
    await Sink(repository, bus).handle(replay)

    assert bus.published[1][1].event_id == stored.event_id
    assert bus.published[1][1] == stored
    assert replay.acked is True

    correction = source.model_copy(
        update={"ingested_at": source.ingested_at + timedelta(seconds=1)}
    )
    await Sink(repository, bus).handle(RecordingMessage(correction, trace))

    corrected = bus.published[2][1]
    assert corrected.event_id != stored.event_id
    assert corrected.payload["source_ingested_at"] == "2026-07-13T10:01:06Z"


@pytest.mark.asyncio
async def test_sink_publishes_a_stored_prediction_reconciliation_trigger() -> None:
    trace: list[tuple[str, object]] = []
    source = prediction_source_event()
    bus = RecordingBus(trace)

    await Sink(RecordingRepository(trace), bus).handle(RecordingMessage(source, trace))

    assert trace == [
        ("insert", source.event_id),
        ("publish", "grid.stored.prediction.imbalance.v1"),
        ("ack", source.event_id),
    ]
    subject, stored = bus.published[0]
    assert subject == "grid.stored.prediction.imbalance.v1"
    assert stored.event_type == "imbalance.prediction.stored"
    assert stored.event_time == datetime(2026, 7, 13, 10, 2, tzinfo=UTC)
    assert stored.payload["prediction_event_id"] == source.event_id


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
    payload = DeadLetterPayload.model_validate(dlq.payload)
    assert payload.reason is DeadLetterReason.DELIVERY_EXHAUSTED
    assert payload.delivery_count == 5
    assert payload.original_event == message.event
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
    payload = DeadLetterPayload.model_validate(dlq.payload)
    assert payload.reason is DeadLetterReason.INVALID_PAYLOAD
    assert payload.delivery_count == 1
    assert payload.original_event == message.event
    replay = DeadLetterPayload.model_validate_json(payload.model_dump_json()).original_event
    assert replay == message.event
    assert replay.model_dump_json() == message.event.model_dump_json()
    assert secret not in dlq.model_dump_json()


@pytest.mark.asyncio
async def test_stored_publish_failure_naks_inserted_source_for_safe_replay() -> None:
    trace: list[tuple[str, object]] = []
    repository = RecordingRepository(trace)
    bus = RecordingBus(trace, RuntimeError("broker unavailable"))
    message = RecordingMessage(imbalance_event(), trace)

    await Sink(repository, bus).handle(message)

    assert trace == [
        ("insert", "source-event-001"),
        ("publish", "grid.stored.elia.imbalance.v1"),
        ("nak", 1.0),
    ]
    assert message.acked is False
    assert message.nak_delay == 1.0


@pytest.mark.asyncio
async def test_dlq_publish_failure_naks_the_rejected_message() -> None:
    trace: list[tuple[str, object]] = []
    repository = RecordingRepository(
        trace,
        PermanentEventError(DeadLetterReason.UNSUPPORTED_SCHEMA_VERSION),
    )
    bus = RecordingBus(trace, RuntimeError("broker unavailable"))
    message = RecordingMessage(imbalance_event(), trace)

    await Sink(repository, bus).handle(message)

    assert message.acked is False
    assert message.nak_delay == 1.0


@pytest.mark.asyncio
async def test_ack_failure_naks_the_source_message() -> None:
    trace: list[tuple[str, object]] = []
    repository = RecordingRepository(trace)
    bus = RecordingBus(trace)
    message = RecordingMessage(
        imbalance_event(),
        trace,
        ack_failure=RuntimeError("ack unavailable"),
    )

    await Sink(repository, bus).handle(message)

    assert trace == [
        ("insert", "source-event-001"),
        ("publish", "grid.stored.elia.imbalance.v1"),
        ("ack", "source-event-001"),
        ("nak", 1.0),
    ]
    assert message.acked is False
    assert message.nak_delay == 1.0


@pytest.mark.asyncio
async def test_nak_failure_is_sanitized_and_does_not_ack() -> None:
    trace: list[tuple[str, object]] = []
    secret = "transport token=do-not-expose"
    repository = RecordingRepository(trace)
    bus = RecordingBus(trace, RuntimeError("broker unavailable"))
    message = RecordingMessage(
        imbalance_event(),
        trace,
        nak_failure=RuntimeError(secret),
    )

    with pytest.raises(RuntimeError, match="failed to request message redelivery") as raised:
        await Sink(repository, bus).handle(message)

    assert secret not in str(raised.value)
    assert message.acked is False
    assert trace[-1] == ("nak", 1.0)


@pytest.mark.asyncio
async def test_one_publish_and_settlement_failure_does_not_cancel_sibling_consumers() -> None:
    trace: list[tuple[str, object]] = []
    failed = RecordingMessage(
        imbalance_event(),
        trace,
        nak_failure=RuntimeError("nak unavailable"),
    )
    succeeding_event = imbalance_event().model_copy(update={"event_id": "source-event-002"})
    succeeded = RecordingMessage(succeeding_event, trace)
    bus = SequencedBus(trace, [failed, succeeded])

    await Sink(RecordingRepository(trace), bus).run()

    assert failed.acked is False
    assert succeeded.acked is True
    assert len(bus.published) == 1
    subject, stored_event = bus.published[0]
    assert subject == "grid.stored.elia.imbalance.v1"
    assert stored_event.causation_id == succeeding_event.event_id


@pytest.mark.asyncio
async def test_cancellation_during_publish_propagates_without_nak() -> None:
    trace: list[tuple[str, object]] = []
    repository = RecordingRepository(trace)
    bus = RecordingBus(trace, asyncio.CancelledError())
    message = RecordingMessage(imbalance_event(), trace)

    with pytest.raises(asyncio.CancelledError):
        await Sink(repository, bus).handle(message)

    assert not any(call[0] == "nak" for call in trace)
