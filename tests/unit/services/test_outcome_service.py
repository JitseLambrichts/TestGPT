from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from imbalance_pipeline.domain.events import EventEnvelope, Subject
from imbalance_pipeline.domain.imbalance import ImbalanceObservation
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.services.outcomes import (
    MAX_PENDING_PREDICTION_DELIVERIES,
    OUTCOME_DLQ_SUBJECT,
    OUTCOME_SUBSCRIPTION,
    PREDICTION_SETTLEMENT_GRACE,
    OutcomeService,
)
from imbalance_pipeline.storage.clickhouse import Prediction

NOW = datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC)


class RecordingBus:
    def __init__(self, trace: list[tuple[str, object]]) -> None:
        self.trace = trace
        self.published: list[tuple[str, EventEnvelope]] = []

    async def publish(self, subject: str, event: EventEnvelope) -> None:
        self.trace.append(("publish", event.event_id))
        self.published.append((subject, event))

    async def messages(self, subject: str, durable: str) -> AsyncIterator[Message]:
        del subject, durable
        if False:
            yield


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
        self._trace = trace
        self.nak_delay: float | None = None

    async def ack(self) -> None:
        self._trace.append(("ack", self.event.event_id))

    async def nak(self, delay_seconds: float) -> None:
        self.nak_delay = delay_seconds
        self._trace.append(("nak", delay_seconds))


class FakeOutcomeRepository:
    def __init__(
        self,
        predictions: list[Prediction],
        *,
        observations: list[ImbalanceObservation] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.predictions = predictions
        self.observations = observations or [realized_observation()]
        self.failure = failure
        self.window_calls: list[tuple[datetime, int, datetime]] = []
        self.prediction_calls: list[datetime] = []

    async def fetch_imbalance_window(
        self,
        event_cutoff: datetime,
        minutes: int,
        *,
        knowledge_cutoff: datetime,
    ) -> list[ImbalanceObservation]:
        self.window_calls.append((event_cutoff, minutes, knowledge_cutoff))
        if self.failure is not None:
            raise self.failure
        return self.observations

    async def fetch_predictions_for_target(self, target_time: datetime) -> list[Prediction]:
        self.prediction_calls.append(target_time)
        return self.predictions


def stored_event() -> EventEnvelope:
    return EventEnvelope(
        event_id="stored-event-001",
        event_type="elia.imbalance.stored",
        source="clickhouse",
        dataset="ods161",
        event_time=NOW,
        observed_at=NOW,
        ingested_at=NOW,
        correlation_id="source-event-001",
        causation_id="source-event-001",
        quality_status="Validated",
        payload={
            "source_event_id": "source-event-001",
            "timestamp": NOW.isoformat().replace("+00:00", "Z"),
            "source_ingested_at": NOW.isoformat().replace("+00:00", "Z"),
        },
    )


def realized_observation(value: float = -25.0) -> ImbalanceObservation:
    return ImbalanceObservation(
        timestamp=NOW,
        quarter_hour=NOW,
        resolution_code="PT1M",
        quality_status="Validated",
        ace_mw=0.0,
        system_imbalance_mw=value,
        alpha_eur_mwh=None,
        alpha_prime_eur_mwh=None,
        marginal_incremental_price_eur_mwh=None,
        marginal_decremental_price_eur_mwh=None,
        imbalance_price_eur_mwh=None,
    )


def prediction(
    event_id: str,
    current_state: str | None,
    *,
    generated_at: datetime = NOW,
) -> Prediction:
    return Prediction(
        event_id=event_id,
        cutoff=NOW,
        target_time=NOW,
        generated_at=generated_at,
        system_imbalance_mw=10.0,
        p10_mw=0.0,
        p90_mw=20.0,
        flip_probability=0.5,
        will_flip=False,
        current_state=current_state,
        predicted_state=current_state,
        prediction_quality="model",
        model_version="model-v1",
        feature_schema_hash="feature-schema-001",
    )


def test_outcome_subscription_only_consumes_stored_imbalance_events() -> None:
    subject, durable = OUTCOME_SUBSCRIPTION

    assert subject == Subject.STORED_ELIA_IMBALANCE.value
    assert subject != Subject.RAW_ELIA_IMBALANCE.value
    assert durable == "outcome-builder-v1"


@pytest.mark.asyncio
async def test_outcome_service_joins_every_matching_prediction_without_mutation() -> None:
    trace: list[tuple[str, object]] = []
    predictions = [
        prediction("prediction-positive", "positive"),
        prediction("prediction-unknown", None),
    ]
    repository = FakeOutcomeRepository(predictions)
    bus = RecordingBus(trace)
    message = RecordingMessage(stored_event(), trace)

    await OutcomeService(repository, bus).handle(message)

    assert repository.window_calls == [(NOW, 1, NOW)]
    assert repository.prediction_calls == [NOW]
    assert [subject for subject, _ in bus.published] == [Subject.OUTCOMES_IMBALANCE.value] * 2
    assert trace[-1] == ("ack", "stored-event-001")
    first, second = [event for _, event in bus.published]
    assert first.payload["prediction_event_id"] == "prediction-positive"
    assert first.payload["realized_state"] == "negative"
    assert first.payload["flip_actual"] is True
    assert second.payload["prediction_event_id"] == "prediction-unknown"
    assert second.payload["flip_actual"] is None
    assert predictions[0].current_state == "positive"


@pytest.mark.asyncio
async def test_outcome_service_naks_until_a_lagging_prediction_is_available() -> None:
    trace: list[tuple[str, object]] = []
    repository = FakeOutcomeRepository([])
    bus = RecordingBus(trace)
    message = RecordingMessage(stored_event(), trace)

    await OutcomeService(repository, bus).handle(message)

    assert trace == [("nak", 5.0)]
    assert bus.published == []
    assert message.nak_delay == 5.0


@pytest.mark.asyncio
async def test_outcome_service_waits_for_the_prediction_settlement_grace_period() -> None:
    trace: list[tuple[str, object]] = []
    repository = FakeOutcomeRepository([prediction("first-member", "positive")])
    bus = RecordingBus(trace)
    now = NOW
    service = OutcomeService(repository, bus, clock=lambda: now)

    await service.handle(RecordingMessage(stored_event(), trace))
    now = NOW + PREDICTION_SETTLEMENT_GRACE
    await service.handle(RecordingMessage(stored_event(), trace, delivery_count=2))

    assert trace[0] == ("nak", 5.0)
    assert [subject for subject, _ in bus.published] == [Subject.OUTCOMES_IMBALANCE.value]
    assert trace[-1] == ("ack", "stored-event-001")


@pytest.mark.asyncio
async def test_outcome_service_dead_letters_missing_predictions_after_bounded_retries() -> None:
    trace: list[tuple[str, object]] = []
    repository = FakeOutcomeRepository([])
    bus = RecordingBus(trace)
    message = RecordingMessage(
        stored_event(),
        trace,
        delivery_count=MAX_PENDING_PREDICTION_DELIVERIES,
    )

    await OutcomeService(repository, bus).handle(message)

    assert trace == [("publish", bus.published[0][1].event_id), ("ack", "stored-event-001")]
    assert bus.published[0][0] == OUTCOME_DLQ_SUBJECT
    assert bus.published[0][1].payload["reason"] == "delivery_exhausted"


@pytest.mark.asyncio
async def test_outcome_service_excludes_predictions_generated_after_the_target() -> None:
    trace: list[tuple[str, object]] = []
    repository = FakeOutcomeRepository(
        [prediction("future-prediction", "positive", generated_at=NOW + timedelta(minutes=1))]
    )
    bus = RecordingBus(trace)
    message = RecordingMessage(stored_event(), trace)

    await OutcomeService(repository, bus).handle(message)

    assert trace == [("nak", 5.0)]
    assert bus.published == []


@pytest.mark.asyncio
async def test_outcome_service_naks_retryable_repository_errors() -> None:
    trace: list[tuple[str, object]] = []
    repository = FakeOutcomeRepository(
        [],
        failure=RuntimeError("ClickHouse temporarily unavailable"),
    )
    bus = RecordingBus(trace)
    message = RecordingMessage(stored_event(), trace)

    await OutcomeService(repository, bus).handle(message)

    assert trace == [("nak", 5.0)]
    assert bus.published == []


@pytest.mark.asyncio
async def test_outcome_service_uses_the_configured_deadband_for_the_realized_state() -> None:
    trace: list[tuple[str, object]] = []
    repository = FakeOutcomeRepository(
        [prediction("prediction-positive", "positive")],
        observations=[realized_observation(-15.0)],
    )
    bus = RecordingBus(trace)
    message = RecordingMessage(stored_event(), trace)

    await OutcomeService(repository, bus, deadband_mw=20.0).handle(message)

    event = bus.published[0][1]
    assert event.payload["realized_state"] == "positive"
    assert event.payload["flip_actual"] is False


@pytest.mark.asyncio
async def test_outcome_events_are_deterministic_for_a_replayed_stored_revision() -> None:
    trace: list[tuple[str, object]] = []
    repository = FakeOutcomeRepository([prediction("prediction-positive", "positive")])
    bus = RecordingBus(trace)
    service = OutcomeService(repository, bus)

    await service.handle(RecordingMessage(stored_event(), trace))
    await service.handle(RecordingMessage(stored_event(), trace))

    first = bus.published[0][1]
    second = bus.published[1][1]
    assert first == second
    assert first.ingested_at == NOW


def test_outcome_test_doubles_match_transport_protocols() -> None:
    trace: list[tuple[str, object]] = []

    assert isinstance(RecordingBus(trace), EventBus)
    assert isinstance(RecordingMessage(stored_event(), trace), Message)
