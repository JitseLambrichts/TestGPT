from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from imbalance_pipeline.domain.events import EventEnvelope, Subject
from imbalance_pipeline.domain.imbalance import ImbalanceObservation
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.services.outcomes import OUTCOME_SUBSCRIPTION, OutcomeService
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
    def __init__(self, event: EventEnvelope, trace: list[tuple[str, object]]) -> None:
        self.event = event
        self.delivery_count = 1
        self._trace = trace

    async def ack(self) -> None:
        self._trace.append(("ack", self.event.event_id))

    async def nak(self, delay_seconds: float) -> None:
        self._trace.append(("nak", delay_seconds))


class FakeOutcomeRepository:
    def __init__(self, predictions: list[Prediction]) -> None:
        self.predictions = predictions
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
        return [realized_observation()]

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
        },
    )


def realized_observation() -> ImbalanceObservation:
    return ImbalanceObservation(
        timestamp=NOW,
        quarter_hour=NOW,
        resolution_code="PT1M",
        quality_status="Validated",
        ace_mw=0.0,
        system_imbalance_mw=-25.0,
        alpha_eur_mwh=None,
        alpha_prime_eur_mwh=None,
        marginal_incremental_price_eur_mwh=None,
        marginal_decremental_price_eur_mwh=None,
        imbalance_price_eur_mwh=None,
    )


def prediction(event_id: str, current_state: str | None) -> Prediction:
    return Prediction(
        event_id=event_id,
        cutoff=NOW,
        target_time=NOW,
        generated_at=NOW,
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

    await OutcomeService(repository, bus, clock=lambda: NOW).handle(message)

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


def test_outcome_test_doubles_match_transport_protocols() -> None:
    trace: list[tuple[str, object]] = []

    assert isinstance(RecordingBus(trace), EventBus)
    assert isinstance(RecordingMessage(stored_event(), trace), Message)
