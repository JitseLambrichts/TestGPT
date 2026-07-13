from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from imbalance_pipeline.domain.events import EventEnvelope, Subject
from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.features.engine import FeatureSnapshot
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.services.features import (
    FEATURE_DLQ_SUBJECT,
    FEATURE_SUBSCRIPTION,
    FeatureService,
)

NOW = datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC)


class RecordingBus:
    def __init__(self, trace: list[tuple[str, object]]) -> None:
        self.trace = trace
        self.published: list[tuple[str, EventEnvelope]] = []

    async def publish(self, subject: str, event: EventEnvelope) -> None:
        self.trace.append(("publish", subject))
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
        self.nak_delay: float | None = None

    async def ack(self) -> None:
        self._trace.append(("ack", self.event.event_id))

    async def nak(self, delay_seconds: float) -> None:
        self.nak_delay = delay_seconds
        self._trace.append(("nak", delay_seconds))


class FakeFeatureEngine:
    def __init__(self, result: FeatureSnapshot) -> None:
        self.result = result
        self.calls: list[tuple[datetime, datetime]] = []

    async def build(self, event_cutoff: datetime, *, knowledge_cutoff: datetime) -> FeatureSnapshot:
        self.calls.append((event_cutoff, knowledge_cutoff))
        return self.result


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


def snapshot() -> FeatureSnapshot:
    return FeatureSnapshot(
        event_id="feature-event-001",
        cutoff=NOW,
        knowledge_cutoff=NOW,
        target_time=NOW + timedelta(minutes=1),
        feature_schema_hash="feature-schema-001",
        local_values=np.asarray([[1.0, 2.0]], dtype=np.float32),
        local_masks=np.asarray([[1, 1]], dtype=np.uint8),
        context_values=np.asarray([[3.0]], dtype=np.float32),
        context_masks=np.asarray([[1]], dtype=np.uint8),
        static_values=np.asarray([4.0], dtype=np.float32),
        static_masks=np.asarray([1], dtype=np.uint8),
        current_state=ConfirmedState.POSITIVE,
        model_eligible=False,
        observed_imbalance_minutes=1,
        created_at=NOW,
    )


def test_feature_subscription_only_consumes_stored_imbalance_events() -> None:
    subject, durable = FEATURE_SUBSCRIPTION

    assert subject == Subject.STORED_ELIA_IMBALANCE.value
    assert subject != Subject.RAW_ELIA_IMBALANCE.value
    assert durable == "feature-builder-v1"


@pytest.mark.asyncio
async def test_feature_service_builds_publishes_and_only_then_acknowledges() -> None:
    trace: list[tuple[str, object]] = []
    engine = FakeFeatureEngine(snapshot())
    bus = RecordingBus(trace)
    message = RecordingMessage(stored_event(), trace)

    await FeatureService(engine, bus).handle(message)

    assert engine.calls == [(NOW, NOW)]
    assert trace == [
        ("publish", Subject.FEATURES_IMBALANCE.value),
        ("ack", "stored-event-001"),
    ]
    subject, event = bus.published[0]
    assert subject == Subject.FEATURES_IMBALANCE.value
    assert event.event_id == "feature-event-001"
    assert event.causation_id == "stored-event-001"
    assert event.payload["model_eligible"] is False
    assert event.payload["local_values"] == [[1.0, 2.0]]


@pytest.mark.asyncio
async def test_feature_service_dead_letters_an_invalid_stored_trigger() -> None:
    trace: list[tuple[str, object]] = []
    engine = FakeFeatureEngine(snapshot())
    bus = RecordingBus(trace)
    invalid = stored_event().model_copy(update={"event_type": "elia.imbalance.observed"})
    message = RecordingMessage(invalid, trace)

    await FeatureService(engine, bus).handle(message)

    assert engine.calls == []
    assert trace == [("publish", FEATURE_DLQ_SUBJECT), ("ack", "stored-event-001")]
    assert bus.published[0][1].payload["reason"] == "unsupported_event_type"


def test_feature_test_doubles_match_transport_protocols() -> None:
    trace: list[tuple[str, object]] = []

    assert isinstance(RecordingBus(trace), EventBus)
    assert isinstance(RecordingMessage(stored_event(), trace), Message)
