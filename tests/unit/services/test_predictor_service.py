from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from imbalance_pipeline.domain.events import EventEnvelope, Subject
from imbalance_pipeline.domain.imbalance import ConfirmedState
from imbalance_pipeline.features.engine import FeatureSnapshot
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.services.predictor import (
    PREDICTOR_DLQ_SUBJECT,
    PREDICTOR_SUBSCRIPTION,
    PredictorService,
)
from imbalance_pipeline.serving.runtime import PredictionValues

NOW = datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC)


class RecordingBus:
    def __init__(self, trace: list[tuple[str, object]]) -> None:
        self.trace = trace
        self.published: list[tuple[str, EventEnvelope]] = []
        self.failure: Exception | None = None

    async def publish(self, subject: str, event: EventEnvelope) -> None:
        if self.failure is not None:
            raise self.failure
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


class FakeRuntime:
    def __init__(self, result: PredictionValues | Exception) -> None:
        self.result = result
        self.calls: list[FeatureSnapshot] = []
        self.model_version = "model-v1"
        self.feature_schema_hash = "feature-schema-001"

    def predict(self, snapshot: FeatureSnapshot) -> PredictionValues:
        self.calls.append(snapshot)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def feature_event(
    *,
    eligible: bool = True,
    current_state: str | None = "positive",
) -> EventEnvelope:
    return EventEnvelope(
        event_id="feature-event-001",
        event_type="imbalance.feature.snapshot",
        source="feature-engine",
        dataset="system-imbalance",
        event_time=NOW,
        observed_at=NOW,
        ingested_at=NOW,
        correlation_id="stored-event-001",
        causation_id="stored-event-001",
        quality_status="eligible" if eligible else "insufficient_history",
        payload={
            "cutoff": NOW.isoformat().replace("+00:00", "Z"),
            "target_time": (NOW + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
            "feature_schema_hash": "feature-schema-001",
            "local_values": [[-20.0, 1.0], [42.0, 2.0]],
            "local_masks": [[1, 1], [1, 1]],
            "context_values": [[3.0]],
            "context_masks": [[1]],
            "static_values": [4.0],
            "static_masks": [1],
            "current_state": current_state,
            "model_eligible": eligible,
            "observed_imbalance_minutes": 180,
            "created_at": NOW.isoformat().replace("+00:00", "Z"),
        },
    )


def model_prediction() -> PredictionValues:
    return PredictionValues(
        model_version="model-v1",
        feature_schema_hash="feature-schema-001",
        system_imbalance_mw=-15.0,
        p10_mw=-30.0,
        p90_mw=5.0,
        flip_probability=0.8,
        will_flip=True,
        current_state=ConfirmedState.POSITIVE,
        predicted_state=ConfirmedState.NEGATIVE,
        delta_mw=-20.0,
    )


def test_predictor_subscription_only_consumes_feature_snapshots() -> None:
    subject, durable = PREDICTOR_SUBSCRIPTION

    assert subject == Subject.FEATURES_IMBALANCE.value
    assert durable == "predictor-v1"


@pytest.mark.asyncio
async def test_predictor_publishes_model_prediction_before_acknowledging() -> None:
    trace: list[tuple[str, object]] = []
    runtime = FakeRuntime(model_prediction())
    bus = RecordingBus(trace)
    message = RecordingMessage(feature_event(), trace)
    service = PredictorService(runtime, bus, clock=lambda: NOW + timedelta(seconds=1))

    await service.handle(message)

    assert len(runtime.calls) == 1
    assert trace == [
        ("publish", Subject.PREDICTIONS_IMBALANCE.value),
        ("ack", "feature-event-001"),
    ]
    subject, event = bus.published[0]
    assert subject == Subject.PREDICTIONS_IMBALANCE.value
    assert event.event_type == "imbalance.prediction.generated"
    assert event.event_time == NOW + timedelta(minutes=1)
    assert event.causation_id == "feature-event-001"
    assert event.payload == {
        "cutoff": "2026-07-13T10:01:05Z",
        "target_time": "2026-07-13T10:02:05Z",
        "generated_at": "2026-07-13T10:01:06Z",
        "system_imbalance_mw": -15.0,
        "p10_mw": -30.0,
        "p90_mw": 5.0,
        "flip_probability": 0.8,
        "will_flip": True,
        "current_state": "positive",
        "predicted_state": "negative",
        "prediction_quality": "model",
        "model_version": "model-v1",
        "feature_schema_hash": "feature-schema-001",
    }


@pytest.mark.asyncio
async def test_predictor_falls_back_for_an_ineligible_snapshot_without_calling_model() -> None:
    trace: list[tuple[str, object]] = []
    runtime = FakeRuntime(model_prediction())
    bus = RecordingBus(trace)
    message = RecordingMessage(feature_event(eligible=False), trace)
    service = PredictorService(runtime, bus, clock=lambda: NOW + timedelta(seconds=1))

    await service.handle(message)

    assert runtime.calls == []
    assert trace == [
        ("publish", Subject.PREDICTIONS_IMBALANCE.value),
        ("ack", "feature-event-001"),
    ]
    payload = bus.published[0][1].payload
    assert payload["prediction_quality"] == "degraded"
    assert payload["degraded_reason"] == "insufficient_feature_history"
    assert payload["system_imbalance_mw"] == 42.0
    assert payload["p10_mw"] == 42.0
    assert payload["p90_mw"] == 42.0
    assert payload["will_flip"] is False
    assert payload["predicted_state"] == "positive"


@pytest.mark.asyncio
async def test_predictor_falls_back_when_model_inference_fails() -> None:
    trace: list[tuple[str, object]] = []
    runtime = FakeRuntime(RuntimeError("onnx inference failed"))
    bus = RecordingBus(trace)
    message = RecordingMessage(feature_event(), trace)
    service = PredictorService(runtime, bus, clock=lambda: NOW + timedelta(seconds=1))

    await service.handle(message)

    assert len(runtime.calls) == 1
    assert trace[-1] == ("ack", "feature-event-001")
    payload = bus.published[0][1].payload
    assert payload["prediction_quality"] == "degraded"
    assert payload["degraded_reason"] == "model_inference_failed"
    assert payload["model_version"] == "model-v1"


@pytest.mark.asyncio
async def test_predictor_naks_a_model_failure_when_fallback_is_disabled() -> None:
    trace: list[tuple[str, object]] = []
    runtime = FakeRuntime(RuntimeError("onnx inference failed"))
    bus = RecordingBus(trace)
    message = RecordingMessage(feature_event(), trace)
    service = PredictorService(runtime, bus, allow_fallback=False)

    await service.handle(message)

    assert trace == [("nak", 5.0)]
    assert bus.published == []


@pytest.mark.asyncio
async def test_predictor_dead_letters_an_invalid_feature_event() -> None:
    trace: list[tuple[str, object]] = []
    runtime = FakeRuntime(model_prediction())
    bus = RecordingBus(trace)
    invalid = feature_event().model_copy(update={"event_type": "elia.imbalance.stored"})
    message = RecordingMessage(invalid, trace)
    service = PredictorService(runtime, bus)

    await service.handle(message)

    assert runtime.calls == []
    assert trace == [("publish", PREDICTOR_DLQ_SUBJECT), ("ack", "feature-event-001")]
    assert bus.published[0][1].payload["reason"] == "unsupported_event_type"


@pytest.mark.asyncio
async def test_predictor_naks_when_prediction_publication_fails() -> None:
    trace: list[tuple[str, object]] = []
    runtime = FakeRuntime(model_prediction())
    bus = RecordingBus(trace)
    bus.failure = RuntimeError("nats unavailable")
    message = RecordingMessage(feature_event(), trace)
    service = PredictorService(runtime, bus)

    await service.handle(message)

    assert len(runtime.calls) == 1
    assert trace == [("nak", 5.0)]


def test_predictor_test_doubles_match_transport_protocols() -> None:
    trace: list[tuple[str, object]] = []

    assert isinstance(RecordingBus(trace), EventBus)
    assert isinstance(cast(Message, RecordingMessage(feature_event(), trace)), Message)
