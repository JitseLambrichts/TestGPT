import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from imbalance_pipeline.config import get_settings
from imbalance_pipeline.domain.events import EventEnvelope, Subject, event_id
from imbalance_pipeline.domain.imbalance import (
    ConfirmedState,
    ImbalanceObservation,
    advance_state,
    flip_label,
)
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.services.sink import DeadLetterPayload
from imbalance_pipeline.storage.clickhouse import (
    ClickHouseRepository,
    DeadLetterReason,
    Prediction,
)

OUTCOME_DLQ_SUBJECT: Final = "grid.dlq.outcomes.v1"
OUTCOME_SUBSCRIPTION: Final = (Subject.STORED_ELIA_IMBALANCE.value, "outcome-builder-v1")


class OutcomeRepository(Protocol):
    async def fetch_imbalance_window(
        self,
        event_cutoff: datetime,
        minutes: int,
        *,
        knowledge_cutoff: datetime,
    ) -> list[ImbalanceObservation]: ...

    async def fetch_predictions_for_target(self, target_time: datetime) -> list[Prediction]: ...


class _StoredImbalancePayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_event_id: str
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("stored imbalance timestamp must be UTC-aware")
        return value.astimezone(UTC)


class OutcomeService:
    def __init__(
        self,
        repository: OutcomeRepository,
        bus: EventBus,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._bus = bus
        self._clock = clock or (lambda: datetime.now(UTC))

    async def handle(self, message: Message) -> None:
        try:
            outcomes = await self._build_outcomes(message.event)
        except (TypeError, ValidationError, ValueError):
            await self._reject(message, _invalid_reason(message.event))
            return
        try:
            for outcome in outcomes:
                await self._bus.publish(Subject.OUTCOMES_IMBALANCE.value, outcome)
            await message.ack()
        except Exception:
            await message.nak(5.0)

    async def run(self) -> None:
        subject, durable = OUTCOME_SUBSCRIPTION
        async for message in self._bus.messages(subject, durable):
            await self.handle(message)

    async def _build_outcomes(self, event: EventEnvelope) -> list[EventEnvelope]:
        trigger = _stored_payload(event)
        observations = await self._repository.fetch_imbalance_window(
            trigger.timestamp,
            minutes=1,
            knowledge_cutoff=event.ingested_at,
        )
        realized = next(
            (
                observation
                for observation in observations
                if observation.timestamp == trigger.timestamp
            ),
            None,
        )
        if realized is None:
            raise RuntimeError("stored imbalance trigger has no realized observation")
        predictions = await self._repository.fetch_predictions_for_target(trigger.timestamp)
        return [
            _outcome_event(event, trigger, prediction, realized, self._clock())
            for prediction in predictions
        ]

    async def _reject(self, message: Message, reason: DeadLetterReason) -> None:
        event = message.event
        payload = DeadLetterPayload(
            original_event=event,
            reason=reason,
            delivery_count=message.delivery_count,
        )
        try:
            await self._bus.publish(
                OUTCOME_DLQ_SUBJECT,
                EventEnvelope(
                    event_id=event_id(
                        "outcome-joiner",
                        "dead-letter",
                        f"{event.event_id}:{reason.value}",
                    ),
                    event_type="outcome.event.rejected",
                    source="outcome-joiner",
                    dataset="dead-letter",
                    event_time=event.event_time,
                    observed_at=event.observed_at,
                    ingested_at=event.ingested_at,
                    correlation_id=event.correlation_id,
                    causation_id=event.event_id,
                    quality_status="rejected",
                    payload=payload.model_dump(mode="json"),
                ),
            )
            await message.ack()
        except Exception:
            await message.nak(5.0)


def _stored_payload(event: EventEnvelope) -> _StoredImbalancePayload:
    if event.event_type != "elia.imbalance.stored":
        raise ValueError("outcome service accepts only stored imbalance triggers")
    if event.schema_version != "1":
        raise ValueError("unsupported stored imbalance schema version")
    payload = _StoredImbalancePayload.model_validate(event.payload)
    if payload.timestamp != event.event_time:
        raise ValueError("stored imbalance timestamp does not match event time")
    return payload


def _invalid_reason(event: EventEnvelope) -> DeadLetterReason:
    if event.event_type != "elia.imbalance.stored":
        return DeadLetterReason.UNSUPPORTED_EVENT_TYPE
    if event.schema_version != "1":
        return DeadLetterReason.UNSUPPORTED_SCHEMA_VERSION
    return DeadLetterReason.INVALID_PAYLOAD


def _outcome_event(
    trigger_event: EventEnvelope,
    trigger: _StoredImbalancePayload,
    prediction: Prediction,
    realized: ImbalanceObservation,
    evaluated_at: datetime,
) -> EventEnvelope:
    current_state = _confirmed_state(prediction.current_state)
    realized_state = advance_state(current_state, realized.system_imbalance_mw)
    flip_actual = flip_label(current_state, realized_state)
    return EventEnvelope(
        event_id=event_id(
            "outcome-joiner",
            "prediction-outcome",
            f"{prediction.event_id}:{trigger.source_event_id}",
        ),
        event_type="imbalance.prediction.evaluated",
        source="outcome-joiner",
        dataset="system-imbalance",
        event_time=trigger.timestamp,
        observed_at=trigger_event.observed_at,
        ingested_at=evaluated_at,
        correlation_id=trigger_event.correlation_id,
        causation_id=prediction.event_id,
        quality_status=realized.quality_status,
        payload={
            "prediction_event_id": prediction.event_id,
            "target_time": trigger.timestamp.isoformat().replace("+00:00", "Z"),
            "realized_event_id": trigger.source_event_id,
            "realized_system_imbalance_mw": realized.system_imbalance_mw,
            "realized_state": realized_state.value if realized_state is not None else None,
            "flip_actual": flip_actual,
            "evaluated_at": evaluated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        },
    )


def _confirmed_state(value: str | None) -> ConfirmedState | None:
    if value is None:
        return None
    try:
        return ConfirmedState(value)
    except ValueError:
        return None


async def _run_service() -> None:
    from imbalance_pipeline.messaging.nats import NatsEventBus

    settings = get_settings()
    bus = await NatsEventBus.connect(settings)
    try:
        repository = await ClickHouseRepository.connect(settings)
        try:
            await bus.ensure_grid_stream()
            await OutcomeService(repository, cast(EventBus, bus)).run()
        finally:
            await repository.aclose()
    finally:
        await bus.aclose()


def main() -> None:
    asyncio.run(_run_service())
