import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from imbalance_pipeline.config import get_settings
from imbalance_pipeline.domain.events import EventEnvelope, Subject, event_id, utc_milliseconds
from imbalance_pipeline.domain.imbalance import (
    ConfirmedState,
    ImbalanceObservation,
    VersionedImbalanceObservation,
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
OUTCOME_PREDICTION_SUBSCRIPTION: Final = (
    Subject.STORED_IMBALANCE_PREDICTION.value,
    "outcome-prediction-reconcile-v1",
)
OUTCOME_SUBSCRIPTIONS: Final = (OUTCOME_SUBSCRIPTION, OUTCOME_PREDICTION_SUBSCRIPTION)


class OutcomeRepository(Protocol):
    async def fetch_imbalance_window(
        self,
        event_cutoff: datetime,
        minutes: int,
        *,
        knowledge_cutoff: datetime,
    ) -> list[ImbalanceObservation]: ...

    async def fetch_predictions_for_target(self, target_time: datetime) -> list[Prediction]: ...

    async def fetch_imbalance_versions(
        self,
        start: datetime,
        end: datetime,
        *,
        knowledge_cutoff: datetime,
    ) -> list[VersionedImbalanceObservation]: ...


class _StoredImbalancePayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_event_id: str
    timestamp: datetime
    source_ingested_at: datetime | None = None

    @field_validator("timestamp", "source_ingested_at")
    @classmethod
    def require_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("stored imbalance timestamp must be UTC-aware")
        return value.astimezone(UTC)


class _StoredPredictionPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_event_id: str
    target_time: datetime
    generated_at: datetime

    @field_validator("target_time", "generated_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("stored prediction timestamp must be UTC-aware")
        return value.astimezone(UTC)


class OutcomeService:
    def __init__(
        self,
        repository: OutcomeRepository,
        bus: EventBus,
        *,
        deadband_mw: float = 10.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if deadband_mw <= 0:
            raise ValueError("deadband_mw must be positive")
        self._repository = repository
        self._bus = bus
        self._deadband_mw = deadband_mw
        self._clock = clock or (lambda: datetime.now(UTC))

    async def handle(self, message: Message) -> None:
        try:
            event_type = message.event.event_type
            if event_type == "elia.imbalance.stored":
                trigger = _stored_payload(message.event)
                prediction_trigger = None
            elif event_type == "imbalance.prediction.stored":
                trigger = None
                prediction_trigger = _stored_prediction_payload(message.event)
            else:
                raise ValueError("outcome service accepts only stored imbalance events")
        except (TypeError, ValidationError, ValueError):
            await self._reject(message, _invalid_reason(message.event))
            return
        try:
            if trigger is not None:
                outcomes = await self._build_outcomes(message.event, trigger)
            else:
                assert prediction_trigger is not None
                outcomes = await self._reconcile_stored_prediction(
                    message.event,
                    prediction_trigger,
                )
        except Exception:
            await message.nak(5.0)
            return
        try:
            for outcome in outcomes:
                await self._bus.publish(Subject.OUTCOMES_IMBALANCE.value, outcome)
            await message.ack()
        except Exception:
            await message.nak(5.0)

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tasks:
            for subject, durable in OUTCOME_SUBSCRIPTIONS:
                tasks.create_task(self._consume(subject, durable), name=durable)

    async def _consume(self, subject: str, durable: str) -> None:
        async for message in self._bus.messages(subject, durable):
            await self.handle(message)

    async def _build_outcomes(
        self,
        event: EventEnvelope,
        trigger: _StoredImbalancePayload,
    ) -> list[EventEnvelope]:
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
        return await self._prediction_outcomes(trigger, realized)

    async def _reconcile_stored_prediction(
        self,
        event: EventEnvelope,
        trigger: _StoredPredictionPayload,
    ) -> list[EventEnvelope]:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() != UTC.utcoffset(now):
            raise ValueError("outcome reconciliation clock must be UTC-aware")
        versions = await self._repository.fetch_imbalance_versions(
            trigger.target_time,
            trigger.target_time,
            knowledge_cutoff=now.astimezone(UTC),
        )
        if not versions:
            return []
        latest = max(
            versions,
            key=lambda version: (version.row_version, version.available_at, version.event_id),
        )
        realization_trigger = _reconciled_realization(latest)
        return await self._prediction_outcomes(
            realization_trigger,
            latest.observation,
        )

    async def _prediction_outcomes(
        self,
        trigger: _StoredImbalancePayload,
        realized: ImbalanceObservation,
    ) -> list[EventEnvelope]:
        predictions = await self._repository.fetch_predictions_for_target(trigger.timestamp)
        eligible_predictions = [
            prediction for prediction in predictions if prediction.generated_at <= trigger.timestamp
        ]
        return [
            _outcome_event(
                trigger,
                prediction,
                realized,
                deadband_mw=self._deadband_mw,
            )
            for prediction in eligible_predictions
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
    if payload.source_ingested_at is not None and payload.source_ingested_at != event.ingested_at:
        raise ValueError("stored imbalance source version does not match envelope")
    return payload


def _stored_prediction_payload(event: EventEnvelope) -> _StoredPredictionPayload:
    if event.event_type != "imbalance.prediction.stored":
        raise ValueError("outcome service accepts only stored prediction triggers")
    if event.schema_version != "1":
        raise ValueError("unsupported stored prediction schema version")
    payload = _StoredPredictionPayload.model_validate(event.payload)
    if payload.target_time != event.event_time:
        raise ValueError("stored prediction target does not match event time")
    return payload


def _invalid_reason(event: EventEnvelope) -> DeadLetterReason:
    if event.event_type not in {"elia.imbalance.stored", "imbalance.prediction.stored"}:
        return DeadLetterReason.UNSUPPORTED_EVENT_TYPE
    if event.schema_version != "1":
        return DeadLetterReason.UNSUPPORTED_SCHEMA_VERSION
    return DeadLetterReason.INVALID_PAYLOAD


def _reconciled_realization(
    realization: VersionedImbalanceObservation,
) -> _StoredImbalancePayload:
    observation = realization.observation
    source_ingested_at = utc_milliseconds(realization.available_at)
    return _StoredImbalancePayload(
        source_event_id=realization.event_id,
        timestamp=observation.timestamp,
        source_ingested_at=source_ingested_at,
    )


def _outcome_event(
    trigger: _StoredImbalancePayload,
    prediction: Prediction,
    realized: ImbalanceObservation,
    *,
    deadband_mw: float,
) -> EventEnvelope:
    current_state = _confirmed_state(prediction.current_state)
    realized_state = advance_state(
        current_state,
        realized.system_imbalance_mw,
        deadband_mw=deadband_mw,
    )
    flip_actual = flip_label(current_state, realized_state)
    evaluated_at = utc_milliseconds(trigger.source_ingested_at or trigger.timestamp)
    return EventEnvelope(
        event_id=event_id(
            "outcome-joiner",
            "prediction-outcome",
            ":".join(
                (
                    prediction.event_id,
                    trigger.source_event_id,
                    evaluated_at.astimezone(UTC).isoformat(),
                )
            ),
        ),
        event_type="imbalance.prediction.evaluated",
        source="outcome-joiner",
        dataset="system-imbalance",
        event_time=trigger.timestamp,
        observed_at=None,
        ingested_at=evaluated_at,
        correlation_id=trigger.source_event_id,
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
            await OutcomeService(
                repository,
                cast(EventBus, bus),
                deadband_mw=settings.flip_deadband_mw,
            ).run()
        finally:
            await repository.aclose()
    finally:
        await bus.aclose()


def main() -> None:
    asyncio.run(_run_service())
