import asyncio
from datetime import UTC, datetime
from typing import Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from imbalance_pipeline.config import get_settings
from imbalance_pipeline.domain.events import EventEnvelope, Subject, event_id
from imbalance_pipeline.features.engine import FeatureSnapshot
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.services.sink import DeadLetterPayload
from imbalance_pipeline.storage.clickhouse import ClickHouseRepository, DeadLetterReason

FEATURE_DLQ_SUBJECT: Final = "grid.dlq.features.v1"
FEATURE_SUBSCRIPTION: Final = (Subject.STORED_ELIA_IMBALANCE.value, "feature-builder-v1")


class FeatureBuilder(Protocol):
    async def build(
        self,
        event_cutoff: datetime,
        *,
        knowledge_cutoff: datetime,
    ) -> FeatureSnapshot: ...


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


class FeatureService:
    def __init__(self, engine: FeatureBuilder, bus: EventBus) -> None:
        self._engine = engine
        self._bus = bus

    async def handle(self, message: Message) -> None:
        try:
            _stored_payload(message.event)
        except (TypeError, ValidationError, ValueError):
            await self._reject(message, _invalid_reason(message.event))
            return
        try:
            snapshot = await self._build_snapshot(message.event)
            await self._bus.publish(
                Subject.FEATURES_IMBALANCE.value,
                _feature_event(message.event, snapshot),
            )
            await message.ack()
        except Exception:
            await message.nak(5.0)

    async def run(self) -> None:
        subject, durable = FEATURE_SUBSCRIPTION
        async for message in self._bus.messages(subject, durable):
            await self.handle(message)

    async def _build_snapshot(self, event: EventEnvelope) -> FeatureSnapshot:
        return await self._engine.build(event.event_time, knowledge_cutoff=event.ingested_at)

    async def _reject(self, message: Message, reason: DeadLetterReason) -> None:
        event = message.event
        payload = DeadLetterPayload(
            original_event=event,
            reason=reason,
            delivery_count=message.delivery_count,
        )
        try:
            await self._bus.publish(
                FEATURE_DLQ_SUBJECT,
                EventEnvelope(
                    event_id=event_id(
                        "feature-engine",
                        "dead-letter",
                        f"{event.event_id}:{reason.value}",
                    ),
                    event_type="feature.event.rejected",
                    source="feature-engine",
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
        raise ValueError("feature service accepts only stored imbalance triggers")
    if event.schema_version != "1":
        raise ValueError("unsupported stored imbalance schema version")
    payload = _StoredImbalancePayload.model_validate(event.payload)
    if payload.timestamp != event.event_time:
        raise ValueError("stored imbalance timestamp does not match event time")
    if payload.source_ingested_at is not None and payload.source_ingested_at != event.ingested_at:
        raise ValueError("stored imbalance source version does not match envelope")
    return payload


def _invalid_reason(event: EventEnvelope) -> DeadLetterReason:
    if event.event_type != "elia.imbalance.stored":
        return DeadLetterReason.UNSUPPORTED_EVENT_TYPE
    if event.schema_version != "1":
        return DeadLetterReason.UNSUPPORTED_SCHEMA_VERSION
    return DeadLetterReason.INVALID_PAYLOAD


def _feature_event(trigger: EventEnvelope, snapshot: FeatureSnapshot) -> EventEnvelope:
    state = snapshot.current_state.value if snapshot.current_state is not None else None
    return EventEnvelope(
        event_id=snapshot.event_id,
        event_type="imbalance.feature.snapshot",
        source="feature-engine",
        dataset="system-imbalance",
        event_time=snapshot.cutoff,
        observed_at=trigger.observed_at,
        ingested_at=snapshot.created_at,
        correlation_id=trigger.correlation_id,
        causation_id=trigger.event_id,
        quality_status="eligible" if snapshot.model_eligible else "insufficient_history",
        payload={
            "cutoff": snapshot.cutoff.isoformat().replace("+00:00", "Z"),
            "target_time": snapshot.target_time.isoformat().replace("+00:00", "Z"),
            "feature_schema_hash": snapshot.feature_schema_hash,
            "local_values": _array_list(snapshot.local_values),
            "local_masks": _array_list(snapshot.local_masks),
            "context_values": _array_list(snapshot.context_values),
            "context_masks": _array_list(snapshot.context_masks),
            "static_values": _array_list(snapshot.static_values),
            "static_masks": _array_list(snapshot.static_masks),
            "current_state": state,
            "model_eligible": snapshot.model_eligible,
            "observed_imbalance_minutes": snapshot.observed_imbalance_minutes,
            "created_at": snapshot.created_at.isoformat().replace("+00:00", "Z"),
        },
    )


def _array_list(values: object) -> object:
    tolist = getattr(values, "tolist", None)
    if not callable(tolist):
        raise TypeError("feature arrays must provide tolist()")
    return cast(object, tolist())


async def _run_service() -> None:
    from imbalance_pipeline.features.engine import FeatureEngine
    from imbalance_pipeline.features.schema import FeatureRegistry
    from imbalance_pipeline.messaging.nats import NatsEventBus

    settings = get_settings()
    bus = await NatsEventBus.connect(settings)
    try:
        repository = await ClickHouseRepository.connect(settings)
        try:
            await bus.ensure_grid_stream()
            registry = FeatureRegistry.default(deadband_mw=settings.flip_deadband_mw)
            await FeatureService(FeatureEngine(repository, registry), cast(EventBus, bus)).run()
        finally:
            await repository.aclose()
    finally:
        await bus.aclose()


def main() -> None:
    asyncio.run(_run_service())
