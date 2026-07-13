import asyncio
from datetime import UTC
from typing import Final, Protocol, cast

from pydantic import BaseModel, ConfigDict

from imbalance_pipeline.config import get_settings
from imbalance_pipeline.domain.events import EventEnvelope, Subject, event_id
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.storage.clickhouse import (
    ClickHouseRepository,
    DeadLetterReason,
    PermanentEventError,
    TransientStorageError,
)

CLICKHOUSE_DLQ_SUBJECT: Final = "grid.dlq.clickhouse.v1"
MODEL_SUBJECT: Final = "grid.models.imbalance.v1"
MAX_DELIVERY_COUNT: Final = 5
DELIVERY_DELAYS_SECONDS: Final = (1.0, 5.0, 30.0, 120.0)
SINK_SUBSCRIPTIONS: Final = (
    (Subject.RAW_ELIA_IMBALANCE.value, "clickhouse-raw-imbalance-v1"),
    (Subject.RAW_ELIA_LOAD.value, "clickhouse-raw-load-v1"),
    (Subject.RAW_ELIA_WIND.value, "clickhouse-raw-wind-v1"),
    (Subject.RAW_ELIA_SOLAR.value, "clickhouse-raw-solar-v1"),
    (Subject.RAW_WEATHER_FORECAST.value, "clickhouse-raw-weather-v1"),
    (Subject.FEATURES_IMBALANCE.value, "clickhouse-features-v1"),
    (Subject.PREDICTIONS_IMBALANCE.value, "clickhouse-predictions-v1"),
    (Subject.OUTCOMES_IMBALANCE.value, "clickhouse-outcomes-v1"),
    (MODEL_SUBJECT, "clickhouse-models-v1"),
)


class EventRepository(Protocol):
    async def insert_event(self, event: EventEnvelope) -> None: ...


class DeadLetterPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_event_id: str
    event_type: str
    schema_version: str
    source: str
    dataset: str
    reason: DeadLetterReason
    delivery_count: int


class Sink:
    def __init__(self, repository: EventRepository, bus: EventBus) -> None:
        self._repository = repository
        self._bus = bus

    async def handle(self, message: Message) -> None:
        try:
            await self._repository.insert_event(message.event)
        except PermanentEventError as exc:
            await self._publish_dead_letter(message, exc.reason)
            await message.ack()
            return
        except TransientStorageError:
            if message.delivery_count >= MAX_DELIVERY_COUNT:
                await self._publish_dead_letter(message, DeadLetterReason.DELIVERY_EXHAUSTED)
                await message.ack()
            else:
                await message.nak(_delivery_delay(message.delivery_count))
            return

        if message.event.event_type == "elia.imbalance.observed":
            await self._bus.publish(
                Subject.STORED_ELIA_IMBALANCE.value,
                _stored_event(message.event),
            )
        await message.ack()

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tasks:
            for subject, durable in SINK_SUBSCRIPTIONS:
                tasks.create_task(self._consume(subject, durable), name=durable)

    async def _consume(self, subject: str, durable: str) -> None:
        async for message in self._bus.messages(subject, durable):
            await self.handle(message)

    async def _publish_dead_letter(
        self,
        message: Message,
        reason: DeadLetterReason,
    ) -> None:
        source = message.event
        payload = DeadLetterPayload(
            source_event_id=source.event_id,
            event_type=source.event_type,
            schema_version=source.schema_version,
            source=source.source,
            dataset=source.dataset,
            reason=reason,
            delivery_count=message.delivery_count,
        )
        dead_letter = EventEnvelope(
            event_id=event_id(
                "clickhouse",
                "dead-letter",
                f"{source.event_id}:{reason.value}",
            ),
            event_type="clickhouse.event.rejected",
            schema_version="1",
            source="clickhouse",
            dataset="dead-letter",
            event_time=source.event_time,
            observed_at=source.observed_at,
            ingested_at=source.ingested_at,
            correlation_id=source.correlation_id,
            causation_id=source.event_id,
            quality_status="rejected",
            payload=payload.model_dump(mode="json"),
        )
        await self._bus.publish(CLICKHOUSE_DLQ_SUBJECT, dead_letter)


def _stored_event(source: EventEnvelope) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id(
            "clickhouse",
            "stored-elia-imbalance",
            source.event_id,
            source.schema_version,
        ),
        event_type="elia.imbalance.stored",
        schema_version="1",
        source="clickhouse",
        dataset=source.dataset,
        event_time=source.event_time,
        observed_at=source.observed_at,
        ingested_at=source.ingested_at,
        correlation_id=source.correlation_id,
        causation_id=source.event_id,
        quality_status=source.quality_status,
        payload={
            "source_event_id": source.event_id,
            "timestamp": source.event_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        },
    )


def _delivery_delay(delivery_count: int) -> float:
    index = max(1, delivery_count) - 1
    return DELIVERY_DELAYS_SECONDS[min(index, len(DELIVERY_DELAYS_SECONDS) - 1)]


async def _run_service() -> None:
    from imbalance_pipeline.messaging.nats import NatsEventBus

    settings = get_settings()
    bus = await NatsEventBus.connect(settings)
    try:
        repository = await ClickHouseRepository.connect(settings)
        try:
            await bus.ensure_grid_stream()
            await Sink(repository, cast(EventBus, bus)).run()
        finally:
            await repository.aclose()
    finally:
        await bus.aclose()


def main() -> None:
    asyncio.run(_run_service())
