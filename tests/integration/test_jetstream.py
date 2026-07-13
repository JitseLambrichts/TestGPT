import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import nats
import pytest
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    ConsumerInfo,
    DeliverPolicy,
    ReplayPolicy,
)
from nats.js.client import JetStreamContext
from nats.js.errors import NotFoundError

import imbalance_pipeline.messaging.nats as nats_adapter
import imbalance_pipeline.services.sink as sink_module
from imbalance_pipeline.config import Settings
from imbalance_pipeline.domain.events import EventEnvelope
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.messaging.nats import NatsEventBus
from imbalance_pipeline.services.sink import CLICKHOUSE_DLQ_SUBJECT, DeadLetterPayload, Sink
from imbalance_pipeline.storage.clickhouse import DeadLetterReason, TransientStorageError

NATS_URL = os.environ.get("IMBALANCE_NATS_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        NATS_URL is None,
        reason="set IMBALANCE_NATS_URL to run the real JetStream integration test",
    ),
]


async def settled_consumer_info(
    jetstream: JetStreamContext,
    durable: str,
) -> ConsumerInfo:
    info = await jetstream.consumer_info("GRID_EVENTS", durable)
    for _ in range(50):
        if info.num_pending == 0 and info.num_ack_pending == 0:
            return info
        await asyncio.sleep(0.02)
        info = await jetstream.consumer_info("GRID_EVENTS", durable)
    return info


async def delete_test_consumer(jetstream: JetStreamContext, durable: str) -> None:
    with suppress(NotFoundError):
        await jetstream.delete_consumer("GRID_EVENTS", durable)


class AlwaysTransientRepository:
    def __init__(self) -> None:
        self.attempts = 0

    async def insert_event(self, event: EventEnvelope) -> None:
        del event
        self.attempts += 1
        raise TransientStorageError


class FailFirstDlqBus:
    def __init__(self, delegate: NatsEventBus) -> None:
        self._delegate = delegate
        self._remaining_failures = 1

    async def publish(self, subject: str, event: EventEnvelope) -> None:
        if subject == CLICKHOUSE_DLQ_SUBJECT and self._remaining_failures:
            self._remaining_failures -= 1
            raise RuntimeError("simulated DLQ outage")
        await self._delegate.publish(subject, event)

    def messages(self, subject: str, durable: str) -> AsyncIterator[Message]:
        return self._delegate.messages(subject, durable)


def imbalance_event(identity: str) -> EventEnvelope:
    timestamp = datetime.now(UTC)
    return EventEnvelope(
        event_id=identity,
        event_type="elia.imbalance.observed",
        schema_version="1",
        source="elia",
        dataset="ods161",
        event_time=timestamp,
        observed_at=timestamp,
        ingested_at=timestamp,
        correlation_id=identity,
        causation_id="integration-test-run",
        quality_status="Validated",
        payload={
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "quarter_hour": timestamp.isoformat().replace("+00:00", "Z"),
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


@pytest.mark.asyncio
async def test_duplicate_publish_is_consumed_once_and_ack_survives_reconnect() -> None:
    assert NATS_URL is not None
    identity = uuid4().hex
    subject = f"grid.integration.nats.{identity}"
    durable = f"integration-{identity}"
    event = EventEnvelope(
        event_id=identity,
        event_type="integration.jetstream.probe",
        schema_version="1",
        source="integration-test",
        dataset="jetstream",
        event_time=datetime.now(UTC),
        observed_at=datetime.now(UTC),
        ingested_at=datetime.now(UTC),
        correlation_id=identity,
        causation_id="integration-test-run",
        quality_status="test",
        payload={"probe": True, "identity": identity},
    )

    admin = await nats.connect(NATS_URL)
    admin_js = admin.jetstream()
    first_bus: NatsEventBus | None = None
    second_bus: NatsEventBus | None = None
    try:
        first_bus = await NatsEventBus.connect(Settings(nats_url=NATS_URL))
        await first_bus.ensure_grid_stream()
        await first_bus.publish(subject, event)
        await first_bus.publish(subject, event)

        stored = await admin_js.stream_info("GRID_EVENTS", subjects_filter=subject)
        assert stored.state.subjects == {subject: 1}

        messages = first_bus.messages(subject, durable)
        received = await asyncio.wait_for(anext(messages), timeout=5)
        assert received.event == event
        assert received.delivery_count == 1
        await received.ack()
        await messages.aclose()
        await first_bus.aclose()

        second_bus = await NatsEventBus.connect(Settings(nats_url=NATS_URL))
        messages_after_reconnect = second_bus.messages(subject, durable)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(messages_after_reconnect), timeout=2)
        await messages_after_reconnect.aclose()

        consumer = await settled_consumer_info(admin_js, durable)
        assert consumer.num_pending == 0
        assert consumer.num_ack_pending == 0
    finally:
        if first_bus is not None:
            await first_bus.aclose()
        if second_bus is not None:
            await second_bus.aclose()
        await delete_test_consumer(admin_js, durable)
        await admin.close()


@pytest.mark.asyncio
async def test_failed_fifth_delivery_is_redelivered_after_restart_then_dlqed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert NATS_URL is not None
    monkeypatch.setattr(sink_module, "DELIVERY_DELAYS_SECONDS", (0.01, 0.01, 0.01, 0.01))
    monkeypatch.setattr(nats_adapter, "CONSUMER_BACKOFF_SECONDS", (0.01, 0.01, 0.01, 0.01))
    identity = uuid4().hex
    subject = f"grid.integration.retry.{identity}"
    durable = f"integration-retry-{identity}"
    source = imbalance_event(identity)
    repository = AlwaysTransientRepository()
    admin = await nats.connect(NATS_URL)
    admin_js = admin.jetstream()
    first_bus: NatsEventBus | None = None
    second_bus: NatsEventBus | None = None
    first_messages: AsyncIterator[Message] | None = None
    second_messages: AsyncIterator[Message] | None = None
    try:
        first_bus = await NatsEventBus.connect(Settings(nats_url=NATS_URL))
        await first_bus.ensure_grid_stream()
        await first_bus.publish(subject, source)
        first_messages = cast(AsyncIterator[Message], first_bus.messages(subject, durable))
        first_sink = Sink(repository, cast(EventBus, first_bus))

        for expected_delivery in range(1, 5):
            message = await asyncio.wait_for(anext(first_messages), timeout=5)
            assert message.delivery_count == expected_delivery
            await first_sink.handle(message)

        fifth = await asyncio.wait_for(anext(first_messages), timeout=5)
        assert fifth.delivery_count == 5
        await Sink(repository, FailFirstDlqBus(first_bus)).handle(fifth)
        await first_messages.aclose()
        first_messages = None
        await first_bus.aclose()
        first_bus = None

        second_bus = await NatsEventBus.connect(Settings(nats_url=NATS_URL))
        second_messages = cast(AsyncIterator[Message], second_bus.messages(subject, durable))
        redelivered = await asyncio.wait_for(anext(second_messages), timeout=5)
        assert redelivered.delivery_count >= 6
        await Sink(repository, cast(EventBus, second_bus)).handle(redelivered)
        await second_messages.aclose()
        second_messages = None

        raw_dlq = await admin_js.get_last_msg("GRID_EVENTS", CLICKHOUSE_DLQ_SUBJECT)
        dlq = EventEnvelope.model_validate_json(raw_dlq.data)
        payload = DeadLetterPayload.model_validate(dlq.payload)
        assert payload.reason is DeadLetterReason.DELIVERY_EXHAUSTED
        assert payload.delivery_count >= 6
        assert payload.original_event == source
        assert repository.attempts >= 6

        consumer = await settled_consumer_info(admin_js, durable)
        assert consumer.num_pending == 0
        assert consumer.num_ack_pending == 0
    finally:
        if first_messages is not None:
            await first_messages.aclose()
        if second_messages is not None:
            await second_messages.aclose()
        if first_bus is not None:
            await first_bus.aclose()
        if second_bus is not None:
            await second_bus.aclose()
        await delete_test_consumer(admin_js, durable)
        await admin.close()


@pytest.mark.parametrize(
    ("field", "drift"),
    [
        ("max_deliver", {"max_deliver": 4, "backoff": [1, 5, 30]}),
        ("headers_only", {"headers_only": True}),
    ],
)
@pytest.mark.asyncio
async def test_existing_durable_policy_conflict_is_rejected_before_binding(
    field: str,
    drift: dict[str, object],
) -> None:
    assert NATS_URL is not None
    identity = uuid4().hex
    subject = f"grid.integration.conflict.{identity}"
    durable = f"integration-conflict-{identity}"
    admin = await nats.connect(NATS_URL)
    admin_js = admin.jetstream()
    bus: NatsEventBus | None = None
    try:
        bus = await NatsEventBus.connect(Settings(nats_url=NATS_URL))
        await bus.ensure_grid_stream()
        config = ConsumerConfig(
            name=durable,
            durable_name=durable,
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            max_deliver=-1,
            backoff=[1, 5, 30, 120],
            filter_subject=subject,
            replay_policy=ReplayPolicy.INSTANT,
        )
        await admin_js.add_consumer("GRID_EVENTS", config=config.evolve(**drift))

        messages = bus.messages(subject, durable)
        with pytest.raises(nats_adapter.ConsumerConfigConflict, match=field):
            await asyncio.wait_for(anext(messages), timeout=2)
    finally:
        if bus is not None:
            await bus.aclose()
        await delete_test_consumer(admin_js, durable)
        await admin.close()
