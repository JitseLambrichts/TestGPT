import asyncio
import os
from contextlib import suppress
from datetime import UTC, datetime
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
from imbalance_pipeline.config import Settings
from imbalance_pipeline.domain.events import EventEnvelope
from imbalance_pipeline.messaging.nats import NatsEventBus

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
            max_deliver=5,
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
