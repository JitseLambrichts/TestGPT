import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

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

    first_bus = await NatsEventBus.connect(Settings(nats_url=NATS_URL))
    try:
        await first_bus.ensure_grid_stream()
        await first_bus.publish(subject, event)
        await first_bus.publish(subject, event)
        messages = first_bus.messages(subject, durable)
        received = await asyncio.wait_for(anext(messages), timeout=5)
        assert received.event == event
        assert received.delivery_count == 1
        await received.ack()
        await messages.aclose()
    finally:
        await first_bus.aclose()

    second_bus = await NatsEventBus.connect(Settings(nats_url=NATS_URL))
    try:
        messages_after_reconnect = second_bus.messages(subject, durable)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(messages_after_reconnect), timeout=2)
        await messages_after_reconnect.aclose()
    finally:
        await second_bus.aclose()
