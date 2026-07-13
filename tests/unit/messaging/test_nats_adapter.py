import asyncio
import json
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from nats.aio.msg import Msg
from nats.errors import ConnectionClosedError
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    DiscardPolicy,
    PubAck,
    RetentionPolicy,
    StorageType,
    StreamConfig,
    StreamInfo,
    StreamState,
)
from nats.js.errors import NotFoundError

import imbalance_pipeline.messaging.nats as nats_adapter
from imbalance_pipeline.config import Settings
from imbalance_pipeline.domain.events import EventEnvelope
from imbalance_pipeline.messaging.base import Message
from imbalance_pipeline.messaging.nats import NatsEventBus

GRID_SUBJECT = "grid.raw.elia.imbalance.v1"
ACK_REPLY = "$JS.ACK._.account.GRID_EVENTS.sink.3.10.4.1783958400000000000.0.token"


def complete_event() -> EventEnvelope:
    return EventEnvelope(
        event_id="event-001",
        event_type="elia.imbalance.observed",
        schema_version="1",
        source="elia",
        dataset="ods161",
        event_time=datetime(2026, 7, 13, 10, 1, tzinfo=UTC),
        observed_at=datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC),
        ingested_at=datetime(2026, 7, 13, 10, 1, 6, tzinfo=UTC),
        correlation_id="correlation-001",
        causation_id="source-request-001",
        quality_status="Validated",
        payload={
            "unit": "MW",
            "reading": 133.96,
            "labels": {"status": "ok", "zone": "BE"},
        },
    )


def canonical_json(event: EventEnvelope) -> bytes:
    return json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def stream_info(config: StreamConfig, *, messages: int = 0) -> StreamInfo:
    return StreamInfo(
        config=config,
        state=StreamState(
            messages=messages,
            bytes=messages * 100,
            first_seq=1 if messages else 0,
            last_seq=messages,
            consumer_count=0,
            deleted=[],
            num_deleted=0,
            lost=None,
            subjects={GRID_SUBJECT: messages} if messages else {},
        ),
        mirror=None,
        sources=None,
        cluster=None,
        did_create=False,
        created=datetime(2026, 7, 13, tzinfo=UTC),
    )


@dataclass(frozen=True)
class Published:
    subject: str
    payload: bytes
    headers: dict[str, Any] | None


@dataclass(frozen=True)
class PullRequest:
    subject: str
    durable: str | None
    stream: str | None
    config: ConsumerConfig | None


class FakePullSubscription:
    def __init__(self, deliveries: list[Msg | Exception] | None = None) -> None:
        self.deliveries = deque(deliveries or [])
        self.fetches: list[tuple[int, float | None]] = []
        self.unsubscribe_count = 0

    async def fetch(  # noqa: ASYNC109 - mirrors the nats-py pull API
        self,
        batch: int = 1,
        timeout: float | None = 5,  # noqa: ASYNC109
    ) -> list[Msg]:
        self.fetches.append((batch, timeout))
        if not self.deliveries:
            raise AssertionError("test subscription has no configured delivery")
        delivery = self.deliveries.popleft()
        if isinstance(delivery, Exception):
            raise delivery
        return [delivery]

    async def unsubscribe(self) -> None:
        self.unsubscribe_count += 1


class BlockingPullSubscription(FakePullSubscription):
    def __init__(self) -> None:
        super().__init__()
        self.fetch_started = asyncio.Event()
        self.released = asyncio.Event()

    async def fetch(  # noqa: ASYNC109 - mirrors the nats-py pull API
        self,
        batch: int = 1,
        timeout: float | None = 5,  # noqa: ASYNC109
    ) -> list[Msg]:
        self.fetches.append((batch, timeout))
        self.fetch_started.set()
        await self.released.wait()
        raise ConnectionClosedError

    async def unsubscribe(self) -> None:
        await super().unsubscribe()
        self.released.set()


class FakeJetStream:
    def __init__(
        self,
        *,
        existing_stream: StreamInfo | None = None,
        subscription: FakePullSubscription | None = None,
    ) -> None:
        self.existing_stream = existing_stream
        self.subscription = subscription or FakePullSubscription()
        self.stream_info_requests: list[str] = []
        self.added_streams: list[StreamConfig] = []
        self.updated_streams: list[StreamConfig] = []
        self.deleted_streams: list[str] = []
        self.published: list[Published] = []
        self.pull_requests: list[PullRequest] = []

    async def stream_info(self, name: str) -> StreamInfo:
        self.stream_info_requests.append(name)
        if self.existing_stream is None:
            raise NotFoundError(code=404, description="stream not found")
        return self.existing_stream

    async def add_stream(self, config: StreamConfig) -> StreamInfo:
        self.added_streams.append(config)
        self.existing_stream = stream_info(config)
        return self.existing_stream

    async def update_stream(self, config: StreamConfig) -> StreamInfo:
        self.updated_streams.append(config)
        assert self.existing_stream is not None
        self.existing_stream = StreamInfo(
            config=config,
            state=self.existing_stream.state,
            mirror=self.existing_stream.mirror,
            sources=self.existing_stream.sources,
            cluster=self.existing_stream.cluster,
            did_create=False,
            created=self.existing_stream.created,
        )
        return self.existing_stream

    async def delete_stream(self, name: str) -> bool:
        self.deleted_streams.append(name)
        self.existing_stream = None
        return True

    async def publish(
        self,
        subject: str,
        payload: bytes = b"",
        *,
        headers: dict[str, Any] | None = None,
    ) -> PubAck:
        self.published.append(Published(subject, payload, headers))
        return PubAck(stream="GRID_EVENTS", seq=len(self.published), domain=None, duplicate=False)

    async def pull_subscribe(
        self,
        subject: str,
        durable: str | None = None,
        stream: str | None = None,
        config: ConsumerConfig | None = None,
    ) -> FakePullSubscription:
        self.pull_requests.append(PullRequest(subject, durable, stream, config))
        return self.subscription


class FakeNatsConnection:
    def __init__(self, jetstream: FakeJetStream) -> None:
        self._jetstream = jetstream
        self.jetstream_calls = 0
        self.close_count = 0

    def jetstream(self) -> FakeJetStream:
        self.jetstream_calls += 1
        return self._jetstream

    async def close(self) -> None:
        self.close_count += 1


@dataclass(frozen=True)
class AckFrame:
    subject: str
    payload: bytes
    reply: str
    headers: dict[str, str] | None


class FakeRawNatsClient:
    def __init__(self) -> None:
        self.frames: list[AckFrame] = []

    async def publish(
        self,
        subject: str,
        payload: bytes = b"",
        reply: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.frames.append(AckFrame(subject, payload, reply, headers))


def raw_message(
    event: EventEnvelope,
    client: FakeRawNatsClient,
    *,
    delivery_count: int = 1,
) -> Msg:
    metadata = Msg.Metadata(
        sequence=Msg.Metadata.SequencePair(consumer=4, stream=10),
        num_pending=0,
        num_delivered=delivery_count,
        timestamp=datetime(2026, 7, 13, 10, 1, 7, tzinfo=UTC),
        stream="GRID_EVENTS",
        consumer="sink",
        domain="",
    )
    return Msg(
        _client=client,  # type: ignore[arg-type]
        subject=GRID_SUBJECT,
        reply=ACK_REPLY,
        data=canonical_json(event),
        headers={"Nats-Msg-Id": event.event_id},
        _metadata=metadata,
        _ackd=False,
        _sid=42,
    )


def make_bus(
    jetstream: FakeJetStream,
) -> tuple[NatsEventBus, FakeNatsConnection]:
    connection = FakeNatsConnection(jetstream)
    return NatsEventBus(connection, jetstream), connection  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_connect_uses_settings_url_and_builds_a_working_jetstream_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jetstream = FakeJetStream()
    connection = FakeNatsConnection(jetstream)
    connected_urls: list[str] = []

    async def connect(url: str) -> FakeNatsConnection:
        connected_urls.append(url)
        return connection

    monkeypatch.setattr(nats_adapter.nats, "connect", connect)

    bus = await NatsEventBus.connect(Settings(nats_url="nats://broker.internal:4222"))
    await bus.publish(GRID_SUBJECT, complete_event())
    await bus.aclose()

    assert connected_urls == ["nats://broker.internal:4222"]
    assert connection.jetstream_calls == 1
    assert connection.close_count == 1
    assert EventEnvelope.model_validate_json(jetstream.published[0].payload) == complete_event()


@pytest.mark.asyncio
async def test_ensure_grid_stream_is_idempotent_with_required_configuration() -> None:
    jetstream = FakeJetStream()
    bus, _ = make_bus(jetstream)

    await bus.ensure_grid_stream()
    await bus.ensure_grid_stream()

    assert len(jetstream.added_streams) == 1
    assert jetstream.updated_streams == []
    assert jetstream.deleted_streams == []
    config = jetstream.added_streams[0]
    assert config.name == "GRID_EVENTS"
    assert config.subjects == ["grid.>"]
    assert config.retention is RetentionPolicy.LIMITS
    assert config.storage is StorageType.FILE
    assert config.max_age == 14 * 24 * 60 * 60
    assert config.duplicate_window == 2 * 60 * 60
    assert config.discard is DiscardPolicy.OLD


@pytest.mark.asyncio
async def test_ensure_grid_stream_reconciles_compatible_drift_without_replacing_data() -> None:
    existing_config = StreamConfig(
        name="GRID_EVENTS",
        subjects=["grid.raw.>"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.FILE,
        max_age=60,
        duplicate_window=30,
        discard=DiscardPolicy.NEW,
        max_bytes=1_000_000,
        deny_delete=True,
    )
    jetstream = FakeJetStream(existing_stream=stream_info(existing_config, messages=17))
    bus, _ = make_bus(jetstream)

    await bus.ensure_grid_stream()

    assert jetstream.added_streams == []
    assert jetstream.deleted_streams == []
    assert len(jetstream.updated_streams) == 1
    updated = jetstream.updated_streams[0]
    assert updated.subjects == ["grid.>"]
    assert updated.max_age == 14 * 24 * 60 * 60
    assert updated.duplicate_window == 2 * 60 * 60
    assert updated.discard is DiscardPolicy.OLD
    assert updated.max_bytes == 1_000_000
    assert updated.deny_delete is True
    assert jetstream.existing_stream is not None
    assert jetstream.existing_stream.state.messages == 17


@pytest.mark.asyncio
async def test_ensure_grid_stream_rejects_storage_drift_instead_of_deleting_data() -> None:
    existing = StreamConfig(
        name="GRID_EVENTS",
        subjects=["grid.>"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.MEMORY,
        max_age=14 * 24 * 60 * 60,
        duplicate_window=2 * 60 * 60,
        discard=DiscardPolicy.OLD,
    )
    jetstream = FakeJetStream(existing_stream=stream_info(existing, messages=17))
    bus, _ = make_bus(jetstream)

    with pytest.raises(RuntimeError, match="storage"):
        await bus.ensure_grid_stream()

    assert jetstream.updated_streams == []
    assert jetstream.deleted_streams == []
    assert jetstream.existing_stream is not None
    assert jetstream.existing_stream.state.messages == 17


@pytest.mark.asyncio
async def test_publish_sends_canonical_envelope_json_with_exact_message_id() -> None:
    jetstream = FakeJetStream()
    bus, _ = make_bus(jetstream)
    event = complete_event()

    await bus.publish(GRID_SUBJECT, event)

    assert jetstream.published == [
        Published(
            subject=GRID_SUBJECT,
            payload=canonical_json(event),
            headers={"Nats-Msg-Id": event.event_id},
        )
    ]
    assert EventEnvelope.model_validate_json(jetstream.published[0].payload) == event


@pytest.mark.asyncio
async def test_messages_configures_durable_pull_delivery_and_exposes_ack_metadata() -> None:
    event = complete_event()
    raw_client = FakeRawNatsClient()
    subscription = FakePullSubscription([raw_message(event, raw_client, delivery_count=3)])
    jetstream = FakeJetStream(subscription=subscription)
    bus, _ = make_bus(jetstream)

    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")
    message = await anext(messages)

    assert isinstance(message, Message)
    assert message.event == event
    assert message.delivery_count == 3
    assert len(jetstream.pull_requests) == 1
    request = jetstream.pull_requests[0]
    assert request.subject == GRID_SUBJECT
    assert request.durable == "clickhouse-sink"
    assert request.stream == "GRID_EVENTS"
    assert request.config is not None
    assert request.config.durable_name == "clickhouse-sink"
    assert request.config.filter_subject == GRID_SUBJECT
    assert request.config.ack_policy is AckPolicy.EXPLICIT
    assert request.config.max_deliver == 5
    assert request.config.backoff == [1, 5, 30, 120]

    await message.ack()
    await messages.aclose()

    assert raw_client.frames == [AckFrame(ACK_REPLY, b"", "", None)]
    assert subscription.unsubscribe_count == 1


@pytest.mark.asyncio
async def test_message_nak_forwards_the_requested_delay_to_the_nats_wire_format() -> None:
    event = complete_event()
    raw_client = FakeRawNatsClient()
    subscription = FakePullSubscription([raw_message(event, raw_client)])
    jetstream = FakeJetStream(subscription=subscription)
    bus, _ = make_bus(jetstream)

    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")
    message = await anext(messages)
    await message.nak(delay_seconds=2.5)
    await messages.aclose()

    assert raw_client.frames == [AckFrame(ACK_REPLY, b'-NAK {"delay": 2500000000}', "", None)]


@pytest.mark.asyncio
async def test_messages_propagates_transport_failures_and_unsubscribes() -> None:
    subscription = FakePullSubscription([ConnectionClosedError()])
    jetstream = FakeJetStream(subscription=subscription)
    bus, _ = make_bus(jetstream)
    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")

    with pytest.raises(ConnectionClosedError):
        await anext(messages)

    assert subscription.unsubscribe_count == 1


@pytest.mark.asyncio
async def test_cancelling_message_iteration_unsubscribes_cleanly() -> None:
    subscription = BlockingPullSubscription()
    jetstream = FakeJetStream(subscription=subscription)
    bus, _ = make_bus(jetstream)
    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")
    pending = asyncio.create_task(anext(messages))
    await subscription.fetch_started.wait()

    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    assert subscription.unsubscribe_count == 1


@pytest.mark.asyncio
async def test_closing_bus_stops_active_message_iteration_and_closes_once() -> None:
    subscription = BlockingPullSubscription()
    jetstream = FakeJetStream(subscription=subscription)
    bus, connection = make_bus(jetstream)
    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")
    pending = asyncio.create_task(anext(messages))
    await subscription.fetch_started.wait()

    await bus.aclose()
    await bus.aclose()

    with pytest.raises(StopAsyncIteration):
        await pending
    assert subscription.unsubscribe_count == 1
    assert connection.close_count == 1
