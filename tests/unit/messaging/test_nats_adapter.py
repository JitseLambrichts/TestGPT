import asyncio
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from nats.aio.msg import Msg
from nats.errors import ConnectionClosedError
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    ConsumerInfo,
    DeliverPolicy,
    DiscardPolicy,
    PubAck,
    RawStreamMsg,
    ReplayPolicy,
    RetentionPolicy,
    SequenceInfo,
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
NOW = datetime(2026, 7, 20, 12, tzinfo=UTC)


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


def stored_message(timestamp: datetime | None) -> RawStreamMsg:
    return RawStreamMsg(
        subject=GRID_SUBJECT,
        seq=1,
        data=b"stored-event",
        hdrs=None,
        headers={"Nats-Msg-Id": "stored-event-001"},
        stream="GRID_EVENTS",
        time=timestamp,
    )


def managed_consumer_config(
    durable: str = "clickhouse-sink",
    subject: str = GRID_SUBJECT,
) -> ConsumerConfig:
    return ConsumerConfig(
        name=durable,
        durable_name=durable,
        deliver_policy=DeliverPolicy.ALL,
        ack_policy=AckPolicy.EXPLICIT,
        max_deliver=-1,
        backoff=[1, 5, 30, 120],
        filter_subject=subject,
        replay_policy=ReplayPolicy.INSTANT,
    )


def consumer_info(config: ConsumerConfig) -> ConsumerInfo:
    durable = config.durable_name or config.name or "clickhouse-sink"
    return ConsumerInfo(
        name=durable,
        stream_name="GRID_EVENTS",
        config=config,
        created=datetime(2026, 7, 13, tzinfo=UTC),
        delivered=SequenceInfo(consumer_seq=0, stream_seq=0, last_active=None),
        ack_floor=SequenceInfo(consumer_seq=0, stream_seq=0, last_active=None),
        num_ack_pending=0,
        num_redelivered=0,
        num_waiting=0,
        num_pending=0,
        cluster=None,
        push_bound=False,
        paused=False,
        pause_remaining=None,
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


@dataclass(frozen=True)
class BindRequest:
    consumer: str | None
    stream: str | None


class FakePullSubscription:
    def __init__(
        self,
        deliveries: list[Msg | Exception] | None = None,
        *,
        unsubscribe_errors: list[BaseException] | None = None,
    ) -> None:
        self.deliveries = deque(deliveries or [])
        self.unsubscribe_errors = deque(unsubscribe_errors or [])
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
        if self.unsubscribe_errors:
            raise self.unsubscribe_errors.popleft()


class BlockingPullSubscription(FakePullSubscription):
    def __init__(self, *, unsubscribe_errors: list[BaseException] | None = None) -> None:
        super().__init__(unsubscribe_errors=unsubscribe_errors)
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
        self.released.set()
        await super().unsubscribe()


class FakeJetStream:
    def __init__(
        self,
        *,
        existing_stream: StreamInfo | None = None,
        oldest_message: RawStreamMsg | None = None,
        consumers: dict[str, ConsumerInfo] | None = None,
        consumer_info_results: list[ConsumerInfo | BaseException] | None = None,
        subscription: FakePullSubscription | None = None,
    ) -> None:
        self.existing_stream = existing_stream
        self.oldest_message = oldest_message
        self.consumers = dict(consumers or {})
        self.consumer_info_results = deque(consumer_info_results or [])
        self.subscription = subscription or FakePullSubscription()
        self.stream_info_requests: list[str] = []
        self.get_msg_requests: list[tuple[str, int | None]] = []
        self.consumer_info_requests: list[tuple[str, str]] = []
        self.added_streams: list[StreamConfig] = []
        self.updated_streams: list[StreamConfig] = []
        self.deleted_streams: list[str] = []
        self.deleted_consumers: list[tuple[str, str]] = []
        self.published: list[Published] = []
        self.pull_requests: list[PullRequest] = []
        self.bind_requests: list[BindRequest] = []

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

    async def get_msg(
        self,
        stream_name: str,
        seq: int | None = None,
    ) -> RawStreamMsg:
        self.get_msg_requests.append((stream_name, seq))
        if self.oldest_message is None:
            raise AssertionError("test stream has no configured oldest message")
        return self.oldest_message

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

    async def consumer_info(self, stream: str, consumer: str) -> ConsumerInfo:
        self.consumer_info_requests.append((stream, consumer))
        if self.consumer_info_results:
            result = self.consumer_info_results.popleft()
            if isinstance(result, BaseException):
                raise result
            return result
        if consumer not in self.consumers:
            raise NotFoundError(code=404, description="consumer not found")
        return self.consumers[consumer]

    async def delete_consumer(self, stream: str, consumer: str) -> bool:
        self.deleted_consumers.append((stream, consumer))
        self.consumers.pop(consumer, None)
        return True

    async def pull_subscribe(
        self,
        subject: str,
        durable: str | None = None,
        stream: str | None = None,
        config: ConsumerConfig | None = None,
    ) -> FakePullSubscription:
        self.pull_requests.append(PullRequest(subject, durable, stream, config))
        if durable is not None and config is not None:
            self.consumers[durable] = consumer_info(config)
        return self.subscription

    async def pull_subscribe_bind(
        self,
        consumer: str | None = None,
        stream: str | None = None,
    ) -> FakePullSubscription:
        self.bind_requests.append(BindRequest(consumer, stream))
        return self.subscription


class CloseDuringSubscribeJetStream(FakeJetStream):
    def __init__(self) -> None:
        super().__init__()
        self.pull_started = asyncio.Event()
        self.connection_closed = asyncio.Event()

    async def pull_subscribe(
        self,
        subject: str,
        durable: str | None = None,
        stream: str | None = None,
        config: ConsumerConfig | None = None,
    ) -> FakePullSubscription:
        self.pull_requests.append(PullRequest(subject, durable, stream, config))
        self.pull_started.set()
        await self.connection_closed.wait()
        raise ConnectionClosedError


class FakeNatsConnection:
    def __init__(
        self,
        jetstream: FakeJetStream,
        *,
        close_errors: list[Exception] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._jetstream = jetstream
        self._close_errors = deque(close_errors or [])
        self._on_close = on_close
        self.jetstream_calls = 0
        self.close_count = 0

    def jetstream(self) -> FakeJetStream:
        self.jetstream_calls += 1
        return self._jetstream

    async def close(self) -> None:
        self.close_count += 1
        if self._on_close is not None:
            self._on_close()
        if self._close_errors:
            raise self._close_errors.popleft()


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
    *,
    clock: Callable[[], datetime] | None = None,
    close_errors: list[Exception] | None = None,
    on_close: Callable[[], None] | None = None,
) -> tuple[NatsEventBus, FakeNatsConnection]:
    connection = FakeNatsConnection(
        jetstream,
        close_errors=close_errors,
        on_close=on_close,
    )
    if clock is None:
        bus = NatsEventBus(connection, jetstream)  # type: ignore[arg-type]
    else:
        bus = NatsEventBus(connection, jetstream, clock=clock)  # type: ignore[call-arg,arg-type]
    return bus, connection


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
    assert jetstream.get_msg_requests == []


@pytest.mark.asyncio
async def test_ensure_grid_stream_can_tighten_an_empty_unlimited_stream() -> None:
    existing = StreamConfig(
        name="GRID_EVENTS",
        subjects=["grid.>"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.FILE,
        max_age=0,
        duplicate_window=2 * 60 * 60,
        discard=DiscardPolicy.OLD,
    )
    jetstream = FakeJetStream(existing_stream=stream_info(existing, messages=0))
    bus, _ = make_bus(jetstream, clock=lambda: NOW)

    await bus.ensure_grid_stream()

    assert len(jetstream.updated_streams) == 1
    assert jetstream.updated_streams[0].max_age == 14 * 24 * 60 * 60
    assert jetstream.get_msg_requests == []
    assert jetstream.deleted_streams == []


@pytest.mark.parametrize("existing_max_age", [0, 30 * 24 * 60 * 60])
@pytest.mark.asyncio
async def test_ensure_grid_stream_rejects_tightening_when_old_data_would_be_deleted(
    existing_max_age: int,
) -> None:
    existing = StreamConfig(
        name="GRID_EVENTS",
        subjects=["grid.>"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.FILE,
        max_age=existing_max_age,
        duplicate_window=2 * 60 * 60,
        discard=DiscardPolicy.OLD,
    )
    jetstream = FakeJetStream(
        existing_stream=stream_info(existing, messages=1),
        oldest_message=stored_message(NOW - timedelta(days=14, seconds=1)),
    )
    bus, _ = make_bus(jetstream, clock=lambda: NOW)

    with pytest.raises(RuntimeError, match="max_age"):
        await bus.ensure_grid_stream()

    assert jetstream.get_msg_requests == [("GRID_EVENTS", 1)]
    assert jetstream.updated_streams == []
    assert jetstream.deleted_streams == []
    assert jetstream.existing_stream is not None
    assert jetstream.existing_stream.state.messages == 1


@pytest.mark.asyncio
async def test_ensure_grid_stream_can_tighten_when_oldest_data_is_within_fourteen_days() -> None:
    existing = StreamConfig(
        name="GRID_EVENTS",
        subjects=["grid.>"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.FILE,
        max_age=30 * 24 * 60 * 60,
        duplicate_window=2 * 60 * 60,
        discard=DiscardPolicy.OLD,
    )
    jetstream = FakeJetStream(
        existing_stream=stream_info(existing, messages=3),
        oldest_message=stored_message(NOW - timedelta(days=13, hours=23)),
    )
    bus, _ = make_bus(jetstream, clock=lambda: NOW)

    await bus.ensure_grid_stream()

    assert jetstream.get_msg_requests == [("GRID_EVENTS", 1)]
    assert len(jetstream.updated_streams) == 1
    assert jetstream.updated_streams[0].max_age == 14 * 24 * 60 * 60
    assert jetstream.existing_stream is not None
    assert jetstream.existing_stream.state.messages == 3


@pytest.mark.asyncio
async def test_ensure_grid_stream_rejects_tightening_without_an_oldest_timestamp() -> None:
    existing = StreamConfig(
        name="GRID_EVENTS",
        subjects=["grid.>"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.FILE,
        max_age=0,
        duplicate_window=2 * 60 * 60,
        discard=DiscardPolicy.OLD,
    )
    jetstream = FakeJetStream(
        existing_stream=stream_info(existing, messages=1),
        oldest_message=stored_message(None),
    )
    bus, _ = make_bus(jetstream, clock=lambda: NOW)

    with pytest.raises(RuntimeError, match="oldest timestamp"):
        await bus.ensure_grid_stream()

    assert jetstream.updated_streams == []
    assert jetstream.deleted_streams == []


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
    assert request.config.deliver_policy is DeliverPolicy.ALL
    assert request.config.ack_policy is AckPolicy.EXPLICIT
    assert request.config.max_deliver == -1
    assert request.config.backoff == [1, 5, 30, 120]
    assert request.config.replay_policy is ReplayPolicy.INSTANT

    await message.ack()
    await messages.aclose()

    assert jetstream.consumer_info_requests == [
        ("GRID_EVENTS", "clickhouse-sink"),
        ("GRID_EVENTS", "clickhouse-sink"),
    ]
    assert raw_client.frames == [AckFrame(ACK_REPLY, b"", "", None)]
    assert subscription.unsubscribe_count == 1


@pytest.mark.parametrize("headers_only", [None, False])
@pytest.mark.asyncio
async def test_messages_binds_an_existing_compatible_durable_without_recreating_it(
    headers_only: bool | None,
) -> None:
    event = complete_event()
    raw_client = FakeRawNatsClient()
    subscription = FakePullSubscription([raw_message(event, raw_client)])
    existing = consumer_info(managed_consumer_config().evolve(headers_only=headers_only))
    jetstream = FakeJetStream(
        consumers={"clickhouse-sink": existing},
        subscription=subscription,
    )
    bus, _ = make_bus(jetstream)

    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")
    message = await anext(messages)
    await messages.aclose()

    assert message.event == event
    assert jetstream.consumer_info_requests == [
        ("GRID_EVENTS", "clickhouse-sink"),
        ("GRID_EVENTS", "clickhouse-sink"),
    ]
    assert jetstream.pull_requests == []
    assert jetstream.bind_requests == [BindRequest("clickhouse-sink", "GRID_EVENTS")]


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("filter_subject", "grid.raw.elia.load.v1"),
        ("ack_policy", AckPolicy.NONE),
        ("max_deliver", 4),
        ("backoff", [1, 5, 30]),
        ("deliver_policy", DeliverPolicy.NEW),
        ("replay_policy", ReplayPolicy.ORIGINAL),
        ("deliver_subject", "_INBOX.existing-push-consumer"),
        ("headers_only", True),
    ],
)
@pytest.mark.asyncio
async def test_messages_rejects_existing_durable_policy_drift_before_binding(
    field: str,
    drifted_value: object,
) -> None:
    event = complete_event()
    raw_client = FakeRawNatsClient()
    subscription = FakePullSubscription([raw_message(event, raw_client)])
    drifted = managed_consumer_config().evolve(**{field: drifted_value})
    jetstream = FakeJetStream(
        consumers={"clickhouse-sink": consumer_info(drifted)},
        subscription=subscription,
    )
    bus, _ = make_bus(jetstream)
    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")

    with pytest.raises(nats_adapter.ConsumerConfigConflict, match=field):
        await anext(messages)

    assert jetstream.pull_requests == []
    assert jetstream.bind_requests == []
    assert subscription.fetches == []


@pytest.mark.asyncio
async def test_messages_rejects_a_conflicting_durable_created_during_subscription_setup() -> None:
    event = complete_event()
    raw_client = FakeRawNatsClient()
    subscription = FakePullSubscription([raw_message(event, raw_client)])
    conflict = consumer_info(managed_consumer_config().evolve(max_deliver=4))
    jetstream = FakeJetStream(
        consumer_info_results=[
            NotFoundError(code=404, description="consumer not found"),
            conflict,
        ],
        subscription=subscription,
    )
    bus, _ = make_bus(jetstream)
    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")

    with pytest.raises(nats_adapter.ConsumerConfigConflict, match="max_deliver"):
        await anext(messages)

    assert jetstream.consumer_info_requests == [
        ("GRID_EVENTS", "clickhouse-sink"),
        ("GRID_EVENTS", "clickhouse-sink"),
    ]
    assert len(jetstream.pull_requests) == 1
    assert jetstream.bind_requests == []
    assert subscription.fetches == []
    assert subscription.unsubscribe_count == 1
    assert jetstream.deleted_consumers == []


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
async def test_unsubscribe_failure_does_not_replace_the_primary_transport_error() -> None:
    subscription = FakePullSubscription(
        [ConnectionClosedError()],
        unsubscribe_errors=[RuntimeError("unsubscribe failed")],
    )
    jetstream = FakeJetStream(subscription=subscription)
    bus, _ = make_bus(jetstream)
    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")

    with pytest.raises(ConnectionClosedError) as raised:
        await anext(messages)

    assert subscription.unsubscribe_count == 1
    assert any("unsubscribe failed" in note for note in raised.value.__notes__)
    await bus.aclose()
    assert subscription.unsubscribe_count == 2


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
async def test_unsubscribe_failure_does_not_replace_iteration_cancellation() -> None:
    subscription = BlockingPullSubscription(
        unsubscribe_errors=[RuntimeError("unsubscribe failed during cancellation")]
    )
    jetstream = FakeJetStream(subscription=subscription)
    bus, _ = make_bus(jetstream)
    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")
    pending = asyncio.create_task(anext(messages))
    await subscription.fetch_started.wait()

    pending.cancel()

    with pytest.raises(asyncio.CancelledError) as raised:
        await pending
    assert subscription.unsubscribe_count == 1
    assert any("unsubscribe failed during cancellation" in note for note in raised.value.__notes__)


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


@pytest.mark.asyncio
async def test_closing_bus_during_pull_subscription_setup_stops_iteration_cleanly() -> None:
    jetstream = CloseDuringSubscribeJetStream()
    bus, connection = make_bus(
        jetstream,
        on_close=jetstream.connection_closed.set,
    )
    messages = bus.messages(GRID_SUBJECT, durable="clickhouse-sink")
    pending = asyncio.create_task(anext(messages))
    await jetstream.pull_started.wait()

    await bus.aclose()

    with pytest.raises(StopAsyncIteration):
        await pending
    assert connection.close_count == 1
    assert jetstream.bind_requests == []


@pytest.mark.asyncio
async def test_connection_close_failure_can_be_retried_until_close_completes() -> None:
    jetstream = FakeJetStream()
    bus, connection = make_bus(
        jetstream,
        close_errors=[RuntimeError("connection close failed")],
    )

    with pytest.raises(RuntimeError, match="connection close failed"):
        await bus.aclose()

    await bus.aclose()
    await bus.aclose()

    assert connection.close_count == 2
