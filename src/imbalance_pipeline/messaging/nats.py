import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Final

import nats
from nats.aio.client import Client as NatsClient
from nats.aio.msg import Msg
from nats.errors import BadSubscriptionError, ConnectionClosedError
from nats.errors import TimeoutError as NatsTimeoutError
from nats.js import api
from nats.js.client import JetStreamContext
from nats.js.errors import NotFoundError

from imbalance_pipeline.config import Settings
from imbalance_pipeline.domain.events import EventEnvelope

GRID_STREAM: Final = "GRID_EVENTS"
GRID_SUBJECTS: Final = ("grid.>",)
STREAM_MAX_AGE_SECONDS: Final = 14 * 24 * 60 * 60
DUPLICATE_WINDOW_SECONDS: Final = 2 * 60 * 60
CONSUMER_BACKOFF_SECONDS: Final = (1, 5, 30, 120)
CONSUMER_MAX_DELIVER: Final = 5
FETCH_TIMEOUT_SECONDS: Final = 1.0


@dataclass(frozen=True, slots=True)
class NatsMessage:
    _raw: Msg
    event: EventEnvelope = field(init=False)
    delivery_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event", EventEnvelope.model_validate_json(self._raw.data))
        object.__setattr__(self, "delivery_count", self._raw.metadata.num_delivered)

    async def ack(self) -> None:
        await self._raw.ack()

    async def nak(self, delay_seconds: float) -> None:
        await self._raw.nak(delay=delay_seconds)


class NatsEventBus:
    def __init__(self, connection: NatsClient, jetstream: JetStreamContext) -> None:
        self._connection = connection
        self._jetstream = jetstream
        self._closed = asyncio.Event()
        self._close_lock = asyncio.Lock()
        self._subscriptions: set[JetStreamContext.PullSubscription] = set()

    @classmethod
    async def connect(cls, settings: Settings) -> "NatsEventBus":
        connection = await nats.connect(settings.nats_url)
        return cls(connection, connection.jetstream())

    async def ensure_grid_stream(self) -> None:
        desired = _grid_stream_config()
        try:
            info = await self._jetstream.stream_info(GRID_STREAM)
        except NotFoundError:
            await self._jetstream.add_stream(config=desired)
            return

        existing = info.config
        if existing.name != GRID_STREAM:
            raise RuntimeError(f"stream {GRID_STREAM} returned incompatible name {existing.name!r}")
        if existing.storage != api.StorageType.FILE:
            raise RuntimeError(
                f"stream {GRID_STREAM} has incompatible storage {existing.storage!r}; "
                "refusing to replace a stream that may contain data"
            )
        if existing.retention != api.RetentionPolicy.LIMITS:
            raise RuntimeError(
                f"stream {GRID_STREAM} has incompatible retention {existing.retention!r}; "
                "refusing to replace a stream that may contain data"
            )

        updates = {
            "subjects": list(GRID_SUBJECTS),
            "max_age": STREAM_MAX_AGE_SECONDS,
            "duplicate_window": DUPLICATE_WINDOW_SECONDS,
            "discard": api.DiscardPolicy.OLD,
        }
        changed = {
            name: value for name, value in updates.items() if getattr(existing, name) != value
        }
        if not changed:
            return
        if existing.sealed:
            raise RuntimeError(
                f"stream {GRID_STREAM} is sealed and cannot be reconciled without replacement"
            )
        await self._jetstream.update_stream(config=existing.evolve(**changed))

    async def publish(self, subject: str, event: EventEnvelope) -> None:
        await self._jetstream.publish(
            subject,
            _canonical_event_json(event),
            headers={api.Header.MSG_ID.value: event.event_id},
        )

    async def messages(self, subject: str, durable: str) -> AsyncIterator[NatsMessage]:
        if self._closed.is_set():
            return

        config = api.ConsumerConfig(
            durable_name=durable,
            ack_policy=api.AckPolicy.EXPLICIT,
            max_deliver=CONSUMER_MAX_DELIVER,
            backoff=list(CONSUMER_BACKOFF_SECONDS),
            filter_subject=subject,
        )
        subscription = await self._jetstream.pull_subscribe(
            subject,
            durable=durable,
            stream=GRID_STREAM,
            config=config,
        )
        if self._closed.is_set():
            await subscription.unsubscribe()
            return

        self._subscriptions.add(subscription)
        try:
            while not self._closed.is_set():
                try:
                    raw_messages = await subscription.fetch(
                        batch=1,
                        timeout=FETCH_TIMEOUT_SECONDS,
                    )
                except NatsTimeoutError:
                    continue
                except (BadSubscriptionError, ConnectionClosedError):
                    if self._closed.is_set():
                        return
                    raise
                for raw_message in raw_messages:
                    yield NatsMessage(raw_message)
        finally:
            if subscription in self._subscriptions:
                self._subscriptions.discard(subscription)
                await subscription.unsubscribe()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            subscriptions = tuple(self._subscriptions)
            self._subscriptions.clear()
            errors: list[Exception] = []
            try:
                for subscription in subscriptions:
                    try:
                        await subscription.unsubscribe()
                    except Exception as exc:
                        errors.append(exc)
            finally:
                try:
                    await self._connection.close()
                except Exception as exc:
                    errors.append(exc)

            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise ExceptionGroup("failed to close NATS resources", errors)


def _grid_stream_config() -> api.StreamConfig:
    return api.StreamConfig(
        name=GRID_STREAM,
        subjects=list(GRID_SUBJECTS),
        retention=api.RetentionPolicy.LIMITS,
        storage=api.StorageType.FILE,
        max_age=STREAM_MAX_AGE_SECONDS,
        duplicate_window=DUPLICATE_WINDOW_SECONDS,
        discard=api.DiscardPolicy.OLD,
    )


def _canonical_event_json(event: EventEnvelope) -> bytes:
    return json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
