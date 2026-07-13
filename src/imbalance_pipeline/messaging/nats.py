import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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
CONSUMER_MAX_DELIVER: Final = -1
FETCH_TIMEOUT_SECONDS: Final = 1.0


class ConsumerConfigConflict(RuntimeError):
    """A durable consumer exists with configuration this service cannot use safely."""


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
    def __init__(
        self,
        connection: NatsClient,
        jetstream: JetStreamContext,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._connection = connection
        self._jetstream = jetstream
        self._clock = clock or _utc_now
        self._shutdown_requested = asyncio.Event()
        self._close_completed = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()
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
        if "max_age" in changed and _tightens_max_age(existing.max_age):
            await self._verify_max_age_tightening_is_safe(info)
        await self._jetstream.update_stream(config=existing.evolve(**changed))

    async def _verify_max_age_tightening_is_safe(self, info: api.StreamInfo) -> None:
        if info.state.messages == 0:
            return
        if info.state.first_seq <= 0:
            raise RuntimeError(
                f"cannot tighten {GRID_STREAM} max_age without a valid oldest sequence"
            )

        oldest = await self._jetstream.get_msg(GRID_STREAM, seq=info.state.first_seq)
        if oldest.time is None or oldest.time.tzinfo is None or oldest.time.utcoffset() is None:
            raise RuntimeError(f"cannot tighten {GRID_STREAM} max_age without an oldest timestamp")
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise RuntimeError("NATS stream reconciliation clock must be timezone-aware")
        cutoff = now.astimezone(UTC) - timedelta(seconds=STREAM_MAX_AGE_SECONDS)
        if oldest.time.astimezone(UTC) < cutoff:
            raise RuntimeError(
                f"refusing to tighten {GRID_STREAM} max_age because retained data is older "
                "than fourteen days"
            )

    async def publish(self, subject: str, event: EventEnvelope) -> None:
        await self._jetstream.publish(
            subject,
            _canonical_event_json(event),
            headers={api.Header.MSG_ID.value: event.event_id},
        )

    async def messages(self, subject: str, durable: str) -> AsyncIterator[NatsMessage]:
        subscription: JetStreamContext.PullSubscription | None = None
        unregistered_subscription = False
        primary_error: BaseException | None = None
        try:
            async with self._lifecycle_lock:
                if self._shutdown_requested.is_set():
                    return

            config = _consumer_config(subject, durable)
            try:
                try:
                    info = await self._jetstream.consumer_info(GRID_STREAM, durable)
                except NotFoundError:
                    if self._shutdown_requested.is_set():
                        return
                    subscription = await self._jetstream.pull_subscribe(
                        subject,
                        durable=durable,
                        stream=GRID_STREAM,
                        config=config,
                    )
                    unregistered_subscription = True
                else:
                    if self._shutdown_requested.is_set():
                        return
                    _validate_consumer_policy(info.config, config, durable)
                    subscription = await self._jetstream.pull_subscribe_bind(
                        consumer=durable,
                        stream=GRID_STREAM,
                    )
                    unregistered_subscription = True

                actual = await self._jetstream.consumer_info(GRID_STREAM, durable)
                _validate_consumer_policy(actual.config, config, durable)
            except (BadSubscriptionError, ConnectionClosedError):
                if self._shutdown_requested.is_set():
                    return
                raise

            async with self._lifecycle_lock:
                close_after_setup = self._shutdown_requested.is_set()
                if not close_after_setup:
                    self._subscriptions.add(subscription)
                    unregistered_subscription = False
            if close_after_setup:
                return

            while not self._shutdown_requested.is_set():
                try:
                    raw_messages = await subscription.fetch(
                        batch=1,
                        timeout=FETCH_TIMEOUT_SECONDS,
                    )
                except NatsTimeoutError:
                    continue
                except (BadSubscriptionError, ConnectionClosedError):
                    if self._shutdown_requested.is_set():
                        return
                    raise
                for raw_message in raw_messages:
                    yield NatsMessage(raw_message)
        except GeneratorExit:
            raise
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup = unregistered_subscription
            if subscription is not None:
                async with self._lifecycle_lock:
                    if subscription in self._subscriptions:
                        self._subscriptions.discard(subscription)
                        cleanup = True
            if cleanup and subscription is not None:
                try:
                    await subscription.unsubscribe()
                except BaseException as cleanup_error:
                    if (
                        isinstance(
                            cleanup_error,
                            (BadSubscriptionError, ConnectionClosedError),
                        )
                        and self._shutdown_requested.is_set()
                    ):
                        pass
                    else:
                        async with self._lifecycle_lock:
                            if not self._close_completed.is_set():
                                self._subscriptions.add(subscription)
                        if primary_error is not None:
                            primary_error.add_note(
                                f"NATS unsubscribe also failed: {cleanup_error!r}"
                            )
                        else:
                            raise

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._close_completed.is_set():
                return

            async with self._lifecycle_lock:
                self._shutdown_requested.set()
                subscriptions = tuple(self._subscriptions)
                self._subscriptions.clear()

            errors: list[Exception] = []
            failed_subscriptions: list[JetStreamContext.PullSubscription] = []
            primary_error: BaseException | None = None
            connection_closed = False
            try:
                for subscription in subscriptions:
                    try:
                        await subscription.unsubscribe()
                    except (BadSubscriptionError, ConnectionClosedError):
                        pass
                    except Exception as exc:
                        errors.append(exc)
                        failed_subscriptions.append(subscription)
            except BaseException as exc:
                primary_error = exc
            finally:
                try:
                    await self._connection.close()
                except Exception as exc:
                    if primary_error is not None:
                        primary_error.add_note(f"NATS connection close also failed: {exc!r}")
                    else:
                        errors.append(exc)
                else:
                    connection_closed = True
                    self._close_completed.set()

            if failed_subscriptions and not connection_closed:
                async with self._lifecycle_lock:
                    self._subscriptions.update(failed_subscriptions)

            if primary_error is not None:
                raise primary_error

            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise ExceptionGroup("failed to close NATS resources", errors)


def _consumer_config(subject: str, durable: str) -> api.ConsumerConfig:
    return api.ConsumerConfig(
        durable_name=durable,
        deliver_policy=api.DeliverPolicy.ALL,
        ack_policy=api.AckPolicy.EXPLICIT,
        max_deliver=CONSUMER_MAX_DELIVER,
        backoff=list(CONSUMER_BACKOFF_SECONDS),
        filter_subject=subject,
        replay_policy=api.ReplayPolicy.INSTANT,
    )


def _consumer_policy_conflicts(
    existing: api.ConsumerConfig,
    desired: api.ConsumerConfig,
) -> list[str]:
    conflicts: list[str] = []
    for field_name in (
        "filter_subject",
        "ack_policy",
        "max_deliver",
        "deliver_policy",
        "replay_policy",
        "deliver_subject",
    ):
        if getattr(existing, field_name) != getattr(desired, field_name):
            conflicts.append(field_name)
    if tuple(existing.backoff or ()) != tuple(desired.backoff or ()):
        conflicts.append("backoff")
    if bool(existing.headers_only) != bool(desired.headers_only):
        conflicts.append("headers_only")
    return conflicts


def _validate_consumer_policy(
    existing: api.ConsumerConfig,
    desired: api.ConsumerConfig,
    durable: str,
) -> None:
    conflicts = _consumer_policy_conflicts(existing, desired)
    if conflicts:
        fields = ", ".join(conflicts)
        raise ConsumerConfigConflict(
            f"durable consumer {durable!r} conflicts with managed fields: {fields}"
        )


def _tightens_max_age(existing_max_age: float | None) -> bool:
    return (
        existing_max_age is None
        or existing_max_age <= 0
        or existing_max_age > STREAM_MAX_AGE_SECONDS
    )


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


def _utc_now() -> datetime:
    return datetime.now(UTC)
