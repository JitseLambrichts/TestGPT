from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from imbalance_pipeline.domain.events import EventEnvelope


@runtime_checkable
class Message(Protocol):
    event: EventEnvelope
    delivery_count: int

    async def ack(self) -> None: ...

    async def nak(self, delay_seconds: float) -> None: ...


@runtime_checkable
class EventBus(Protocol):
    async def publish(self, subject: str, event: EventEnvelope) -> None: ...

    def messages(self, subject: str, durable: str) -> AsyncIterator[Message]: ...
