import asyncio
import importlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from prometheus_client import CollectorRegistry, Counter
from pydantic import BaseModel

from imbalance_pipeline.config import Settings, get_settings
from imbalance_pipeline.domain.events import EventEnvelope, Subject
from imbalance_pipeline.messaging.base import EventBus
from imbalance_pipeline.sources.elia import (
    EliaClient,
    normalize_imbalance,
    normalize_load,
    normalize_solar,
    normalize_wind,
)
from imbalance_pipeline.sources.weather import WeatherForecast

LOGGER = logging.getLogger(__name__)

ODS002_DATASET = "ods002"
ODS086_DATASET = "ods086"
ODS087_DATASET = "ods087"
WEATHER_DATASET = "forecast"
MAX_PAGE_RANGE = timedelta(days=1)

IMBALANCE_LOOKBACK = timedelta(minutes=5)
SUPPORTING_LOOKBACK = timedelta(minutes=30)
WEATHER_LOOKBACK = timedelta(hours=1)


class EliaSource(Protocol):
    def iter_records(
        self,
        dataset: str,
        start: datetime | None,
        end: datetime | None,
    ) -> AsyncIterator[dict[str, object]]: ...


class WeatherSource(Protocol):
    def iter_current_state(
        self,
        start: datetime,
        end: datetime,
        historical: bool,
    ) -> AsyncIterator[WeatherForecast]: ...


@dataclass(frozen=True, slots=True)
class PollIntervals:
    imbalance: float = 15.0
    load: float = 300.0
    wind: float = 300.0
    solar: float = 300.0
    weather: float = 900.0

    def __post_init__(self) -> None:
        if any(interval <= 0 for interval in self.values()):
            raise ValueError("poll intervals must be positive")

    def values(self) -> tuple[float, float, float, float, float]:
        return self.imbalance, self.load, self.wind, self.solar, self.weather


_BuildEvent = Callable[
    [Mapping[str, object]],
    tuple[BaseModel, datetime, str, str],
]
_Poll = Callable[[], Awaitable[int]]


class Ingestor:
    def __init__(
        self,
        client: EliaSource,
        bus: EventBus,
        settings: Settings,
        *,
        weather_client: WeatherSource | None = None,
        registry: CollectorRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        intervals: PollIntervals | None = None,
    ) -> None:
        self._client = client
        self._bus = bus
        self._settings = settings
        self._weather_client = weather_client
        self._clock = clock or _utc_now
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._intervals = intervals or PollIntervals()
        self.metrics_registry = registry if registry is not None else CollectorRegistry()
        self._published = Counter(
            "imbalance_ingestor_published",
            "Source records published by the imbalance ingestor.",
            ("source", "dataset"),
            registry=self.metrics_registry,
        )
        self._source_errors = Counter(
            "imbalance_ingestor_source_errors",
            "Source or publication failures encountered by the imbalance ingestor.",
            ("source", "dataset"),
            registry=self.metrics_registry,
        )

    async def poll_imbalance_once(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        return await self._poll_elia_once(
            dataset=self._settings.elia_imbalance_live_dataset,
            subject=Subject.RAW_ELIA_IMBALANCE,
            event_type="elia.imbalance.observed",
            build_event=_build_imbalance_event,
            lookback=IMBALANCE_LOOKBACK,
            start=start,
            end=end,
        )

    async def poll_load_once(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        return await self._poll_elia_once(
            dataset=ODS002_DATASET,
            subject=Subject.RAW_ELIA_LOAD,
            event_type="elia.load.observed",
            build_event=_build_load_event,
            lookback=SUPPORTING_LOOKBACK,
            start=start,
            end=end,
        )

    async def poll_wind_once(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        return await self._poll_elia_once(
            dataset=ODS086_DATASET,
            subject=Subject.RAW_ELIA_WIND,
            event_type="elia.wind.observed",
            build_event=_build_wind_event,
            lookback=SUPPORTING_LOOKBACK,
            start=start,
            end=end,
        )

    async def poll_solar_once(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        return await self._poll_elia_once(
            dataset=ODS087_DATASET,
            subject=Subject.RAW_ELIA_SOLAR,
            event_type="elia.solar.observed",
            build_event=_build_solar_event,
            lookback=SUPPORTING_LOOKBACK,
            start=start,
            end=end,
        )

    async def poll_weather_once(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        historical: bool = False,
    ) -> int:
        if self._weather_client is None:
            return 0
        window_start, window_end = self._poll_window(start, end, WEATHER_LOOKBACK)
        try:
            count = 0
            async for forecast in self._weather_client.iter_current_state(
                window_start,
                window_end,
                historical,
            ):
                await self._publish(
                    subject=Subject.RAW_WEATHER_FORECAST,
                    event_type="weather.forecast.observed",
                    source="open-meteo",
                    dataset=WEATHER_DATASET,
                    event_time=forecast.valid_time,
                    natural_key=_timestamp_key(forecast.valid_time),
                    payload=forecast,
                    quality_status="not_provided",
                    observed_at=forecast.available_at,
                )
                count += 1
            return count
        except Exception:
            self._source_errors.labels(source="open-meteo", dataset=WEATHER_DATASET).inc()
            raise

    async def run_forever(self) -> None:
        pollers: list[tuple[str, _Poll, float]] = [
            ("imbalance", self.poll_imbalance_once, self._intervals.imbalance),
            ("load", self.poll_load_once, self._intervals.load),
            ("wind", self.poll_wind_once, self._intervals.wind),
            ("solar", self.poll_solar_once, self._intervals.solar),
        ]
        if self._weather_client is not None:
            pollers.append(("weather", self.poll_weather_once, self._intervals.weather))

        async with asyncio.TaskGroup() as tasks:
            for name, poll, interval in pollers:
                tasks.create_task(
                    self._run_periodically(name, poll, interval),
                    name=f"ingestor-{name}",
                )

    async def _poll_elia_once(
        self,
        *,
        dataset: str,
        subject: Subject,
        event_type: str,
        build_event: _BuildEvent,
        lookback: timedelta,
        start: datetime | None,
        end: datetime | None,
    ) -> int:
        window_start, window_end = self._poll_window(start, end, lookback)
        try:
            count = 0
            async for source_record in self._client.iter_records(
                dataset,
                window_start,
                window_end,
            ):
                payload, event_time, natural_key, quality_status = build_event(source_record)
                await self._publish(
                    subject=subject,
                    event_type=event_type,
                    source="elia",
                    dataset=dataset,
                    event_time=event_time,
                    natural_key=natural_key,
                    payload=payload,
                    quality_status=quality_status,
                )
                count += 1
            return count
        except Exception:
            self._source_errors.labels(source="elia", dataset=dataset).inc()
            raise

    async def _publish(
        self,
        *,
        subject: Subject,
        event_type: str,
        source: str,
        dataset: str,
        event_time: datetime,
        natural_key: str,
        payload: BaseModel,
        quality_status: str,
        observed_at: datetime | None = None,
    ) -> None:
        event = EventEnvelope.create(
            event_type=event_type,
            source=source,
            dataset=dataset,
            event_time=event_time,
            natural_key=natural_key,
            payload=payload,
            quality_status=quality_status,
        )
        if observed_at is not None:
            event = event.model_copy(update={"observed_at": observed_at})
        await self._bus.publish(subject, event)
        self._published.labels(source=source, dataset=dataset).inc()

    def _poll_window(
        self,
        start: datetime | None,
        end: datetime | None,
        lookback: timedelta,
    ) -> tuple[datetime, datetime]:
        window_end = _normalize_boundary(end if end is not None else self._clock())
        window_start = _normalize_boundary(start if start is not None else window_end - lookback)
        if window_start >= window_end:
            raise ValueError("poll start must be before end")
        if window_end - window_start > MAX_PAGE_RANGE:
            raise ValueError("poll window exceeds the maximum page range")
        return window_start, window_end

    async def _run_periodically(
        self,
        name: str,
        poll: _Poll,
        interval: float,
    ) -> None:
        deadline = self._monotonic()
        while True:
            try:
                await poll()
            except Exception:
                LOGGER.exception("%s source poll failed", name)

            deadline += interval
            now = self._monotonic()
            if deadline <= now:
                missed_intervals = int((now - deadline) // interval) + 1
                deadline += missed_intervals * interval
            await self._sleep(deadline - now)


def _build_imbalance_event(
    record: Mapping[str, object],
) -> tuple[BaseModel, datetime, str, str]:
    observation = normalize_imbalance(record)
    return (
        observation,
        observation.timestamp,
        _timestamp_key(observation.timestamp),
        observation.quality_status,
    )


def _build_load_event(
    record: Mapping[str, object],
) -> tuple[BaseModel, datetime, str, str]:
    observation = normalize_load(record)
    return (
        observation,
        observation.timestamp,
        _timestamp_key(observation.timestamp),
        "not_provided",
    )


def _build_wind_event(
    record: Mapping[str, object],
) -> tuple[BaseModel, datetime, str, str]:
    observation = normalize_wind(record)
    natural_key = _dimension_key(
        timestamp=observation.timestamp,
        offshore_onshore=observation.offshore_onshore,
        region=observation.region,
        grid_connection_type=observation.grid_connection_type,
    )
    return observation, observation.timestamp, natural_key, "not_provided"


def _build_solar_event(
    record: Mapping[str, object],
) -> tuple[BaseModel, datetime, str, str]:
    observation = normalize_solar(record)
    natural_key = _dimension_key(
        timestamp=observation.timestamp,
        region=observation.region,
    )
    return observation, observation.timestamp, natural_key, "not_provided"


def _timestamp_key(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _dimension_key(*, timestamp: datetime, **dimensions: str) -> str:
    identity = {"timestamp": _timestamp_key(timestamp), **dimensions}
    return json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_boundary(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("poll boundaries must be timezone-aware")
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _run_service() -> None:
    settings = get_settings()
    nats_module = importlib.import_module("imbalance_pipeline.messaging.nats")
    bus = await nats_module.NatsEventBus.connect(settings)
    await bus.ensure_grid_stream()
    try:
        async with EliaClient(base_url=settings.elia_base_url) as client:
            await Ingestor(client=client, bus=bus, settings=settings).run_forever()
    finally:
        await bus.aclose()


def main() -> None:
    asyncio.run(_run_service())
