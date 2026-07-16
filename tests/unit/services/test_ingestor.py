import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import httpx
import pytest
from prometheus_client import CollectorRegistry

import imbalance_pipeline.services.ingestor as ingestor_module
from imbalance_pipeline.config import Settings
from imbalance_pipeline.domain.events import EventEnvelope, Subject
from imbalance_pipeline.messaging.base import EventBus, Message
from imbalance_pipeline.services.ingestor import (
    MAX_PAGE_RANGE,
    Ingestor,
    PollIntervals,
)
from imbalance_pipeline.sources.weather import (
    WEATHER_LOCATIONS,
    WEATHER_VARIABLES,
    WeatherClient,
    WeatherForecast,
    WeatherVariableAggregate,
)

FIXTURE_ROOT = Path(__file__).parents[2] / "contract" / "fixtures"
NOW = datetime(2026, 7, 20, 12, 34, 56, tzinfo=UTC)


def fixture_records(dataset: str) -> list[dict[str, object]]:
    payload = cast(
        dict[str, object],
        json.loads((FIXTURE_ROOT / f"elia_{dataset}.json").read_text()),
    )
    return cast(list[dict[str, object]], payload["results"])


class FakeEliaClient:
    def __init__(
        self,
        records: Mapping[str, list[dict[str, object]]] | None = None,
        errors: Mapping[str, Exception] | None = None,
    ) -> None:
        self.records = records or {}
        self.errors = errors or {}
        self.calls: list[tuple[str, datetime | None, datetime | None]] = []

    async def iter_records(
        self,
        dataset: str,
        start: datetime | None,
        end: datetime | None,
    ) -> AsyncIterator[dict[str, object]]:
        self.calls.append((dataset, start, end))
        error = self.errors.get(dataset)
        if error is not None:
            raise error
        for record in self.records.get(dataset, []):
            yield dict(record)


class FakeWeatherClient:
    def __init__(self, forecasts: list[WeatherForecast]) -> None:
        self.forecasts = forecasts
        self.calls: list[tuple[datetime, datetime, bool, datetime | None]] = []

    async def iter_current_state(
        self,
        start: datetime,
        end: datetime,
        historical: bool,
        availability_cutoff: datetime | None = None,
    ) -> AsyncIterator[WeatherForecast]:
        self.calls.append((start, end, historical, availability_cutoff))
        for forecast in self.forecasts:
            yield forecast


class InMemoryEventBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, EventEnvelope]] = []

    async def publish(self, subject: str, event: EventEnvelope) -> None:
        self.published.append((subject, event))

    async def messages(self, subject: str, durable: str) -> AsyncIterator[Message]:
        del subject, durable
        if False:
            yield FakeMessage(event=_placeholder_event())


class FakeMessage:
    def __init__(self, event: EventEnvelope) -> None:
        self.event = event
        self.delivery_count = 1
        self.acked = False
        self.nak_delay: float | None = None

    async def ack(self) -> None:
        self.acked = True

    async def nak(self, delay_seconds: float) -> None:
        self.nak_delay = delay_seconds


def _placeholder_event() -> EventEnvelope:
    record = fixture_records("ods161")[0]
    return EventEnvelope(
        event_id="event-id",
        event_type="elia.imbalance.observed",
        source="elia",
        dataset="ods161",
        event_time=datetime.fromisoformat(cast(str, record["datetime"])).astimezone(UTC),
        correlation_id="event-id",
        quality_status=cast(str, record["qualitystatus"]),
        payload={},
    )


def fixed_clock() -> datetime:
    return NOW


def make_ingestor(
    client: FakeEliaClient,
    bus: InMemoryEventBus,
    *,
    weather_client: FakeWeatherClient | None = None,
    registry: CollectorRegistry | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    intervals: PollIntervals | None = None,
) -> Ingestor:
    return Ingestor(
        client=client,
        bus=bus,
        settings=Settings(),
        weather_client=weather_client,
        registry=registry,
        clock=fixed_clock,
        monotonic=monotonic,
        sleep=sleep,
        intervals=intervals,
    )


def test_transport_contracts_are_runtime_checkable() -> None:
    bus = InMemoryEventBus()
    message = FakeMessage(_placeholder_event())

    assert isinstance(bus, EventBus)
    assert isinstance(message, Message)


@pytest.mark.asyncio
async def test_poll_imbalance_publishes_every_delivery_with_one_deterministic_id() -> None:
    record = fixture_records("ods161")[0]
    client = FakeEliaClient({"ods161": [record, record]})
    bus = InMemoryEventBus()

    count = await make_ingestor(client, bus).poll_imbalance_once()

    assert count == 2
    assert len(bus.published) == 2
    assert [subject for subject, _ in bus.published] == [
        Subject.RAW_ELIA_IMBALANCE,
        Subject.RAW_ELIA_IMBALANCE,
    ]
    first = bus.published[0][1]
    duplicate = bus.published[1][1]
    assert first.event_id == duplicate.event_id
    assert first.event_time == datetime(2026, 7, 13, 16, 14, tzinfo=UTC)
    assert first.payload["system_imbalance_mw"] == 325.224
    assert first.quality_status == "Non-validated"
    assert client.calls[0][0] == "ods161"
    assert client.calls[0][1] is not None
    assert client.calls[0][2] == NOW
    assert client.calls[0][2] - client.calls[0][1] <= MAX_PAGE_RANGE


@pytest.mark.asyncio
async def test_poll_imbalance_uses_history_dataset_for_explicit_backfill_window() -> None:
    client = FakeEliaClient()
    bus = InMemoryEventBus()
    ingestor = make_ingestor(client, bus)

    await ingestor.poll_imbalance_once(
        start=datetime(2026, 7, 13, 0, tzinfo=UTC),
        end=datetime(2026, 7, 13, 1, tzinfo=UTC),
    )

    assert client.calls[0][0] == "ods133"


@pytest.mark.asyncio
async def test_poll_imbalance_uses_live_dataset_without_explicit_window() -> None:
    client = FakeEliaClient()
    bus = InMemoryEventBus()

    await make_ingestor(client, bus).poll_imbalance_once()

    assert client.calls[0][0] == "ods161"


@pytest.mark.asyncio
async def test_poll_imbalance_gives_a_changed_source_payload_a_new_revision_identity() -> None:
    original = fixture_records("ods161")[0]
    corrected = {**original, "systemimbalance": 326.0}
    client = FakeEliaClient({"ods161": [original, corrected, original]})
    bus = InMemoryEventBus()

    await make_ingestor(client, bus).poll_imbalance_once()

    first, correction, repeated_original = [event for _, event in bus.published]
    assert first.event_id != correction.event_id
    assert first.event_id == repeated_original.event_id
    assert correction.payload["system_imbalance_mw"] == 326.0


@pytest.mark.parametrize(
    ("dataset", "method_name", "subject", "payload_field", "payload_value"),
    [
        ("ods002", "poll_load_once", Subject.RAW_ELIA_LOAD, "most_recent_forecast_mw", 8597.61),
        ("ods086", "poll_wind_once", Subject.RAW_ELIA_WIND, "region", "Flanders"),
        ("ods087", "poll_solar_once", Subject.RAW_ELIA_SOLAR, "region", "Namur"),
    ],
)
@pytest.mark.asyncio
async def test_supporting_source_pollers_publish_typed_events(
    dataset: str,
    method_name: str,
    subject: Subject,
    payload_field: str,
    payload_value: object,
) -> None:
    client = FakeEliaClient({dataset: [fixture_records(dataset)[0]]})
    bus = InMemoryEventBus()
    ingestor = make_ingestor(client, bus)

    method = cast(Callable[[], Awaitable[int]], getattr(ingestor, method_name))
    count = await method()

    assert count == 1
    published_subject, event = bus.published[0]
    assert published_subject == subject
    assert event.dataset == dataset
    assert event.event_time == datetime.fromisoformat(
        cast(str, fixture_records(dataset)[0]["datetime"])
    ).astimezone(UTC)
    assert event.payload[payload_field] == payload_value
    assert event.quality_status == "not_provided"


@pytest.mark.asyncio
async def test_wind_identity_uses_all_source_dimensions_without_deduplicating() -> None:
    first, second = fixture_records("ods086")
    client = FakeEliaClient({"ods086": [first, second, first]})
    bus = InMemoryEventBus()

    count = await make_ingestor(client, bus).poll_wind_once()

    event_ids = [event.event_id for _, event in bus.published]
    assert count == 3
    assert event_ids[0] != event_ids[1]
    assert event_ids[0] == event_ids[2]


def weather_forecast(
    available_at: datetime = datetime(2026, 7, 20, 11, 58, 30, tzinfo=UTC),
) -> WeatherForecast:
    return WeatherForecast(
        valid_time=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
        available_at=available_at,
        locations=("Brussels", "Zeebrugge", "Hasselt", "Liège", "Arlon"),
        variables={
            "temperature_2m": WeatherVariableAggregate(
                mean=24.0,
                minimum=22.0,
                maximum=26.0,
                point_mask={
                    "Brussels": True,
                    "Zeebrugge": True,
                    "Hasselt": True,
                    "Liège": True,
                    "Arlon": True,
                },
            )
        },
    )


@pytest.mark.asyncio
async def test_weather_poll_is_optional_and_publishes_availability_metadata() -> None:
    disabled_bus = InMemoryEventBus()
    assert await make_ingestor(FakeEliaClient(), disabled_bus).poll_weather_once() == 0
    assert disabled_bus.published == []

    weather = FakeWeatherClient([weather_forecast()])
    bus = InMemoryEventBus()
    count = await make_ingestor(
        FakeEliaClient(),
        bus,
        weather_client=weather,
    ).poll_weather_once()

    assert count == 1
    subject, event = bus.published[0]
    assert subject == Subject.RAW_WEATHER_FORECAST
    assert event.source == "open-meteo"
    assert event.dataset == "forecast"
    assert event.event_time == weather_forecast().valid_time
    assert event.observed_at == weather_forecast().available_at
    assert event.payload["variables"] == weather_forecast().model_dump(mode="json")["variables"]
    assert weather.calls[0][2] is False
    assert weather.calls[0][3] is None


@pytest.mark.asyncio
async def test_weather_identity_distinguishes_availability_vintages() -> None:
    first = weather_forecast(datetime(2026, 7, 20, 11, 58, tzinfo=UTC))
    second = weather_forecast(datetime(2026, 7, 20, 11, 59, tzinfo=UTC))
    bus = InMemoryEventBus()

    count = await make_ingestor(
        FakeEliaClient(),
        bus,
        weather_client=FakeWeatherClient([first, second, first]),
    ).poll_weather_once()

    event_ids = [event.event_id for _, event in bus.published]
    assert count == 3
    assert event_ids[0] != event_ids[1]
    assert event_ids[0] == event_ids[2]


def live_weather_response() -> list[dict[str, object]]:
    responses: list[dict[str, object]] = []
    for point_index, location in enumerate(WEATHER_LOCATIONS):
        hourly: dict[str, object] = {"time": ["2026-07-20T12:00"]}
        for variable_index, variable in enumerate(WEATHER_VARIABLES):
            hourly[variable] = [float(10 * (variable_index + 1) + point_index)]
        responses.append(
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "generationtime_ms": 0.1,
                "utc_offset_seconds": 0,
                "timezone": "GMT",
                "timezone_abbreviation": "GMT",
                "elevation": 20.0 + point_index,
                "hourly_units": {
                    "time": "iso8601",
                    **{variable: "source-unit" for variable in WEATHER_VARIABLES},
                },
                "hourly": hourly,
            }
        )
    return responses


@pytest.mark.asyncio
async def test_live_weather_received_after_poll_end_is_published() -> None:
    receipt_time = NOW + timedelta(seconds=2)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=live_weather_response(), request=request)

    transport = httpx.MockTransport(respond)
    async with httpx.AsyncClient(transport=transport) as http_client:
        weather = WeatherClient(
            enabled=True,
            client=http_client,
            clock=lambda: receipt_time,
        )
        bus = InMemoryEventBus()
        count = await make_ingestor(
            FakeEliaClient(),
            bus,
            weather_client=weather,
        ).poll_weather_once()

    assert count == 1
    assert bus.published[0][1].event_time == datetime(2026, 7, 20, 12, tzinfo=UTC)
    assert bus.published[0][1].observed_at == receipt_time


@pytest.mark.asyncio
async def test_poll_rejects_a_range_larger_than_the_safe_page_window() -> None:
    client = FakeEliaClient()
    ingestor = make_ingestor(client, InMemoryEventBus())
    start = datetime(2026, 7, 18, tzinfo=UTC)

    with pytest.raises(ValueError, match="maximum page range"):
        await ingestor.poll_imbalance_once(
            start=start,
            end=start + MAX_PAGE_RANGE + timedelta(microseconds=1),
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_metrics_use_the_supplied_registry_and_count_source_errors() -> None:
    registry = CollectorRegistry()
    client = FakeEliaClient(errors={"ods161": RuntimeError("source unavailable")})
    ingestor = make_ingestor(client, InMemoryEventBus(), registry=registry)

    with pytest.raises(RuntimeError, match="source unavailable"):
        await ingestor.poll_imbalance_once()

    assert (
        registry.get_sample_value(
            "imbalance_ingestor_source_errors_total",
            {"source": "elia", "dataset": "ods161"},
        )
        == 1.0
    )
    assert (
        registry.get_sample_value(
            "imbalance_ingestor_published_total",
            {"source": "elia", "dataset": "ods161"},
        )
        is None
    )

    # Separate ingestors get isolated metric collectors instead of colliding globally.
    make_ingestor(FakeEliaClient(), InMemoryEventBus())
    make_ingestor(FakeEliaClient(), InMemoryEventBus())


class PairwiseMonotonic:
    def __init__(self) -> None:
        self._calls = 0

    def __call__(self) -> float:
        value = 100.0 if self._calls % 2 == 0 else 102.0
        self._calls += 1
        return value


@pytest.mark.asyncio
async def test_run_forever_uses_monotonic_deadlines_and_propagates_cancellation() -> None:
    intervals = PollIntervals(
        imbalance=15.0,
        load=30.0,
        wind=45.0,
        solar=60.0,
        weather=90.0,
    )
    delays: list[float] = []
    all_sleeping = asyncio.Event()
    never = asyncio.Event()

    async def recording_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 4:
            all_sleeping.set()
        await never.wait()

    ingestor = make_ingestor(
        FakeEliaClient(),
        InMemoryEventBus(),
        monotonic=PairwiseMonotonic(),
        sleep=recording_sleep,
        intervals=intervals,
    )
    task = asyncio.create_task(ingestor.run_forever())
    await asyncio.wait_for(all_sleeping.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sorted(delays) == [13.0, 28.0, 43.0, 58.0]


class ExactDeadlineMonotonic:
    def __init__(self) -> None:
        self._calls = 0

    def __call__(self) -> float:
        value = 100.0 if self._calls % 2 == 0 else 115.0
        self._calls += 1
        return value


@pytest.mark.asyncio
async def test_run_forever_allows_zero_sleep_at_an_exact_deadline() -> None:
    delays: list[float] = []
    all_sleeping = asyncio.Event()
    never = asyncio.Event()

    async def recording_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 4:
            all_sleeping.set()
        await never.wait()

    intervals = PollIntervals(
        imbalance=15.0,
        load=15.0,
        wind=15.0,
        solar=15.0,
        weather=15.0,
    )
    ingestor = make_ingestor(
        FakeEliaClient(),
        InMemoryEventBus(),
        monotonic=ExactDeadlineMonotonic(),
        sleep=recording_sleep,
        intervals=intervals,
    )
    task = asyncio.create_task(ingestor.run_forever())
    await asyncio.wait_for(all_sleeping.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert delays == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "invalid_interval",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_poll_intervals_require_positive_finite_values(invalid_interval: float) -> None:
    with pytest.raises(ValueError, match="poll intervals"):
        PollIntervals(imbalance=invalid_interval)


class SetupFailingEventBus(InMemoryEventBus):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def ensure_grid_stream(self) -> None:
        raise RuntimeError("stream setup failed")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_bus_closes_when_stream_setup_fails() -> None:
    bus = SetupFailingEventBus()

    async def connect(settings: Settings) -> SetupFailingEventBus:
        del settings
        return bus

    async def serve(event_bus: EventBus) -> None:
        del event_bus
        pytest.fail("service must not start when stream setup fails")

    with pytest.raises(RuntimeError, match="stream setup failed"):
        await ingestor_module._run_with_event_bus(
            settings=Settings(),
            connect=connect,
            serve=serve,
        )

    assert bus.closed is True
