import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import httpx
import pytest
import respx

from imbalance_pipeline.sources.elia import (
    EliaClient,
    normalize_imbalance,
    normalize_load,
    normalize_solar,
    normalize_wind,
)
from imbalance_pipeline.sources.weather import WeatherClient

ELIA_BASE_URL = "https://opendata.elia.be/api/explore/v2.1"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_VARIABLES = (
    "temperature_2m",
    "cloud_cover",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
)


@pytest.fixture(scope="module")
def elia_fixture() -> dict[str, object]:
    fixture_path = Path(__file__).parent / "fixtures" / "elia_ods161.json"
    return cast(dict[str, object], json.loads(fixture_path.read_text()))


@pytest.fixture
def fixture_record(elia_fixture: dict[str, object]) -> dict[str, object]:
    results = cast(list[dict[str, object]], elia_fixture["results"])
    return results[0]


def supporting_record(elia_fixture: Mapping[str, object], dataset: str) -> dict[str, object]:
    supporting = cast(dict[str, dict[str, object]], elia_fixture["supporting"])
    return supporting[dataset]


def weather_response() -> list[dict[str, object]]:
    times = [
        "2026-07-13T12:00",
        "2026-07-13T13:00",
        "2026-07-13T14:00",
    ]
    responses: list[dict[str, object]] = []
    for point_index in range(5):
        hourly: dict[str, object] = {"time": times}
        for variable_index, variable in enumerate(WEATHER_VARIABLES):
            base = float(10 * (variable_index + 1) + point_index)
            hourly[variable] = [base, base + 1.0, base + 2.0]
        if point_index == 4:
            temperature = cast(list[float | None], hourly["temperature_2m"])
            temperature[0] = None
        responses.append(
            {
                "latitude": 50.0 + point_index / 10,
                "longitude": 4.0 + point_index / 10,
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


def test_normalize_ods161_record(fixture_record: dict[str, object]) -> None:
    observation = normalize_imbalance(fixture_record)

    assert observation.resolution_code == "PT1M"
    assert observation.system_imbalance_mw == 325.224
    assert observation.timestamp == datetime(2026, 7, 13, 16, 14, tzinfo=UTC)
    assert observation.quarter_hour == datetime(2026, 7, 13, 16, 0, tzinfo=UTC)
    assert observation.quality_status == "Non-validated"
    assert observation.ace_mw == -118.511
    assert observation.alpha_prime_eur_mwh is None
    assert observation.imbalance_price_eur_mwh is None


def test_normalize_imbalance_rejects_non_minute_resolution(
    fixture_record: dict[str, object],
) -> None:
    record = {**fixture_record, "resolutioncode": "PT15M"}

    with pytest.raises(ValueError, match="PT1M"):
        normalize_imbalance(record)


def test_normalizers_reject_naive_source_timestamps(fixture_record: dict[str, object]) -> None:
    record = {**fixture_record, "datetime": "2026-07-13T16:14:00"}

    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_imbalance(record)


def test_normalize_ods002_preserves_null_measurement(
    elia_fixture: dict[str, object],
) -> None:
    observation = normalize_load(supporting_record(elia_fixture, "ods002"))

    assert observation.timestamp == datetime(2026, 7, 17, 21, 45, tzinfo=UTC)
    assert observation.resolution_code == "PT15M"
    assert observation.measured_mw is None
    assert observation.most_recent_forecast_mw == 8597.61
    assert observation.most_recent_confidence_10_mw == 7845.65
    assert observation.most_recent_confidence_90_mw == 9349.57
    assert observation.day_ahead_forecast_mw == 8597.61
    assert observation.week_ahead_forecast_mw == 8714.32


def test_normalize_ods086_preserves_source_dimensions_and_nulls(
    elia_fixture: dict[str, object],
) -> None:
    observation = normalize_wind(supporting_record(elia_fixture, "ods086"))

    assert observation.offshore_onshore == "Onshore"
    assert observation.region == "Flanders"
    assert observation.grid_connection_type == "Elia"
    assert observation.real_time_mw is None
    assert observation.load_factor is None
    assert observation.decremental_bid_id is None
    assert observation.monitored_capacity_mw == 333.4


def test_normalize_ods087_preserves_region_and_zero_values(
    elia_fixture: dict[str, object],
) -> None:
    observation = normalize_solar(supporting_record(elia_fixture, "ods087"))

    assert observation.region == "Namur"
    assert observation.real_time_mw is None
    assert observation.most_recent_forecast_mw == 0.0
    assert observation.load_factor == 0.0
    assert observation.monitored_capacity_mw == 444.557


@pytest.mark.asyncio
async def test_fetch_page_uses_explore_v21_path_query_and_ten_second_timeout(
    respx_mock: respx.MockRouter,
    elia_fixture: dict[str, object],
) -> None:
    captured_request: httpx.Request | None = None

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=elia_fixture)

    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods161/records").mock(
        side_effect=respond
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client)
        records = await client.fetch_page(
            dataset="ods161",
            limit=2,
            offset=4,
            where='qualitystatus = "Validated"',
            order_by="datetime asc",
        )

    assert records == elia_fixture["results"]
    assert captured_request is not None
    assert captured_request.url.path == "/api/explore/v2.1/catalog/datasets/ods161/records"
    assert dict(captured_request.url.params) == {
        "limit": "2",
        "offset": "4",
        "where": 'qualitystatus = "Validated"',
        "order_by": "datetime asc",
    }
    timeout = cast(dict[str, float], captured_request.extensions["timeout"])
    assert timeout["connect"] == 10.0
    assert timeout["read"] == 10.0


@pytest.mark.asyncio
async def test_iter_records_uses_utc_boundaries_ordering_and_pagination(
    respx_mock: respx.MockRouter,
    elia_fixture: dict[str, object],
) -> None:
    source_records = cast(list[dict[str, object]], elia_fixture["results"])
    captured_requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        offset = int(request.url.params["offset"])
        page = [source_records[1], source_records[0]] if offset == 0 else []
        return httpx.Response(200, json={"total_count": 2, "results": page})

    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods161/records").mock(
        side_effect=respond
    )
    source_timezone = timezone(timedelta(hours=2))
    start = datetime(2026, 7, 13, 18, 12, tzinfo=source_timezone)
    end = datetime(2026, 7, 13, 18, 15, tzinfo=source_timezone)
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client, page_size=2)
        records = [record async for record in client.iter_records("ods161", start, end)]

    assert records == [source_records[1], source_records[0]]
    assert len(captured_requests) == 2
    for request in captured_requests:
        assert request.url.params["where"] == (
            'datetime >= "2026-07-13T16:12:00+00:00" '
            'AND datetime < "2026-07-13T16:15:00+00:00"'
        )
        assert request.url.params["order_by"] == "datetime asc"
    assert [request.url.params["offset"] for request in captured_requests] == ["0", "2"]


@pytest.mark.asyncio
async def test_iter_records_rejects_naive_query_boundaries() -> None:
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client)
        iterator = client.iter_records("ods161", datetime(2026, 7, 13), None)

        with pytest.raises(ValueError, match="timezone-aware"):
            await anext(iterator)


@pytest.mark.asyncio
async def test_fetch_page_retries_only_retryable_http_failures(
    respx_mock: respx.MockRouter,
    elia_fixture: dict[str, object],
) -> None:
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("connection refused", request=request)
        if attempts == 2:
            return httpx.Response(429, headers={"Retry-After": "0"})
        if attempts == 3:
            return httpx.Response(503)
        return httpx.Response(200, json=elia_fixture)

    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods161/records").mock(
        side_effect=respond
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(
            base_url=ELIA_BASE_URL,
            client=http_client,
            retry_base_seconds=0,
        )
        records = await client.fetch_page("ods161", 2, 0, None, "datetime asc")

    assert records == elia_fixture["results"]


@pytest.mark.asyncio
async def test_fetch_page_does_not_retry_other_4xx(
    respx_mock: respx.MockRouter,
    elia_fixture: dict[str, object],
) -> None:
    route = respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods161/records").mock(
        side_effect=[httpx.Response(400), httpx.Response(200, json=elia_fixture)]
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(
            base_url=ELIA_BASE_URL,
            client=http_client,
            retry_base_seconds=0,
        )
        with pytest.raises(httpx.HTTPStatusError) as error:
            await client.fetch_page("ods161", 2, 0, None, "datetime asc")

    assert error.value.response.status_code == 400
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_fetch_page_stops_after_five_attempts(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods161/records").mock(
        return_value=httpx.Response(503)
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(
            base_url=ELIA_BASE_URL,
            client=http_client,
            retry_base_seconds=0,
        )
        with pytest.raises(httpx.HTTPStatusError) as error:
            await client.fetch_page("ods161", 2, 0, None, "datetime asc")

    assert error.value.response.status_code == 503
    assert route.call_count == 5


@pytest.mark.asyncio
async def test_weather_is_disabled_by_default_without_an_http_request() -> None:
    def fail_on_request(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"unexpected weather request: {request.url}")

    transport = httpx.MockTransport(fail_on_request)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = WeatherClient(client=http_client)
        forecasts = [
            forecast
            async for forecast in client.iter_current_state(
                datetime(2026, 7, 13, 12, tzinfo=UTC),
                datetime(2026, 7, 13, 14, tzinfo=UTC),
                historical=False,
            )
        ]

    assert forecasts == []


@pytest.mark.asyncio
async def test_historical_weather_queries_fixed_points_and_aggregates_complete_rows(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.get(HISTORICAL_FORECAST_URL).mock(
        return_value=httpx.Response(200, json=weather_response())
    )
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(enabled=True, client=http_client)
        forecasts = [
            forecast
            async for forecast in client.iter_current_state(
                datetime(2026, 7, 13, 12, tzinfo=UTC),
                datetime(2026, 7, 13, 14, tzinfo=UTC),
                historical=True,
            )
        ]

    assert [forecast.valid_time for forecast in forecasts] == [
        datetime(2026, 7, 13, 12, tzinfo=UTC),
        datetime(2026, 7, 13, 13, tzinfo=UTC),
    ]
    assert [forecast.available_at for forecast in forecasts] == [
        forecast.valid_time for forecast in forecasts
    ]
    temperature = forecasts[0].variables["temperature_2m"]
    assert temperature.mean == 11.5
    assert temperature.minimum == 10.0
    assert temperature.maximum == 13.0
    assert temperature.point_mask == {
        "Brussels": True,
        "Zeebrugge": True,
        "Hasselt": True,
        "Liège": True,
        "Arlon": False,
    }
    direction = forecasts[0].variables["wind_direction_100m"]
    assert direction.minimum is None
    assert direction.maximum is None
    assert math.isfinite(cast(float, direction.mean))
    request = route.calls[0].request
    assert request.url.params["timezone"] == "UTC"
    assert request.url.params["start_date"] == "2026-07-13"
    assert request.url.params["end_date"] == "2026-07-13"
    assert request.url.params["hourly"] == ",".join(WEATHER_VARIABLES)
    assert request.url.params["latitude"] == "50.8503,51.33,50.9307,50.6337,49.6833"
    assert request.url.params["longitude"] == "4.3517,3.2,5.3325,5.5675,5.8167"


@pytest.mark.asyncio
async def test_live_weather_uses_receipt_time_as_availability(
    respx_mock: respx.MockRouter,
) -> None:
    receipt_time = datetime(2026, 7, 13, 12, 30, tzinfo=UTC)
    route = respx_mock.get(FORECAST_URL).mock(
        return_value=httpx.Response(200, json=weather_response())
    )
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(enabled=True, client=http_client, clock=lambda: receipt_time)
        forecasts = [
            forecast
            async for forecast in client.iter_current_state(
                datetime(2026, 7, 13, 12, tzinfo=UTC),
                datetime(2026, 7, 13, 14, tzinfo=UTC),
                historical=False,
            )
        ]

    assert route.called
    assert forecasts
    assert all(forecast.available_at == receipt_time for forecast in forecasts)


@pytest.mark.asyncio
async def test_weather_drops_rows_available_after_feature_cutoff(
    respx_mock: respx.MockRouter,
) -> None:
    receipt_time = datetime(2026, 7, 13, 14, 1, tzinfo=UTC)
    respx_mock.get(FORECAST_URL).mock(
        return_value=httpx.Response(200, json=weather_response())
    )
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(enabled=True, client=http_client, clock=lambda: receipt_time)
        forecasts = [
            forecast
            async for forecast in client.iter_current_state(
                datetime(2026, 7, 13, 12, tzinfo=UTC),
                datetime(2026, 7, 13, 14, tzinfo=UTC),
                historical=False,
            )
        ]

    assert forecasts == []
