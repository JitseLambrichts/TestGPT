import json
import math
from collections.abc import Callable, Mapping, Sequence
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
    return load_elia_fixture("ods161")


def load_elia_fixture(dataset: str) -> dict[str, object]:
    fixture_path = Path(__file__).parent / "fixtures" / f"elia_{dataset}.json"
    return cast(dict[str, object], json.loads(fixture_path.read_text()))


@pytest.fixture
def fixture_record(elia_fixture: dict[str, object]) -> dict[str, object]:
    results = cast(list[dict[str, object]], elia_fixture["results"])
    return results[0]


def fixture_results(fixture: Mapping[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], fixture["results"])


def weather_response(
    *,
    utc_offset_seconds: int = 0,
    wind_directions: Sequence[float | None] | None = None,
) -> list[dict[str, object]]:
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
        if wind_directions is not None:
            directions = cast(list[float | None], hourly["wind_direction_100m"])
            directions[0] = wind_directions[point_index]
        if point_index == 4:
            temperature = cast(list[float | None], hourly["temperature_2m"])
            temperature[0] = None
        responses.append(
            {
                "latitude": 50.0 + point_index / 10,
                "longitude": 4.0 + point_index / 10,
                "generationtime_ms": 0.1,
                "utc_offset_seconds": utc_offset_seconds,
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


@pytest.mark.asyncio
async def test_fetch_and_normalize_ods002_preserves_null_measurement(
    respx_mock: respx.MockRouter,
) -> None:
    fixture = load_elia_fixture("ods002")
    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods002/records").mock(
        return_value=httpx.Response(200, json=fixture)
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client)
        records = await client.fetch_page("ods002", 100, 0, None, "datetime asc")
    observation = normalize_load(records[0])

    assert observation.timestamp == datetime(2026, 7, 17, 21, 45, tzinfo=UTC)
    assert observation.resolution_code == "PT15M"
    assert observation.measured_mw is None
    assert observation.most_recent_forecast_mw == 8597.61
    assert observation.most_recent_confidence_10_mw == 7845.65
    assert observation.most_recent_confidence_90_mw == 9349.57
    assert observation.day_ahead_forecast_mw == 8597.61
    assert observation.week_ahead_forecast_mw == 8714.32


@pytest.mark.asyncio
async def test_fetch_and_normalize_ods086_preserves_source_dimensions_and_nulls(
    respx_mock: respx.MockRouter,
) -> None:
    fixture = load_elia_fixture("ods086")
    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods086/records").mock(
        return_value=httpx.Response(200, json=fixture)
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client)
        records = await client.fetch_page(
            "ods086",
            100,
            0,
            None,
            "datetime asc, offshoreonshore asc, region asc, gridconnectiontype asc",
        )
    observation = normalize_wind(records[0])

    assert observation.offshore_onshore == "Onshore"
    assert observation.region == "Flanders"
    assert observation.grid_connection_type == "Elia"
    assert observation.real_time_mw is None
    assert observation.load_factor is None
    assert observation.decremental_bid_id is None
    assert observation.monitored_capacity_mw == 333.4


@pytest.mark.asyncio
async def test_fetch_and_normalize_ods087_preserves_region_and_zero_values(
    respx_mock: respx.MockRouter,
) -> None:
    fixture = load_elia_fixture("ods087")
    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods087/records").mock(
        return_value=httpx.Response(200, json=fixture)
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client)
        records = await client.fetch_page(
            "ods087",
            100,
            0,
            None,
            "datetime asc, region asc",
        )
    observation = normalize_solar(records[0])

    assert observation.region == "Namur"
    assert observation.real_time_mw is None
    assert observation.most_recent_forecast_mw == 0.0
    assert observation.load_factor == 0.0
    assert observation.monitored_capacity_mw == 444.557


@pytest.mark.parametrize(
    ("dataset", "normalizer"),
    [
        ("ods002", normalize_load),
        ("ods086", normalize_wind),
        ("ods087", normalize_solar),
    ],
)
def test_supporting_normalizers_reject_non_quarter_hour_resolution(
    dataset: str,
    normalizer: Callable[[Mapping[str, object]], object],
) -> None:
    record = {**fixture_results(load_elia_fixture(dataset))[0], "resolutioncode": "PT1M"}

    with pytest.raises(ValueError, match="PT15M"):
        normalizer(record)


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
    third_record = {**source_records[0], "datetime": "2026-07-13T16:14:30+00:00"}
    captured_requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        offset = int(request.url.params["offset"])
        page = [source_records[1], source_records[0]] if offset == 0 else [third_record]
        return httpx.Response(200, json={"total_count": 3, "results": page})

    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods161/records").mock(
        side_effect=respond
    )
    source_timezone = timezone(timedelta(hours=2))
    start = datetime(2026, 7, 13, 18, 12, tzinfo=source_timezone)
    end = datetime(2026, 7, 13, 18, 15, tzinfo=source_timezone)
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client, page_size=2)
        records = [record async for record in client.iter_records("ods161", start, end)]

    assert records == [source_records[1], source_records[0], third_record]
    assert len(captured_requests) == 2
    for request in captured_requests:
        assert request.url.params["where"] == (
            'datetime >= "2026-07-13T16:12:00+00:00" '
            'AND datetime < "2026-07-13T16:15:00+00:00"'
        )
        assert request.url.params["order_by"] == "datetime asc"
    assert [request.url.params["offset"] for request in captured_requests] == ["0", "2"]


@pytest.mark.asyncio
async def test_iter_records_stops_when_exact_full_page_consumes_total_count(
    respx_mock: respx.MockRouter,
    elia_fixture: dict[str, object],
) -> None:
    records = fixture_results(elia_fixture)

    def respond(request: httpx.Request) -> httpx.Response:
        page = records if request.url.params["offset"] == "0" else []
        return httpx.Response(200, json={"total_count": 2, "results": page})

    route = respx_mock.get(
        f"{ELIA_BASE_URL}/catalog/datasets/ods161/records"
    ).mock(side_effect=respond)

    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client, page_size=2)
        fetched = [
            record
            async for record in client.iter_records(
                "ods161",
                datetime(2026, 7, 13, tzinfo=UTC),
                datetime(2026, 7, 14, tzinfo=UTC),
            )
        ]

    assert fetched == records
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_iter_records_rejects_a_repeated_page(
    respx_mock: respx.MockRouter,
    elia_fixture: dict[str, object],
) -> None:
    records = fixture_results(elia_fixture)

    def respond(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        page = records if offset < 4 else []
        return httpx.Response(200, json={"total_count": 6, "results": page})

    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods161/records").mock(
        side_effect=respond
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client, page_size=2)

        with pytest.raises(RuntimeError, match="repeated page"):
            _ = [
                record
                async for record in client.iter_records(
                    "ods161",
                    datetime(2026, 7, 13, tzinfo=UTC),
                    datetime(2026, 7, 14, tzinfo=UTC),
                )
            ]


@pytest.mark.asyncio
async def test_iter_records_rejects_an_empty_page_before_total_count(
    respx_mock: respx.MockRouter,
    elia_fixture: dict[str, object],
) -> None:
    records = fixture_results(elia_fixture)

    def respond(request: httpx.Request) -> httpx.Response:
        page = records if request.url.params["offset"] == "0" else []
        return httpx.Response(200, json={"total_count": 3, "results": page})

    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods161/records").mock(
        side_effect=respond
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client, page_size=2)

        with pytest.raises(RuntimeError, match="no progress"):
            _ = [
                record
                async for record in client.iter_records(
                    "ods161",
                    datetime(2026, 7, 13, tzinfo=UTC),
                    datetime(2026, 7, 14, tzinfo=UTC),
                )
            ]


@pytest.mark.parametrize(
    ("dataset", "expected_order"),
    [
        (
            "ods086",
            "datetime asc, offshoreonshore asc, region asc, gridconnectiontype asc",
        ),
        ("ods087", "datetime asc, region asc"),
    ],
)
@pytest.mark.asyncio
async def test_iter_records_orders_tied_timestamps_by_source_dimensions(
    respx_mock: respx.MockRouter,
    dataset: str,
    expected_order: str,
) -> None:
    captured_request: httpx.Request | None = None

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json={"total_count": 0, "results": []})

    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/{dataset}/records").mock(
        side_effect=respond
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client)
        records = [
            record
            async for record in client.iter_records(
                dataset,
                datetime(2026, 7, 13, tzinfo=UTC),
                datetime(2026, 7, 14, tzinfo=UTC),
            )
        ]

    assert records == []
    assert captured_request is not None
    assert captured_request.url.params["order_by"] == expected_order


@pytest.mark.parametrize("page_size", [0, 101])
def test_elia_client_enforces_explore_page_size(page_size: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        EliaClient(page_size=page_size)


@pytest.mark.asyncio
async def test_iter_records_splits_large_bounded_ranges_into_utc_day_windows(
    respx_mock: respx.MockRouter,
) -> None:
    windows = {
        (
            'datetime >= "2026-07-01T00:00:00+00:00" '
            'AND datetime < "2026-07-02T00:00:00+00:00"'
        ): ("2026-07-01", 5001),
        (
            'datetime >= "2026-07-02T00:00:00+00:00" '
            'AND datetime < "2026-07-03T00:00:00+00:00"'
        ): ("2026-07-02", 5001),
    }
    captured_requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        window = windows.get(request.url.params["where"])
        if window is None:
            return httpx.Response(400, json={"error": "range was not split by UTC day"})
        day, total_count = window
        limit = int(request.url.params["limit"])
        offset = int(request.url.params["offset"])
        row_count = max(0, min(limit, total_count - offset))
        results = [
            {
                "datetime": f"{day}T00:00:00+00:00",
                "synthetic_row_id": offset + index,
            }
            for index in range(row_count)
        ]
        return httpx.Response(200, json={"total_count": total_count, "results": results})

    respx_mock.get(f"{ELIA_BASE_URL}/catalog/datasets/ods161/records").mock(
        side_effect=respond
    )
    async with httpx.AsyncClient() as http_client:
        client = EliaClient(base_url=ELIA_BASE_URL, client=http_client, page_size=100)
        records = [
            record
            async for record in client.iter_records(
                "ods161",
                datetime(2026, 7, 1, tzinfo=UTC),
                datetime(2026, 7, 3, tzinfo=UTC),
            )
        ]

    assert len(records) == 10_002
    assert len(captured_requests) == 102
    assert all(request.url.path.endswith("/records") for request in captured_requests)
    assert all(
        int(request.url.params["offset"]) + int(request.url.params["limit"]) < 10_000
        for request in captured_requests
    )


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


@pytest.mark.parametrize(
    ("historical", "endpoint"),
    [
        (False, FORECAST_URL),
        (True, HISTORICAL_FORECAST_URL),
    ],
)
@pytest.mark.asyncio
async def test_weather_retries_retryable_failures_for_live_and_historical_endpoints(
    respx_mock: respx.MockRouter,
    historical: bool,
    endpoint: str,
) -> None:
    attempts = 0
    receipt_time = datetime(2026, 7, 13, 12, 30, tzinfo=UTC)

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("connection refused", request=request)
        if attempts == 2:
            return httpx.Response(429)
        if attempts == 3:
            return httpx.Response(503)
        return httpx.Response(200, json=weather_response())

    respx_mock.get(endpoint).mock(side_effect=respond)
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(
            enabled=True,
            client=http_client,
            clock=lambda: receipt_time,
            retry_base_seconds=0,
        )
        forecasts = [
            forecast
            async for forecast in client.iter_current_state(
                datetime(2026, 7, 13, 12, tzinfo=UTC),
                datetime(2026, 7, 13, 14, tzinfo=UTC),
                historical=historical,
            )
        ]

    assert forecasts
    assert attempts == 4


@pytest.mark.asyncio
async def test_weather_does_not_retry_other_4xx(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.get(FORECAST_URL).mock(
        side_effect=[httpx.Response(400), httpx.Response(200, json=weather_response())]
    )
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(enabled=True, client=http_client)
        iterator = client.iter_current_state(
            datetime(2026, 7, 13, 12, tzinfo=UTC),
            datetime(2026, 7, 13, 14, tzinfo=UTC),
            historical=False,
        )
        with pytest.raises(httpx.HTTPStatusError) as error:
            await anext(iterator)

    assert error.value.response.status_code == 400
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_weather_stops_after_five_attempts(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.get(FORECAST_URL).mock(return_value=httpx.Response(503))
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(
            enabled=True,
            client=http_client,
            retry_base_seconds=0,
        )
        iterator = client.iter_current_state(
            datetime(2026, 7, 13, 12, tzinfo=UTC),
            datetime(2026, 7, 13, 14, tzinfo=UTC),
            historical=False,
        )
        with pytest.raises(httpx.HTTPStatusError) as error:
            await anext(iterator)

    assert error.value.response.status_code == 503
    assert route.call_count == 5


@pytest.mark.asyncio
async def test_weather_rejects_nonzero_source_utc_offset_at_http_boundary(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get(FORECAST_URL).mock(
        return_value=httpx.Response(200, json=weather_response(utc_offset_seconds=3600))
    )
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(
            enabled=True,
            client=http_client,
            clock=lambda: datetime(2026, 7, 13, 12, 30, tzinfo=UTC),
        )
        iterator = client.iter_current_state(
            datetime(2026, 7, 13, 12, tzinfo=UTC),
            datetime(2026, 7, 13, 14, tzinfo=UTC),
            historical=False,
        )
        with pytest.raises(ValueError, match="must use UTC"):
            await anext(iterator)


@pytest.mark.asyncio
async def test_weather_rejects_mismatched_time_and_value_arrays_at_http_boundary(
    respx_mock: respx.MockRouter,
) -> None:
    payload = weather_response()
    hourly = cast(dict[str, object], payload[0]["hourly"])
    temperatures = cast(list[float], hourly["temperature_2m"])
    temperatures.pop()
    respx_mock.get(FORECAST_URL).mock(return_value=httpx.Response(200, json=payload))

    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(
            enabled=True,
            client=http_client,
            clock=lambda: datetime(2026, 7, 13, 12, 30, tzinfo=UTC),
        )
        iterator = client.iter_current_state(
            datetime(2026, 7, 13, 12, tzinfo=UTC),
            datetime(2026, 7, 13, 14, tzinfo=UTC),
            historical=False,
        )
        with pytest.raises(ValueError, match="equal lengths"):
            await anext(iterator)


@pytest.mark.asyncio
async def test_live_receipt_time_is_captured_before_response_decode(
    respx_mock: respx.MockRouter,
) -> None:
    receipt_time = datetime(2026, 7, 13, 12, 30, tzinfo=UTC)
    clock_calls: list[datetime] = []

    def clock() -> datetime:
        clock_calls.append(receipt_time)
        return receipt_time

    respx_mock.get(FORECAST_URL).mock(
        return_value=httpx.Response(200, content=b"not valid JSON")
    )
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(enabled=True, client=http_client, clock=clock)
        iterator = client.iter_current_state(
            datetime(2026, 7, 13, 12, tzinfo=UTC),
            datetime(2026, 7, 13, 14, tzinfo=UTC),
            historical=False,
        )
        with pytest.raises(json.JSONDecodeError):
            await anext(iterator)

    assert clock_calls == [receipt_time]


@pytest.mark.asyncio
async def test_historical_weather_availability_does_not_call_live_clock(
    respx_mock: respx.MockRouter,
) -> None:
    def fail_clock() -> datetime:
        raise AssertionError("historical weather must not read the live receipt clock")

    respx_mock.get(HISTORICAL_FORECAST_URL).mock(
        return_value=httpx.Response(200, json=weather_response())
    )
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(enabled=True, client=http_client, clock=fail_clock)
        forecasts = [
            forecast
            async for forecast in client.iter_current_state(
                datetime(2026, 7, 13, 12, tzinfo=UTC),
                datetime(2026, 7, 13, 14, tzinfo=UTC),
                historical=True,
            )
        ]

    assert forecasts
    assert all(forecast.available_at == forecast.valid_time for forecast in forecasts)


@pytest.mark.asyncio
async def test_circular_wind_direction_wraps_across_north(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get(HISTORICAL_FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json=weather_response(wind_directions=[350.0, 10.0, 350.0, 10.0, 0.0]),
        )
    )
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(enabled=True, client=http_client)
        forecasts = [
            forecast
            async for forecast in client.iter_current_state(
                datetime(2026, 7, 13, 12, tzinfo=UTC),
                datetime(2026, 7, 13, 13, tzinfo=UTC),
                historical=True,
            )
        ]

    mean = cast(float, forecasts[0].variables["wind_direction_100m"].mean)
    assert min(abs(mean), abs(mean - 360.0)) < 1e-9


@pytest.mark.asyncio
async def test_circular_wind_direction_is_missing_when_vectors_cancel(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get(HISTORICAL_FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json=weather_response(wind_directions=[0.0, 180.0, 0.0, 180.0, None]),
        )
    )
    async with httpx.AsyncClient() as http_client:
        client = WeatherClient(enabled=True, client=http_client)
        forecasts = [
            forecast
            async for forecast in client.iter_current_state(
                datetime(2026, 7, 13, 12, tzinfo=UTC),
                datetime(2026, 7, 13, 13, tzinfo=UTC),
                historical=True,
            )
        ]

    assert forecasts[0].variables["wind_direction_100m"].mean is None


@pytest.mark.asyncio
async def test_adapters_leave_injected_http_client_open() -> None:
    async with httpx.AsyncClient() as http_client:
        elia = EliaClient(client=http_client)
        weather = WeatherClient(client=http_client)

        await elia.aclose()
        await weather.aclose()

        assert not http_client.is_closed


@pytest.mark.asyncio
async def test_adapters_close_owned_http_clients() -> None:
    elia = EliaClient()
    weather = WeatherClient(enabled=True)

    await elia.aclose()
    await weather.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await elia.fetch_page("ods161", 1, 0, None, "datetime asc")
    weather_iterator = weather.iter_current_state(
        datetime(2026, 7, 13, 12, tzinfo=UTC),
        datetime(2026, 7, 13, 14, tzinfo=UTC),
        historical=False,
    )
    with pytest.raises(RuntimeError, match="closed"):
        await anext(weather_iterator)
