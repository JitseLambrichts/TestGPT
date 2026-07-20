from datetime import UTC, datetime

from fastapi.testclient import TestClient

from imbalance_pipeline.serving.api import ModelStatus, create_app
from imbalance_pipeline.storage.clickhouse import DashboardRow, Prediction, TransientStorageError

NOW = datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC)


class FakePredictionRepository:
    def __init__(
        self,
        result: Prediction | None | Exception,
        dashboard_result: list[DashboardRow] | Exception | None = None,
    ) -> None:
        self.result = result
        self.dashboard_result = dashboard_result if dashboard_result is not None else []
        self.calls = 0
        self.list_calls: list[dict[str, object]] = []
        self.dashboard_calls: list[dict[str, object]] = []

    async def latest_prediction(self) -> Prediction | None:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    async def list_predictions(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        after_target_time: datetime | None,
        after_event_id: str | None,
    ) -> list[Prediction]:
        self.list_calls.append(
            {
                "start": start,
                "end": end,
                "limit": limit,
                "after_target_time": after_target_time,
                "after_event_id": after_event_id,
            }
        )
        if isinstance(self.result, Exception):
            raise self.result
        if self.result is None:
            return []
        return [self.result, self.result.model_copy(update={"event_id": "two"})]

    async def list_dashboard_rows(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[DashboardRow]:
        self.dashboard_calls.append({"start": start, "end": end, "limit": limit})
        if isinstance(self.dashboard_result, Exception):
            raise self.dashboard_result
        return self.dashboard_result


def prediction() -> Prediction:
    return Prediction(
        event_id="prediction-event-001",
        cutoff=NOW,
        target_time=NOW,
        generated_at=NOW,
        system_imbalance_mw=-15.0,
        p10_mw=-30.0,
        p90_mw=5.0,
        flip_probability=0.8,
        will_flip=True,
        current_state="positive",
        predicted_state="negative",
        prediction_quality="model",
        model_version="model-v1",
        feature_schema_hash="feature-schema-001",
    )


def dashboard_row() -> DashboardRow:
    return DashboardRow(
        **prediction().model_dump(),
        realized_system_imbalance_mw=-12.0,
        realized_state="negative",
        flip_actual=True,
        evaluated_at=NOW,
    )


def test_liveness_does_not_depend_on_clickhouse() -> None:
    repository = FakePredictionRepository(TransientStorageError())
    client = TestClient(create_app(repository))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert repository.calls == 0


def test_readiness_checks_clickhouse_connectivity() -> None:
    repository = FakePredictionRepository(prediction())
    client = TestClient(create_app(repository))

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert repository.calls == 1


def test_documented_readiness_alias_checks_the_same_dependencies() -> None:
    repository = FakePredictionRepository(prediction())
    client = TestClient(create_app(repository))

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert repository.calls == 1


def test_latest_prediction_returns_the_storage_record() -> None:
    repository = FakePredictionRepository(prediction())
    client = TestClient(create_app(repository))

    response = client.get("/v1/predictions/latest")

    assert response.status_code == 200
    assert response.json()["event_id"] == "prediction-event-001"
    assert response.json()["will_flip"] is True
    assert response.json()["flip_probability"] == 0.8
    assert repository.calls == 1


def test_latest_prediction_returns_not_found_before_any_prediction() -> None:
    repository = FakePredictionRepository(None)
    client = TestClient(create_app(repository))

    response = client.get("/v1/predictions/latest")

    assert response.status_code == 404
    assert response.json()["detail"] == "no prediction is available yet"


def test_storage_failure_is_reported_as_not_ready_and_unavailable() -> None:
    repository = FakePredictionRepository(TransientStorageError())
    client = TestClient(create_app(repository))

    ready = client.get("/readyz")
    latest = client.get("/v1/predictions/latest")

    assert ready.status_code == 503
    assert latest.status_code == 503
    assert latest.json()["detail"] == "prediction store is temporarily unavailable"


def test_metrics_are_exposed_in_prometheus_text_format() -> None:
    repository = FakePredictionRepository(prediction())
    client = TestClient(create_app(repository))

    client.get("/healthz")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "imbalance_api_requests_total" in response.text


def test_prediction_range_uses_utc_keyset_pagination_and_is_bounded() -> None:
    repository = FakePredictionRepository(prediction())
    client = TestClient(create_app(repository))

    response = client.get(
        "/v1/predictions",
        params={
            "start": "2026-07-13T10:00:00Z",
            "end": "2026-07-13T11:00:00Z",
            "limit": 1,
            "after_target_time": "2026-07-13T10:01:05Z",
            "after_event_id": "previous",
        },
    )

    assert response.status_code == 200
    assert [item["event_id"] for item in response.json()["items"]] == ["prediction-event-001"]
    assert response.json()["next_cursor"] == {
        "event_id": "prediction-event-001",
        "target_time": "2026-07-13T10:01:05Z",
    }
    assert repository.list_calls == [
        {
            "start": datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
            "end": datetime(2026, 7, 13, 11, 0, tzinfo=UTC),
            "limit": 2,
            "after_target_time": datetime(2026, 7, 13, 10, 1, 5, tzinfo=UTC),
            "after_event_id": "previous",
        }
    ]


def test_prediction_range_rejects_reversed_or_excessive_requests() -> None:
    client = TestClient(create_app(FakePredictionRepository(prediction())))

    reversed_range = client.get(
        "/v1/predictions?start=2026-07-13T11:00:00Z&end=2026-07-13T10:00:00Z"
    )
    excessive = client.get(
        "/v1/predictions?start=2026-07-13T10:00:00Z&end=2026-07-13T11:00:00Z&limit=10001"
    )

    assert reversed_range.status_code == 422
    assert excessive.status_code == 422


def test_current_model_omits_any_filesystem_location() -> None:
    client = TestClient(
        create_app(
            FakePredictionRepository(prediction()),
            model_status=ModelStatus(
                model_version="model-v1",
                feature_schema_hash="feature-schema-001",
                fallback_enabled=True,
            ),
        )
    )

    response = client.get("/v1/models/current")

    assert response.status_code == 200
    assert response.json() == {
        "fallback_enabled": True,
        "feature_schema_hash": "feature-schema-001",
        "model_version": "model-v1",
    }
    assert "path" not in response.text


def test_dashboard_endpoint_returns_live_rows() -> None:
    repository = FakePredictionRepository(prediction(), [dashboard_row()])
    client = TestClient(create_app(repository))

    response = client.get(
        "/v1/dashboard",
        params={
            "start": "2026-07-13T04:00:00Z",
            "end": "2026-07-13T10:02:00Z",
        },
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["realized_system_imbalance_mw"] == -12.0
    assert response.json()["items"][0]["flip_actual"] is True
    assert repository.dashboard_calls == [
        {
            "start": datetime(2026, 7, 13, 4, 0, tzinfo=UTC),
            "end": datetime(2026, 7, 13, 10, 2, tzinfo=UTC),
            "limit": 500,
        }
    ]


def test_dashboard_endpoint_validates_range_and_limit() -> None:
    client = TestClient(create_app(FakePredictionRepository(None)))

    reversed_range = client.get(
        "/v1/dashboard?start=2026-07-13T10:02:00Z&end=2026-07-13T04:00:00Z"
    )
    naive = client.get(
        "/v1/dashboard?start=2026-07-13T04:00:00&end=2026-07-13T10:02:00Z"
    )
    excessive = client.get(
        "/v1/dashboard?start=2026-07-13T04:00:00Z&end=2026-07-13T10:02:00Z&limit=10001"
    )

    assert reversed_range.status_code == 422
    assert naive.status_code == 422
    assert excessive.status_code == 422


def test_dashboard_endpoint_sanitizes_storage_failure() -> None:
    client = TestClient(
        create_app(FakePredictionRepository(None, TransientStorageError("secret")))
    )

    response = client.get(
        "/v1/dashboard?start=2026-07-13T04:00:00Z&end=2026-07-13T10:02:00Z"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "prediction store is temporarily unavailable"


def test_dashboard_page_is_served_without_remote_assets() -> None:
    response = TestClient(create_app(FakePredictionRepository(None))).get("/dashboard")

    assert response.status_code == 200
    assert 'src="/dashboard-assets/vendor/chart.umd.min.js"' in response.text
    assert "https://" not in response.text
