# Real-time System-Imbalance Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a Dockerized, replayable pipeline that ingests Elia data through NATS JetStream, stores it in ClickHouse, trains a probabilistic TCN-Transformer ensemble, and serves next-minute system-imbalance and flip predictions.

**Architecture:** Python services communicate only through versioned JetStream events and persist canonical rows in ClickHouse. A point-in-time feature engine is shared by online feature construction and training export. Three compact PyTorch TCN-Transformer members emit Gaussian-mixture and flip outputs, are validated and exported to ONNX, and are served by a CPU FastAPI process.

**Tech Stack:** Python 3.12, Pydantic 2, httpx, nats-py, clickhouse-connect, NumPy, PyTorch 2, scikit-learn, ONNX Runtime, FastAPI, Prometheus client, pytest, Ruff, mypy, Docker Compose, NATS JetStream, ClickHouse, optional Prometheus and Grafana.

## Global Constraints

- Predict Elia system imbalance exactly one minute after the latest accepted ODS161 observation.
- Define confirmed positive above +10 MW, confirmed negative below -10 MW, and retain the prior confirmed state inside the inclusive neutral band.
- Use only features whose availability timestamp is at or before the prediction cutoff.
- Use a 180-step one-minute local window and a 96-step 15-minute context window covering 24 hours.
- Train three independently seeded causal TCN-Transformer members.
- Emit a five-component Gaussian-mixture distribution plus a separately calibrated flip probability.
- Keep standard inference CPU-compatible; allow mixed-precision GPU training in Google Colab.
- Publish predictions no later than ten seconds after the new imbalance record is received under normal local operation.
- Preserve acknowledged events through restart and use deterministic identifiers for at-least-once idempotence.
- Store all timestamps internally as UTC DateTime64 with millisecond precision.
- Expose a marked persistence fallback when the model or feature history is unavailable.
- Do not promote a learned model unless the quantitative gates in the approved design pass.
- Never place trades or control grid assets.

## Planned File Structure

The project uses a single installable package with narrow modules:

- pyproject.toml: dependency groups, package metadata, entry points, test and lint configuration.
- compose.yaml, Dockerfile, Makefile, .env.example: reproducible runtime and developer commands.
- config/nats/nats.conf: file-backed JetStream configuration.
- infra/clickhouse/001_schema.sql: canonical tables and materialized indexes.
- infra/prometheus/prometheus.yml and infra/grafana/: optional observability profile.
- src/imbalance_pipeline/config.py: validated environment configuration.
- src/imbalance_pipeline/domain/events.py: event envelope, subject constants, identifiers.
- src/imbalance_pipeline/domain/imbalance.py: source observation, state, label, and prediction types.
- src/imbalance_pipeline/sources/elia.py: Explore API 2.1 client and normalizers.
- src/imbalance_pipeline/sources/weather.py: disabled-by-default point-in-time weather adapter.
- src/imbalance_pipeline/messaging/base.py: event-bus protocol used by services and tests.
- src/imbalance_pipeline/messaging/nats.py: JetStream implementation.
- src/imbalance_pipeline/storage/clickhouse.py: repositories and query contracts.
- src/imbalance_pipeline/features/schema.py: ordered feature registry and fingerprint.
- src/imbalance_pipeline/features/engine.py: causal offline/online feature construction.
- src/imbalance_pipeline/model/distribution.py: Gaussian-mixture math and quantiles.
- src/imbalance_pipeline/model/network.py: TCN, Transformer, fusion, and output heads.
- src/imbalance_pipeline/model/losses.py: mixture NLL, focal, auxiliary, and consistency loss.
- src/imbalance_pipeline/model/bundle.py: manifest, checksum, validation, and promotion.
- src/imbalance_pipeline/training/: datasets, splits, baselines, metrics, calibration, trainer, exporter.
- src/imbalance_pipeline/serving/runtime.py: ONNX ensemble runtime.
- src/imbalance_pipeline/serving/api.py: FastAPI application.
- src/imbalance_pipeline/services/: ingestor, sink, feature, predictor, and outcome entry points.
- tests/unit/: pure behavior tests.
- tests/contract/fixtures/: recorded source payloads and contract tests.
- tests/integration/: JetStream and ClickHouse tests.
- tests/e2e/: deterministic full-pipeline test and live smoke test.
- notebooks/train_colab.ipynb: thin Colab driver over the package.
- docs/runbook.md: startup, backfill, model promotion, replay, and incident procedures.

---

### Task 1: Project Foundation and Validated Configuration

**Files:**
- Create: pyproject.toml
- Create: src/imbalance_pipeline/__init__.py
- Create: src/imbalance_pipeline/config.py
- Create: tests/unit/test_config.py
- Create: .gitignore
- Create: .env.example

**Interfaces:**
- Produces: Settings() -> Settings with nested NATS, ClickHouse, source, feature, model, and API values.
- Produces: get_settings() -> cached Settings.

- [ ] **Step 1: Create package metadata and the first failing configuration test**

Use this project metadata and keep runtime, ML, test, and observability dependencies in explicit groups:

~~~toml
[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[project]
name = "imbalance-pipeline"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = [
  "clickhouse-connect>=0.8,<1",
  "fastapi>=0.115,<1",
  "holidays>=0.67,<1",
  "httpx>=0.28,<1",
  "nats-py>=2.10,<3",
  "numpy>=2.1,<3",
  "prometheus-client>=0.21,<1",
  "pydantic>=2.10,<3",
  "pydantic-settings>=2.7,<3",
  "tenacity>=9,<10",
  "uvicorn[standard]>=0.34,<1"
]

[project.optional-dependencies]
ml = [
  "onnx>=1.17,<2",
  "onnxruntime>=1.20,<2",
  "scikit-learn>=1.6,<2",
  "scipy>=1.15,<2",
  "torch>=2.6,<3"
]
test = [
  "mypy>=1.14,<2",
  "pytest>=8.3,<9",
  "pytest-asyncio>=0.25,<1",
  "pytest-cov>=6,<7",
  "respx>=0.22,<1",
  "ruff>=0.9,<1"
]

[project.scripts]
imbalance-ingestor = "imbalance_pipeline.services.ingestor:main"
imbalance-sink = "imbalance_pipeline.services.sink:main"
imbalance-features = "imbalance_pipeline.services.features:main"
imbalance-predictor = "imbalance_pipeline.services.predictor:main"
imbalance-outcomes = "imbalance_pipeline.services.outcomes:main"
imbalance-train = "imbalance_pipeline.training.train:main"
imbalance-export = "imbalance_pipeline.training.export_data:main"

[tool.hatch.build.targets.wheel]
packages = ["src/imbalance_pipeline"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
  "integration: requires Docker services",
  "live: calls an external API"
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "ASYNC"]

[tool.mypy]
python_version = "3.12"
strict = true
packages = ["imbalance_pipeline"]
~~~

Create tests/unit/test_config.py:

~~~python
import pytest
from pydantic import ValidationError

from imbalance_pipeline.config import Settings


def test_settings_use_safe_local_defaults() -> None:
    settings = Settings()
    assert settings.flip_deadband_mw == 10.0
    assert settings.local_window_minutes == 180
    assert settings.context_steps == 96
    assert settings.ensemble_size == 3
    assert settings.gmm_components == 5


def test_deadband_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(flip_deadband_mw=0)
~~~

- [ ] **Step 2: Install the editable package and verify the test fails**

Run: python -m pip install -e ".[ml,test]"

Run: python -m pytest tests/unit/test_config.py -q

Expected: collection fails because imbalance_pipeline.config does not exist.

- [ ] **Step 3: Implement immutable validated settings**

Create config.py with BaseSettings, ConfigDict(extra="forbid"), positive constrained fields, URLs for NATS and ClickHouse, Elia base URL, dataset identifiers, polling intervals, model directory, fallback flag, and API host/port. Use an lru_cache-backed get_settings function and the IMBALANCE_ environment prefix.

The required public shape is:

~~~python
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IMBALANCE_",
        env_file=".env",
        extra="forbid",
        frozen=True,
    )

    nats_url: str = "nats://localhost:4222"
    clickhouse_url: str = "http://localhost:8123"
    clickhouse_database: str = "imbalance"
    clickhouse_user: str = "imbalance"
    clickhouse_password: str = "imbalance"
    elia_base_url: str = "https://opendata.elia.be/api/explore/v2.1"
    elia_imbalance_live_dataset: str = "ods161"
    elia_imbalance_history_dataset: str = "ods133"
    flip_deadband_mw: float = Field(default=10.0, gt=0)
    local_window_minutes: int = Field(default=180, ge=30)
    context_steps: int = Field(default=96, ge=24)
    ensemble_size: int = Field(default=3, ge=1)
    gmm_components: int = Field(default=5, ge=2)
    model_dir: Path = Path("/models/production")
    allow_fallback: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
~~~

Add .gitignore entries for virtual environments, Python caches, coverage, .env, local ClickHouse/NATS volumes, exported datasets, training runs, and models/production while retaining models/test.

Add .env.example containing every Settings key with local Compose values and no secret not suitable for local development.

- [ ] **Step 4: Verify foundation quality**

Run: python -m pytest tests/unit/test_config.py -q

Expected: 2 passed.

Run: ruff check src tests

Expected: All checks passed.

Run: mypy

Expected: Success: no issues found.

- [ ] **Step 5: Commit the foundation**

Run:

~~~bash
git add pyproject.toml src/imbalance_pipeline .gitignore .env.example tests/unit/test_config.py
git commit -m "build: initialize imbalance pipeline package"
~~~

### Task 2: Event Contracts, Deterministic Identity, and Flip State

**Files:**
- Create: src/imbalance_pipeline/domain/__init__.py
- Create: src/imbalance_pipeline/domain/events.py
- Create: src/imbalance_pipeline/domain/imbalance.py
- Create: tests/unit/domain/test_events.py
- Create: tests/unit/domain/test_hysteresis.py

**Interfaces:**
- Produces: event_id(source: str, dataset: str, natural_key: str, version: str = "1") -> str.
- Produces: EventEnvelope.create(event_type: str, source: str, dataset: str, event_time: datetime, natural_key: str, payload: BaseModel, quality_status: str) -> EventEnvelope.
- Produces: advance_state(previous: ConfirmedState | None, value_mw: float, deadband_mw: float) -> ConfirmedState | None.
- Produces: flip_label(current: ConfirmedState | None, next_state: ConfirmedState | None) -> bool | None.

- [ ] **Step 1: Write failing event and hysteresis tests**

Cover stable SHA-256 identifiers, UTC enforcement, JSON round trips, strict subject constants, exact +10/-10 boundaries, retained neutral state, first-neutral unknown state, and flips in both directions:

~~~python
from imbalance_pipeline.domain.imbalance import ConfirmedState, advance_state, flip_label


def test_neutral_band_retains_confirmed_state() -> None:
    assert advance_state(ConfirmedState.POSITIVE, 10.0, 10.0) is ConfirmedState.POSITIVE
    assert advance_state(ConfirmedState.NEGATIVE, -10.0, 10.0) is ConfirmedState.NEGATIVE


def test_crossing_opposite_boundary_is_a_flip() -> None:
    current = advance_state(None, 11.0, 10.0)
    future = advance_state(current, -11.0, 10.0)
    assert flip_label(current, future) is True


def test_unknown_initial_state_is_masked() -> None:
    assert advance_state(None, 0.0, 10.0) is None
    assert flip_label(None, ConfirmedState.POSITIVE) is None
~~~

- [ ] **Step 2: Verify the domain tests fail**

Run: python -m pytest tests/unit/domain -q

Expected: collection fails because the domain modules do not exist.

- [ ] **Step 3: Implement strict domain types**

Use timezone-aware datetime validators, Decimal-compatible source fields represented as float, and frozen Pydantic models. Define exact subjects in a StrEnum and reject non-UTC envelope datetimes. Store payload as a dict produced from a typed Pydantic payload so envelopes remain transport-neutral.

Implement hysteresis exactly:

~~~python
from enum import StrEnum


class ConfirmedState(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


def advance_state(
    previous: ConfirmedState | None,
    value_mw: float,
    deadband_mw: float = 10.0,
) -> ConfirmedState | None:
    if deadband_mw <= 0:
        raise ValueError("deadband_mw must be positive")
    if value_mw > deadband_mw:
        return ConfirmedState.POSITIVE
    if value_mw < -deadband_mw:
        return ConfirmedState.NEGATIVE
    return previous


def flip_label(
    current: ConfirmedState | None,
    future: ConfirmedState | None,
) -> bool | None:
    if current is None or future is None:
        return None
    return current is not future
~~~

Define ImbalanceObservation with timestamp, quarter_hour, quality_status, ace_mw, system_imbalance_mw, alpha_eur_mwh, alpha_prime_eur_mwh, marginal_incremental_price_eur_mwh, marginal_decremental_price_eur_mwh, and imbalance_price_eur_mwh.

- [ ] **Step 4: Run and harden domain tests**

Run: python -m pytest tests/unit/domain -q

Expected: all domain tests pass.

Run: ruff check src/imbalance_pipeline/domain tests/unit/domain

Expected: All checks passed.

- [ ] **Step 5: Commit domain contracts**

Run:

~~~bash
git add src/imbalance_pipeline/domain tests/unit/domain
git commit -m "feat: define versioned grid event contracts"
~~~

### Task 3: Elia Explore API Client and Source Normalization

**Files:**
- Create: src/imbalance_pipeline/sources/__init__.py
- Create: src/imbalance_pipeline/sources/elia.py
- Create: src/imbalance_pipeline/sources/weather.py
- Create: tests/contract/fixtures/elia_ods161.json
- Create: tests/contract/test_elia_client.py

**Interfaces:**
- Produces: EliaClient.fetch_page(dataset: str, limit: int, offset: int, where: str | None, order_by: str) -> list[dict[str, object]].
- Produces: EliaClient.iter_records(dataset: str, start: datetime | None, end: datetime | None) -> AsyncIterator[dict[str, object]].
- Produces: normalize_imbalance(record: Mapping[str, object]) -> ImbalanceObservation.
- Produces: WeatherClient.iter_current_state(start: datetime, end: datetime, historical: bool) -> AsyncIterator[WeatherForecast].

- [ ] **Step 1: Record a minimal real-shape ODS161 fixture and write failing contract tests**

The fixture must include total_count and at least two results with datetime, resolutioncode, quarterhour, qualitystatus, ace, systemimbalance, alpha, alpha_prime, both marginal prices, and imbalanceprice. Add representative ODS002, ODS086, and ODS087 records. Tests assert Explore API 2.1 path construction, UTC-aware query boundaries, deterministic ordering by datetime, pagination, null-preserving normalization, source dimensions, and rejection of PT15M for the one-minute contract.

Core normalization assertion:

~~~python
def test_normalize_ods161_record(fixture_record: dict[str, object]) -> None:
    observation = normalize_imbalance(fixture_record)
    assert observation.resolution_code == "PT1M"
    assert observation.system_imbalance_mw == 325.224
    assert observation.timestamp.tzinfo is not None
~~~

- [ ] **Step 2: Verify the contract tests fail**

Run: python -m pytest tests/contract/test_elia_client.py -q

Expected: collection fails because imbalance_pipeline.sources.elia is absent.

- [ ] **Step 3: Implement the resilient client and normalizers**

Use one shared httpx.AsyncClient with ten-second connect and read timeouts. Call /catalog/datasets/{dataset}/records with limit, offset, where, and order_by parameters. Escape ISO timestamps as Explore query string literals. Retry only connection failures, 429, and 5xx with exponential jitter and a bounded five-attempt budget. Do not retry other 4xx responses.

Normalize each Elia source into a typed domain record while preserving source quality, dimensions, and null fields.

Implement the optional WeatherClient against the Open-Meteo live Forecast API and Historical Forecast API, whose response formats match. Query UTC hourly values for five fixed Belgian representative points: Brussels, Zeebrugge, Hasselt, Liège, and Arlon. Request temperature_2m, cloud_cover, surface_pressure, precipitation, wind_speed_10m, wind_speed_100m, wind_direction_100m, wind_gusts_10m, shortwave_radiation, direct_radiation, and diffuse_radiation. Aggregate mean/min/max where meaningful while retaining point masks. Set available_at to the valid time for the historical stitched first-hour forecast and to local receipt time for live forecast responses. Never use a weather row with available_at after the feature cutoff. When weather_enabled is false, make no HTTP request and publish no weather event.

- [ ] **Step 4: Run contract and static checks**

Run: python -m pytest tests/contract/test_elia_client.py -q

Expected: all contract tests pass without network access.

Run: ruff check src/imbalance_pipeline/sources tests/contract

Expected: All checks passed.

- [ ] **Step 5: Commit the source adapters**

Run:

~~~bash
git add src/imbalance_pipeline/sources tests/contract
git commit -m "feat: add resilient Elia source adapters"
~~~

### Task 4: Event-Bus Protocol and Ingestion Service

**Files:**
- Create: src/imbalance_pipeline/messaging/__init__.py
- Create: src/imbalance_pipeline/messaging/base.py
- Create: src/imbalance_pipeline/services/__init__.py
- Create: src/imbalance_pipeline/services/ingestor.py
- Create: tests/unit/services/test_ingestor.py

**Interfaces:**
- Produces: EventBus.publish(subject: str, event: EventEnvelope) -> None.
- Produces: EventBus.messages(subject: str, durable: str) -> AsyncIterator[Message].
- Produces: Message.ack() and Message.nak(delay_seconds: float).
- Consumes: EliaClient and event domain from Tasks 2-3.
- Produces: Ingestor.poll_imbalance_once() -> int count of newly published source rows.

- [ ] **Step 1: Write a failing ingestion test with fakes**

Use a fake Elia client returning the same record twice and an in-memory bus. Assert that both source deliveries generate the same event_id and subject, that the event_time is the Elia minute, and that poll_imbalance_once reports source records rather than inventing a uniqueness guarantee before JetStream and ClickHouse.

~~~python
@pytest.mark.asyncio
async def test_poll_publishes_versioned_imbalance_event() -> None:
    client = FakeEliaClient([ODS161_RECORD])
    bus = InMemoryEventBus()
    ingestor = Ingestor(client=client, bus=bus, settings=Settings())
    assert await ingestor.poll_imbalance_once() == 1
    subject, event = bus.published[0]
    assert subject == "grid.raw.elia.imbalance.v1"
    assert event.payload["system_imbalance_mw"] == 325.224
~~~

- [ ] **Step 2: Verify the ingestion test fails**

Run: python -m pytest tests/unit/services/test_ingestor.py -q

Expected: collection fails because messaging.base and services.ingestor are absent.

- [ ] **Step 3: Implement dependency-injected ingestion**

Define runtime-checkable protocols and a Message wrapper independent of nats-py. Implement one-shot methods for each source plus run_forever loops with source-specific polling intervals, monotonic scheduling, clean cancellation, and a maximum page range. Publish typed envelopes and Prometheus counters. Keep CLI wiring in main and business behavior in Ingestor.

- [ ] **Step 4: Verify ingestion behavior**

Run: python -m pytest tests/unit/services/test_ingestor.py -q

Expected: tests cover subject, payload, deterministic identity, cancellation, and source error metrics and all pass.

- [ ] **Step 5: Commit ingestion**

Run:

~~~bash
git add src/imbalance_pipeline/messaging src/imbalance_pipeline/services tests/unit/services/test_ingestor.py
git commit -m "feat: publish Elia observations to event bus"
~~~

### Task 5: NATS JetStream Transport

**Files:**
- Create: src/imbalance_pipeline/messaging/nats.py
- Create: config/nats/nats.conf
- Create: tests/unit/messaging/test_nats_adapter.py
- Create: tests/integration/test_jetstream.py

**Interfaces:**
- Implements: NatsEventBus.connect(settings: Settings) -> NatsEventBus.
- Implements: EventBus from Task 4.
- Produces: ensure_grid_stream() with GRID_EVENTS and grid.> subjects.

- [ ] **Step 1: Write failing unit tests around a fake nats-py context**

Assert JSON serialization, Msg-Id equal to event_id, explicit acknowledgements, durable pull-consumer configuration, max_deliver of five, backoff values, and conversion of raw NATS messages into the transport-neutral Message interface.

- [ ] **Step 2: Verify the adapter tests fail**

Run: python -m pytest tests/unit/messaging/test_nats_adapter.py -q

Expected: collection fails because messaging.nats does not exist.

- [ ] **Step 3: Implement JetStream setup and transport**

Create a file-backed GRID_EVENTS stream covering grid.>, max age fourteen days, duplicate window two hours, and discard-old retention. Publish with Nats-Msg-Id. Consumers use explicit ack, durable names supplied by services, five deliveries, and delays of 1, 5, 30, and 120 seconds before dead-letter handling by the service.

Use this NATS server configuration:

~~~conf
port: 4222
http_port: 8222
jetstream {
  store_dir: /data/jetstream
  max_mem_store: 256MB
  max_file_store: 10GB
}
~~~

- [ ] **Step 4: Add an opt-in real JetStream integration test**

The integration test connects to IMBALANCE_NATS_URL, ensures the stream, publishes one envelope twice with the same message ID, consumes it through a durable, acknowledges it, reconnects, and verifies no pending redelivery. Mark it integration so the default unit suite does not require Docker.

- [ ] **Step 5: Verify unit behavior**

Run: python -m pytest tests/unit/messaging/test_nats_adapter.py -q

Expected: all adapter tests pass.

- [ ] **Step 6: Commit NATS transport**

Run:

~~~bash
git add src/imbalance_pipeline/messaging/nats.py config/nats tests/unit/messaging tests/integration/test_jetstream.py
git commit -m "feat: add durable JetStream transport"
~~~

### Task 6: ClickHouse Schema, Repository, and Idempotent Sink

**Files:**
- Create: infra/clickhouse/001_schema.sql
- Create: src/imbalance_pipeline/storage/__init__.py
- Create: src/imbalance_pipeline/storage/clickhouse.py
- Create: src/imbalance_pipeline/services/sink.py
- Create: tests/unit/storage/test_sink.py
- Create: tests/integration/test_clickhouse.py

**Interfaces:**
- Produces: ClickHouseRepository.insert_event(event: EventEnvelope) -> None.
- Produces: ClickHouseRepository.fetch_imbalance_window(cutoff: datetime, minutes: int) -> list[ImbalanceObservation].
- Produces: ClickHouseRepository.insert_feature_snapshot(snapshot: FeatureSnapshot) -> None.
- Produces: ClickHouseRepository.latest_prediction() -> Prediction | None.
- Produces: Sink.handle(message: Message) -> None.
- Consumes: EventBus and EventEnvelope from Tasks 2, 4, and 5.

- [ ] **Step 1: Write failing sink idempotence and ordering tests**

Use fake repository and bus implementations. Assert that insertion occurs before the stored event, the stored event keeps the source correlation and deterministic identity, acknowledgement occurs only after both succeed, a transient insert failure naks without publishing, and a schema error publishes to grid.dlq.clickhouse.v1 before acknowledgement.

~~~python
@pytest.mark.asyncio
async def test_sink_publishes_stored_trigger_before_ack() -> None:
    repository = RecordingRepository()
    bus = InMemoryEventBus()
    message = FakeMessage(IMBALANCE_EVENT)
    await Sink(repository, bus).handle(message)
    assert repository.calls == [("insert", IMBALANCE_EVENT.event_id)]
    assert bus.published[0][0] == "grid.stored.elia.imbalance.v1"
    assert message.acked is True
~~~

- [ ] **Step 2: Verify sink tests fail**

Run: python -m pytest tests/unit/storage/test_sink.py -q

Expected: collection fails because storage.clickhouse and services.sink are absent.

- [ ] **Step 3: Create the complete ClickHouse migration**

Create the imbalance database and these tables with explicit columns:

- raw_events: event_id, event_type, schema_version, source, dataset, event_time, observed_at Nullable, ingested_at, correlation_id, causation_id Nullable, quality_status, payload_json, row_version.
- imbalance_observations: event_id, timestamp, quarter_hour, resolution_code, quality_status, ace_mw Nullable, system_imbalance_mw, alpha_eur_mwh Nullable, alpha_prime_eur_mwh Nullable, marginal_incremental_price_eur_mwh Nullable, marginal_decremental_price_eur_mwh Nullable, imbalance_price_eur_mwh Nullable, ingested_at, row_version.
- load_observations, wind_observations, and solar_observations: event_id, timestamp, dimensions, realtime Nullable, most_recent_forecast Nullable, confidence bounds Nullable, day-ahead values Nullable, capacity/load factor Nullable, ingested_at, row_version.
- feature_snapshots: event_id, cutoff, target_time, feature_schema_hash, local_values Array(Float32), local_masks Array(UInt8), context_values Array(Float32), context_masks Array(UInt8), static_values Array(Float32), static_masks Array(UInt8), current_state LowCardinality(Nullable(String)), created_at, row_version.
- predictions: event_id, cutoff, target_time, generated_at, system_imbalance_mw, p10_mw, p90_mw, flip_probability, will_flip, current_state Nullable, predicted_state Nullable, prediction_quality LowCardinality(String), model_version, feature_schema_hash, row_version.
- prediction_outcomes: prediction_event_id, target_time, realized_event_id, realized_system_imbalance_mw, realized_state Nullable, flip_actual Nullable(UInt8), evaluated_at.
- model_versions: model_version, feature_schema_hash, manifest_json, metrics_json, promoted_at Nullable, created_at.

Use ReplacingMergeTree(row_version), monthly partitions for time-series tables, ORDER BY natural series keys, and raw-event TTL of 90 days. Add schema_migrations so startup applies each migration exactly once. Grant the application user only database-scoped read/write access.

- [ ] **Step 4: Implement typed repository operations**

Use clickhouse_connect.get_async_client and parameterized inserts. Convert datetimes to UTC before every write. Query canonical rows with argMax tuple aggregation or ORDER BY row_version DESC LIMIT 1; never assume background replacement has finished. Make insert_event route only known event types and raise PermanentEventError for schema/version violations.

- [ ] **Step 5: Implement the sink state machine**

Subscribe only to raw, feature, prediction, outcome, and model subjects. For a raw imbalance event:

1. insert raw and normalized rows under the same deterministic version;
2. publish grid.stored.elia.imbalance.v1 with an identifier derived from the source event;
3. acknowledge the source message.

On TransientStorageError, nak with the delivery-specific delay. On a permanent error or fifth delivery, publish a DeadLetterPayload and acknowledge. Reprocessing the same source event must leave canonical queries with one row.

- [ ] **Step 6: Add real ClickHouse integration coverage**

Against IMBALANCE_CLICKHOUSE_URL, apply the migration, insert the same event twice, query without FINAL through the canonical repository method, and assert one logical observation. Insert two versions of one natural key and assert the newest is returned.

- [ ] **Step 7: Verify storage**

Run: python -m pytest tests/unit/storage/test_sink.py -q

Expected: all unit storage tests pass.

Run with ClickHouse running: python -m pytest tests/integration/test_clickhouse.py -m integration -q

Expected: idempotence and latest-version integration tests pass.

- [ ] **Step 8: Commit storage**

Run:

~~~bash
git add infra/clickhouse src/imbalance_pipeline/storage src/imbalance_pipeline/services/sink.py tests/unit/storage tests/integration/test_clickhouse.py
git commit -m "feat: persist grid events in ClickHouse"
~~~

### Task 7: Ordered Feature Schema and Point-in-Time Feature Engine

**Files:**
- Create: src/imbalance_pipeline/features/__init__.py
- Create: src/imbalance_pipeline/features/schema.py
- Create: src/imbalance_pipeline/features/engine.py
- Create: tests/unit/features/test_schema.py
- Create: tests/unit/features/test_engine.py
- Create: tests/unit/features/test_no_leakage.py

**Interfaces:**
- Produces: FeatureRegistry.local_names, context_names, static_names, and fingerprint.
- Produces: FeatureSnapshot with local [180, F_local], context [96, F_context], static [F_static], matching masks, cutoff, target_time, and current_state.
- Produces: FeatureEngine.build(cutoff: datetime) -> FeatureSnapshot.
- Consumes: repository window-query methods from Task 6.

- [ ] **Step 1: Write failing schema, shape, and causality tests**

Test that registry ordering is stable and SHA-256 fingerprinted; a 180-minute source series creates exactly 180 local rows; the context has 96 rows; the target is cutoff plus one minute; the state follows hysteresis; missing exogenous values are zero-filled with mask 0 and increasing age; and changing any database row after cutoff leaves the complete snapshot byte-identical.

~~~python
def test_future_rows_cannot_change_snapshot(repository: MemoryFeatureSource) -> None:
    engine = FeatureEngine(repository, DEFAULT_FEATURE_REGISTRY)
    before = engine.build(CUTOFF)
    repository.add_imbalance(CUTOFF + timedelta(minutes=1), 9999.0)
    after = engine.build(CUTOFF)
    np.testing.assert_array_equal(before.local_values, after.local_values)
    assert before.feature_schema_hash == after.feature_schema_hash
~~~

- [ ] **Step 2: Verify feature tests fail**

Run: python -m pytest tests/unit/features -q

Expected: collection fails because the feature package is absent.

- [ ] **Step 3: Define an explicit feature registry**

Register every value with name, group, dtype, scaling kind, and maximum source age. Include:

- local raw: system_imbalance_mw, ace_mw, imbalance_price_eur_mwh, marginal prices, quality masks;
- local dynamics: delta_1, delta_2, acceleration, EWM 5/15/60, rolling median 5/15, robust scale 15/60, slope 5/15, min/max 15, boundary distance, confirmed-state sign, state duration;
- context: aggregated imbalance mean/min/max/std, load actual/forecast/error, wind actual/forecast/error/spread, solar actual/forecast/error/spread, source ages and masks;
- static: quarter-hour phase sin/cos, minute/hour/weekday/day-of-year sin/cos, weekend, Belgian holiday, DST transition.

Fingerprint canonical JSON containing ordered names, groups, dtypes, scaling kinds, window sizes, and deadband.

- [ ] **Step 4: Implement causal construction**

Query each source with available_at <= cutoff. Reindex local data to exact UTC minute boundaries ending at cutoff and context data to exact 15-minute boundaries ending at floor(cutoff, 15 minutes). Compute rolling values with right-closed past-only windows. Fit no scaler online: the engine emits engineering-space floats and masks; model-bundle preprocessing performs stored robust scaling.

Pad absent history with zero values and zero masks. Require at least 30 observed imbalance minutes for model quality; set snapshot.model_eligible false below that threshold so serving uses fallback.

- [ ] **Step 5: Prove offline/online parity**

Add build_many(cutoffs: Sequence[datetime]) implemented by repeatedly calling the same private transform primitives as build. For ten deterministic cutoffs, assert each offline row exactly matches its online snapshot and that no future mutation changes earlier rows.

- [ ] **Step 6: Verify features**

Run: python -m pytest tests/unit/features -q

Expected: all feature, parity, shape, and leakage tests pass.

Run: ruff check src/imbalance_pipeline/features tests/unit/features

Expected: All checks passed.

- [ ] **Step 7: Commit feature engine**

Run:

~~~bash
git add src/imbalance_pipeline/features tests/unit/features
git commit -m "feat: build point-in-time imbalance features"
~~~

### Task 8: Stored-Event Feature Service and Outcome Joiner

**Files:**
- Create: src/imbalance_pipeline/services/features.py
- Create: src/imbalance_pipeline/services/outcomes.py
- Create: tests/unit/services/test_feature_service.py
- Create: tests/unit/services/test_outcome_service.py

**Interfaces:**
- Consumes: grid.stored.elia.imbalance.v1.
- Produces: grid.features.imbalance.v1 with FeatureSnapshot payload.
- Produces: grid.outcomes.imbalance.v1 after matching target-time predictions.

- [ ] **Step 1: Write failing orchestration tests**

Assert the feature service never subscribes to grid.raw.elia.imbalance.v1, builds only after the stored trigger, publishes one deterministic snapshot event, and acknowledges only after publication. Assert the outcome service joins every prediction whose target_time equals the realized observation timestamp, preserves prediction immutability, and emits flip_actual null when the prior confirmed state is unknown.

- [ ] **Step 2: Verify service tests fail**

Run: python -m pytest tests/unit/services/test_feature_service.py tests/unit/services/test_outcome_service.py -q

Expected: collection fails because the service modules are absent.

- [ ] **Step 3: Implement both durable consumers**

FeatureService uses durable feature-builder-v1. OutcomeService uses durable outcome-builder-v1. Each detects duplicate output through deterministic event identity rather than in-memory state. Treat insufficient history as a valid feature snapshot with model_eligible false. Send permanent payload/schema failures to the service-specific dead-letter subject.

- [ ] **Step 4: Verify orchestration**

Run: python -m pytest tests/unit/services/test_feature_service.py tests/unit/services/test_outcome_service.py -q

Expected: all orchestration tests pass.

- [ ] **Step 5: Commit feature and outcome services**

Run:

~~~bash
git add src/imbalance_pipeline/services/features.py src/imbalance_pipeline/services/outcomes.py tests/unit/services
git commit -m "feat: orchestrate features and realized outcomes"
~~~

### Task 9: Gaussian-Mixture Math and TCN-Transformer Network

**Files:**
- Create: src/imbalance_pipeline/model/__init__.py
- Create: src/imbalance_pipeline/model/distribution.py
- Create: src/imbalance_pipeline/model/network.py
- Create: tests/unit/model/test_distribution.py
- Create: tests/unit/model/test_network.py

**Interfaces:**
- Produces: gaussian_mixture_nll(target, logits, means, log_scales) -> Tensor.
- Produces: gaussian_mixture_cdf(value, logits, means, log_scales) -> Tensor.
- Produces: gaussian_mixture_quantile(probability, logits, means, log_scales) -> Tensor.
- Produces: ImbalanceForecaster.forward(local, local_mask, context, context_mask, static, static_mask) -> ModelOutput.
- ModelOutput contains mixture_logits [B,5], mixture_means [B,5], mixture_log_scales [B,5], flip_logit [B], delta [B], and auxiliary_horizons [B,3].

- [ ] **Step 1: Write failing distribution tests**

Use a one-component standard normal to assert CDF(0)=0.5, median=0, monotonic quantiles, finite NLL for extreme but valid targets, positive scales after transformation, and normalized mixture weights. Use a two-component symmetric mixture to assert a zero median within numerical tolerance.

- [ ] **Step 2: Write failing network contract tests**

Instantiate a tiny network with local_features=8, context_features=6, static_features=4, d_model=16, two TCN blocks, one Transformer layer, and five mixture components. Assert all output shapes, finite values under missing masks, gradients reaching both temporal branches, and deterministic eval output.

- [ ] **Step 3: Verify model tests fail**

Run: python -m pytest tests/unit/model -q

Expected: collection fails because the model package is absent.

- [ ] **Step 4: Implement numerically stable mixture functions**

Use log_softmax, scale = softplus(log_scale) + 1e-4, logsumexp NLL, and torch.erf for Gaussian CDF. Implement quantiles with exactly 64 bisection iterations over the component mean range expanded by twelve maximum scales. Keep all functions vectorized and ONNX-exportable except the serving quantile helper, which may run in NumPy/SciPy over raw ONNX outputs.

- [ ] **Step 5: Implement the compact hybrid network**

Build:

1. masked per-feature input projections;
2. six residual causal Conv1d blocks with kernel 3 and dilations 1, 2, 4, 8, 16, 32;
3. non-overlapping temporal patches and positional embeddings;
4. two batch-first TransformerEncoder layers with four heads;
5. a gated residual MLP for static and latest exogenous context;
6. gated fusion with LayerNorm;
7. mixture, flip, delta, and 2/5/10-minute heads.

Mask zero-filled values before projection and concatenate mask channels so missingness remains observable. Keep the default member below five million trainable parameters and expose count_parameters().

- [ ] **Step 6: Verify model math and architecture**

Run: python -m pytest tests/unit/model -q

Expected: all distribution and network tests pass.

Run: python -c "from imbalance_pipeline.model.network import default_model; assert default_model().count_parameters() < 5_000_000"

Expected: command exits zero.

- [ ] **Step 7: Commit the model architecture**

Run:

~~~bash
git add src/imbalance_pipeline/model tests/unit/model
git commit -m "feat: add probabilistic TCN transformer"
~~~

### Task 10: Training Dataset, Labels, Splits, and Multi-Task Loss

**Files:**
- Create: src/imbalance_pipeline/model/losses.py
- Create: src/imbalance_pipeline/training/__init__.py
- Create: src/imbalance_pipeline/training/data.py
- Create: src/imbalance_pipeline/training/splits.py
- Create: tests/unit/training/test_data.py
- Create: tests/unit/training/test_splits.py
- Create: tests/unit/model/test_losses.py

**Interfaces:**
- Produces: TrainingExample with feature tensors, y_next, y_delta, y_aux [3], current_state, flip_target, flip_mask.
- Produces: RobustPreprocessor.fit(training_examples) and transform(snapshot).
- Produces: walk_forward_splits(timestamps, folds: int, gap_minutes: int) -> list[TimeSplit].
- Produces: MultiTaskLoss.forward(output: ModelOutput, batch: TrainingBatch) -> LossBreakdown.

- [ ] **Step 1: Write failing label and preprocessing tests**

Assert targets use t+1, t+2, t+5, and t+10; a neutral first state masks flip loss; positive to neutral is not a flip; positive through neutral to negative is a flip only when the target minute establishes negative; robust location and scale are fit on training rows only; masks remain binary; and transformation never changes padded zeros into observed values.

- [ ] **Step 2: Write failing chronological split tests**

For one year of minute timestamps, create three folds. Assert train < gap < validation < gap < test, every gap is at least 1,440 minutes because the largest input context is 24 hours, no timestamp appears in multiple roles inside a fold, and adding future rows does not alter earlier folds.

- [ ] **Step 3: Write failing loss tests**

Assert:

- perfect mixture location lowers NLL relative to a distant location;
- masked flip targets contribute exactly zero;
- focal loss is finite for logits of +/-100;
- positive current state compares flip probability to mixture CDF at -10 MW;
- negative current state compares it to one minus mixture CDF at +10 MW;
- all four model heads receive nonzero gradients.

- [ ] **Step 4: Verify training-foundation tests fail**

Run: python -m pytest tests/unit/training/test_data.py tests/unit/training/test_splits.py tests/unit/model/test_losses.py -q

Expected: collection fails because the files are absent.

- [ ] **Step 5: Implement samples and train-only preprocessing**

Create examples only when y_next exists. Keep MW regression examples when state is unknown, with flip_mask=0. Robust preprocessing stores per-feature median and IQR clipped to a minimum scale of 1e-6, ignores mask-zero values, clips transformed observed values to [-12, 12], restores missing entries to zero, and serializes names, values, and schema fingerprint to JSON.

- [ ] **Step 6: Implement purged walk-forward splits**

Use chronological indices and an explicit gap equal to max(1,440, configured context minutes). Fail with a clear ValueError if a dataset cannot form the requested folds with nonempty train, validation, calibration, and test segments. Return immutable TimeSplit records with inclusive start and exclusive end timestamps.

- [ ] **Step 7: Implement stable multi-task loss**

Use these initial weights:

- mixture NLL: 1.0;
- class-balanced focal flip loss with gamma 2: 0.5;
- Huber one-minute delta: 0.2;
- mean Huber 2/5/10-minute auxiliary loss: 0.1;
- flip/distribution consistency BCE: 0.1.

Compute positive flip weight from the training fold, clipped to [1, 20]. Log every unweighted component. Reject nonfinite output with a batch identifier instead of continuing optimization.

- [ ] **Step 8: Verify dataset, splits, and losses**

Run: python -m pytest tests/unit/training/test_data.py tests/unit/training/test_splits.py tests/unit/model/test_losses.py -q

Expected: all tests pass.

- [ ] **Step 9: Commit training foundations**

Run:

~~~bash
git add src/imbalance_pipeline/model/losses.py src/imbalance_pipeline/training tests/unit/training tests/unit/model/test_losses.py
git commit -m "feat: add leakage-safe training foundations"
~~~

### Task 11: Baselines, Calibration, Trainer, and Promotion Report

**Files:**
- Create: src/imbalance_pipeline/training/baselines.py
- Create: src/imbalance_pipeline/training/metrics.py
- Create: src/imbalance_pipeline/training/calibration.py
- Create: src/imbalance_pipeline/training/train.py
- Create: src/imbalance_pipeline/training/export_data.py
- Create: tests/unit/training/test_metrics.py
- Create: tests/unit/training/test_calibration.py
- Create: tests/unit/training/test_trainer_smoke.py

**Interfaces:**
- Produces: evaluate_predictions(frame) -> EvaluationReport.
- Produces: IsotonicCalibrator.fit(probability, target) and JSON round trip.
- Produces: train_ensemble(dataset_path: Path, output_dir: Path, config: TrainingConfig) -> Path to candidate bundle.
- Produces: promotion_decision(candidate: EvaluationReport, baselines: BaselineReports) -> PromotionDecision.

- [ ] **Step 1: Write failing metric and promotion-gate tests**

Use fixed arrays with hand-computed MAE, RMSE, Brier, log loss, precision, recall, F1, and interval coverage. Assert promotion requires at least 2% aggregate MAE improvement over persistence, no critical cohort worse by more than 5%, higher flip PR-AUC, at least 1% lower Brier score than the classical baseline, and P10-P90 coverage between 75% and 85%.

- [ ] **Step 2: Write failing calibration tests**

Fit isotonic calibration on an intentionally overconfident sequence. Assert calibrated Brier does not worsen on that fitting fixture, output stays in [0,1], breakpoints are monotonic, and JSON restore returns identical probabilities. Threshold selection maximizes F1 with deterministic lower-threshold tie breaking.

- [ ] **Step 3: Write a failing tiny training smoke test**

Generate 512 deterministic synthetic examples, train a tiny member for two epochs on CPU, assert finite decreasing training loss, write three seeded member checkpoints for seeds 17, 29, and 43, and create evaluation.json. Mark the test slow only if it exceeds ten seconds on the reference machine.

- [ ] **Step 4: Verify training tests fail**

Run: python -m pytest tests/unit/training/test_metrics.py tests/unit/training/test_calibration.py tests/unit/training/test_trainer_smoke.py -q

Expected: collection fails because the training modules are absent.

- [ ] **Step 5: Implement baselines and metrics**

Implement persistence, clipped recent linear drift, robust rolling median, HistGradientBoostingRegressor, and HistGradientBoostingClassifier on flattened point-in-time features. Evaluate each on exactly the same folds and masks as the neural model. Compute aggregate metrics plus volatility tercile, quarter-hour phase, source-quality, and current-state cohorts.

- [ ] **Step 6: Implement serializable calibration**

Wrap sklearn IsotonicRegression(out_of_bounds="clip"). Persist only numeric x/y thresholds and the selected decision threshold as JSON, never a pickle. Reimplement inference with NumPy interpolation and prove parity with sklearn in the unit test.

- [ ] **Step 7: Implement deterministic ensemble training**

Training defaults:

- seeds 17, 29, 43;
- AdamW learning rate 3e-4 and weight decay 1e-4;
- batch size 256;
- gradient norm clipping at 1.0;
- automatic mixed precision only on CUDA;
- cosine decay after five warmup epochs;
- maximum 100 epochs;
- early stopping patience 10 on validation mixture NLL plus 0.25 times flip Brier;
- keep the best checkpoint per seed.

Fit preprocessing only on the training fold, select architecture and loss settings on validation, fit calibration on the separate calibration range, and report the untouched test once. Save run_config.json, preprocessing.json, evaluation.json, baseline_evaluation.json, calibration.json, and member checkpoints.

- [ ] **Step 8: Implement point-in-time data export**

Query ClickHouse cutoffs in chronological batches, call FeatureEngine.build_many, attach labels only after features are frozen, and write compressed NPZ shards plus metadata.json. Include source min/max timestamps, schema hash, counts, masks, and SHA-256 per shard. Refuse an overwrite unless --force is explicitly supplied.

- [ ] **Step 9: Verify training**

Run: python -m pytest tests/unit/training -q

Expected: metric, calibration, split, data, and smoke tests pass.

Run: ruff check src/imbalance_pipeline/training tests/unit/training

Expected: All checks passed.

- [ ] **Step 10: Commit training and evaluation**

Run:

~~~bash
git add src/imbalance_pipeline/training tests/unit/training
git commit -m "feat: train and evaluate imbalance ensemble"
~~~

### Task 12: ONNX Export, Checksummed Bundles, Validation, and Runtime

**Files:**
- Create: src/imbalance_pipeline/model/bundle.py
- Create: src/imbalance_pipeline/model/export_onnx.py
- Create: src/imbalance_pipeline/serving/__init__.py
- Create: src/imbalance_pipeline/serving/runtime.py
- Create: scripts/build_test_bundle.py
- Create: models/test/manifest.json
- Generate: models/test/member-0.onnx
- Generate: models/test/member-1.onnx
- Generate: models/test/member-2.onnx
- Create: models/test/preprocessing.json
- Create: models/test/calibration.json
- Create: tests/unit/model/test_bundle.py
- Create: tests/unit/serving/test_runtime.py

**Interfaces:**
- Produces: ModelManifest.load(bundle_dir: Path) -> ModelManifest.
- Produces: validate_bundle(bundle_dir: Path, expected_schema_hash: str) -> ValidationResult.
- Produces: export_member(model, sample_batch, output_path) -> None.
- Produces: OnnxEnsemble.predict(snapshot: FeatureSnapshot) -> PredictionValues.
- Produces: promote_bundle(candidate: Path, model_root: Path) -> Path using an atomic production symlink replacement.

- [ ] **Step 1: Write failing bundle-security and runtime tests**

Assert missing artifacts, checksum changes, wrong schema hashes, non-three-member bundles, invalid ONNX inputs, and out-of-range calibration fail closed. Assert a valid tiny bundle returns finite median/P10/P90, ordered quantiles, probability in [0,1], and deterministic repeated output.

- [ ] **Step 2: Write a failing ONNX parity test**

Export a tiny seeded PyTorch member and run 32 feature snapshots through PyTorch and ONNX Runtime. Compare mixture logits, means, log scales, flip logit, delta, and auxiliary output with maximum absolute difference below 1e-4.

- [ ] **Step 3: Verify artifact tests fail**

Run: python -m pytest tests/unit/model/test_bundle.py tests/unit/serving/test_runtime.py -q

Expected: collection fails because bundle and runtime modules are absent.

- [ ] **Step 4: Implement safe manifests and atomic promotion**

Manifest fields are model_version, created_at, training_period, feature_schema_hash, members, preprocessing_file, calibration_file, evaluation_file, runtime, input_shapes, output_names, checksums, and promotion metrics. Reject absolute paths and parent traversal. Hash every artifact before opening ONNX sessions. Promotion copies to a versioned immutable directory, fsyncs files, validates there, and atomically replaces a relative production symlink.

- [ ] **Step 5: Export ONNX members**

Export with named inputs local, local_mask, context, context_mask, static, static_mask and named raw outputs. Use opset 18, fixed feature dimensions, dynamic batch axis only, model.eval(), and constant folding. Validate with onnx.checker and the parity test before adding checksums to the manifest.

- [ ] **Step 6: Implement CPU ensemble inference**

Create one ONNX Runtime session per member with CPUExecutionProvider and bounded intra-op threads. Apply stored preprocessing, average member Gaussian mixtures with equal 1/3 component weights, average flip logits, apply JSON isotonic calibration, compute quantiles by deterministic mixture-CDF bisection, derive predicted state, and return model_version and schema hash.

- [ ] **Step 7: Generate and commit a deterministic test bundle**

Run: python scripts/build_test_bundle.py --output models/test

Expected: three small ONNX members plus manifest, preprocessing, calibration, and evaluation files are created and validate successfully. The script fixes seeds and tiny dimensions so reruns are reproducible.

- [ ] **Step 8: Verify artifacts**

Run: python -m pytest tests/unit/model/test_bundle.py tests/unit/serving/test_runtime.py -q

Expected: bundle validation, parity, corruption, and runtime tests pass.

- [ ] **Step 9: Commit artifacts and runtime**

Run:

~~~bash
git add src/imbalance_pipeline/model src/imbalance_pipeline/serving scripts/build_test_bundle.py models/test tests/unit/model tests/unit/serving
git commit -m "feat: export and serve validated ONNX bundles"
~~~

### Task 13: Predictor Consumer, Persistence Fallback, and FastAPI

**Files:**
- Create: src/imbalance_pipeline/services/predictor.py
- Create: src/imbalance_pipeline/serving/api.py
- Create: src/imbalance_pipeline/serving/schemas.py
- Create: tests/unit/services/test_predictor.py
- Create: tests/unit/serving/test_api.py

**Interfaces:**
- Consumes: grid.features.imbalance.v1.
- Produces: grid.predictions.imbalance.v1.
- Exposes: create_app(repository, readiness, metrics) -> FastAPI.

- [ ] **Step 1: Write failing predictor tests**

Assert eligible snapshots invoke ONNX, ineligible snapshots produce y_hat equal to latest system imbalance with prediction_quality degraded, current-state-aware predicted state, calibrated thresholding, deterministic event IDs, and acknowledgement only after publication. Model errors use fallback only when allow_fallback is true.

- [ ] **Step 2: Write failing API contract tests**

Using FastAPI TestClient, assert:

- GET /v1/predictions/latest returns the documented JSON shape;
- time-range pagination is UTC and bounded to 10,000 rows;
- invalid or reversed ranges return 422;
- GET /v1/models/current omits filesystem paths;
- liveness returns 200 while the process runs;
- readiness returns 503 without dependencies and 200 with ClickHouse, NATS, and model or enabled fallback;
- GET /metrics uses Prometheus text format.

- [ ] **Step 3: Verify serving tests fail**

Run: python -m pytest tests/unit/services/test_predictor.py tests/unit/serving/test_api.py -q

Expected: tests fail because predictor and API modules are absent.

- [ ] **Step 4: Implement the durable predictor**

Load and warm the model before marking model readiness. Consume through durable predictor-v1. Record end-to-end latency from source ingested_at. For fallback, use the cutoff imbalance as point/P10/P90, derive next state without inventing a flip, set flip_probability to 0.0, will_flip false, and include a machine-readable degraded_reason.

- [ ] **Step 5: Implement API and metrics**

Use typed response models, UTC serialization with Z, structured error bodies, and repository keyset pagination by target_time plus event_id. Expose source freshness, consumer lag, insert latency, feature eligibility, inference latency, fallback count, model version, prediction distribution, and rolling outcome metrics without high-cardinality event labels.

- [ ] **Step 6: Verify serving**

Run: python -m pytest tests/unit/services/test_predictor.py tests/unit/serving/test_api.py -q

Expected: all predictor and API tests pass.

Run: mypy

Expected: Success: no issues found.

- [ ] **Step 7: Commit predictor and API**

Run:

~~~bash
git add src/imbalance_pipeline/services/predictor.py src/imbalance_pipeline/serving tests/unit/services/test_predictor.py tests/unit/serving
git commit -m "feat: serve next-minute imbalance predictions"
~~~

### Task 14: Docker Compose, Migrations, and Observability

**Files:**
- Create: Dockerfile
- Create: compose.yaml
- Create: Makefile
- Create: src/imbalance_pipeline/storage/migrate.py
- Create: infra/prometheus/prometheus.yml
- Create: infra/grafana/provisioning/datasources/prometheus.yml
- Create: infra/grafana/provisioning/dashboards/default.yml
- Create: infra/grafana/dashboards/pipeline.json
- Create: tests/unit/storage/test_migrations.py

**Interfaces:**
- Produces: python -m imbalance_pipeline.storage.migrate idempotently applies numbered SQL migrations.
- Produces: standard, training, and observability Compose profiles.
- Produces: make up, backfill, test, train, smoke, down, and clean commands.

- [ ] **Step 1: Write failing migration-ledger tests**

Use a fake ClickHouse command executor. Assert migrations are sorted numerically, each unapplied file runs once, checksum drift in an already applied migration is fatal, and failure does not record a ledger row.

- [ ] **Step 2: Verify migration tests fail**

Run: python -m pytest tests/unit/storage/test_migrations.py -q

Expected: collection fails because storage.migrate is absent.

- [ ] **Step 3: Implement the migration runner**

Read infra/clickhouse/[0-9][0-9][0-9]_*.sql, compute SHA-256, create schema_migrations if absent, compare applied checksums, execute one file at a time with multiquery enabled, and insert the ledger row only after successful completion. A lock row with a ten-minute expiry prevents concurrent migration services.

- [ ] **Step 4: Create a hardened multi-stage image**

Use python:3.12-slim as the pinned runtime family. Install the package into /opt/venv in a builder, copy only the virtual environment and application into runtime, create UID/GID 10001, run as that user, set PYTHONUNBUFFERED=1 and PYTHONDONTWRITEBYTECODE=1, and provide an HTTP health-check command. Use one image for every Python service with different Compose commands.

- [ ] **Step 5: Create the standard Compose graph**

Pin NATS to the 2.11-alpine family and ClickHouse to the 25.3-alpine LTS family. Define:

- nats with config/nats/nats.conf, health check on port 8222, and nats-data volume;
- clickhouse with application/database environment, health check using clickhouse-client SELECT 1, and clickhouse-data volume;
- migrate depending on healthy ClickHouse and completing successfully;
- ingestor depending on healthy NATS;
- sink, features, predictor-api, and outcomes depending on healthy NATS, healthy ClickHouse, and completed migration;
- predictor-api mounting models/production read-only and exposing port 8000;
- named application services with init true, stop_grace_period 30s, read_only true, tmpfs /tmp, no-new-privileges, bounded memory, and restart unless-stopped.

The training profile mounts data/exports and models/candidates. The observability profile pins Prometheus 3 and Grafana 12 families, persists Grafana state, and exposes ports 9090 and 3000 only when selected.

- [ ] **Step 6: Provision metrics and dashboard**

Prometheus scrapes every Python service and NATS monitoring. The Grafana JSON has panels for newest Elia age, ingest failures, JetStream pending/redelivery/DLQ, ClickHouse insert latency, feature eligibility, prediction latency, fallback rate, MW forecast distribution, flip probability, rolling MAE, and rolling Brier score. Use bounded labels and a default six-hour range.

- [ ] **Step 7: Add deterministic developer commands**

The Makefile targets execute:

~~~make
up:
	docker compose up --build -d

backfill:
	docker compose run --rm ingestor python -m imbalance_pipeline.services.ingestor --start 2024-05-22T00:00:00Z

test:
	python -m pytest -m "not integration and not live"

train:
	docker compose --profile training run --rm trainer

smoke:
	python -m pytest tests/e2e/test_live_smoke.py -m live -q

down:
	docker compose down

clean:
	docker compose down --volumes --remove-orphans
~~~

Document in the target help that clean destroys local pipeline data and require CONFIRM_CLEAN=1 before executing the destructive command.

- [ ] **Step 8: Verify container configuration**

Run: python -m pytest tests/unit/storage/test_migrations.py -q

Expected: all migration tests pass.

Run: docker compose config --quiet

Expected: command exits zero.

Run: docker build -t imbalance-pipeline:test .

Expected: image builds and its configured non-root user is 10001.

- [ ] **Step 9: Commit deployment**

Run:

~~~bash
git add Dockerfile compose.yaml Makefile infra src/imbalance_pipeline/storage/migrate.py tests/unit/storage/test_migrations.py
git commit -m "build: compose the complete prediction stack"
~~~

### Task 15: Full Pipeline Integration, Replay, and Live Elia Smoke Test

**Files:**
- Create: tests/e2e/fixture_source.py
- Create: tests/e2e/test_pipeline.py
- Create: tests/e2e/test_failure_recovery.py
- Create: tests/e2e/test_live_smoke.py
- Create: compose.test.yaml
- Create: scripts/wait_for_stack.py
- Create: scripts/smoke.sh

**Interfaces:**
- Produces: deterministic fixture API implementing the Explore records response.
- Verifies: source -> JetStream -> ClickHouse -> stored trigger -> features -> ONNX -> prediction -> API.

- [ ] **Step 1: Write the failing deterministic pipeline test**

Serve 240 historical fixture minutes followed by one new minute. Start the stack with the fixture base URL and models/test bundle. Wait for readiness, trigger one poll, and assert:

- the source event exists once in canonical ClickHouse queries;
- a feature snapshot targets source timestamp plus one minute;
- the snapshot schema matches the model bundle;
- a model-quality prediction appears at GET /v1/predictions/latest;
- prediction and source correlation IDs are connected;
- measured source-receipt-to-prediction latency is below ten seconds.

- [ ] **Step 2: Write failing recovery tests**

Cover:

1. send the same source row twice and assert one canonical observation and prediction;
2. stop the sink after ClickHouse insert but before ack, restart it, and assert no logical duplicate;
3. publish malformed payload and assert one DLQ event;
4. omit one minute and assert masks/staleness rather than forward-looking interpolation;
5. hide the model bundle and assert degraded fallback;
6. corrupt one model checksum and assert fallback plus readiness behavior;
7. restart NATS and ClickHouse containers and assert acknowledged data remains.

- [ ] **Step 3: Verify the E2E tests initially fail**

Run: docker compose -f compose.yaml -f compose.test.yaml up --build -d

Run: python -m pytest tests/e2e/test_pipeline.py tests/e2e/test_failure_recovery.py -m integration -q

Expected: tests expose missing wiring or configuration before the implementation is corrected.

- [ ] **Step 4: Complete entry-point wiring and readiness**

Wire settings, NATS, repository, model runtime, Prometheus server, signal handling, and durable names in every main function. Ensure all clients close on SIGTERM and incomplete messages remain unacked. Add startup logs with service/version/config fingerprints but no password.

- [ ] **Step 5: Implement the live smoke test**

Call ODS161 through EliaClient, require at least one PT1M result, normalize it, publish through the running pipeline, and verify a prediction or explicitly degraded fallback for the next minute. Mark live and skip only when IMBALANCE_RUN_LIVE_TESTS is not 1. Never assert a fixed current value.

- [ ] **Step 6: Run full integration and replay verification**

Run: python -m pytest tests/e2e/test_pipeline.py tests/e2e/test_failure_recovery.py -m integration -q

Expected: all deterministic pipeline and recovery scenarios pass.

Run: IMBALANCE_RUN_LIVE_TESTS=1 python -m pytest tests/e2e/test_live_smoke.py -m live -q

Expected: the current Elia response normalizes and flows to a next-minute prediction.

- [ ] **Step 7: Commit E2E coverage**

Run:

~~~bash
git add tests/e2e compose.test.yaml scripts/wait_for_stack.py scripts/smoke.sh src/imbalance_pipeline
git commit -m "test: verify the complete realtime pipeline"
~~~

### Task 16: Colab Training Driver, Documentation, and Final Verification

**Files:**
- Create: notebooks/train_colab.ipynb
- Create: README.md
- Create: docs/runbook.md
- Create: docs/model-card.md
- Modify: .env.example
- Modify: Makefile

**Interfaces:**
- Produces: a Colab notebook that calls the package exporter, trainer, evaluator, ONNX exporter, and bundle validator.
- Produces: operator procedures with exact commands and expected health states.

- [ ] **Step 1: Create a notebook structure test**

Add a unit test that parses notebooks/train_colab.ipynb as JSON and asserts cells exist in this order: environment check, repository install, dataset acquisition/upload, configuration, GPU training, evaluation display, ONNX export, validation, bundle download. Assert the notebook imports project functions and contains no copied network class or hard-coded credential.

- [ ] **Step 2: Build the thin Colab notebook**

The notebook:

1. selects CUDA when available and prints device/runtime versions;
2. clones or uploads the repository and installs .[ml];
3. accepts a ClickHouse export NPZ or optional Google Drive path;
4. runs train_ensemble with seeds 17/29/43 and mixed precision;
5. renders aggregate/cohort metrics and calibration plots;
6. runs promotion_decision;
7. exports and validates three ONNX members;
8. zips only the checksummed candidate bundle for download.

The notebook must stop with a clear assertion when a promotion gate fails; it may still allow downloading an explicitly named rejected-candidate bundle for analysis.

- [ ] **Step 3: Write the user and operator documentation**

README covers architecture, prerequisites, one-command startup, service URLs, backfill, fallback behavior, Colab workflow, model installation, all Make targets, test layers, directory map, API examples, and data-license links.

The runbook gives exact commands for source staleness, JetStream lag, DLQ inspection and bounded replay, ClickHouse health, model rollback, checksum failure, backup/restore, safe shutdown, and destructive local cleanup.

The model card documents target/label definitions, training range, features, architecture, metrics and cohorts, calibration, limitations, MARI regime boundary, non-trading warning, and model/version provenance. Generated reports fill numeric metrics after a real full training run; the committed test model card identifies itself as deterministic synthetic test data rather than claiming real performance.

- [ ] **Step 4: Run the complete local quality gate**

Run: ruff check .

Expected: All checks passed.

Run: mypy

Expected: Success: no issues found.

Run: python -m pytest -m "not integration and not live" --cov=imbalance_pipeline --cov-report=term-missing

Expected: all unit and contract tests pass with at least 85% package line coverage.

Run: docker compose config --quiet

Expected: command exits zero.

Run: docker compose -f compose.yaml -f compose.test.yaml up --build -d

Run: python -m pytest -m integration tests/integration tests/e2e -q

Expected: all deterministic integration and E2E tests pass.

Run: docker compose -f compose.yaml -f compose.test.yaml down --volumes

Expected: test containers and volumes are removed.

- [ ] **Step 5: Run the live, non-destructive acceptance smoke**

Run: docker compose up --build -d

Run: IMBALANCE_RUN_LIVE_TESTS=1 python -m pytest tests/e2e/test_live_smoke.py -m live -q

Expected: live Elia ingestion reaches the API as a model or marked fallback prediction.

Run: curl --fail http://localhost:8000/health/ready

Expected: HTTP 200 with dependency and model/fallback readiness fields.

- [ ] **Step 6: Commit documentation and verified handoff**

Run:

~~~bash
git add notebooks README.md docs .env.example Makefile
git commit -m "docs: complete pipeline training and operations guide"
~~~

### Task 17: Independent Review Fixes and Release Snapshot

**Files:**
- Modify: only files identified by specification and code review.
- Create: docs/verification-report.md

**Interfaces:**
- Produces: a final repository state whose evidence maps every approved design requirement to a passing check.

- [ ] **Step 1: Request specification compliance review**

Have a fresh reviewer compare the implementation to docs/superpowers/specs/2026-07-13-realtime-system-imbalance-pipeline-design.md and return only concrete missing or contradictory requirements with file and line evidence.

- [ ] **Step 2: Request code-quality review**

Have a different fresh reviewer inspect correctness, event ordering, idempotence, leakage, numerical stability, error handling, security, and test strength. Require severity, reproducible evidence, and a proposed narrow fix for each finding.

- [ ] **Step 3: Reproduce every accepted finding with a failing test**

Add the smallest test that fails for each confirmed issue. Run each target test and record the expected failure before changing implementation.

- [ ] **Step 4: Fix confirmed findings and rerun focused tests**

Implement only evidence-backed fixes. Run focused tests until each passes, then run all unit, integration, and E2E gates from Task 16.

- [ ] **Step 5: Write the verification report**

Record exact commands, exit codes, test counts, coverage, image build result, integration result, live smoke result, active model type, and known environmental limitations. Do not claim full-model accuracy until the real historical Colab training and untouched test evaluation have completed.

- [ ] **Step 6: Commit the release snapshot**

Run:

~~~bash
git add src tests docs compose.yaml Dockerfile Makefile notebooks models/test
git commit -m "chore: finalize verified imbalance pipeline"
~~~
