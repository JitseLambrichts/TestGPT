# Live Imbalance Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live-only, auto-refreshing FastAPI dashboard that compares predicted and realized Belgian system imbalance and shows whether each flip prediction was correct.

**Architecture:** Extend `ClickHouseRepository` with one canonical prediction/outcome join and expose it through a bounded `/v1/dashboard` endpoint. Serve a dependency-light HTML/CSS/JavaScript UI from the same FastAPI process, with a locally vendored Chart.js build and no runtime network dependency.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, ClickHouse, pytest, HTML5, CSS, browser JavaScript, Chart.js 4.x

## Global Constraints

- Use only live records already stored by the pipeline; never synthesize dashboard values.
- Refresh every 15 seconds and retain the last successful view after a failed refresh.
- Query from `now - 6 hours` through `now + 2 minutes` so the latest t+1 forecast is included.
- Display time in `Europe/Brussels`; preserve UTC in the API.
- Serve Chart.js locally with its license; do not add a CDN, Node runtime, build pipeline, or frontend container.
- Keep the existing prediction and outcome event contracts and existing API response shapes unchanged.
- A flip is `Correct` only when `flip_actual` is known and equals `will_flip`; null is `In afwachting`.
- Preserve all pre-existing uncommitted worktree changes and stage only dashboard-related paths.

---

## File structure

- `src/imbalance_pipeline/storage/clickhouse.py`: immutable dashboard row and bounded canonical join query.
- `src/imbalance_pipeline/serving/api.py`: repository protocol, JSON route, page route, and static mount.
- `src/imbalance_pipeline/serving/dashboard/index.html`: semantic dashboard shell and stable DOM hooks.
- `src/imbalance_pipeline/serving/dashboard/dashboard.css`: complete responsive visual system.
- `src/imbalance_pipeline/serving/dashboard/dashboard.js`: refresh lifecycle, formatting, cards, Chart.js update, table, and status handling.
- `src/imbalance_pipeline/serving/dashboard/vendor/chart.umd.min.js`: pinned Chart.js browser distribution.
- `src/imbalance_pipeline/serving/dashboard/vendor/LICENSE.md`: matching Chart.js license.
- `tests/unit/storage/test_dashboard_query.py`: query contract and dashboard model tests using a recording ClickHouse client.
- `tests/unit/serving/test_api.py`: route, validation, error, page, and asset behavior.
- `tests/unit/serving/test_dashboard_assets.py`: HTML/JavaScript/CSS contract and local-vendor checks.
- `tests/integration/test_clickhouse.py`: real canonical prediction/outcome join behavior.
- `pyproject.toml`: wheel inclusion for dashboard assets.
- `README.md`: dashboard URL and live-only behavior.

---

### Task 1: Canonical prediction/outcome dashboard rows

**Files:**
- Modify: `src/imbalance_pipeline/storage/clickhouse.py`
- Create: `tests/unit/storage/test_dashboard_query.py`
- Modify: `tests/integration/test_clickhouse.py`

**Interfaces:**
- Consumes: existing `Prediction`, `prediction_outcomes`, `_clickhouse_utc`, and `TransientStorageError`.
- Produces: `DashboardRow` and `ClickHouseRepository.list_dashboard_rows(*, start: datetime, end: datetime, limit: int) -> list[DashboardRow]`.

- [ ] **Step 1: Write failing model and recording-client tests**

Create tests that validate nullable outcomes and inspect the generated bounded join:

```python
@pytest.mark.asyncio
async def test_dashboard_rows_join_canonical_outcomes() -> None:
    client = RecordingClient(rows=[dashboard_row(realized_mw=-18.0, flip_actual=True)])
    repository = ClickHouseRepository(client, database="imbalance")

    rows = await repository.list_dashboard_rows(start=START, end=END, limit=500)

    assert rows[0].realized_system_imbalance_mw == -18.0
    assert rows[0].flip_actual is True
    assert "LEFT JOIN canonical_outcomes" in client.query_text
    assert client.parameters == {"start": START, "end": END, "limit": 500}


@pytest.mark.asyncio
async def test_dashboard_rows_preserve_pending_outcomes() -> None:
    client = RecordingClient(rows=[dashboard_row(realized_mw=None, flip_actual=None)])
    rows = await ClickHouseRepository(client).list_dashboard_rows(
        start=START, end=END, limit=500
    )
    assert rows[0].realized_system_imbalance_mw is None
    assert rows[0].flip_actual is None
```

Also assert that reversed ranges, naive timestamps, and limits outside `1..10_000` raise `ValueError`, and that ClickHouse/OSError failures become `TransientStorageError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest tests/unit/storage/test_dashboard_query.py -q`  
Expected: collection/import failure because `DashboardRow` and `list_dashboard_rows` do not exist.

- [ ] **Step 3: Add the immutable model and canonical join**

Add a frozen Pydantic model that repeats the prediction contract and adds nullable outcome fields:

```python
class DashboardRow(Prediction):
    realized_system_imbalance_mw: float | None = None
    realized_state: str | None = None
    flip_actual: bool | None = None
    evaluated_at: datetime | None = None
```

Validate `evaluated_at` as UTC when present. Implement `list_dashboard_rows` with two CTEs. `canonical_predictions` uses the existing prediction `argMax` tuple grouped by `event_id`; `canonical_outcomes` uses `argMax(tuple(realized_system_imbalance_mw, realized_state, flip_actual, evaluated_at), tuple(row_version, evaluated_at, realized_event_id))` grouped by `(prediction_event_id, target_time)`. Left-join on both identifiers, filter prediction target time with typed parameters, order ascending by `(target_time, event_id)`, and use a typed `UInt32` limit.

- [ ] **Step 4: Run unit tests and verify GREEN**

Run: `pytest tests/unit/storage/test_dashboard_query.py tests/unit/storage/test_sink.py -q`  
Expected: all selected tests pass.

- [ ] **Step 5: Add the live ClickHouse integration test**

Insert one prediction with a matching outcome and one later prediction without an outcome. Assert the result list contains both rows, the first is realized, and the second has null outcome fields. Insert a newer revision of the outcome before reading and assert only the newer realized values are returned.

- [ ] **Step 6: Run integration coverage when the test stack is available**

Run: `docker compose -f compose.test.yaml up --build --abort-on-container-exit --exit-code-from tests`  
Expected: the integration suite exits with code 0. If Docker is unavailable, record the environmental reason and retain the unit query-contract coverage.

- [ ] **Step 7: Commit the storage slice**

```bash
git add src/imbalance_pipeline/storage/clickhouse.py tests/unit/storage/test_dashboard_query.py tests/integration/test_clickhouse.py
git commit -m "feat: query live dashboard outcomes"
```

---

### Task 2: Dashboard JSON and static delivery routes

**Files:**
- Modify: `src/imbalance_pipeline/serving/api.py`
- Modify: `tests/unit/serving/test_api.py`
- Create: `src/imbalance_pipeline/serving/dashboard/index.html`
- Create: `src/imbalance_pipeline/serving/dashboard/dashboard.css`
- Create: `src/imbalance_pipeline/serving/dashboard/dashboard.js`

**Interfaces:**
- Consumes: `DashboardRow` and `PredictionRepository.list_dashboard_rows`.
- Produces: `GET /v1/dashboard`, `GET /dashboard`, and `/dashboard-assets/*`.

- [ ] **Step 1: Extend the fake repository and write failing route tests**

Add `dashboard_result` and `list_dashboard_rows` to `FakePredictionRepository`. Test:

```python
def test_dashboard_endpoint_returns_live_rows() -> None:
    row = dashboard_row(realized_system_imbalance_mw=-12.0, flip_actual=True)
    client = TestClient(create_app(FakePredictionRepository(prediction(), [row])))
    response = client.get(
        "/v1/dashboard",
        params={"start": "2026-07-13T04:00:00Z", "end": "2026-07-13T10:02:00Z"},
    )
    assert response.status_code == 200
    assert response.json()["items"][0]["flip_actual"] is True


def test_dashboard_page_uses_only_local_assets() -> None:
    response = TestClient(create_app(FakePredictionRepository(None, []))).get("/dashboard")
    assert response.status_code == 200
    assert 'src="/dashboard-assets/vendor/chart.umd.min.js"' in response.text
    assert "https://" not in response.text
```

Add validation cases for reversed/naive/non-UTC ranges, limit `10_001`, and a `503` repository failure.

- [ ] **Step 2: Run API tests and verify RED**

Run: `pytest tests/unit/serving/test_api.py -q`  
Expected: new dashboard cases fail with `404` or missing protocol methods.

- [ ] **Step 3: Implement API and static routes**

Add `DashboardPage`, extend the protocol, and mount package-relative assets:

```python
DASHBOARD_DIR = Path(__file__).with_name("dashboard")
app.mount("/dashboard-assets", StaticFiles(directory=DASHBOARD_DIR), name="dashboard-assets")

@app.get("/dashboard", response_class=FileResponse, include_in_schema=False)
async def dashboard_page() -> FileResponse:
    return FileResponse(DASHBOARD_DIR / "index.html")

@app.get("/v1/dashboard", response_model=DashboardPage)
async def dashboard_data(start: datetime, end: datetime, limit: int = Query(500, ge=1, le=10_000)) -> DashboardPage:
    start = _utc_query_timestamp(start, "start")
    end = _utc_query_timestamp(end, "end")
    if end < start:
        raise HTTPException(status_code=422, detail="end must not precede start")
    try:
        rows = await repository.list_dashboard_rows(start=start, end=end, limit=limit)
    except TransientStorageError as exc:
        raise HTTPException(status_code=503, detail="prediction store is temporarily unavailable") from exc
    return DashboardPage(items=rows)
```

Use `_utc_query_timestamp` for both bounds and the existing sanitized `503` message. Add the semantic HTML shell with the required status region, five card hooks, chart canvas plus text summary, retry button, and exact table headings.

- [ ] **Step 4: Run API tests and verify GREEN**

Run: `pytest tests/unit/serving/test_api.py -q`  
Expected: all API tests pass, including existing prediction routes.

- [ ] **Step 5: Commit the route slice**

```bash
git add src/imbalance_pipeline/serving/api.py src/imbalance_pipeline/serving/dashboard tests/unit/serving/test_api.py
git commit -m "feat: serve live imbalance dashboard"
```

---

### Task 3: Chart.js UI, status lifecycle, cards, and history table

**Files:**
- Modify: `src/imbalance_pipeline/serving/dashboard/index.html`
- Modify: `src/imbalance_pipeline/serving/dashboard/dashboard.css`
- Modify: `src/imbalance_pipeline/serving/dashboard/dashboard.js`
- Create: `src/imbalance_pipeline/serving/dashboard/vendor/chart.umd.min.js`
- Create: `src/imbalance_pipeline/serving/dashboard/vendor/LICENSE.md`
- Create: `tests/unit/serving/test_dashboard_assets.py`

**Interfaces:**
- Consumes: `{items: DashboardRow[]}` from `/v1/dashboard` and global `Chart` from the local UMD bundle.
- Produces: `window.ImbalanceDashboard` pure helpers for contract tests plus the complete user-facing dashboard.

- [ ] **Step 1: Pin and vendor Chart.js from the official npm package**

Resolve the current stable Chart.js 4.x release from the npm registry, record the exact version and integrity in the license notice, and copy only `dist/chart.umd.min.js` plus the package license into `dashboard/vendor/`. Verify that HTML contains no `http://`, `https://`, or protocol-relative runtime asset reference.

- [ ] **Step 2: Write failing asset-contract tests**

Tests read the local assets and assert:

```python
def test_dashboard_has_required_live_regions() -> None:
    html = INDEX.read_text()
    for hook in ("live-status", "last-update", "model-version", "prediction-chart", "history-body"):
        assert f'id="{hook}"' in html
    assert 'aria-live="polite"' in html


def test_dashboard_javascript_uses_live_window_and_refresh_interval() -> None:
    javascript = SCRIPT.read_text()
    assert "15_000" in javascript
    assert "6 * 60 * 60 * 1000" in javascript
    assert "2 * 60 * 1000" in javascript
    assert "Europe/Brussels" in javascript
    assert "flip_actual === row.will_flip" in javascript
```

Also assert the table contains all eight approved columns, CSS has responsive breakpoints, visible focus styling, tabular numerals, and reduced-motion handling, and the Chart.js file plus license exist and are non-empty.

- [ ] **Step 3: Run asset tests and verify RED**

Run: `pytest tests/unit/serving/test_dashboard_assets.py -q`  
Expected: failures for the incomplete CSS/JavaScript behavior and missing vendor assets.

- [ ] **Step 4: Implement pure formatting and result helpers**

Expose a frozen helper object for inspection and use it internally:

```javascript
function resultFor(row) {
  if (row.flip_actual === null || row.flip_actual === undefined) return "pending";
  return row.flip_actual === row.will_flip ? "correct" : "incorrect";
}

function formatMw(value, fallback = "Nog niet gekend") {
  if (value === null || value === undefined || !Number.isFinite(value)) return fallback;
  return `${new Intl.NumberFormat("nl-BE", { signDisplay: "always", minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(value)} MW`;
}

window.ImbalanceDashboard = Object.freeze({ resultFor, formatMw });
```

Validate every received row before changing the page. Required prediction fields must be finite/typed; outcome fields may be null.

- [ ] **Step 5: Implement refresh and rendering behavior**

Build a request with ISO UTC bounds from six hours ago through two minutes ahead. Prevent overlapping fetches with an `AbortController`/in-flight guard. On success, update existing content and schedule the next 15-second refresh. On failure, preserve content, set `Verouderd` or `Niet bereikbaar`, show the retry action, and retain the last-success time. Pause the timer on hidden documents and refresh immediately on visibility return.

Derive forecast/state/quality from the newest prediction and actual/error from the newest realized row. Reverse rows only for the table. Render status text in addition to status color.

- [ ] **Step 6: Configure and update one Chart.js instance**

Use a linear x-scale with epoch milliseconds. Render p90 and p10 first so `fill: "-1"` creates the interval, then predicted and realized lines. Add a zero-line plugin drawn from the y=0 pixel without another dependency. Format x ticks and tooltips with `Intl.DateTimeFormat("nl-BE", { timeZone: "Europe/Brussels", hour: "2-digit", minute: "2-digit" })`. Disable parsing, use accessible chart summary text, update datasets in place, and use `chart.update("none")` after the initial render.

- [ ] **Step 7: Implement the responsive control-room styling**

Define CSS custom properties for navy ink, slate surfaces, forecast blue, realized teal, signal green, red, amber, spacing, radii, and shadows. Use a restrained top status rail, responsive five-card grid, a chart panel with a fixed minimum height, and a table that becomes horizontally scrollable inside its own region on narrow screens. Add `:focus-visible`, `font-variant-numeric: tabular-nums`, 44px minimum interactive targets, and `@media (prefers-reduced-motion: reduce)`.

- [ ] **Step 8: Run asset and API tests and verify GREEN**

Run: `pytest tests/unit/serving/test_dashboard_assets.py tests/unit/serving/test_api.py -q`  
Expected: all selected tests pass.

- [ ] **Step 9: Commit the complete UI slice**

```bash
git add src/imbalance_pipeline/serving/dashboard tests/unit/serving/test_dashboard_assets.py
git commit -m "feat: render live imbalance dashboard"
```

---

### Task 4: Package, document, and verify the dashboard

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `tests/unit/serving/test_dashboard_assets.py`

**Interfaces:**
- Consumes: complete dashboard assets and routes from Tasks 1-3.
- Produces: wheel/Docker inclusion, user documentation, and verification evidence.

- [ ] **Step 1: Write the failing wheel-content test**

Build a wheel into a temporary directory and inspect its ZIP members. Assert it contains:

```python
required = {
    "imbalance_pipeline/serving/dashboard/index.html",
    "imbalance_pipeline/serving/dashboard/dashboard.css",
    "imbalance_pipeline/serving/dashboard/dashboard.js",
    "imbalance_pipeline/serving/dashboard/vendor/chart.umd.min.js",
    "imbalance_pipeline/serving/dashboard/vendor/LICENSE.md",
}
assert required <= set(wheel_members)
```

- [ ] **Step 2: Run the packaging test and verify RED**

Run: `pytest tests/unit/serving/test_dashboard_assets.py -q`  
Expected: the wheel-content assertion fails until package inclusion is explicit.

- [ ] **Step 3: Include static assets and document usage**

Add dashboard assets to Hatch's wheel force-includes or artifacts configuration. Add a README section stating that `http://localhost:8000/dashboard` reads only live ClickHouse records, refreshes every 15 seconds, displays Brussels time, and may be empty until the pipeline has produced predictions/outcomes.

- [ ] **Step 4: Run quality and regression checks**

Run:

```bash
ruff check src tests
mypy src/imbalance_pipeline
pytest -m "not integration and not live" -q
```

Expected: all commands exit with code 0.

- [ ] **Step 5: Build the package and verify static contents**

Run: `python -m build --wheel` and inspect the resulting wheel.  
Expected: all five dashboard asset paths are present.

- [ ] **Step 6: Verify the containerized page when Docker is available**

Run `docker compose build api`, start the required stack, and request `/dashboard` plus a valid `/v1/dashboard` range.  
Expected: HTML returns `200`; JSON returns `200` with only live rows or an empty `items` list.

- [ ] **Step 7: Perform visual QA**

Open the local dashboard in the in-app browser. Inspect desktop and mobile widths, live/degraded/pending states that naturally exist, table overflow, tooltips, zero baseline, exact value formatting, refresh status, and reduced-motion behavior. Do not insert demonstration data to manufacture a state.

- [ ] **Step 8: Commit packaging and documentation**

```bash
git add pyproject.toml README.md tests/unit/serving/test_dashboard_assets.py
git commit -m "docs: ship live dashboard assets"
```

---

## Completion checklist

- [ ] Review `docs/superpowers/specs/2026-07-15-live-imbalance-dashboard-design.md` line by line and map every requirement to Tasks 1-4.
- [ ] Scan the plan for placeholder-marker phrases and confirm none remain.
- [ ] Confirm `DashboardRow`, `list_dashboard_rows`, `/v1/dashboard`, and every JavaScript field name are spelled consistently.
- [ ] Run `git status --short` and confirm unrelated pre-existing changes were not staged or modified by dashboard commits.
