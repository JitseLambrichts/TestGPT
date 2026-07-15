# Live imbalance dashboard

Date: 2026-07-15  
Status: approved design  
Audience: users who need to understand the current Belgian system-imbalance forecast without using operational tooling

## 1. Purpose and success criteria

Add a presentation-ready, read-only dashboard to the existing FastAPI service. It must use only live records already stored by the pipeline, refresh automatically, and explain both the current forecast and recent forecast performance.

The dashboard succeeds when it:

- opens at `GET /dashboard` from the existing API container;
- shows the latest prediction and realized value without fixture or demo data;
- plots predicted and realized MW from six hours ago through the current one-minute forecast horizon, with the p10-p90 interval;
- shows exact current values in summary cards;
- lists recent predictions and marks the flip decision as correct, incorrect, or pending;
- refreshes every 15 seconds without blanking previously loaded content;
- remains usable on desktop, tablet, and mobile;
- works without a CDN, a Node runtime, or a separate frontend service.

The dashboard is informational. It does not provide trading or dispatch advice.

## 2. Scope

### Included

- One combined dashboard read endpoint backed by ClickHouse.
- One FastAPI-served HTML page with local CSS and JavaScript assets.
- A locally vendored Chart.js browser build and its license notice.
- Live status, latest-update metadata, summary cards, chart, and history table.
- Loading, empty, stale, and temporary-error states.
- Repository, API, frontend-contract, and static-asset tests.
- Container/package changes needed to ship the static assets.

### Excluded

- Synthetic or demonstration records.
- User accounts, permissions, or multi-tenant views.
- Editable controls, alerts, trading recommendations, or model retraining controls.
- A separate React application, Node build pipeline, or frontend container.
- Changes to the prediction or outcome event contracts.
- Long-range analytics beyond the recent operational view.

## 3. User experience and visual direction

The page is a calm grid-control-room view rather than a generic analytics template. It uses a deep navy ink, cool slate surfaces, electric grid blue for forecasts, dark teal for realized values, signal green for correct decisions, red for incorrect decisions, and amber for pending or degraded data. Typography uses the local system sans-serif stack and tabular numerals for measurements, so the page has no font-network dependency.

The distinctive element is the main chart: a strong zero-MW baseline anchors the positive and negative imbalance regions, while the uncertainty band stays visually subordinate to the two value lines. Decoration is restrained; color always communicates data or status.

### Page structure

1. A header identifies the dashboard, shows `Live`, `Verouderd`, or `Niet bereikbaar`, the most recent successful refresh, and the model version.
2. Five summary cards show:
   - predicted system imbalance in MW;
   - realized system imbalance in MW, or `Nog niet gekend`;
   - absolute prediction error in MW, or `In afwachting`;
   - flip probability and the yes/no decision;
   - predicted state and prediction quality.
3. The six-hour Chart.js chart contains:
   - predicted MW;
   - realized MW where outcomes exist;
   - p10 and p90 bounds rendered as one filled interval;
   - a visible zero baseline;
   - local-time tooltips with exact values.
4. The history table shows newest records first with columns for target time, predicted MW, realized MW, signed error, flip probability, predicted flip, actual flip, and result.

`Result` is:

- `Correct` when `flip_actual` is known and equals `will_flip`;
- `Fout` when `flip_actual` is known and differs from `will_flip`;
- `In afwachting` when no outcome exists or `flip_actual` is null.

Pending records are not counted as incorrect. Missing values are shown as text, never as zero.

## 4. Architecture and component boundaries

### Storage repository

Add an immutable dashboard row model containing the existing prediction fields plus nullable outcome fields:

- `realized_system_imbalance_mw`;
- `realized_state`;
- `flip_actual`;
- `evaluated_at`.

Add a bounded repository method that accepts UTC-aware `start`, `end`, and `limit`. It canonicalizes `predictions` and `prediction_outcomes` independently with `argMax`, then left-joins each canonical prediction to its canonical outcome by prediction event ID and target time. It returns rows in ascending target-time/event-ID order for charting. A missing outcome remains a valid row with nullable outcome fields.

The query limit is restricted to 1-10,000 rows. The default browser request uses `start = now - 6 hours` and `end = now + 2 minutes`, with a limit sufficient for minute data. The small forward allowance ensures that the newest t+1 prediction is included despite clock skew, while still excluding anything outside the immediate forecast horizon. Query failures are translated through the existing `TransientStorageError` boundary.

### HTTP API

Add:

- `GET /v1/dashboard?start=<UTC>&end=<UTC>&limit=<n>` returning `{items: [...]}`;
- `GET /dashboard` returning the HTML shell;
- a mounted static asset path for dashboard CSS, JavaScript, Chart.js, and license files.

The dashboard endpoint uses the existing UTC validation rules, returns `422` for invalid ranges or limits, and returns `503` with the existing safe storage error message when ClickHouse is unavailable. Existing API routes and response shapes remain unchanged.

Static resources are resolved from the installed Python package rather than the process working directory. Package metadata includes all dashboard assets so the Docker image and editable development install behave the same way.

### Browser application

The browser code is split by responsibility even though it remains dependency-light:

- data loading builds the rolling UTC request from six hours ago through two minutes ahead and owns the 15-second refresh timer;
- view-model helpers format Brussels-local timestamps, MW values, probabilities, state labels, and result labels;
- summary rendering derives cards from the newest row;
- chart rendering creates one Chart.js instance and updates its datasets in place;
- table rendering replaces table rows from the latest successful response;
- status rendering owns loading, live, stale, empty, and error messages.

All API and asset requests are same-origin. The page performs an immediate request on load and then refreshes every 15 seconds. It prevents overlapping requests. A failed refresh keeps the last successful chart and table visible, marks the page stale or unavailable, and records the failure accessibly. Refresh timing pauses while the page is hidden and resumes immediately when it becomes visible.

## 5. Data and display rules

- The API contract uses UTC timestamps. The dashboard displays them in `Europe/Brussels` with `Intl.DateTimeFormat`.
- MW values use a sign and one decimal place. Errors are `predicted - realized`, preserving direction.
- Probabilities are displayed as whole percentages while exact fractional data remains in the JSON contract.
- The newest prediction supplies the forecast, probability, state, and quality cards, plus the model-version label in the header.
- The newest row with an outcome supplies the latest realized value and its matching prediction error. This avoids making a future prediction appear to have a realization already.
- Predictions with equal target times remain separate by event ID; the repository preserves current event identity semantics rather than inventing client-side deduplication.
- `prediction_quality="degraded"` is visibly marked amber. Other quality strings remain visible as provided by the backend.
- The chart uses numeric epoch-millisecond x-values on a linear scale, avoiding the need for a Chart.js date adapter. Tick and tooltip callbacks format Brussels time.
- Chart animations are disabled after the initial render and fully disabled when `prefers-reduced-motion` is set.

## 6. Error, empty, and accessibility behavior

- Initial load: show a compact loading skeleton and status text.
- No predictions: show an explicit `Nog geen live voorspellingen beschikbaar` state; cards and table do not invent values.
- Temporary refresh failure after a success: retain content, show `Verouderd`, the last successful update time, and a retry action.
- Initial failure: show `Niet bereikbaar`, a short actionable explanation, and a retry action.
- Malformed successful payload: treat it as a client-visible load failure and do not partially update the interface.
- All status changes use an `aria-live` region.
- The chart has a concise textual summary, the table remains the accessible source of exact historical values, controls have visible keyboard focus, color is never the only status cue, and layouts reflow without horizontal page scrolling.

## 7. Verification strategy

### Repository tests

- canonical prediction rows are joined with their matching latest outcome;
- predictions without outcomes return nullable realized fields;
- UTC, range, and limit validation is enforced;
- ClickHouse failures become `TransientStorageError`.

### API tests

- the dashboard endpoint serializes completed and pending rows;
- reversed ranges, naive/non-UTC timestamps, and excessive limits return `422`;
- storage unavailability returns `503`;
- existing prediction endpoints remain unchanged;
- `/dashboard` and every referenced local static asset are served successfully.

### Browser contract tests

The JavaScript formatting and result-state helpers are kept pure and covered without a browser build step. An HTML contract test verifies the required landmarks, card hooks, status region, chart canvas, table columns, and local Chart.js reference. The vendored file is pinned and its license is shipped.

### Final verification

- run the focused repository/API/dashboard test set;
- run the complete non-live test suite;
- build or start the Docker API service and verify `/dashboard` plus the dashboard JSON endpoint;
- inspect a desktop and mobile screenshot against the approved hierarchy, including an empty or pending outcome state when naturally present in live data.

## 8. Deployment and operational impact

The API service remains read-only. The new query is bounded by time and row count and uses the prediction/outcome table ordering keys. No schema migration, new port, environment variable, service, or external network request is required. The README documents the dashboard URL and its live-data-only behavior.
