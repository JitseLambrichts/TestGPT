# Imbalance-only Training Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export reproducible imbalance-only training datasets from historical ClickHouse observations and make them trainable through the existing model pipeline.

**Architecture:** Historical backfill explicitly selects ODS133 while live polling continues to use ODS161. A dedicated exporter replays canonical ClickHouse imbalance observations through the existing causal `FeatureEngine`, creates labels with `build_training_examples`, and persists the existing checksummed NPZ format. Training and promotion remain separate operator steps.

**Tech Stack:** Python 3.12, asyncio, ClickHouse Connect, Pydantic settings, NumPy, pytest, Docker Compose, Make.

## Global Constraints

- Version 1 uses only the ODS133 imbalance history; no load, wind, solar, or weather ingestion is added.
- Live polling continues to use ODS161.
- Historical features must use the existing `FeatureEngine` and `FeatureReplaySession`; no duplicated feature math.
- Training labels remain next-minute, auxiliary horizons `(2, 5, 10)`, and existing deadband flip semantics.
- Exported datasets use the existing `write_training_dataset` checksums and metadata.
- A candidate is never promoted automatically by the exporter or trainer.

---

### Task 1: Make explicit historical backfill use ODS133

**Files:**
- Modify: `src/imbalance_pipeline/services/ingestor.py` in `poll_imbalance_once` and bounded `_run_service` flow
- Test: `tests/unit/services/test_ingestor.py`

**Interfaces:**
- Consumes: `Settings.elia_imbalance_history_dataset`, existing `poll_imbalance_once(start, end)`.
- Produces: historical bounded polling that uses ODS133 while unbounded polling still uses ODS161.

- [ ] **Step 1: Write failing tests** asserting that an explicit `start/end` chooses `settings.elia_imbalance_history_dataset`, while the no-boundary live path chooses `settings.elia_imbalance_live_dataset`.
- [ ] **Step 2: Run the focused tests** with `uv run pytest -q tests/unit/services/test_ingestor.py -k 'history or dataset or imbalance'`; expect the new assertions to fail against the current unconditional ODS161 selection.
- [ ] **Step 3: Implement the smallest source-selection change** by adding an explicit historical-dataset parameter/path to the bounded poll and preserving the live default for `start=None, end=None`.
- [ ] **Step 4: Run the focused ingestor tests** and confirm all pass.
- [ ] **Step 5: Commit** with `git add src/imbalance_pipeline/services/ingestor.py tests/unit/services/test_ingestor.py && git commit -m "fix: use historical Elia dataset for backfills"`.

### Task 2: Add the ClickHouse-to-training exporter

**Files:**
- Create: `src/imbalance_pipeline/training/export_clickhouse.py`
- Modify: `src/imbalance_pipeline/training/export_data.py` only if a shared writer helper is needed
- Test: `tests/unit/training/test_export_clickhouse.py`

**Interfaces:**
- Consumes: `ClickHouseRepository.fetch_imbalance_versions`, `FeatureEngine.open_replay`, `FeatureEngine.build_many`, and `build_training_examples`.
- Produces: `async def export_clickhouse_training_dataset(repository, output, start, end, *, batch_size=512, deadband_mw=10.0) -> Path` and CLI `imbalance-export-training --start ISO --end ISO --output PATH`.

- [ ] **Step 1: Write failing unit tests** for UTC/range validation, empty input, replay reuse, future-leakage protection, and successful writer invocation with a fake repository/engine.
- [ ] **Step 2: Run `uv run pytest -q tests/unit/training/test_export_clickhouse.py`; expect collection or assertion failures because the module and CLI do not yet exist.
- [ ] **Step 3: Implement the exporter** with these exact behaviors:
  - normalize and validate timezone-aware UTC boundaries and require `end > start`;
  - require enough history for the 180-minute local window and fetch canonical versions through the repository;
  - use each observation timestamp as a candidate cutoff, exclude the final target-less cutoff, and pass knowledge cutoffs that do not include future observations;
  - open one replay session, build snapshots in bounded batches, convert snapshots to `TrainingExample` labels, and call `write_training_dataset` once;
  - raise a clear error for no usable examples and remove any partial output on failure.
- [ ] **Step 4: Run the focused exporter tests** and confirm all pass.
- [ ] **Step 5: Commit** with `git add src/imbalance_pipeline/training/export_clickhouse.py src/imbalance_pipeline/training/export_data.py tests/unit/training/test_export_clickhouse.py && git commit -m "feat: export imbalance history for training"`.

### Task 3: Expose the exporter through packaging, Make, and documentation

**Files:**
- Modify: `pyproject.toml` console scripts
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `docs/runbook.md`
- Test: `tests/unit/test_training_cli.py` and `tests/unit/test_colab_notebook.py` if command references need coverage

**Interfaces:**
- Consumes: `imbalance-export-training --start --end --output` from Task 2.
- Produces: `make export-training START=... END=... OUTPUT=...` and documented backfill/export/train/promote workflow.

- [ ] **Step 1: Add a failing CLI/Make contract test** that checks the console entry point and documented command names.
- [ ] **Step 2: Run the focused contract test** and confirm it fails before packaging changes.
- [ ] **Step 3: Add the console script and Make target** with explicit required variables and UTC ISO-8601 forwarding; do not add automatic promotion.
- [ ] **Step 4: Document the exact workflow**, including ODS133 history backfill, export, `make train`, promotion through `promote_bundle`, and `docker compose restart predictor api`.
- [ ] **Step 5: Run the contract tests and `uv run python -m build --wheel`; confirm the wheel exposes the exporter command.
- [ ] **Step 6: Commit** with `git add pyproject.toml Makefile README.md docs/runbook.md tests/unit/test_training_cli.py tests/unit/test_colab_notebook.py && git commit -m "docs: expose imbalance training workflow"`.

### Task 4: Validate the complete export path

**Files:**
- Create or modify: `tests/integration/test_training_export.py` only if the existing ClickHouse fixture supports a bounded export fixture
- No source changes unless validation exposes a defect

**Interfaces:**
- Consumes: the exporter CLI and a seeded ClickHouse observation range.
- Produces: a verified dataset with metadata, checksummed shards, monotonic cutoffs, and labels aligned to the next minute.

- [ ] **Step 1: Seed a small canonical imbalance range** containing at least 180 minutes of history plus one target minute in the existing integration fixture.
- [ ] **Step 2: Run the integration exporter test** and verify the output `metadata.json`, shard checksums, feature schema hash, and nonzero examples.
- [ ] **Step 3: Run the complete non-integration suite** with `uv run pytest -q -m 'not integration'`.
- [ ] **Step 4: Run quality checks** with `uv run ruff check src tests`, `uv run mypy src`, and `python -m build --wheel`.
- [ ] **Step 5: Run a live-safe command check** that confirms ODS133 bounded backfill selection and leaves ODS161 live polling unchanged.
- [ ] **Step 6: Commit any test-only changes** with `git add tests/integration/test_training_export.py && git commit -m "test: verify imbalance training export"`.

## Self-review checklist

- Historical source selection is covered by Task 1.
- Causal replay, labels, no-future leakage, checksums, empty output, and batching are covered by Task 2.
- CLI/Make/documentation workflow is covered by Task 3.
- End-to-end ClickHouse verification and full quality gates are covered by Task 4.
- No task adds unrelated supporting datasets or automatic model promotion.
