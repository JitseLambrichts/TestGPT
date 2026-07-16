from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import imbalance_pipeline.training.export_clickhouse as exporter
from imbalance_pipeline.training.export_clickhouse import export_clickhouse_training_dataset


def test_rejects_naive_boundaries(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        import asyncio

        asyncio.run(
            export_clickhouse_training_dataset(
                object(), tmp_path / "dataset", datetime(2025, 1, 1), datetime(2025, 1, 2)
            )
        )


def test_rejects_reversed_range(tmp_path):
    import asyncio

    start = datetime(2025, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError, match="end must be after start"):
        asyncio.run(
            export_clickhouse_training_dataset(object(), tmp_path / "dataset", start, start)
        )


@pytest.mark.asyncio
async def test_empty_input_raises_and_does_not_create_output(tmp_path):
    class Repository:
        async def fetch_imbalance_versions(self, *args, **kwargs):
            return []

    with pytest.raises(ValueError, match="no usable"):
        await export_clickhouse_training_dataset(
            Repository(),
            tmp_path / "dataset",
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        )
    assert not (tmp_path / "dataset").exists()


@pytest.mark.asyncio
async def test_failure_preserves_existing_output(tmp_path):
    class Repository:
        async def fetch_imbalance_versions(self, *args, **kwargs):
            return []

    output = tmp_path / "dataset"
    output.write_text("caller-owned", encoding="utf-8")
    with pytest.raises(ValueError, match="no usable"):
        await export_clickhouse_training_dataset(
            Repository(), output, datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 2, tzinfo=UTC)
        )
    assert output.read_text(encoding="utf-8") == "caller-owned"


@pytest.mark.asyncio
async def test_rejects_gapped_local_history(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    versions = _versions(start)
    versions.pop(10)

    class Repository:
        async def fetch_imbalance_versions(self, *args, **kwargs):
            return versions

    with pytest.raises(ValueError, match="contiguous"):
        await export_clickhouse_training_dataset(
            Repository(), tmp_path / "dataset", start, start + timedelta(minutes=2)
        )


def _versions(start: datetime):
    timestamps = [start + timedelta(minutes=offset) for offset in range(-180, 3)]
    return [
        SimpleNamespace(
            observation=SimpleNamespace(timestamp=timestamp, system_imbalance_mw=float(index)),
            available_at=timestamp,
            row_version=0,
            event_id=f"event-{index}",
        )
        for index, timestamp in enumerate(timestamps)
    ]


@pytest.mark.asyncio
async def test_reuses_one_replay_and_passes_causal_knowledge_cutoffs(tmp_path, monkeypatch):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    calls = {"open": [], "build": []}

    class Repository:
        async def fetch_imbalance_versions(self, *args, **kwargs):
            return _versions(start)

    class Engine:
        def __init__(self, repository, registry):
            del repository, registry

        async def open_replay(self, **kwargs):
            calls["open"].append(kwargs)
            return object()

        async def build_many(self, cutoffs, *, knowledge_cutoffs, replay):
            calls["build"].append((list(cutoffs), list(knowledge_cutoffs), replay))
            return [SimpleNamespace(cutoff=cutoff) for cutoff in cutoffs]

    monkeypatch.setattr(exporter, "FeatureEngine", Engine)
    monkeypatch.setattr(
        exporter,
        "build_training_examples",
        lambda snapshots, values, deadband_mw: [object()],
    )
    written = []

    def write(examples, output):
        written.append((list(examples), output))
        return output

    monkeypatch.setattr(exporter, "write_training_dataset", write)

    output = tmp_path / "dataset"
    result = await export_clickhouse_training_dataset(
        Repository(), output, start, start + timedelta(minutes=2), batch_size=1
    )
    assert result == output
    assert len(calls["open"]) == 1
    assert len(calls["build"]) == 2
    assert calls["build"][0][1] == [start]
    assert calls["build"][1][1] == [start + timedelta(minutes=1)]
    assert calls["build"][0][2] is calls["build"][1][2]
    assert len(written) == 1


@pytest.mark.asyncio
async def test_excludes_future_observations_from_feature_knowledge_cutoff(tmp_path, monkeypatch):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    observed = _versions(start)
    observed.append(
        SimpleNamespace(
            observation=SimpleNamespace(
                timestamp=start + timedelta(minutes=2), system_imbalance_mw=99.0
            ),
            available_at=start + timedelta(minutes=2),
            row_version=0,
            event_id="future",
        )
    )
    knowledge = []

    class Repository:
        async def fetch_imbalance_versions(self, *args, **kwargs):
            return observed

    class Engine:
        def __init__(self, repository, registry):
            del repository, registry

        async def open_replay(self, **kwargs):
            return object()

        async def build_many(self, cutoffs, *, knowledge_cutoffs, replay):
            del cutoffs, replay
            knowledge.extend(knowledge_cutoffs)
            return []

    monkeypatch.setattr(exporter, "FeatureEngine", Engine)
    monkeypatch.setattr(exporter, "build_training_examples", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="no usable training examples"):
        await export_clickhouse_training_dataset(
            Repository(), tmp_path / "dataset", start, start + timedelta(minutes=2)
        )
    assert knowledge == [start, start + timedelta(minutes=1)]
