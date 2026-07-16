"""Point-in-time ClickHouse export of imbalance training examples."""

import argparse
import asyncio
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from imbalance_pipeline.config import get_settings
from imbalance_pipeline.domain.imbalance import VersionedImbalanceObservation
from imbalance_pipeline.features.engine import (
    LOCAL_HISTORY_MINUTES,
    FeatureEngine,
    FeatureSnapshot,
)
from imbalance_pipeline.features.schema import FeatureRegistry
from imbalance_pipeline.storage.clickhouse import ClickHouseRepository
from imbalance_pipeline.training.data import build_training_examples
from imbalance_pipeline.training.export_data import write_training_dataset


def _utc_boundary(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_values(versions: Sequence[VersionedImbalanceObservation]) -> dict[datetime, float]:
    latest: dict[datetime, VersionedImbalanceObservation] = {}
    for version in versions:
        timestamp = version.observation.timestamp.astimezone(UTC)
        previous = latest.get(timestamp)
        if previous is None or (version.row_version, version.available_at, version.event_id) >= (
            previous.row_version,
            previous.available_at,
            previous.event_id,
        ):
            latest[timestamp] = version
    return {
        timestamp: version.observation.system_imbalance_mw
        for timestamp, version in latest.items()
    }


def _remove_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()


async def export_clickhouse_training_dataset(
    repository: Any,
    output: Path,
    start: datetime,
    end: datetime,
    *,
    batch_size: int = 512,
    deadband_mw: float = 10.0,
) -> Path:
    start = _utc_boundary(start, "start")
    end = _utc_boundary(end, "end")
    if end <= start:
        raise ValueError("end must be after start")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    history_start = start - timedelta(minutes=LOCAL_HISTORY_MINUTES)
    versions = await repository.fetch_imbalance_versions(
        history_start, end, knowledge_cutoff=end
    )
    values = _canonical_values(versions)
    cutoffs = sorted(timestamp for timestamp in values if start <= timestamp < end)
    if not cutoffs:
        raise ValueError("no usable imbalance observations in requested range")
    if min(values) > history_start:
        raise ValueError("insufficient imbalance history for the 180-minute local window")

    registry = FeatureRegistry.default(deadband_mw=deadband_mw)
    engine = FeatureEngine(repository, registry)
    replay = await engine.open_replay(end=end, knowledge_cutoff=end)
    examples = []
    try:
        for offset in range(0, len(cutoffs), batch_size):
            batch = cutoffs[offset : offset + batch_size]
            snapshots: list[FeatureSnapshot] = await engine.build_many(
                batch,
                knowledge_cutoffs=batch,
                replay=replay,
            )
            examples.extend(build_training_examples(snapshots, values, deadband_mw=deadband_mw))
        if not examples:
            raise ValueError("no usable training examples (observations lack next-minute targets)")
        return write_training_dataset(examples, Path(output))
    except Exception:
        _remove_output(Path(output))
        raise


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO timestamp: {value}") from exc


async def _run_cli(start: datetime, end: datetime, output: Path) -> Path:
    settings = get_settings()
    repository = await ClickHouseRepository.connect(settings)
    try:
        return await export_clickhouse_training_dataset(repository, output, start, end)
    finally:
        await repository.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="imbalance-export-training")
    parser.add_argument("--start", required=True, type=_parse_iso)
    parser.add_argument("--end", required=True, type=_parse_iso)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(_run_cli(args.start, args.end, args.output))


__all__ = ["export_clickhouse_training_dataset", "main"]
