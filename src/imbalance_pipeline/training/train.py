import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from imbalance_pipeline.model.bundle import validate_bundle
from imbalance_pipeline.model.distribution import gaussian_mixture_quantile
from imbalance_pipeline.model.export_onnx import OUTPUT_NAMES, export_member
from imbalance_pipeline.model.losses import MultiTaskLoss
from imbalance_pipeline.model.network import ImbalanceForecaster
from imbalance_pipeline.training.baselines import (
    classical_baseline,
    clipped_linear_drift_baseline,
    persistence_baseline,
    rolling_median_baseline,
)
from imbalance_pipeline.training.calibration import IsotonicCalibrator
from imbalance_pipeline.training.data import (
    RobustPreprocessor,
    TrainingBatch,
    TrainingExample,
)
from imbalance_pipeline.training.export_data import TrainingDataset, open_training_dataset
from imbalance_pipeline.training.metrics import EvaluationReport, evaluate_predictions
from imbalance_pipeline.training.splits import (
    IndexRange,
    TimeSplit,
    latest_purged_split,
    validate_time_split,
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seeds: tuple[int, int, int] = (17, 29, 43)
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    d_model: int = 128
    tcn_blocks: int = 6
    transformer_layers: int = 2
    transformer_heads: int = 4
    mixture_components: int = 5
    deadband_mw: float = 10.0
    device: str | None = None
    mixed_precision: bool = True
    deterministic: bool = True
    split: TimeSplit | None = None

    def __post_init__(self) -> None:
        if len(set(self.seeds)) != 3:
            raise ValueError("training requires exactly three unique ensemble seeds")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.warmup_epochs,
                self.early_stopping_patience,
                self.d_model,
                self.tcn_blocks,
                self.transformer_layers,
                self.transformer_heads,
                self.mixture_components,
            )
            <= 0
        ):
            raise ValueError("training dimensions and schedule values must be positive")
        if self.d_model % self.transformer_heads != 0:
            raise ValueError("d_model must be divisible by transformer_heads")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.gradient_clip_norm <= 0
            or self.deadband_mw <= 0
        ):
            raise ValueError("training optimizer and deadband values are invalid")


@dataclass(frozen=True, slots=True)
class _MemberResult:
    seed: int
    model: ImbalanceForecaster
    best_validation_loss: float
    epochs_trained: int


@dataclass(frozen=True, slots=True)
class _EnsemblePredictions:
    predicted_mw: NDArray[np.float64]
    p10_mw: NDArray[np.float64]
    p90_mw: NDArray[np.float64]
    raw_flip_probability: NDArray[np.float64]
    actual_mw: NDArray[np.float64]
    flip_target: NDArray[np.int64]
    flip_mask: NDArray[np.bool_]
    current_state: NDArray[np.int64]
    volatility: NDArray[np.float64]
    quarter_hour_phase: NDArray[np.int8]
    source_quality: NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class _Partition:
    dataset: TrainingDataset
    section: IndexRange

    @property
    def count(self) -> int:
        return self.section.stop - self.section.start

    def sample(self) -> TrainingExample:
        return self.dataset.example_at(self.section.start)

    def examples(self) -> Iterator[TrainingExample]:
        return self.dataset.iter_range(self.section)

    def batches(
        self,
        batch_size: int,
        *,
        seed: int | None = None,
    ) -> Iterator[list[TrainingExample]]:
        return self.dataset.iter_batches(self.section, batch_size=batch_size, seed=seed)


def train_ensemble(
    dataset_path: Path,
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> Path:
    config = config or TrainingConfig()
    dataset = open_training_dataset(dataset_path)
    timestamps = dataset.cutoffs()
    _require_unique_timestamps(timestamps)
    split = _resolve_split(timestamps, config)
    training = _Partition(dataset, split.train)
    validation = _Partition(dataset, split.validation)
    calibration = _Partition(dataset, split.calibration)
    test = _Partition(dataset, split.test)
    preprocessor = RobustPreprocessor.fit_stream(training.examples())
    device = _device(config.device)
    results = tuple(
        _train_member(seed, training, validation, preprocessor, config, device)
        for seed in config.seeds
    )
    calibration_prediction = _ensemble_predictions(
        results,
        calibration,
        preprocessor,
        config.batch_size,
        device,
    )
    calibrator = _fit_calibrator(calibration_prediction)
    test_prediction = _ensemble_predictions(
        results,
        test,
        preprocessor,
        config.batch_size,
        device,
    )
    candidate_report = _evaluation_report(test_prediction, calibrator)
    baseline_reports = _baseline_reports(training, test, test_prediction)
    return _write_candidate(
        output_dir,
        results,
        preprocessor,
        calibrator,
        candidate_report,
        baseline_reports,
        training,
        validation,
        calibration,
        test,
        split,
        device,
        config,
    )


def _train_member(
    seed: int,
    training: _Partition,
    validation: _Partition,
    preprocessor: RobustPreprocessor,
    config: TrainingConfig,
    device: torch.device,
) -> _MemberResult:
    _seed_everything(seed, device, deterministic=config.deterministic)
    sample = training.sample()
    model = ImbalanceForecaster(
        local_features=sample.local_values.shape[-1],
        context_features=sample.context_values.shape[-1],
        static_features=sample.static_values.shape[-1],
        d_model=config.d_model,
        tcn_blocks=config.tcn_blocks,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        mixture_components=config.mixture_components,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss = MultiTaskLoss(
        positive_weight=_positive_weight(training.examples()),
        deadband_mw=config.deadband_mw,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.mixed_precision and device.type == "cuda",
    )
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    stalled_epochs = 0
    completed_epochs = 0
    for epoch in range(config.epochs):
        _set_learning_rate(optimizer, config, epoch)
        model.train()
        for examples in training.batches(config.batch_size, seed=seed + epoch):
            batch = _prepared_batch(
                examples,
                preprocessor,
                batch_id=f"seed-{seed}-epoch-{epoch}",
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.mixed_precision):
                breakdown = loss(model(*_model_inputs(batch)), batch)
            scaler.scale(breakdown.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        validation_loss = _validation_loss(
            model,
            validation,
            preprocessor,
            loss,
            config.batch_size,
            device,
        )
        completed_epochs = epoch + 1
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= config.early_stopping_patience:
                break
    if best_state is None:
        raise ValueError(f"training did not produce a finite validation result for seed {seed}")
    model.load_state_dict(best_state)
    return _MemberResult(
        seed=seed,
        model=model,
        best_validation_loss=best_loss,
        epochs_trained=completed_epochs,
    )


def _validation_loss(
    model: ImbalanceForecaster,
    validation: _Partition,
    preprocessor: RobustPreprocessor,
    loss: MultiTaskLoss,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    nll_sum = 0.0
    nll_count = 0
    brier_sum = 0.0
    brier_count = 0
    with torch.inference_mode():
        for items in validation.batches(batch_size):
            batch = _prepared_batch(items, preprocessor, batch_id="validation", device=device)
            output = model(*_model_inputs(batch))
            breakdown = loss(output, batch)
            count = len(items)
            nll_sum += float(breakdown.mixture_nll.cpu()) * count
            nll_count += count
            squared_error = (torch.sigmoid(output.flip_logit) - batch.flip_target).square()
            brier_sum += float((squared_error * batch.flip_mask).sum().cpu())
            brier_count += int(batch.flip_mask.sum().item())
    return _validation_objective(nll_sum, nll_count, brier_sum, brier_count)


def _ensemble_predictions(
    results: Sequence[_MemberResult],
    partition: _Partition,
    preprocessor: RobustPreprocessor,
    batch_size: int,
    device: torch.device,
) -> _EnsemblePredictions:
    point: list[NDArray[np.float64]] = []
    p10: list[NDArray[np.float64]] = []
    p90: list[NDArray[np.float64]] = []
    flip: list[NDArray[np.float64]] = []
    actual: list[NDArray[np.float64]] = []
    flip_target: list[NDArray[np.int64]] = []
    flip_mask: list[NDArray[np.bool_]] = []
    current_state: list[NDArray[np.int64]] = []
    volatility: list[NDArray[np.float64]] = []
    quarter_hour_phase: list[NDArray[np.int8]] = []
    source_quality: list[NDArray[np.int8]] = []
    for result in results:
        result.model.eval()
    with torch.inference_mode():
        for items in partition.batches(batch_size):
            batch = _prepared_batch(items, preprocessor, batch_id="evaluation", device=device)
            outputs = [result.model(*_model_inputs(batch)) for result in results]
            mixture_logits = torch.cat(
                [
                    torch.log_softmax(output.mixture_logits, dim=-1) - math.log(len(outputs))
                    for output in outputs
                ],
                dim=-1,
            )
            mixture_means = torch.cat([output.mixture_means for output in outputs], dim=-1)
            mixture_log_scales = torch.cat(
                [output.mixture_log_scales for output in outputs],
                dim=-1,
            )
            point.append(
                gaussian_mixture_quantile(
                    torch.full_like(batch.target_next, 0.5),
                    mixture_logits,
                    mixture_means,
                    mixture_log_scales,
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            p10.append(
                gaussian_mixture_quantile(
                    torch.full_like(batch.target_next, 0.1),
                    mixture_logits,
                    mixture_means,
                    mixture_log_scales,
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            p90.append(
                gaussian_mixture_quantile(
                    torch.full_like(batch.target_next, 0.9),
                    mixture_logits,
                    mixture_means,
                    mixture_log_scales,
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            mean_flip_logit = torch.stack([output.flip_logit for output in outputs]).mean(dim=0)
            flip.append(torch.sigmoid(mean_flip_logit).cpu().numpy().astype(np.float64))
            actual.append(batch.target_next.cpu().numpy().astype(np.float64))
            flip_target.append(batch.flip_target.cpu().numpy().astype(np.int64))
            flip_mask.append(batch.flip_mask.cpu().numpy().astype(bool))
            current_state.append(batch.current_state.cpu().numpy().astype(np.int64))
            volatility.append(_volatility(items))
            quarter_hour_phase.append(
                np.asarray([item.cutoff.minute % 15 for item in items], dtype=np.int8)
            )
            source_quality.append(_source_quality(items))
    return _EnsemblePredictions(
        predicted_mw=np.concatenate(point),
        p10_mw=np.concatenate(p10),
        p90_mw=np.concatenate(p90),
        raw_flip_probability=np.concatenate(flip),
        actual_mw=np.concatenate(actual),
        flip_target=np.concatenate(flip_target),
        flip_mask=np.concatenate(flip_mask),
        current_state=np.concatenate(current_state),
        volatility=np.concatenate(volatility),
        quarter_hour_phase=np.concatenate(quarter_hour_phase),
        source_quality=np.concatenate(source_quality),
    )


def _fit_calibrator(
    prediction: _EnsemblePredictions,
) -> IsotonicCalibrator:
    mask = prediction.flip_mask
    if not np.any(mask):
        return IsotonicCalibrator(
            x_thresholds=np.asarray([0.0, 1.0]),
            y_thresholds=np.asarray([0.0, 0.0]),
            decision_threshold=0.5,
        )
    return IsotonicCalibrator.fit(
        prediction.raw_flip_probability[mask],
        prediction.flip_target[mask],
    )


def _evaluation_report(
    prediction: _EnsemblePredictions,
    calibrator: IsotonicCalibrator,
) -> EvaluationReport:
    if not np.any(prediction.flip_mask):
        raise ValueError("test partition requires at least one known flip label")
    calibrated = calibrator.predict(prediction.raw_flip_probability)
    return evaluate_predictions(
        actual_mw=prediction.actual_mw,
        predicted_mw=prediction.predicted_mw,
        p10_mw=prediction.p10_mw,
        p90_mw=prediction.p90_mw,
        flip_target=prediction.flip_target,
        flip_probability=calibrated,
        flip_mask=prediction.flip_mask,
        threshold=calibrator.decision_threshold,
        cohorts=_cohort_masks(prediction),
    )


def _baseline_reports(
    training: _Partition,
    test: _Partition,
    evaluation: _EnsemblePredictions,
) -> dict[str, EvaluationReport]:
    if not np.any(evaluation.flip_mask):
        raise ValueError("test partition requires at least one known flip label")
    persistence = persistence_baseline(test.examples())
    drift = clipped_linear_drift_baseline(test.examples())
    rolling_median = rolling_median_baseline(test.examples())
    classical = classical_baseline(training.examples(), test.examples())
    cohorts = _cohort_masks(evaluation)
    return {
        "persistence": evaluate_predictions(
            actual_mw=evaluation.actual_mw,
            predicted_mw=persistence,
            p10_mw=persistence,
            p90_mw=persistence,
            flip_target=evaluation.flip_target,
            flip_probability=np.zeros(len(evaluation.actual_mw), dtype=np.float64),
            flip_mask=evaluation.flip_mask,
            threshold=0.5,
            cohorts=cohorts,
        ),
        "clipped_linear_drift": evaluate_predictions(
            actual_mw=evaluation.actual_mw,
            predicted_mw=drift,
            p10_mw=drift,
            p90_mw=drift,
            flip_target=evaluation.flip_target,
            flip_probability=np.zeros(len(evaluation.actual_mw), dtype=np.float64),
            flip_mask=evaluation.flip_mask,
            threshold=0.5,
            cohorts=cohorts,
        ),
        "rolling_median": evaluate_predictions(
            actual_mw=evaluation.actual_mw,
            predicted_mw=rolling_median,
            p10_mw=rolling_median,
            p90_mw=rolling_median,
            flip_target=evaluation.flip_target,
            flip_probability=np.zeros(len(evaluation.actual_mw), dtype=np.float64),
            flip_mask=evaluation.flip_mask,
            threshold=0.5,
            cohorts=cohorts,
        ),
        "classical": evaluate_predictions(
            actual_mw=evaluation.actual_mw,
            predicted_mw=classical.predicted_mw,
            p10_mw=classical.predicted_mw,
            p90_mw=classical.predicted_mw,
            flip_target=evaluation.flip_target,
            flip_probability=classical.flip_probability,
            flip_mask=evaluation.flip_mask,
            threshold=0.5,
            cohorts=cohorts,
        ),
    }


def _write_candidate(
    output: Path,
    results: Sequence[_MemberResult],
    preprocessor: RobustPreprocessor,
    calibrator: IsotonicCalibrator,
    report: EvaluationReport,
    baselines: dict[str, EvaluationReport],
    training: _Partition,
    validation: _Partition,
    calibration: _Partition,
    test: _Partition,
    split: TimeSplit,
    device: torch.device,
    config: TrainingConfig,
) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"candidate staging path already exists: {staging}")
    staging.mkdir()
    try:
        members: list[str] = []
        sample = training.sample()
        sample_batch = TrainingBatch.from_examples([sample], batch_id="onnx-export")
        sample_inputs = _model_inputs(sample_batch)
        for index, result in enumerate(results):
            model = result.model.to("cpu").eval()
            torch.save(
                {"seed": result.seed, "state_dict": model.state_dict()},
                staging / f"member-{index}.pt",
            )
            member = f"member-{index}.onnx"
            export_member(model, sample_inputs, staging / member)
            members.append(member)
        (staging / "preprocessing.json").write_text(preprocessor.to_json(), encoding="utf-8")
        (staging / "calibration.json").write_text(calibrator.to_json(), encoding="utf-8")
        _write_json(
            staging / "evaluation.json",
            {
                "candidate": report.to_dict(),
                "members": [
                    {
                        "best_validation_loss": result.best_validation_loss,
                        "epochs_trained": result.epochs_trained,
                        "seed": result.seed,
                    }
                    for result in results
                ],
            },
        )
        _write_json(
            staging / "baseline_evaluation.json",
            {name: baseline.to_dict() for name, baseline in baselines.items()},
        )
        _write_json(
            staging / "run_config.json",
            _config_dict(config, split, device, training, validation, calibration, test),
        )
        artifacts = (*members, "preprocessing.json", "calibration.json", "evaluation.json")
        _write_json(
            staging / "manifest.json",
            {
                "checksums": {
                    artifact: hashlib.sha256((staging / artifact).read_bytes()).hexdigest()
                    for artifact in artifacts
                },
                "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "evaluation_file": "evaluation.json",
                "feature_schema_hash": preprocessor.feature_schema_hash,
                "input_shapes": {
                    "context": [
                        -1,
                        sample.context_values.shape[0],
                        sample.context_values.shape[1],
                    ],
                    "local": [
                        -1,
                        sample.local_values.shape[0],
                        sample.local_values.shape[1],
                    ],
                    "static": [-1, sample.static_values.shape[0]],
                },
                "members": members,
                "model_version": f"imbalance-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
                "output_names": list(OUTPUT_NAMES),
                "preprocessing_file": "preprocessing.json",
                "calibration_file": "calibration.json",
                "runtime": {"opset": 18, "provider": "CPUExecutionProvider"},
                "training_period": {
                    "end": training.dataset.example_at(
                        training.section.stop - 1
                    ).cutoff.isoformat(),
                    "start": sample.cutoff.isoformat(),
                },
            },
        )
        manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        bundle_validation = validate_bundle(
            staging,
            expected_schema_hash=str(manifest["feature_schema_hash"]),
        )
        if not bundle_validation.valid:
            raise ValueError(bundle_validation.reason or "candidate bundle validation failed")
        _fsync_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _positive_weight(examples: Iterator[TrainingExample]) -> float:
    known = 0
    positives = 0
    for example in examples:
        if example.flip_mask > 0.5:
            known += 1
            positives += int(example.flip_target > 0.5)
    return (known - positives) / positives if positives else 1.0


def _prepared_batch(
    examples: Sequence[TrainingExample],
    preprocessor: RobustPreprocessor,
    *,
    batch_id: str,
    device: torch.device,
) -> TrainingBatch:
    raw = TrainingBatch.from_examples(examples, batch_id=batch_id)
    return preprocessor.transform_batch(raw).to(device)


def _validation_objective(
    nll_sum: float,
    nll_count: int,
    brier_sum: float,
    brier_count: int,
) -> float:
    if nll_count <= 0:
        raise ValueError("validation requires at least one regression example")
    nll = nll_sum / nll_count
    brier = brier_sum / brier_count if brier_count else 0.0
    return nll + 0.25 * brier


def _volatility(examples: Sequence[TrainingExample]) -> NDArray[np.float64]:
    values: list[float] = []
    for example in examples:
        history = example.local_values[:, 0][example.local_masks[:, 0] == 1]
        values.append(float(np.std(history, dtype=np.float64)) if len(history) else 0.0)
    return np.asarray(values, dtype=np.float64)


def _cohort_masks(prediction: _EnsemblePredictions) -> dict[str, NDArray[np.bool_]]:
    masks: dict[str, NDArray[np.bool_]] = {
        "current_positive": prediction.current_state == 1,
        "current_negative": prediction.current_state == -1,
        "current_unknown": prediction.current_state == 0,
        "quarter_hour_phase_start": prediction.quarter_hour_phase < 5,
        "quarter_hour_phase_middle": (prediction.quarter_hour_phase >= 5)
        & (prediction.quarter_hour_phase < 10),
        "quarter_hour_phase_end": prediction.quarter_hour_phase >= 10,
        "source_quality_validated": prediction.source_quality == 1,
        "source_quality_unvalidated": prediction.source_quality == 0,
        "source_quality_unknown": prediction.source_quality == -1,
    }
    low, medium, high = _volatility_terciles(prediction.volatility)
    masks["low_volatility"] = low
    masks["medium_volatility"] = medium
    masks["high_volatility"] = high
    return masks


def _volatility_terciles(
    volatility: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    if not len(volatility):
        raise ValueError("cohorts require at least one prediction")
    masks = [np.zeros(len(volatility), dtype=bool) for _ in range(3)]
    # Rank-based tertiles remain disjoint and cover the full sample, even for
    # a quiet period where multiple histories have identical volatility.
    for group, indices in enumerate(np.array_split(np.argsort(volatility, kind="stable"), 3)):
        masks[group][indices] = True
    return masks[0], masks[1], masks[2]


def _source_quality(items: Sequence[TrainingExample]) -> NDArray[np.int8]:
    """Expose default-schema quality validation without inventing it for custom schemas."""
    values: list[int] = []
    for item in items:
        if item.local_values.shape[-1] <= 5 or item.local_masks.shape[-1] <= 5:
            values.append(-1)
            continue
        if item.local_masks[-1, 5] != 1:
            values.append(-1)
            continue
        values.append(int(item.local_values[-1, 5] > 0.5))
    return np.asarray(values, dtype=np.int8)


def _model_inputs(batch: TrainingBatch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        batch.local,
        batch.local_mask,
        batch.context,
        batch.context_mask,
        batch.static,
        batch.static_mask,
    )


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
) -> None:
    if epoch < config.warmup_epochs:
        scale = (epoch + 1) / config.warmup_epochs
    else:
        remaining = max(config.epochs - config.warmup_epochs, 1)
        progress = (epoch - config.warmup_epochs) / remaining
        scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate * scale


def _autocast(device: torch.device, mixed_precision: bool) -> AbstractContextManager[object]:
    if mixed_precision and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _device(configured: str | None) -> torch.device:
    if configured is not None:
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int, device: torch.device, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)


def _require_unique_timestamps(timestamps: Sequence[datetime]) -> None:
    if not timestamps:
        raise ValueError("training dataset is empty")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("training dataset cutoffs must be unique")


def _resolve_split(timestamps: Sequence[datetime], config: TrainingConfig) -> TimeSplit:
    split = config.split or latest_purged_split(timestamps)
    validate_time_split(timestamps, split)
    return split


def _config_dict(
    config: TrainingConfig,
    split: TimeSplit,
    device: torch.device,
    training: _Partition,
    validation: _Partition,
    calibration: _Partition,
    test: _Partition,
) -> dict[str, object]:
    return {
        "batch_size": config.batch_size,
        "d_model": config.d_model,
        "deadband_mw": config.deadband_mw,
        "deterministic": config.deterministic,
        "device": str(device),
        "dataset": {
            "count": training.dataset.count,
            "digest": training.dataset.dataset_digest,
            "feature_schema_hash": training.dataset.feature_schema_hash,
            "shards": [
                {"checksum": shard.checksum, "count": shard.count, "name": shard.name}
                for shard in training.dataset.shards
            ],
        },
        "early_stopping_patience": config.early_stopping_patience,
        "epochs": config.epochs,
        "gradient_clip_norm": config.gradient_clip_norm,
        "learning_rate": config.learning_rate,
        "mixture_components": config.mixture_components,
        "mixed_precision": config.mixed_precision,
        "partitions": {
            "calibration": _partition_metadata(calibration),
            "test": _partition_metadata(test),
            "train": _partition_metadata(training),
            "validation": _partition_metadata(validation),
        },
        "seeds": list(config.seeds),
        "split": {
            name: {"start": section.start, "stop": section.stop}
            for name, section in (
                ("train", split.train),
                ("validation", split.validation),
                ("calibration", split.calibration),
                ("test", split.test),
            )
        },
        "tcn_blocks": config.tcn_blocks,
        "transformer_heads": config.transformer_heads,
        "transformer_layers": config.transformer_layers,
        "warmup_epochs": config.warmup_epochs,
        "weight_decay": config.weight_decay,
    }


def _partition_metadata(partition: _Partition) -> dict[str, object]:
    return {
        "count": partition.count,
        "end": partition.dataset.example_at(partition.section.stop - 1).cutoff.isoformat(),
        "start": partition.sample().cutoff.isoformat(),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export an imbalance ensemble bundle.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    train_ensemble(arguments.dataset, arguments.output)
