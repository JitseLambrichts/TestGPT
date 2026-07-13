import argparse
import hashlib
import json
import math
import os
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
from imbalance_pipeline.training.baselines import classical_baseline, persistence_baseline
from imbalance_pipeline.training.calibration import IsotonicCalibrator
from imbalance_pipeline.training.data import (
    RobustPreprocessor,
    TrainingBatch,
    TrainingExample,
    preprocess_training_examples,
)
from imbalance_pipeline.training.export_data import load_training_dataset
from imbalance_pipeline.training.metrics import EvaluationReport, evaluate_predictions
from imbalance_pipeline.training.splits import TimeSplit, walk_forward_splits


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
    split: TimeSplit | None = None

    def __post_init__(self) -> None:
        if len(set(self.seeds)) != 3:
            raise ValueError("training requires exactly three unique ensemble seeds")
        if min(
            self.epochs,
            self.batch_size,
            self.warmup_epochs,
            self.early_stopping_patience,
            self.d_model,
            self.tcn_blocks,
            self.transformer_layers,
            self.transformer_heads,
            self.mixture_components,
        ) <= 0:
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


def train_ensemble(
    dataset_path: Path,
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> Path:
    config = config or TrainingConfig()
    examples = sorted(load_training_dataset(dataset_path), key=lambda example: example.cutoff)
    _require_unique_cutoffs(examples)
    split = config.split or walk_forward_splits(
        [example.cutoff for example in examples],
        folds=1,
    )[-1]
    train_raw, validation_raw, calibration_raw, test_raw = _partitions(examples, split)
    preprocessor = RobustPreprocessor.fit(train_raw)
    train_examples = preprocess_training_examples(train_raw, preprocessor)
    validation_examples = preprocess_training_examples(validation_raw, preprocessor)
    calibration_examples = preprocess_training_examples(calibration_raw, preprocessor)
    test_examples = preprocess_training_examples(test_raw, preprocessor)
    device = _device(config.device)
    results = tuple(
        _train_member(seed, train_examples, validation_examples, config, device)
        for seed in config.seeds
    )
    calibration_prediction = _ensemble_predictions(
        results,
        calibration_examples,
        config.batch_size,
        device,
    )
    calibrator = _fit_calibrator(calibration_prediction.raw_flip_probability, calibration_raw)
    test_prediction = _ensemble_predictions(results, test_examples, config.batch_size, device)
    candidate_report = _evaluation_report(test_prediction, test_raw, calibrator)
    baseline_reports = _baseline_reports(train_raw, test_raw)
    return _write_candidate(
        output_dir,
        results,
        preprocessor,
        calibrator,
        candidate_report,
        baseline_reports,
        train_examples,
        train_raw,
        test_raw,
        config,
    )


def _train_member(
    seed: int,
    training: Sequence[TrainingExample],
    validation: Sequence[TrainingExample],
    config: TrainingConfig,
    device: torch.device,
) -> _MemberResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ImbalanceForecaster(
        local_features=training[0].local_values.shape[-1],
        context_features=training[0].context_values.shape[-1],
        static_features=training[0].static_values.shape[-1],
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
        positive_weight=_positive_weight(training),
        deadband_mw=config.deadband_mw,
    )
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    stalled_epochs = 0
    completed_epochs = 0
    for epoch in range(config.epochs):
        _set_learning_rate(optimizer, config, epoch)
        model.train()
        for examples in _batches(training, config.batch_size, seed + epoch):
            batch = TrainingBatch.from_examples(
                examples,
                batch_id=f"seed-{seed}-epoch-{epoch}",
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.mixed_precision):
                breakdown = loss(model(*_model_inputs(batch)), batch)
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
        validation_loss = _validation_loss(model, validation, loss, config.batch_size, device)
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
    examples: Sequence[TrainingExample],
    loss: MultiTaskLoss,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for items in _ordered_batches(examples, batch_size):
            batch = TrainingBatch.from_examples(items, batch_id="validation").to(device)
            output = model(*_model_inputs(batch))
            breakdown = loss(output, batch)
            brier = _masked_brier(output.flip_logit, batch.flip_target, batch.flip_mask)
            values.append(float((breakdown.mixture_nll + 0.25 * brier).cpu()))
    return float(np.mean(values))


def _ensemble_predictions(
    results: Sequence[_MemberResult],
    examples: Sequence[TrainingExample],
    batch_size: int,
    device: torch.device,
) -> _EnsemblePredictions:
    point: list[NDArray[np.float64]] = []
    p10: list[NDArray[np.float64]] = []
    p90: list[NDArray[np.float64]] = []
    flip: list[NDArray[np.float64]] = []
    for result in results:
        result.model.eval()
    with torch.inference_mode():
        for items in _ordered_batches(examples, batch_size):
            batch = TrainingBatch.from_examples(items, batch_id="evaluation").to(device)
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
    return _EnsemblePredictions(
        predicted_mw=np.concatenate(point),
        p10_mw=np.concatenate(p10),
        p90_mw=np.concatenate(p90),
        raw_flip_probability=np.concatenate(flip),
    )


def _fit_calibrator(
    probability: NDArray[np.float64],
    examples: Sequence[TrainingExample],
) -> IsotonicCalibrator:
    mask = np.asarray([example.flip_mask > 0.5 for example in examples])
    if not np.any(mask):
        return IsotonicCalibrator(
            x_thresholds=np.asarray([0.0, 1.0]),
            y_thresholds=np.asarray([0.0, 0.0]),
            decision_threshold=0.5,
        )
    target = np.asarray([example.flip_target for example in examples], dtype=np.int64)[mask]
    return IsotonicCalibrator.fit(probability[mask], target)


def _evaluation_report(
    prediction: _EnsemblePredictions,
    examples: Sequence[TrainingExample],
    calibrator: IsotonicCalibrator,
) -> EvaluationReport:
    mask = np.asarray([example.flip_mask > 0.5 for example in examples])
    if not np.any(mask):
        raise ValueError("test partition requires at least one known flip label")
    calibrated = calibrator.predict(prediction.raw_flip_probability[mask])
    return evaluate_predictions(
        actual_mw=np.asarray([example.target_next for example in examples], dtype=np.float64)[mask],
        predicted_mw=prediction.predicted_mw[mask],
        p10_mw=prediction.p10_mw[mask],
        p90_mw=prediction.p90_mw[mask],
        flip_target=np.asarray([example.flip_target for example in examples], dtype=np.int64)[mask],
        flip_probability=calibrated,
        threshold=calibrator.decision_threshold,
    )


def _baseline_reports(
    training: Sequence[TrainingExample],
    test: Sequence[TrainingExample],
) -> dict[str, EvaluationReport]:
    mask = np.asarray([example.flip_mask > 0.5 for example in test])
    if not np.any(mask):
        raise ValueError("test partition requires at least one known flip label")
    actual = np.asarray([example.target_next for example in test], dtype=np.float64)[mask]
    target = np.asarray([example.flip_target for example in test], dtype=np.int64)[mask]
    persistence = persistence_baseline(test)[mask]
    classical = classical_baseline(training, test)
    return {
        "persistence": evaluate_predictions(
            actual_mw=actual,
            predicted_mw=persistence,
            p10_mw=persistence,
            p90_mw=persistence,
            flip_target=target,
            flip_probability=np.zeros(len(actual), dtype=np.float64),
            threshold=0.5,
        ),
        "classical": evaluate_predictions(
            actual_mw=actual,
            predicted_mw=classical.predicted_mw[mask],
            p10_mw=classical.predicted_mw[mask],
            p90_mw=classical.predicted_mw[mask],
            flip_target=target,
            flip_probability=classical.flip_probability[mask],
            threshold=0.5,
        ),
    }


def _write_candidate(
    output: Path,
    results: Sequence[_MemberResult],
    preprocessor: RobustPreprocessor,
    calibrator: IsotonicCalibrator,
    report: EvaluationReport,
    baselines: dict[str, EvaluationReport],
    training: Sequence[TrainingExample],
    training_raw: Sequence[TrainingExample],
    test_raw: Sequence[TrainingExample],
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
        sample_batch = TrainingBatch.from_examples([training[0]], batch_id="onnx-export")
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
        _write_json(staging / "run_config.json", _config_dict(config))
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
                        training[0].context_values.shape[0],
                        training[0].context_values.shape[1],
                    ],
                    "local": [
                        -1,
                        training[0].local_values.shape[0],
                        training[0].local_values.shape[1],
                    ],
                    "static": [-1, training[0].static_values.shape[0]],
                },
                "members": members,
                "model_version": f"imbalance-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
                "output_names": list(OUTPUT_NAMES),
                "preprocessing_file": "preprocessing.json",
                "calibration_file": "calibration.json",
                "runtime": {"opset": 18, "provider": "CPUExecutionProvider"},
                "training_period": {
                    "end": max(example.cutoff for example in training_raw).isoformat(),
                    "start": min(example.cutoff for example in training_raw).isoformat(),
                },
            },
        )
        manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        validation = validate_bundle(
            staging,
            expected_schema_hash=str(manifest["feature_schema_hash"]),
        )
        if not validation.valid:
            raise ValueError(validation.reason or "candidate bundle validation failed")
        _fsync_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _partitions(
    examples: Sequence[TrainingExample],
    split: TimeSplit,
) -> tuple[
    list[TrainingExample],
    list[TrainingExample],
    list[TrainingExample],
    list[TrainingExample],
]:
    for section in (split.train, split.validation, split.calibration, split.test):
        if section.stop > len(examples):
            raise ValueError("training split is outside the dataset")
    return (
        list(examples[split.train.start : split.train.stop]),
        list(examples[split.validation.start : split.validation.stop]),
        list(examples[split.calibration.start : split.calibration.stop]),
        list(examples[split.test.start : split.test.stop]),
    )


def _positive_weight(examples: Sequence[TrainingExample]) -> float:
    labels = [example.flip_target for example in examples if example.flip_mask > 0.5]
    positives = sum(label > 0.5 for label in labels)
    return (len(labels) - positives) / positives if positives else 1.0


def _batches(
    examples: Sequence[TrainingExample],
    batch_size: int,
    seed: int,
) -> Iterator[list[TrainingExample]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(examples), generator=generator).tolist()
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]


def _ordered_batches(
    examples: Sequence[TrainingExample],
    batch_size: int,
) -> Iterator[list[TrainingExample]]:
    for start in range(0, len(examples), batch_size):
        yield list(examples[start : start + batch_size])


def _model_inputs(batch: TrainingBatch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        batch.local,
        batch.local_mask,
        batch.context,
        batch.context_mask,
        batch.static,
        batch.static_mask,
    )


def _masked_brier(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    typed_mask = mask.to(logits.dtype)
    return (
        ((torch.sigmoid(logits) - target).square() * typed_mask).sum()
        / typed_mask.sum().clamp_min(1.0)
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


def _require_unique_cutoffs(examples: Sequence[TrainingExample]) -> None:
    if not examples:
        raise ValueError("training dataset is empty")
    cutoffs = [example.cutoff for example in examples]
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("training dataset cutoffs must be unique")


def _config_dict(config: TrainingConfig) -> dict[str, object]:
    value = {
        "batch_size": config.batch_size,
        "d_model": config.d_model,
        "deadband_mw": config.deadband_mw,
        "early_stopping_patience": config.early_stopping_patience,
        "epochs": config.epochs,
        "gradient_clip_norm": config.gradient_clip_norm,
        "learning_rate": config.learning_rate,
        "mixture_components": config.mixture_components,
        "seeds": list(config.seeds),
        "tcn_blocks": config.tcn_blocks,
        "transformer_heads": config.transformer_heads,
        "transformer_layers": config.transformer_layers,
        "warmup_epochs": config.warmup_epochs,
        "weight_decay": config.weight_decay,
    }
    if config.split is not None:
        value["split"] = {
            name: {"start": section.start, "stop": section.stop}
            for name, section in (
                ("train", config.split.train),
                ("validation", config.split.validation),
                ("calibration", config.split.calibration),
                ("test", config.split.test),
            )
        }
    return value


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
