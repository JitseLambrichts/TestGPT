import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    mae: float
    rmse: float
    brier: float
    log_loss: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    interval_coverage: float
    cohort_mae: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "brier": self.brier,
            "cohort_mae": dict(self.cohort_mae),
            "f1": self.f1,
            "interval_coverage": self.interval_coverage,
            "log_loss": self.log_loss,
            "mae": self.mae,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "rmse": self.rmse,
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]


def evaluate_predictions(
    *,
    actual_mw: NDArray[np.float64],
    predicted_mw: NDArray[np.float64],
    p10_mw: NDArray[np.float64],
    p90_mw: NDArray[np.float64],
    flip_target: NDArray[np.int64],
    flip_probability: NDArray[np.float64],
    threshold: float,
    cohorts: Mapping[str, NDArray[np.bool_]] | None = None,
) -> EvaluationReport:
    arrays = (actual_mw, predicted_mw, p10_mw, p90_mw, flip_target, flip_probability)
    count = len(actual_mw)
    if count == 0 or any(len(values) != count for values in arrays[1:]):
        raise ValueError("evaluation arrays must be non-empty and have matching lengths")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("flip threshold must be within [0, 1]")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("evaluation arrays must be finite")
    if np.any(p10_mw > p90_mw) or np.any((flip_target != 0) & (flip_target != 1)):
        raise ValueError("prediction intervals and flip labels are invalid")
    error = predicted_mw - actual_mw
    predicted_flip = flip_probability >= threshold
    truth = flip_target.astype(bool)
    true_positive = int(np.count_nonzero(predicted_flip & truth))
    false_positive = int(np.count_nonzero(predicted_flip & ~truth))
    false_negative = int(np.count_nonzero(~predicted_flip & truth))
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    probability = np.clip(flip_probability, 1e-6, 1.0 - 1e-6)
    cohort_mae = _cohort_mae(np.abs(error), cohorts)
    return EvaluationReport(
        mae=float(np.mean(np.abs(error))),
        rmse=math.sqrt(float(np.mean(error**2))),
        brier=float(np.mean((flip_probability - flip_target) ** 2)),
        log_loss=float(
            -np.mean(flip_target * np.log(probability) + (1 - flip_target) * np.log1p(-probability))
        ),
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=_average_precision(flip_target, flip_probability),
        interval_coverage=float(np.mean((actual_mw >= p10_mw) & (actual_mw <= p90_mw))),
        cohort_mae=cohort_mae,
    )


def promotion_decision(
    candidate: EvaluationReport,
    persistence: EvaluationReport,
    classical: EvaluationReport,
) -> PromotionDecision:
    reasons: list[str] = []
    if candidate.mae > persistence.mae * 0.98:
        reasons.append("candidate MAE must improve persistence by at least 2%")
    if candidate.pr_auc <= classical.pr_auc:
        reasons.append("candidate flip PR-AUC must exceed the classical baseline")
    if candidate.brier > classical.brier - 0.01:
        reasons.append("candidate flip Brier score must improve the classical baseline by 1%")
    if not 0.75 <= candidate.interval_coverage <= 0.85:
        reasons.append("interval coverage must be between 75% and 85%")
    for cohort, baseline_mae in persistence.cohort_mae.items():
        candidate_mae = candidate.cohort_mae.get(cohort)
        if candidate_mae is None or candidate_mae > baseline_mae * 1.05:
            reasons.append(f"critical cohort {cohort} cannot regress by more than 5%")
    return PromotionDecision(promote=not reasons, reasons=tuple(reasons))


def _cohort_mae(
    absolute_error: NDArray[np.float64],
    cohorts: Mapping[str, NDArray[np.bool_]] | None,
) -> dict[str, float]:
    if cohorts is None:
        return {}
    result: dict[str, float] = {}
    for name, mask in cohorts.items():
        if len(mask) != len(absolute_error):
            raise ValueError(f"cohort {name} has a mismatched length")
        if np.any(mask):
            result[name] = float(np.mean(absolute_error[mask]))
    return result


def _average_precision(target: NDArray[np.int64], probability: NDArray[np.float64]) -> float:
    positives = int(np.count_nonzero(target))
    if positives == 0:
        return 0.0
    order = np.argsort(-probability, kind="stable")
    sorted_target = target[order]
    cumulative_positive = np.cumsum(sorted_target)
    precision = cumulative_positive / np.arange(1, len(target) + 1)
    return float(np.sum(precision * sorted_target) / positives)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
