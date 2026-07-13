import math
from dataclasses import replace

import numpy as np

from imbalance_pipeline.training.metrics import (
    EvaluationReport,
    evaluate_predictions,
    promotion_decision,
)


def test_metrics_match_known_regression_probability_and_interval_values() -> None:
    report = evaluate_predictions(
        actual_mw=np.asarray([0.0, 2.0, 4.0, 6.0]),
        predicted_mw=np.asarray([0.0, 4.0, 2.0, 6.0]),
        p10_mw=np.asarray([-1.0, 3.0, 3.0, 7.0]),
        p90_mw=np.asarray([1.0, 4.0, 5.0, 8.0]),
        flip_target=np.asarray([0, 1, 1, 0]),
        flip_probability=np.asarray([0.1, 0.8, 0.6, 0.2]),
        threshold=0.5,
    )

    assert report.mae == 1.0
    assert report.rmse == math.sqrt(2.0)
    assert report.interval_coverage == 0.5
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert 0.0 <= report.brier <= 1.0
    assert 0.0 <= report.pr_auc <= 1.0


def test_promotion_requires_every_safety_and_quality_gate() -> None:
    candidate = EvaluationReport(
        mae=9.7,
        rmse=12.0,
        brier=0.12,
        log_loss=0.3,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        pr_auc=0.40,
        interval_coverage=0.80,
        cohort_mae={"high_volatility": 11.0},
    )
    persistence = EvaluationReport(
        mae=10.0,
        rmse=13.0,
        brier=0.3,
        log_loss=0.6,
        precision=0.2,
        recall=0.2,
        f1=0.2,
        pr_auc=0.1,
        interval_coverage=0.0,
        cohort_mae={"high_volatility": 11.0},
    )
    classical = EvaluationReport(
        mae=9.9,
        rmse=12.5,
        brier=0.14,
        log_loss=0.4,
        precision=0.4,
        recall=0.4,
        f1=0.4,
        pr_auc=0.35,
        interval_coverage=0.8,
        cohort_mae={"high_volatility": 10.9},
    )

    accepted = promotion_decision(candidate, persistence, classical)
    rejected = promotion_decision(
        replace(candidate, interval_coverage=0.9),
        persistence,
        classical,
    )

    assert accepted.promote is True
    assert accepted.reasons == ()
    assert rejected.promote is False
    assert "interval coverage must be between 75% and 85%" in rejected.reasons
