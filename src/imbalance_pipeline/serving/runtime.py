import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.special import ndtr  # type: ignore[import-untyped]

from imbalance_pipeline.domain.imbalance import ConfirmedState, advance_state
from imbalance_pipeline.features.engine import FeatureSnapshot
from imbalance_pipeline.model.bundle import ModelManifest, validate_bundle
from imbalance_pipeline.model.export_onnx import INPUT_NAMES, OUTPUT_NAMES
from imbalance_pipeline.training.calibration import IsotonicCalibrator
from imbalance_pipeline.training.data import PreparedFeatures, RobustPreprocessor

_MIN_SCALE = 1e-4
_QUANTILE_ITERATIONS = 64


@dataclass(frozen=True, slots=True)
class PredictionValues:
    model_version: str
    feature_schema_hash: str
    system_imbalance_mw: float
    p10_mw: float
    p90_mw: float
    flip_probability: float
    will_flip: bool
    current_state: ConfirmedState | None
    predicted_state: ConfirmedState | None
    delta_mw: float


class OnnxEnsemble:
    """Validated, CPU-only runtime for the three-member probabilistic ensemble."""

    def __init__(
        self,
        bundle_dir: Path,
        *,
        expected_schema_hash: str,
        deadband_mw: float = 10.0,
        intra_op_num_threads: int = 1,
    ) -> None:
        if deadband_mw <= 0:
            raise ValueError("deadband_mw must be positive")
        if intra_op_num_threads <= 0:
            raise ValueError("intra_op_num_threads must be positive")
        self._bundle_dir = Path(bundle_dir)
        validation = validate_bundle(self._bundle_dir, expected_schema_hash=expected_schema_hash)
        if not validation.valid:
            raise ValueError(validation.reason or "model bundle validation failed")
        self._manifest = ModelManifest.load(self._bundle_dir)
        _require_runtime_contract(self._manifest)
        self._preprocessor = _load_preprocessor(
            self._bundle_dir / self._manifest.preprocessing_file,
            expected_schema_hash,
        )
        self._calibrator = _load_calibrator(self._bundle_dir / self._manifest.calibration_file)
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._sessions = tuple(
            _load_session(self._bundle_dir / member, options) for member in self._manifest.members
        )
        self._deadband_mw = deadband_mw

    @property
    def model_version(self) -> str:
        return self._manifest.model_version

    @property
    def feature_schema_hash(self) -> str:
        return self._manifest.feature_schema_hash

    def predict(self, snapshot: FeatureSnapshot) -> PredictionValues:
        prepared = self._preprocessor.transform(snapshot)
        outputs = tuple(
            session.run(list(OUTPUT_NAMES), _session_inputs(prepared)) for session in self._sessions
        )
        weights, means, scales = _ensemble_distribution(outputs)
        p10 = _mixture_quantile(0.1, weights, means, scales)
        median = _mixture_quantile(0.5, weights, means, scales)
        p90 = _mixture_quantile(0.9, weights, means, scales)
        raw_flip_probability = _sigmoid(_average_scalar(outputs, 3, "flip_logit"))
        flip_probability = float(self._calibrator.predict(np.asarray([raw_flip_probability]))[0])
        delta = _average_scalar(outputs, 4, "delta")
        predicted_state = advance_state(snapshot.current_state, median, self._deadband_mw)
        will_flip = (
            snapshot.current_state is not None
            and flip_probability >= self._calibrator.decision_threshold
        )
        return PredictionValues(
            model_version=self._manifest.model_version,
            feature_schema_hash=self._manifest.feature_schema_hash,
            system_imbalance_mw=median,
            p10_mw=p10,
            p90_mw=p90,
            flip_probability=flip_probability,
            will_flip=will_flip,
            current_state=snapshot.current_state,
            predicted_state=predicted_state,
            delta_mw=delta,
        )


def _require_runtime_contract(manifest: ModelManifest) -> None:
    if manifest.output_names != OUTPUT_NAMES:
        raise ValueError("model manifest output names do not match the runtime contract")


def _load_preprocessor(path: Path, expected_schema_hash: str) -> RobustPreprocessor:
    try:
        preprocessor = RobustPreprocessor.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("invalid preprocessing artifact") from exc
    if preprocessor.feature_schema_hash != expected_schema_hash:
        raise ValueError("preprocessing schema hash does not match")
    return preprocessor


def _load_calibrator(path: Path) -> IsotonicCalibrator:
    try:
        return IsotonicCalibrator.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("invalid calibration artifact") from exc


def _load_session(path: Path, options: ort.SessionOptions) -> ort.InferenceSession:
    try:
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ValueError(f"cannot load ONNX member {path.name}") from exc
    if "CPUExecutionProvider" not in session.get_providers():
        raise ValueError("ONNX CPU execution provider is unavailable")
    input_names = tuple(item.name for item in session.get_inputs())
    output_names = tuple(item.name for item in session.get_outputs())
    if input_names != INPUT_NAMES or output_names != OUTPUT_NAMES:
        raise ValueError("ONNX member input or output contract does not match")
    return session


def _session_inputs(prepared: PreparedFeatures) -> Mapping[str, NDArray[np.float32]]:
    return {
        "local": prepared.local_values[None, ...],
        "local_mask": prepared.local_masks[None, ...].astype(np.float32),
        "context": prepared.context_values[None, ...],
        "context_mask": prepared.context_masks[None, ...].astype(np.float32),
        "static": prepared.static_values[None, ...],
        "static_mask": prepared.static_masks[None, ...].astype(np.float32),
    }


def _ensemble_distribution(
    outputs: tuple[list[NDArray[np.float32]], ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    weights: list[NDArray[np.float64]] = []
    means: list[NDArray[np.float64]] = []
    scales: list[NDArray[np.float64]] = []
    for member in outputs:
        logits = _member_vector(member, 0, "mixture_logits")
        member_means = _member_vector(member, 1, "mixture_means")
        log_scales = _member_vector(member, 2, "mixture_log_scales")
        if logits.shape != member_means.shape or logits.shape != log_scales.shape:
            raise ValueError("ONNX member mixture outputs have inconsistent shapes")
        weights.append(_softmax(logits) / len(outputs))
        means.append(member_means)
        scales.append(_softplus(log_scales) + _MIN_SCALE)
    return (
        np.concatenate(weights),
        np.concatenate(means),
        np.concatenate(scales),
    )


def _member_vector(
    outputs: list[NDArray[np.float32]],
    index: int,
    name: str,
) -> NDArray[np.float64]:
    if len(outputs) != len(OUTPUT_NAMES):
        raise ValueError("ONNX member returned an unexpected number of outputs")
    values = np.asarray(outputs[index], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 1 or not np.isfinite(values).all():
        raise ValueError(f"ONNX member returned invalid {name}")
    return np.asarray(values[0], dtype=np.float64)


def _average_scalar(outputs: tuple[list[NDArray[np.float32]], ...], index: int, name: str) -> float:
    values: list[float] = []
    for member in outputs:
        output = np.asarray(member[index], dtype=np.float64)
        if output.shape not in ((1,), (1, 1)) or not np.isfinite(output).all():
            raise ValueError(f"ONNX member returned invalid {name}")
        values.append(float(output.reshape(-1)[0]))
    return float(np.mean(values))


def _mixture_quantile(
    probability: float,
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("quantile probability must be within (0, 1)")
    width = 12.0 * float(np.max(scales))
    lower = float(np.min(means)) - width
    upper = float(np.max(means)) + width
    for _ in range(_QUANTILE_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        if _mixture_cdf(midpoint, weights, means, scales) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _mixture_cdf(
    value: float,
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> float:
    return float(np.sum(weights * ndtr((value - means) / scales)))


def _softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return np.asarray(exponentials / np.sum(exponentials), dtype=np.float64)


def _softplus(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
