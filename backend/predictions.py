"""Standalone prediction functions for the four verified crop-yield models."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.preprocessing import StandardScaler


BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
FEATURES = ("Fertilizer", "temp", "N", "P", "K")
TARGET = "yeild"


@dataclass(frozen=True)
class ClassicalModel:
    estimator: Any
    scaler: StandardScaler
    model_path: Path


@dataclass(frozen=True)
class QuantumModel:
    classical: ClassicalModel
    feature_map: ZZFeatureMap
    alpha: np.ndarray
    quantum_training_inputs: np.ndarray
    vqr_qnn: EstimatorQNN
    vqr_parameters: np.ndarray
    artifact_paths: tuple[Path, ...]
    _training_states: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _training_states_lock: Lock = field(
        default_factory=Lock, init=False, repr=False, compare=False
    )

    @property
    def quantum_training_states(self) -> np.ndarray:
        states = self._training_states
        if states is None:
            with self._training_states_lock:
                states = self._training_states
                if states is None:
                    parameters = tuple(self.feature_map.parameters)
                    states = np.stack(
                        [
                            Statevector.from_instruction(
                                self.feature_map.assign_parameters(
                                    dict(zip(parameters, values)), inplace=False
                                )
                            ).data
                            for values in self.quantum_training_inputs
                        ]
                    )
                    states.setflags(write=False)
                    object.__setattr__(self, "_training_states", states)
        return states


@dataclass(frozen=True)
class PredictionModels:
    linear_regression: ClassicalModel
    linear_regression_quantum: QuantumModel
    random_forest: ClassicalModel
    random_forest_quantum: QuantumModel


def _read_model(model_path: Path) -> ClassicalModel:
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing classical model artifact: {model_path}")
    model_data = joblib.load(model_path)
    if not isinstance(model_data, dict):
        raise TypeError(f"Expected a model dictionary in {model_path}")

    required_keys = {"model", "scaler", "feature_cols", "target_col"}
    missing_keys = required_keys - model_data.keys()
    if missing_keys:
        raise ValueError(
            f"Missing model metadata in {model_path}: {sorted(missing_keys)}"
        )
    if list(model_data["feature_cols"]) != list(FEATURES):
        raise ValueError(
            f"Feature order mismatch in {model_path}: "
            f"{model_data['feature_cols']!r}"
        )
    if model_data["target_col"] != TARGET:
        raise ValueError(
            f"Target mismatch in {model_path}: {model_data['target_col']!r}"
        )

    scaler = model_data["scaler"]
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Expected StandardScaler in {model_path}")
    if scaler.n_features_in_ != len(FEATURES):
        raise ValueError(
            f"Scaler feature count mismatch in {model_path}: "
            f"{scaler.n_features_in_}"
        )
    return ClassicalModel(
        estimator=model_data["model"],
        scaler=scaler,
        model_path=model_path,
    )


def _load_quantum_model(
    classical: ClassicalModel,
    artifact_dir: Path,
) -> QuantumModel:
    qkrr_path = artifact_dir / "qkrr_model.npz"
    vqr_path = artifact_dir / "vqr_parameters.npz"
    config_path = artifact_dir / "vqr_circuits.json"
    for required_path in (qkrr_path, vqr_path, config_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing quantum artifact: {required_path}")

    with np.load(qkrr_path, allow_pickle=False) as qkrr_data:
        alpha = np.asarray(qkrr_data["alpha"], dtype=float).reshape(-1)
        quantum_training_inputs = np.asarray(
            qkrr_data["X_train_q"], dtype=float
        )
        lambda_reg = float(np.asarray(qkrr_data["lambda_reg"]).reshape(-1)[0])

    if quantum_training_inputs.ndim != 2:
        raise ValueError(
            f"Quantum training inputs must be a matrix: {qkrr_path}"
        )
    if quantum_training_inputs.shape[1] != len(FEATURES):
        raise ValueError(
            f"Quantum training input feature count mismatch: "
            f"{quantum_training_inputs.shape}"
        )
    if alpha.shape != (quantum_training_inputs.shape[0],):
        raise ValueError(
            f"QKRR coefficient/training sample shape mismatch in {qkrr_path}"
        )
    if not np.isfinite(alpha).all() or not np.isfinite(quantum_training_inputs).all():
        raise ValueError(f"Non-finite QKRR artifact values in {qkrr_path}")
    if not math.isfinite(lambda_reg) or lambda_reg <= 0:
        raise ValueError(f"Invalid QKRR regularization value in {qkrr_path}")

    with np.load(vqr_path, allow_pickle=False) as vqr_data:
        vqr_parameters = np.asarray(
            vqr_data["optimal_parameters"], dtype=float
        ).reshape(-1)
    if not np.isfinite(vqr_parameters).all():
        raise ValueError(f"Non-finite VQR parameters in {vqr_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_config = {
        "n_qubits",
        "feature_map_reps",
        "ansatz_reps",
        "entanglement",
    }
    if not required_config.issubset(config):
        raise ValueError(
            f"Incomplete VQR configuration in {config_path}: "
            f"{sorted(required_config - config.keys())}"
        )
    if int(config["n_qubits"]) != len(FEATURES):
        raise ValueError(f"Unexpected qubit count in {config_path}")

    feature_map = ZZFeatureMap(
        feature_dimension=int(config["n_qubits"]),
        reps=int(config["feature_map_reps"]),
        entanglement=config.get(
            "feature_map_entanglement", config["entanglement"]
        ),
    )
    ansatz = RealAmplitudes(
        num_qubits=int(config["n_qubits"]),
        reps=int(config["ansatz_reps"]),
        entanglement=config["entanglement"],
    )
    if len(vqr_parameters) != ansatz.num_parameters:
        raise ValueError(
            f"VQR parameter count mismatch in {vqr_path}: "
            f"saved={len(vqr_parameters)}, expected={ansatz.num_parameters}"
        )

    estimator_seed = int(config.get("inference_estimator_seed", 42))
    vqr_qnn = EstimatorQNN(
        circuit=feature_map.compose(ansatz),
        estimator=StatevectorEstimator(seed=estimator_seed),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    return QuantumModel(
        classical=classical,
        feature_map=feature_map,
        alpha=alpha,
        quantum_training_inputs=quantum_training_inputs,
        vqr_qnn=vqr_qnn,
        vqr_parameters=vqr_parameters,
        artifact_paths=(qkrr_path, vqr_path, config_path),
    )


def _load_models() -> PredictionModels:
    lr_classical = _read_model(
        MODELS_DIR / "linear_regression" / "linear_regression.joblib"
    )
    rf_classical = _read_model(
        MODELS_DIR / "random_forest" / "random_forest.joblib"
    )
    return PredictionModels(
        linear_regression=lr_classical,
        linear_regression_quantum=_load_quantum_model(
            lr_classical,
            MODELS_DIR / "linear_regression_quantum",
        ),
        random_forest=rf_classical,
        random_forest_quantum=_load_quantum_model(
            rf_classical,
            MODELS_DIR / "random_forest_quantum",
        ),
    )


# Model files and quantum circuits are loaded once when this module is imported.
_MODELS = _load_models()


def _input_array(input_data: Mapping[str, object]) -> np.ndarray:
    if not isinstance(input_data, Mapping):
        raise TypeError("input_data must be a mapping keyed by the feature names.")
    supplied = set(input_data)
    required = set(FEATURES)
    if supplied != required:
        missing = sorted(required - supplied)
        extra = sorted(supplied - required)
        raise ValueError(f"Input feature mismatch; missing={missing}, extra={extra}")

    values: list[float] = []
    for feature in FEATURES:
        value = input_data[feature]
        if isinstance(value, bool):
            raise TypeError(f"{feature} must be a finite number, not bool.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{feature} must be a finite number.") from error
        if not math.isfinite(numeric_value):
            raise ValueError(f"{feature} must be finite.")
        values.append(numeric_value)
    return np.asarray([values], dtype=float)


def _predict_classical(
    model: ClassicalModel,
    input_data: Mapping[str, object],
) -> float:
    values = _input_array(input_data)
    scaled = model.scaler.transform(values)
    prediction = float(np.asarray(model.estimator.predict(scaled)).reshape(-1)[0])
    if not math.isfinite(prediction):
        raise RuntimeError(f"Model returned a non-finite prediction: {model.model_path}")
    return prediction


def _predict_quantum(
    model: QuantumModel,
    input_data: Mapping[str, object],
) -> float:
    values = _input_array(input_data)
    scaled = model.classical.scaler.transform(values)
    baseline = float(
        np.asarray(model.classical.estimator.predict(scaled)).reshape(-1)[0]
    )
    parameters = tuple(model.feature_map.parameters)
    query_state = Statevector.from_instruction(
        model.feature_map.assign_parameters(
            dict(zip(parameters, scaled[0])), inplace=False
        )
    )
    kernel_values = np.abs(
        model.quantum_training_states.conj() @ query_state.data
    ) ** 2
    qkrr_correction = float(kernel_values @ model.alpha)
    vqr_correction = float(
        np.asarray(
            model.vqr_qnn.forward(scaled, model.vqr_parameters)
        ).reshape(-1)[0]
    )
    prediction = baseline + qkrr_correction + vqr_correction
    if not math.isfinite(prediction):
        raise RuntimeError("Quantum-integrated model returned a non-finite prediction.")
    return prediction


def predict_linear_regression(input_data: Mapping[str, object]) -> float:
    """Predict yield with the saved classical Linear Regression model."""
    return _predict_classical(_MODELS.linear_regression, input_data)


def predict_linear_regression_quantum(
    input_data: Mapping[str, object],
) -> float:
    """Predict with LR + saved QKRR correction + saved VQR correction."""
    return _predict_quantum(
        _MODELS.linear_regression_quantum, input_data
    )


def predict_random_forest(input_data: Mapping[str, object]) -> float:
    """Predict yield with the saved classical Random Forest model."""
    return _predict_classical(_MODELS.random_forest, input_data)


def predict_random_forest_quantum(
    input_data: Mapping[str, object],
) -> float:
    """Predict with RF + saved QKRR correction + saved VQR correction."""
    return _predict_quantum(
        _MODELS.random_forest_quantum, input_data
    )


def predict_all_models(input_data: Mapping[str, object]) -> dict[str, float]:
    """Return a prediction from each classical and quantum-integrated model."""
    return {
        "linear_regression": predict_linear_regression(input_data),
        "linear_regression_quantum": predict_linear_regression_quantum(input_data),
        "random_forest": predict_random_forest(input_data),
        "random_forest_quantum": predict_random_forest_quantum(input_data),
    }
