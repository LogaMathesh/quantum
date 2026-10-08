"""Regenerate LR quantum artifacts from saved notebook kernels and smoke-test both hybrids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.regressors import VQR
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Crop Yiled with Soil and Weather.csv"
LR_MODEL = ROOT / "backend" / "models" / "linear_regression" / "linear_regression.joblib"
LR_KERNEL_DIR = ROOT / "linear_regression_model"
LR_ARTIFACT_DIR = ROOT / "backend" / "models" / "linear_regression_quantum"
RF_MODEL = ROOT / "backend" / "models" / "random_forest" / "random_forest.joblib"
RF_ARTIFACT_DIR = ROOT / "backend" / "models" / "random_forest_quantum"
FEATURES = ["Fertilizer", "temp", "N", "P", "K"]
TARGET = "yeild"
TEST_INPUT = {
    "Fertilizer": 120.0,
    "temp": 100.0,
    "N": 100.0,
    "P": 99.0,
    "K": 98.0,
}
EXPECTED_LR_FINAL_METRICS = {
    "r2": -17.110232,
    "mae": 6.081554,
    "mse": 60.613948,
    "rmse": 7.785496,
    "mape_percent": 74.142012,
}


def load_lr_experiment():
    dataframe = pd.read_csv(DATASET)
    X = dataframe[FEATURES].to_numpy()
    y = np.ravel(dataframe[[TARGET]].to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_data = joblib.load(LR_MODEL)
    scaler = model_data["scaler"]
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_q, _, y_train_q, _ = train_test_split(
        X_train_scaled, y_train, train_size=300, random_state=42
    )
    X_test_q, _, y_test_q, _ = train_test_split(
        X_test_scaled, y_test, train_size=150, random_state=42
    )

    saved_train = np.load(LR_KERNEL_DIR / "qkrr_model" / "X_train_q.npy")
    if not np.array_equal(X_train_q, saved_train):
        raise RuntimeError(
            "The LR quantum training subset does not match the saved notebook experiment."
        )

    K_train_q = np.load(LR_KERNEL_DIR / "K_train_q.npy")
    K_test_q = np.load(LR_KERNEL_DIR / "K_test_q.npy")
    if K_train_q.shape != (300, 300) or K_test_q.shape != (150, 300):
        raise RuntimeError(
            f"Unexpected saved LR kernel shapes: {K_train_q.shape}, {K_test_q.shape}"
        )

    return (
        model_data,
        X_train_q,
        y_train_q,
        X_test_q,
        y_test_q,
        K_train_q,
        K_test_q,
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape_percent": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def train_and_save_lr_artifacts() -> None:
    np.random.seed(42)
    algorithm_globals.random_seed = 42
    (
        model_data,
        X_train_q,
        y_train_q,
        X_test_q,
        y_test_q,
        K_train_q,
        K_test_q,
    ) = load_lr_experiment()

    lr_model = model_data["model"]
    lambda_reg = 1e-3
    lr_train_residuals_q = y_train_q - lr_model.predict(X_train_q)
    alpha_q = np.linalg.solve(
        K_train_q + lambda_reg * np.eye(len(X_train_q)),
        lr_train_residuals_q,
    )
    y_pred_lr_q = lr_model.predict(X_test_q)
    y_pred_hybrid_q = y_pred_lr_q + K_test_q @ alpha_q
    vqr_targets_q = y_train_q - (
        lr_model.predict(X_train_q) + K_train_q @ alpha_q
    )

    n_qubits = len(FEATURES)
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=2,
        entanglement="full",
    )
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=2,
        entanglement="full",
    )
    vqr = VQR(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=SPSA(maxiter=100),
        estimator=StatevectorEstimator(),
        loss="squared_error",
    )

    print("Training only the LR VQR from the notebook recipe...", flush=True)
    fit_started = time.perf_counter()
    vqr.fit(X_train_q, np.asarray(vqr_targets_q).reshape(-1))
    fit_seconds = time.perf_counter() - fit_started
    parameters = np.asarray(vqr._fit_result.x, dtype=float).reshape(-1)
    vqr_correction_q = np.asarray(vqr.predict(X_test_q)).reshape(-1)
    y_pred_final_q = np.asarray(y_pred_hybrid_q).reshape(-1) + vqr_correction_q
    metrics = calculate_metrics(y_test_q, y_pred_final_q)
    print("Reproduced LR + QKRR + VQR metrics:", metrics, flush=True)
    matches_notebook = all(
        np.isclose(
            metrics[metric_name], expected_value, rtol=0.0, atol=0.0000005
        )
        for metric_name, expected_value in EXPECTED_LR_FINAL_METRICS.items()
    )

    LR_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        LR_ARTIFACT_DIR / "qkrr_model.npz",
        alpha=alpha_q,
        X_train_q=X_train_q,
        lambda_reg=np.asarray(lambda_reg),
    )
    np.savez_compressed(
        LR_ARTIFACT_DIR / "vqr_parameters.npz",
        optimal_parameters=parameters,
    )
    config = {
        "n_qubits": n_qubits,
        "feature_map_reps": 2,
        "feature_map_entanglement": "full",
        "ansatz_reps": 2,
        "entanglement": "full",
        "inference_sampler_seed": 42,
        "ansatz_parameter_order": [str(parameter) for parameter in ansatz.parameters],
    }
    (LR_ARTIFACT_DIR / "vqr_circuits.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction_parity = test_lr_standalone_vs_notebook_vqr(
        model_data=model_data,
        alpha=alpha_q,
        X_train_q=X_train_q,
        parameters=parameters,
        feature_map=feature_map,
        fitted_vqr=vqr,
    )
    validation = {
        "model": "Linear Regression + QKRR + VQR",
        "training_method": "Existing notebook methodology; used saved LR kernel matrices without recomputing the quantum kernel.",
        "vqr_fit_seconds": fit_seconds,
        "notebook_metrics": EXPECTED_LR_FINAL_METRICS,
        "regenerated_metrics": metrics,
        "matches_notebook_to_6_decimal_places": matches_notebook,
        "metric_comparison_tolerance": 5e-7,
        "single_input_prediction_parity": prediction_parity,
    }
    (LR_ARTIFACT_DIR / "training_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Saved verified LR-only artifacts in", LR_ARTIFACT_DIR, flush=True)
    if not matches_notebook:
        print(
            "WARNING: the newly fitted VQR does not reproduce the notebook's "
            "recorded final metrics; both runs are preserved in "
            "training_validation.json.",
            flush=True,
        )


def create_kernel(config: dict) -> tuple[FidelityQuantumKernel, object, object]:
    feature_map = ZZFeatureMap(
        feature_dimension=int(config.get("n_qubits", 5)),
        reps=int(config.get("feature_map_reps", 2)),
        entanglement=config.get("feature_map_entanglement", "full"),
    )
    sampler = StatevectorSampler(
        seed=int(config.get("inference_sampler_seed", 42))
    )
    fidelity = ComputeUncompute(sampler=sampler)
    return (
        FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity),
        feature_map,
        sampler,
    )


def create_vqr_qnn(config: dict, estimator: StatevectorEstimator):
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
    qnn = EstimatorQNN(
        circuit=feature_map.compose(ansatz),
        estimator=estimator,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    return feature_map, ansatz, qnn


def predict_from_artifacts(
    model_dir: Path,
    model_path: Path,
    use_notebook_vqr_path: bool = False,
) -> float:
    model_data = joblib.load(model_path)
    qkrr_path = next(
        (
            path
            for path in (
                model_dir / "qkrr_model.npz",
                model_dir / "qkrr_model_real.npz",
            )
            if path.is_file()
        ),
        None,
    )
    vqr_path = next(
        (
            path
            for path in (
                model_dir / "vqr_parameters.npz",
                model_dir / "vqr_parameters_real.npz",
            )
            if path.is_file()
        ),
        None,
    )
    if qkrr_path is None or vqr_path is None:
        raise FileNotFoundError(
            f"Missing QKRR or VQR parameters in quantum artifact directory {model_dir}."
        )
    with np.load(qkrr_path, allow_pickle=False) as qkrr:
        alpha = np.asarray(qkrr["alpha"], dtype=float).reshape(-1)
        X_train_q = np.asarray(qkrr["X_train_q"], dtype=float)
    with np.load(vqr_path, allow_pickle=False) as vqr_data:
        parameters = np.asarray(
            vqr_data["optimal_parameters"], dtype=float
        ).reshape(-1)
    if alpha.size != X_train_q.shape[0]:
        raise RuntimeError(
            f"QKRR alpha/training-row mismatch in {model_dir}: "
            f"{alpha.size} vs {X_train_q.shape[0]}."
        )
    config = json.loads((model_dir / "vqr_circuits.json").read_text(encoding="utf-8"))

    values = np.asarray([[TEST_INPUT[name] for name in FEATURES]], dtype=float)
    scaled = model_data["scaler"].transform(values)
    baseline = float(model_data["model"].predict(scaled)[0])
    kernel, _, _ = create_kernel(config)
    kernel_row = kernel.evaluate(x_vec=scaled, y_vec=X_train_q)
    qkrr_correction = float((kernel_row @ alpha).reshape(-1)[0])

    estimator = StatevectorEstimator(seed=42)
    feature_map, ansatz, qnn = create_vqr_qnn(config, estimator)
    if len(parameters) != ansatz.num_parameters:
        raise RuntimeError(
            f"VQR parameter count mismatch in {model_dir}: "
            f"saved {len(parameters)}, circuit expects {ansatz.num_parameters}."
        )
    if use_notebook_vqr_path:
        notebook_vqr = VQR(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=SPSA(maxiter=100),
            estimator=StatevectorEstimator(seed=42),
            loss="squared_error",
        )
        vqr_output = notebook_vqr.neural_network.forward(scaled, parameters)
    else:
        vqr_output = qnn.forward(scaled, parameters)
    vqr_correction = float(np.asarray(vqr_output).reshape(-1)[0])
    prediction = baseline + qkrr_correction + vqr_correction
    if not np.isfinite(prediction):
        raise RuntimeError(f"Non-finite quantum prediction from {model_dir}.")
    return prediction


def test_lr_standalone_vs_notebook_vqr(
    model_data: dict,
    alpha: np.ndarray,
    X_train_q: np.ndarray,
    parameters: np.ndarray,
    feature_map,
    fitted_vqr: VQR,
) -> dict[str, float]:
    values = np.asarray([[TEST_INPUT[name] for name in FEATURES]], dtype=float)
    scaled = model_data["scaler"].transform(values)
    kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=ComputeUncompute(sampler=StatevectorSampler(seed=42)),
    )
    kernel_row = kernel.evaluate(x_vec=scaled, y_vec=X_train_q)
    baseline = float(model_data["model"].predict(scaled)[0])
    qkrr_correction = float((kernel_row @ alpha).reshape(-1)[0])
    notebook_vqr_correction = float(
        np.asarray(fitted_vqr.predict(scaled)).reshape(-1)[0]
    )

    estimator = StatevectorEstimator()
    notebook_feature_map = ZZFeatureMap(
        feature_dimension=len(FEATURES), reps=2, entanglement="full"
    )
    notebook_ansatz = RealAmplitudes(
        num_qubits=len(FEATURES), reps=2, entanglement="full"
    )
    notebook_qnn = EstimatorQNN(
        circuit=notebook_feature_map.compose(notebook_ansatz),
        estimator=estimator,
        input_params=notebook_feature_map.parameters,
        weight_params=notebook_ansatz.parameters,
    )
    standalone_vqr_correction = float(
        np.asarray(notebook_qnn.forward(scaled, parameters)).reshape(-1)[0]
    )
    notebook_prediction = baseline + qkrr_correction + notebook_vqr_correction
    standalone_prediction = baseline + qkrr_correction + standalone_vqr_correction
    prediction_delta = abs(notebook_prediction - standalone_prediction)
    if prediction_delta > 0.001:
        raise RuntimeError(
            "Standalone LR VQR prediction does not match notebook VQR.predict: "
            f"{standalone_prediction} vs {notebook_prediction} "
            f"(absolute difference {prediction_delta:.9f}, tolerance 0.001)."
        )
    print(
        "LR standalone/notebook single-input parity (tolerance 0.001):",
        standalone_prediction,
        notebook_prediction,
        "absolute difference:",
        prediction_delta,
        flush=True,
    )
    return {
        "standalone_prediction": standalone_prediction,
        "notebook_vqr_prediction": notebook_prediction,
        "absolute_difference": prediction_delta,
        "tolerance": 0.001,
    }


def test_both_standalone_predictions() -> None:
    lr_prediction = predict_from_artifacts(
        LR_ARTIFACT_DIR,
        LR_MODEL,
    )
    rf_prediction = predict_from_artifacts(
        RF_ARTIFACT_DIR,
        RF_MODEL,
    )
    lr_notebook_prediction = predict_from_artifacts(
        LR_ARTIFACT_DIR,
        LR_MODEL,
        use_notebook_vqr_path=True,
    )
    rf_notebook_prediction = predict_from_artifacts(
        RF_ARTIFACT_DIR,
        RF_MODEL,
        use_notebook_vqr_path=True,
    )
    for name, standalone, notebook in (
        ("LR + Quantum", lr_prediction, lr_notebook_prediction),
        ("RF + Quantum", rf_prediction, rf_notebook_prediction),
    ):
        if not np.isclose(standalone, notebook, rtol=0.0, atol=1e-9):
            raise RuntimeError(
                f"{name} standalone prediction {standalone} differs from "
                f"the notebook VQR path {notebook}."
            )
        print(
            f"{name} standalone/notebook prediction difference: "
            f"{abs(standalone - notebook):.12g}"
        )
    print(f"LR + Quantum prediction for {TEST_INPUT}: {lr_prediction:.12f}")
    print(f"RF + Quantum prediction for {TEST_INPUT}: {rf_prediction:.12f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate-lr",
        action="store_true",
        help="Rebuild the LR QKRR/VQR artifacts using the notebook's saved kernels.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run standalone predictions for LR + Quantum and RF + Quantum.",
    )
    args = parser.parse_args()
    if not args.regenerate_lr and not args.test:
        parser.error("Choose --regenerate-lr and/or --test.")
    if args.regenerate_lr:
        train_and_save_lr_artifacts()
    if args.test:
        test_both_standalone_predictions()


if __name__ == "__main__":
    main()
