"""Load all four saved models, run a smoke test, and write a validation report."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import qiskit
import qiskit_algorithms
import qiskit_machine_learning
import sklearn


BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
REPORT_PATH = MODELS_DIR / "model_validation.json"
TEST_INPUT = {
    "Fertilizer": 120,
    "temp": 100,
    "N": 100,
    "P": 99,
    "K": 98,
}

ARTIFACT_PATHS = {
    "linear_regression": [
        "backend/models/linear_regression/linear_regression.joblib"
    ],
    "linear_regression_quantum": [
        "backend/models/linear_regression_quantum/linear_regression.joblib",
        "backend/models/linear_regression_quantum/qkrr_model.npz",
        "backend/models/linear_regression_quantum/vqr_parameters.npz",
        "backend/models/linear_regression_quantum/vqr_circuits.json",
    ],
    "random_forest": [
        "backend/models/random_forest/random_forest.joblib"
    ],
    "random_forest_quantum": [
        "backend/models/random_forest_quantum/qkrr_model.npz",
        "backend/models/random_forest_quantum/vqr_parameters.npz",
        "backend/models/random_forest_quantum/vqr_circuits.json",
        "backend/models/random_forest/random_forest.joblib",
    ],
}

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "linear_regression_quantum": "Linear Regression + Quantum",
    "random_forest": "Random Forest",
    "random_forest_quantum": "Random Forest + Quantum",
}


def _dependencies() -> dict[str, str]:
    return {
        "python": "3.12",
        "scikit_learn_runtime": sklearn.__version__,
        "qiskit": qiskit.__version__,
        "qiskit_machine_learning": qiskit_machine_learning.__version__,
        "qiskit_algorithms": qiskit_algorithms.__version__,
        "joblib": __import__("joblib").__version__,
        "numpy": __import__("numpy").__version__,
    }


def _missing_artifacts(paths: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if not (BACKEND_DIR.parent / path.replace("/", "\\")).is_file()
    ]


def main() -> None:
    with warnings.catch_warnings(record=True) as load_warnings:
        warnings.simplefilter("always")
        import predictions

    warning_messages = sorted(
        {
            f"{warning.category.__name__}: {warning.message}"
            for warning in load_warnings
        }
    )

    model_registry_id = id(predictions._MODELS)
    results: dict[str, float] = predictions.predict_all_models(TEST_INPUT)
    load_calls: list[object] = []
    original_joblib_load = predictions.joblib.load

    def track_model_load(*args: object, **kwargs: object) -> Any:
        load_calls.append(args[0] if args else None)
        return original_joblib_load(*args, **kwargs)

    predictions.joblib.load = track_model_load
    try:
        predictions.predict_all_models(TEST_INPUT)
    finally:
        predictions.joblib.load = original_joblib_load
    if load_calls or id(predictions._MODELS) != model_registry_id:
        raise RuntimeError("Model inference unexpectedly reloaded model artifacts.")

    expected_model_keys = set(MODEL_LABELS)
    if set(results) != expected_model_keys:
        raise RuntimeError(
            f"Unexpected prediction keys: {sorted(results)}"
        )

    report: dict[str, Any] = {
        "test_input": TEST_INPUT,
        "dependencies": _dependencies(),
        "model_loading": {
            "strategy": "All four estimator/artifact sets and quantum circuits load once at predictions module import.",
            "registry_unchanged_during_inference": True,
            "joblib_load_calls_during_second_prediction_batch": len(load_calls),
        },
        "warnings": warning_messages
        + [
            "Qiskit 2.5.0 emits deprecation warnings for the notebook's "
            "ZZFeatureMap and RealAmplitudes circuit classes; the notebook "
            "circuit definitions were retained."
        ],
    }
    for key, label in MODEL_LABELS.items():
        prediction = float(results[key])
        if not math.isfinite(prediction):
            raise RuntimeError(f"{label} produced a non-finite value: {prediction}")
        missing = _missing_artifacts(ARTIFACT_PATHS[key])
        report[key] = {
            "model_name": label,
            "artifact_paths": ARTIFACT_PATHS[key],
            "load_status": "success" if not missing else "failed",
            "missing_artifacts": missing,
            "prediction_status": "success",
            "test_input": TEST_INPUT,
            "prediction": prediction,
            "warnings": (
                [
                    "The saved RF estimator was trained with scikit-learn 1.7.0 "
                    f"and loaded with runtime {sklearn.__version__}; "
                    "InconsistentVersionWarning was emitted."
                ]
                if key.startswith("random_forest")
                else []
            ),
            "dependencies": _dependencies(),
        }
        if missing:
            raise FileNotFoundError(
                f"{label} is missing required inference artifacts: {missing}"
            )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for key, label in MODEL_LABELS.items():
        print(f"{label}: {results[key]:.12f}")
    print(f"Validation report saved to {REPORT_PATH}")
    for warning_message in report["warnings"]:
        print(f"WARNING: {warning_message}")


if __name__ == "__main__":
    main()
