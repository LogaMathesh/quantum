# Model artifact layout

The prediction module loads the four canonical model families from this directory once at import. Legacy files are retained; no existing model artifacts were deleted.

```text
models/
├── linear_regression/
│   ├── linear_regression.joblib
│   └── artifact_manifest.json
├── linear_regression_quantum/
│   ├── linear_regression.joblib
│   ├── qkrr_model.npz
│   ├── vqr_parameters.npz
│   ├── vqr_circuits.json
│   ├── training_validation.json
│   └── artifact_manifest.json
├── random_forest/
│   ├── random_forest.joblib
│   └── artifact_manifest.json
└── random_forest_quantum/
    ├── qkrr_model.npz
    ├── vqr_parameters.npz
    ├── vqr_circuits.json
    └── artifact_manifest.json
```

## Inference requirements by model

- **Linear Regression:** `linear_regression/linear_regression.joblib` (estimator, fitted scaler, ordered feature names, target metadata).
- **Linear Regression + Quantum:** `linear_regression_quantum/linear_regression.joblib`, `qkrr_model.npz`, `vqr_parameters.npz`, and `vqr_circuits.json`. Its `training_validation.json` documents provenance and is not loaded for prediction. The old `linear_regression_model/qkrr_model/alpha.npy` is incorrect and must not be used.
- **Random Forest:** `random_forest/random_forest.joblib` (estimator, fitted scaler, ordered feature names, target metadata).
- **Random Forest + Quantum:** `random_forest_quantum/qkrr_model.npz`, `vqr_parameters.npz`, and `vqr_circuits.json`, plus the shared classical estimator/scaler at `random_forest/random_forest.joblib`.

Quantum inference computes one new-input-versus-saved-training-input kernel row. It does not fit or retrain QKRR/VQR. The RF pickle was serialized with scikit-learn 1.7.0 and currently loads under 1.9.1 with `InconsistentVersionWarning`; the warning is recorded in `model_validation.json`.

The older root-level model files and the quantum files formerly colocated in `random_forest/` are retained but are not used by the canonical prediction module.
