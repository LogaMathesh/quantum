from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import joblib
import json

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.neural_networks import EstimatorQNN

from qiskit_algorithms.state_fidelities import ComputeUncompute


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Crop Yield Prediction API",
    description="Classical Random Forest vs Hybrid Quantum ML",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 1. LOAD RANDOM FOREST + SCALER
# ============================================================

print("\nLoading Random Forest...")

model_data = joblib.load(
    "models/random_forest.joblib"
)

rf_model = model_data["model"]
scaler = model_data["scaler"]

feature_cols = model_data["feature_cols"]
target_col = model_data["target_col"]

print("Random Forest loaded.")
print("Features:", feature_cols)
print("Target:", target_col)


# ============================================================
# 2. LOAD QKRR MODEL
# ============================================================

print("\nLoading QKRR model...")

qkrr_data = np.load(
    "models/qkrr_model_real.npz"
)

alpha_q = qkrr_data["alpha"]

X_train_q = qkrr_data["X_train_q"]

lambda_reg = float(
    np.asarray(qkrr_data["lambda_reg"]).reshape(-1)[0]
)

print("QKRR loaded.")
print("X_train_q shape:", X_train_q.shape)
print("alpha shape:", alpha_q.shape)
print("lambda:", lambda_reg)


# ============================================================
# 3. CREATE QUANTUM FEATURE MAP
# ============================================================

print("\nCreating quantum feature map...")

n_qubits = 5

feature_map = ZZFeatureMap(
    feature_dimension=n_qubits,
    reps=2,
    entanglement="full"
)

print("Feature map created.")
print("Qubits:", feature_map.num_qubits)


# ============================================================
# 4. CREATE QUANTUM KERNEL
# ============================================================

print("\nCreating quantum kernel...")

sampler = StatevectorSampler()

fidelity = ComputeUncompute(
    sampler=sampler
)

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map,
    fidelity=fidelity
)

print("Quantum kernel loaded.")


# ============================================================
# 5. LOAD VQR PARAMETERS
# ============================================================

print("\nLoading VQR parameters...")

vqr_data = np.load(
    "models/vqr_parameters_real.npz"
)

vqr_parameters = np.asarray(
    vqr_data["optimal_parameters"],
    dtype=float
).reshape(-1)

print(
    "VQR parameters shape:",
    vqr_parameters.shape
)


# ============================================================
# 6. LOAD VQR CIRCUIT CONFIGURATION
# ============================================================

print("\nLoading VQR circuit configuration...")

with open(
    "models/vqr_circuits.json",
    "r",
    encoding="utf-8"
) as f:

    vqr_config = json.load(f)

print(
    "VQR configuration:",
    vqr_config
)


# ============================================================
# 7. RECREATE REAL AMPLITUDES ANSATZ
# ============================================================

print("\nCreating VQR ansatz...")

ansatz = RealAmplitudes(
    num_qubits=vqr_config["n_qubits"],
    reps=vqr_config["ansatz_reps"],
    entanglement=vqr_config["entanglement"]
)

print("Ansatz created.")

print(
    "Number of ansatz parameters:",
    ansatz.num_parameters
)


# ============================================================
# 8. CREATE ESTIMATOR
# ============================================================

print("\nCreating StatevectorEstimator...")

estimator = StatevectorEstimator()

print("Estimator created.")


# ============================================================
# 9. CREATE ESTIMATOR QNN
# ============================================================

print("\nCreating EstimatorQNN...")

# Combine:
#
# ZZFeatureMap
#       +
# RealAmplitudes
#
# into one quantum circuit.

vqr_circuit = feature_map.compose(
    ansatz
)


vqr_qnn = EstimatorQNN(
    circuit=vqr_circuit,
    estimator=estimator,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

print("EstimatorQNN created.")

print(
    "Expected VQR parameters:",
    ansatz.num_parameters
)

print(
    "Saved VQR parameters:",
    len(vqr_parameters)
)


# ============================================================
# CHECK VQR PARAMETER COUNT
# ============================================================

if len(vqr_parameters) != ansatz.num_parameters:

    raise RuntimeError(
        f"VQR parameter mismatch! "
        f"Saved={len(vqr_parameters)}, "
        f"Expected={ansatz.num_parameters}"
    )

print("VQR parameter count matches.")


# ============================================================
# 10. INPUT MODEL
# ============================================================

class CropInput(BaseModel):

    fertilizer: float

    temp: float

    N: float

    P: float

    K: float


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Crop Yield Prediction API is running",
        "models": {
            "random_forest": True,
            "qkrr": True,
            "vqr": True
        }
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "random_forest": True,
        "qkrr": True,
        "vqr": True
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: CropInput):

    # ========================================================
    # 1. CREATE INPUT
    # ========================================================

    X = np.array(
        [[
            data.fertilizer,
            data.temp,
            data.N,
            data.P,
            data.K
        ]],
        dtype=float
    )


    # ========================================================
    # 2. SCALE INPUT
    # ========================================================

    # IMPORTANT:
    # The original notebook trained RF and quantum models
    # using StandardScaler.

    X_scaled = scaler.transform(X)


    # ========================================================
    # 3. RANDOM FOREST PREDICTION
    # ========================================================

    rf_prediction = rf_model.predict(
        X_scaled
    )[0]

    rf_prediction = float(
        rf_prediction
    )


    # ========================================================
    # 4. QUANTUM KERNEL
    # ========================================================

    # Calculate the quantum kernel between:
    #
    # new website input
    #
    # and
    #
    # the 300 quantum training samples.
    #
    # Result:
    #
    # (1, 300)

    K_test = quantum_kernel.evaluate(
        x_vec=X_scaled,
        y_vec=X_train_q
    )


    # ========================================================
    # 5. QKRR RESIDUAL CORRECTION
    # ========================================================

    qkrr_correction = (
        K_test @ alpha_q
    )


    qkrr_correction = float(
        np.asarray(
            qkrr_correction
        ).reshape(-1)[0]
    )


    # ========================================================
    # 6. RANDOM FOREST + QKRR
    # ========================================================

    hybrid_prediction = (
        rf_prediction
        +
        qkrr_correction
    )

    hybrid_prediction = float(
        hybrid_prediction
    )


    # ========================================================
    # 7. VQR CORRECTION
    # ========================================================

    # The original notebook trained VQR on:
    #
    # remaining residual =
    # actual yield - RF - QKRR
    #
    # Here we use the saved optimal VQR parameters
    # to predict the correction for the new input.

    vqr_output = vqr_qnn.forward(
        X_scaled,
        vqr_parameters
    )


    vqr_correction = float(
        np.asarray(
            vqr_output
        ).reshape(-1)[0]
    )


    # ========================================================
    # 8. FINAL HYBRID QUANTUM PREDICTION
    # ========================================================

    quantum_prediction = (
        hybrid_prediction
        +
        vqr_correction
    )

    quantum_prediction = float(
        quantum_prediction
    )


    # ========================================================
    # 9. DIFFERENCE
    # ========================================================

    difference = (
        quantum_prediction
        -
        rf_prediction
    )


    # ========================================================
    # 10. PERCENTAGE DIFFERENCE
    # ========================================================

    if abs(rf_prediction) > 1e-12:

        difference_percent = (
            difference
            /
            abs(rf_prediction)
        ) * 100

    else:

        difference_percent = 0.0


    # ========================================================
    # 11. PRINT RESULT IN TERMINAL
    # ========================================================

    print("\n" + "=" * 60)

    print("NEW PREDICTION")

    print("=" * 60)

    print(
        f"Input: {X.tolist()}"
    )

    print(
        f"Random Forest       : {rf_prediction:.6f}"
    )

    print(
        f"QKRR correction     : {qkrr_correction:.6f}"
    )

    print(
        f"RF + QKRR           : {hybrid_prediction:.6f}"
    )

    print(
        f"VQR correction      : {vqr_correction:.6f}"
    )

    print(
        f"Final Quantum       : {quantum_prediction:.6f}"
    )

    print(
        f"Difference          : {difference:.6f}"
    )

    print(
        f"Difference %        : {difference_percent:.2f}%"
    )

    print("=" * 60)


    # ========================================================
    # 12. RETURN TO FRONTEND
    # ========================================================

    return {

        "random_forest":
            rf_prediction,

        "qkrr_correction":
            qkrr_correction,

        "rf_qkrr":
            hybrid_prediction,

        "vqr_correction":
            vqr_correction,

        "quantum":
            quantum_prediction,

        "difference":
            float(difference),

        "difference_percent":
            float(difference_percent)
    }