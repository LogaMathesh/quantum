# Crop-yield model comparison

Dataset: `Crop Yiled with Soil and Weather.csv` (2,596 rows). Features: `Fertilizer`, `temp`, `N`, `P`, `K`. Target: `yeild`.

## Matched-subset results

All four rows below use the same 150 samples selected from the held-out 20% test split (`random_state=42`). `StandardScaler` was fitted on the original training split only. MAPE is reported as a percentage.

| Model | R² | MAE | MSE | RMSE | MAPE (%) |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 0.826835 | 0.625815 | 0.579573 | 0.761297 | 7.311760 |
| Linear Regression + Quantum (LR + QKRR + VQR) | -17.110232 | 6.081554 | 60.613948 | 7.785496 | 74.142012 |
| Random Forest | 0.991049 | 0.113924 | 0.029959 | 0.173087 | 1.433949 |
| Random Forest + Quantum (RF + QKRR + VQR) | -1.652454 | 2.387545 | 8.877619 | 2.979533 | 28.517846 |

On this matched evaluation subset, the classical Random Forest has the best R² and the lowest error metrics. The recorded quantum-integrated final predictions are worse than their respective classical baselines; these results do not demonstrate a quantum advantage.

## Full held-out test metrics for classical baselines

The notebooks also report the classical models on the full 520-row test split. These are included for completeness and are not directly comparable to quantum metrics calculated on the 150-row subset.

| Model | R² | MAE | MSE | RMSE | MAPE (%) |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 0.862648 | 0.579096 | 0.516205 | 0.718474 | 6.888511 |
| Random Forest | 0.990757 | 0.122881 | 0.034737 | 0.186379 | 1.530272 |

## Sources and caveats

- LR metrics and LR + QKRR + VQR final metrics: [`linear_regression_quantum.ipynb`](../../linear_regression_quantum.ipynb). The quantum figures are from the executed notebook output for `y_pred_final_q`. The original fitted LR VQR parameters were not saved. A new fit using the same recipe produced different metrics; the original values remain unchanged and the discrepancy is recorded in [`training_validation.json`](../models/linear_regression_quantum/training_validation.json).
- RF metrics and RF + QKRR + VQR final metrics: [`backend/models/with_quantum.ipynb`](../models/with_quantum.ipynb). Its full-test RF baseline was independently recalculated from the saved backend RF estimator and current CSV, and matched the notebook output.
- The separate root [`with_quantum.ipynb`](../../with_quantum.ipynb) contains another RF run, including a QKRR-only output and different rounded baseline values. It was not used for this comparison because the backend notebook contains the final QKRR + VQR prediction used by the backend implementation.
- The previous LR QKRR `alpha` was exactly equal to the RF `alpha` and did not solve the LR residual system. Correct LR QKRR and VQR artifacts have now been generated under `backend/models/linear_regression_quantum/`; RF artifacts were left untouched. Since the original LR VQR weights are unavailable, the preserved LR quantum metrics describe the notebook run and are not proven to describe the newly regenerated VQR weights.
