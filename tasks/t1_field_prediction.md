# T1 — Field prediction from operating parameters

## Goal
Predict the full steady flow and temperature field in the melt from the three
operating parameters, replacing a ~3,500-iteration Fluent run.

## Input
| Name | Unit | Training range |
|---|---|---|
| crystal_rpm | rpm | 4 to 20 |
| crucible_rpm | rpm | −10 to −1 (negative = counter-rotation) |
| t_hot | K | 1730 to 1785 |

## Output
All five fields on every node of the base mesh (81 × 101 = 8,181 nodes):
u_r, u_z, u_swirl (m/s), p (Pa, gauge, hydrostatic part removed), T (K).

## Data
| Split | Cases | Use |
|---|---|---|
| train | 254 | training; model selection only via the fixed folds in `meta/splits.json` |
| test_interp | 26 | inside the box: the three one-parameter sweeps through (8 rpm, −3 rpm, 1745 K) |
| test_extrap | 12 | outside the box: up to 30 rpm, −12 rpm, 1810 K |

The test splits are used **once**, for final reporting. No tuning on them.

## Metrics (computed by `czmeltbench.metrics.evaluate`)
1. **Relative L2 error per field**, volume-weighted (2πr dA):
   ‖pred − true‖ / ‖true‖ per case, then averaged over cases.
   For T, use T − 1685 K, so the large offset doesn't hide errors.
2. **Max error per field**, normalized by that field's range in the case.
3. **Quantities of interest** (relative error):
   max |u_z|, max |u_r|, u_z on the axis at mid-depth, pressure difference across
   the melt, swirl at r = 0.24 m mid-depth, T at r = 0.15 m mid-depth.

The primary score is the mean relative L2 over the five fields, reported separately
for test_interp and test_extrap.

## Rules
- Normalization statistics come from train only (`meta/normalization.json`).
- Report mean ± std over 5 seeds.
- Physics knowledge (equations, BCs) may be used. State it if you do.
- Report training time, inference time per case and parameter count.

## Known difficulties
- **Unsteady region:** crystal ≈ 5–8 rpm, crucible −1 to −5 rpm, t_hot > ~1753 K has
  no steady solution and no training data. Predictions there are not backed by data.
- **Extrapolation:** at 30 rpm the laminar model is outside the realistic regime.
  This tests the model, not the furnace.
- **Scale:** velocities ~1e-2 m/s vs. T ~1.7e3 K, so fields must be normalized.

## Result format
One row per (model, seed, split) in `leaderboard/results.csv`:
model, task, split, seed, relL2_u_r, relL2_u_z, relL2_u_swirl, relL2_p, relL2_T,
relL2_mean, qoi_err_mean, train_time_s, infer_ms_per_case, n_params, commit