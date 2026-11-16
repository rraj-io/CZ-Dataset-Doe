# CZ melt v2 — ML-ready dataset schema (schema_version 1.0)

## Files
| File | Cases | Mesh | Purpose |
| train_doe_base.h5 | 254 | base 81×101 | training / CV |
| test_holdout_interp.h5 | 26 | base | interpolation test |
| test_holdout_extrap.h5 | 12 | base | extrapolation test |
| mf_coarse.h5 / mf_fine.h5 | 30 | 41×51 / 161×200 | multi-fidelity |

## Grid
- Logically structured quad mesh, rows = constant z (bottom→top), columns = axis→wall
- R, Z stored per node (columns lean; the crystal-edge column is not vertical)
- Cell (i,j) = nodes [(i,j),(i,j+1),(i+1,j+1),(i+1,j)], counter-clockwise
- Weights: 2π r_c A per cell, split equally to its 4 nodes; sum = π R² H

## Datasets inside each file
| Path | Shape | dtype | Units | Notes |
| /fields | (N,nz,nr,5) | f8 | m/s, Pa, K | order: u_r,u_z,u_swirl,p,T |
| /params | (N,3) | f8 | rpm, rpm, K | crystal_rpm, crucible_rpm, t_hot |
| ... (case_id, group, grid/*, masks/*, qoi/*)

## Conventions
- p: gauge, hydrostatic part removed, 0 at r=0, z=0
- crucible_rpm negative = opposite to crystal
- Only "kept" cases; excluded ones listed in meta/excluded_cases.csv
- Boundary mask definitions identical to cz_doe/certify.py

## Attributes
mesh, schema_version, dataset_version, convergence=1e-6, build_commit, source_doi
