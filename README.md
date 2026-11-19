# CZMeltBench

Benchmark for machine-learning surrogates of the silicon melt flow in a Czochralski
crystal puller, built on the CZ-Dataset-Doe v2 CFD simulations (Kurisinkal Shine).

**Status:** under construction.

## Source data
Steady, laminar, 2D axisymmetric ANSYS Fluent simulations over crystal rotation,
crucible rotation and hot-wall temperature. Source: github.com/bertwinshine/CZ-Dataset-Doe
(CC BY 4.0).

CZMeltBench/
│
├── data/
│   ├── raw/
│   │   ├── doe/
│   │   ├── holdout/
│   │   └── meshes/
│   │
│   ├── processed/
│   │   ├── canonical/
│   │   ├── graph/
│   │   ├── pointcloud/
│   │   └── structured/
│   │
│   └── metadata/
│       ├── cases.parquet
│       ├── mesh_metadata.json
│       └── feature_schema.json
│
├── czmeltbench/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   └── transforms.py
│   │
│   ├── representations/
│   │   ├── graph.py
│   │   ├── pointcloud.py
│   │   └── structured.py
│   │
│   ├── models/
│   │   ├── baselines/
│   │   ├── gnn/
│   │   └── neural_operators/
│   │
│   └── evaluation/
│       ├── metrics.py
│       ├── splits.py
│       └── physical_metrics.py
│
├── tasks/
│   ├── field_prediction/
│   ├── parameter_generalization/
│   └── mesh_generalization/
│
├── examples/
├── tests/
├── configs/
├── notebooks/
└── README.md
