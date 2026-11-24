"""
File Description: Error metrics for evaluating CZMeltBench model predictions.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-24
Version: 1.0
"""
"""
Metrics compare predictions with original CFD nodal fields.

Field order:
    [u_r, u_z, u_swirl, p, T]

No spatial reshaping, innterpolation, normalization, pressure-guage
adjustment or analytical boundary-value replacemetn is performed.
"""

from typing import TypeAlias

import numpy as np
import pandas as pd 
from numpy.typing import NDArray

from czmeltbench.constants import FIELD_COLUMNS

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def calculate_error_metrics(
    y_true: FloatArray,
    y_pred: FloatArray,
    *,
    weights: FloatArray | None = None,
) -> dict [str, float]:
    """calculate error metrics for one scalar CFD field.

    Parameters
    ----------
    y_true : FloatArray
        Referencce CFD value, shape (N,).

    y_pred : FloatArray
        Model predictions, shape (N,).

    weights : FloatArray | None, optional
        Non-negative integration weights, shape (N,).
        Typically node_volume_weights from MeshQuadrature.

    Returns
    -------
    dict[str, float]
        Always contains
            MAE
            RMSE
            Relative_L2
        
        If weights are provided, also contains:
            volume_mse
            volume_rmse
            volume_relative_l2

    Notes
    ------
    Relative L2 errors are dimentionless fractions, not percentages.

    When the reference L2 norm is exactly zero, the relative L2 erro is 
    undefined and is returned as np.nan.

    The weighted metrics approximate physical volume integration when 
    valid axisymmetric quadrature weights are supplied.
    """

    reference: FloatArray = np.asarray(
        y_true, dtype=np.float64,
    )

    prediction: FloatArray = np.asarray(
        y_pred, dtype=np.float64,
    )

    if reference.ndim != 1 or reference.size == 0:
        raise ValueError(
            "y_true must have non-empty shape (N,)."
        )

    if prediction.shape != reference.shape:
        raise ValueError("y_pred must have the same shape as y_true.")

    if not np.isfinite(reference).all():
        raise ValueError("y_true contains nnon-finite values.")

    if not np.isfinite(prediction).all():
        raise ValueError("y_pred contains non-finite values.")


    error: FloatArray = prediction - reference
    abs_error: FloatArray = np.abs(error)

    # RMS-based implementation of relative L2.
    # this si equivalent to ratio of Euclidean norms.
    rms_error: float = float(
        np.sqrt(np.mean(error**2))
    )

    rms_reference: float = float(
        np.sqrt(np.mean(reference**2))
    )

    relative_l2: float = (
        rms_error / rms_reference
        if rms_reference > 0.0 else float("nan")
    )

    results: dict[str, float] = {
        "mae": float(np.mean(abs_error)),
        "rmse": rms_error,
        "relative_l2": relative_l2,
    }

    if weights is not None:
        w: FloatArray = np.asarray(
            weights, dtype=np.float64,
        )

        if w.shape != reference.shape:
            raise ValueError(
                "weights must have shape (N,)."
            )

        if not np.isfinite(w).all():
            raise ValueError(
                "weights contain non-finite values."
            )

        if np.any(w < 0.0):
            raise ValueError(
                "Weights cannot contain negative values."
            )

        weight_sum: float = float(np.sum(w))

        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError(
                "Total weight must be positive and finite."
            )

        weighted_mae: float = float(
            np.average(abs_error, weights=w)
        )

        weighted_mse: float = float(
            np.average(error**2, weights=w)
        )

        weighted_reference_mse: float = float(
            np.average(reference**2, weights=w)
        )

        weighted_rmse: float = float(
            np.sqrt(weighted_mse)
        )

        weighted_reference_rms: float = float(
            np.sqrt(weighted_reference_mse)
        )

        weighted_relative_l2: float = (
            weighted_rmse / weighted_reference_rms
            if weighted_reference_rms > 0.0 else float("nan")
        )

        results.update({
            "volume_mae": weighted_mae,
            "volume_rmse": weighted_rmse,
            "volume_relative_l2": weighted_relative_l2,
        })

    return results


def evaluate_fields(
    y_true: FloatArray,
    y_pred: FloatArray,
    *,
    volume_weights: FloatArray | None = None,
    node_mask: BoolArray | None = None,
) -> pd.DataFrame:
    """Evaluate predictions for all five CFD fields.

    Paramters
    ---------
    y_true : FloatArray
        Reference CFD fields, shape (N, 5).

    y_pres : FloatArray
        Modal-predicted fields, shape (N, 5).

    volume_weights : FloatArray | None, optional
        Axisymmetric nodal volume weights, shape (N,).

    node_mask : BoolArray | None, optional
        Boolean selection mask, shape (N,).

        True selects a node for evaluation.
        If None, all nodes are evalulated.

    Returns
    --------
    pd.DataFrame
        Index containing the five name in FIELD_COLUMNS.

        Columns:
            [MAE, RMSE, Relative_L2]
        Additional columns whenn volume_weights is provided:
            [Volume_MAE, Volume_RMSE, Volume_Relative_L2]

    Notes
    -----
    Both input arrays must use identical original CFD node ordering.
    this funnction does not align coordinates.

    Volume-weighted metrics refer to the selected region when node_mask
    is provided.
    """
    reference: FloatArray = np.asarray(
        y_true, dtype=np.float64,
    )

    prediction: FloatArray = np.asarray(
        y_pred, dtype=np.float64,
    )

    if (
        reference.ndim != 2 or
        reference.shape[1] != len(FIELD_COLUMNS) or
        reference.shape[0] == 0
    ):
        raise ValueError(
            "y_true must have non-empty shape (N, 5)."
        )

    if prediction.shape != reference.shape:
        raise ValueError(
            "y_pred must have the same shape as y_true."
        )

    if not np.isfinite(reference).all():
        raise ValueError(
            "y_true contains non-finite values."
        )

    if not np.isfinite(prediction).all():
        raise ValueError(
            "y_pred contains non-finite values."
        )

    n_nodes: int = reference.shape[0]

    if node_mask is None:
        selected: BoolArray = np.ones(
            n_nodes, dtype=np.bool_
        )
    else:
        mask_array: NDArray[np.generic] = np.asarray(node_mask)

        if (mask_array.shape != (n_nodes,) or mask_array.dtype != np.bool_):
            raise ValueError(
                "node_mask must be Boolean with sahpe (N,)."
            )
        selected = np.asarray(mask_array, dtype=np.bool_)

    if not np.any(selected):
        raise ValueError(
            "node_mask must select at lest one node."
        )

    selected_weights: FloatArray | None = None

    if volume_weights is not None:
        weights: FloatArray = np.asarray(
            volume_weights, dtype=np.float64
        )

        if weights.shape != (n_nodes,):
            raise ValueError(
                "volume_weights must have shape (N,)."
            )

        if not np.isfinite(weights).all():
            raise ValueError(
                "volume_weights contain non-finite values."
            )

        if np.any(weights < 0.0):
            raise ValueError(
                "volume_weights cannot be negative."
            )

        selected_weights = weights[selected]


    records: dict[str, dict[str, float]] = {}

    for index, field in enumerate(FIELD_COLUMNS):
        records[field] = calculate_error_metrics(
            y_true=reference[selected, index],
            y_pred=prediction[selected, index],
            weights=selected_weights,
        )

    
    return pd.DataFrame.from_dict(
        records,
        orient="index",
    ).rename_axis("field")