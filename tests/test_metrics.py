"""
File Description: Unit test for CZmeltbench evaluation metrics.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-24
Version: 1.0
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from czmeltbench.evaluation.metrics import (
    calculate_error_metrics,
    evaluate_fields,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@pytest.fixture
def sample_predictions(
) -> tuple[FloatArray, FloatArray]:
    """Return synthetic reference and predicted CFD fields."""
    y_true: FloatArray = np.array([
        [1.0, 0.0,  0.0, 2.0, 1700.0],
        [3.0, 2.0, -2.0, 4.0, 1800.0],
        [5.0, 4.0,  4.0, 6.0, 1750.0],
    ], dtype=np.float64)

    y_pred: FloatArray = y_true.copy()

    y_pred[:, 0] += np.array(
        [1.0, 0.0, -1.0]
    )

    y_pred[:, 4] += np.array(
        [10.0, -10.0, 0.0]
    )

    return y_true, y_pred


def test_exact_predictions(
    sample_predictions: tuple[FloatArray, FloatArray],
) -> None:
    """Perfect predictions must have zero error."""
    y_true, _ = sample_predictions

    results = evaluate_fields(
        y_true,
        y_true.copy(),
    )

    np.testing.assert_allclose(
        results["MAE"].to_numpy(),
        0.0,
    )

    np.testing.assert_allclose(
        results["RMSE"].to_numpy(),
        0.0,
    )

    np.testing.assert_allclose(
        results.loc["T", "relative_L2"],
        0.0,
    )


def test_known_scalar_errors() -> None:
    """Verify scalar MAE, RMSE and relative L2."""
    reference: FloatArray = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    prediction: FloatArray = np.array(
        [2.0, 2.0, 1.0],
        dtype=np.float64,
    )

    metrics = calculate_error_metrics(
        reference,
        prediction,
    )

    assert metrics["MAE"] == pytest.approx(1.0)

    assert metrics["RMSE"] == pytest.approx(
        np.sqrt(5.0 / 3.0)
    )

    assert metrics["relative_L2"] == pytest.approx(
        np.sqrt(5.0 / 14.0)
    )


def test_weighted_scalar_errors() -> None:
    """Verify weighted metrics against analytical values."""
    reference: FloatArray = np.array(
        [1.0, 3.0, 5.0],
        dtype=np.float64,
    )

    prediction: FloatArray = np.array(
        [2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    weights: FloatArray = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    metrics = calculate_error_metrics(
        reference,
        prediction,
        weights=weights,
    )

    assert metrics["Volume_MAE"] == pytest.approx(
        4.0 / 6.0
    )

    assert metrics["Volume_RMSE"] == pytest.approx(
        np.sqrt(4.0 / 6.0)
    )

    assert metrics["Volume_Relative_L2"] == pytest.approx(
        np.sqrt(4.0 / 94.0)
    )


def test_field_wise_evaluation(
    sample_predictions: tuple[FloatArray, FloatArray],
) -> None:
    """All five fields must be evaluated separately."""
    y_true, y_pred = sample_predictions

    results = evaluate_fields(
        y_true,
        y_pred,
    )

    assert results.shape == (5, 3)

    assert list(results.index) == [
        "u_r", "u_z", "u_swirl", "p", "T"
    ]

    assert results.loc["u_r", "MAE"] == pytest.approx(
        2.0 / 3.0
    )

    assert results.loc["T", "MAE"] == pytest.approx(
        20.0 / 3.0
    )

    assert results.loc["u_z", "MAE"] == 0.0
    assert results.loc["p", "RMSE"] == 0.0


def test_interior_node_mask(
    sample_predictions: tuple[FloatArray, FloatArray],
) -> None:
    """Node masks must restrict evaluation to selected nodes."""
    y_true, y_pred = sample_predictions

    mask: BoolArray = np.array(
        [False, True, True],
        dtype=np.bool_,
    )

    results = evaluate_fields(
        y_true,
        y_pred,
        node_mask=mask,
    )

    assert results.loc["u_r", "MAE"] == pytest.approx(
        0.5
    )

    assert results.loc["T", "MAE"] == pytest.approx(
        5.0
    )


def test_zero_reference_norm() -> None:
    """Undefined relative error must be reported as NaN."""
    reference: FloatArray = np.zeros(
        3, dtype=np.float64
    )

    prediction: FloatArray = np.ones(
        3, dtype=np.float64
    )

    metrics = calculate_error_metrics(
        reference,
        prediction,
    )

    assert metrics["MAE"] == pytest.approx(1.0)
    assert np.isnan(metrics["relative_L2"])


def test_invalid_inputs(
    sample_predictions: tuple[FloatArray, FloatArray],
) -> None:
    """Reject incompatible field arrays and invalid weights."""
    y_true, y_pred = sample_predictions

    with pytest.raises(ValueError):
        evaluate_fields(
            y_true,
            y_pred[:, :4],
        )

    with pytest.raises(ValueError):
        evaluate_fields(
            y_true,
            y_pred,
            node_mask=np.zeros(3, dtype=np.bool_),
        )

    with pytest.raises(ValueError):
        evaluate_fields(
            y_true,
            y_pred,
            volume_weights=np.array([1.0, -1.0, 1.0]),
        )

    with pytest.raises(ValueError):
        calculate_error_metrics(
            np.array([1.0, np.nan]),
            np.array([1.0, 2.0]),
        )


def test_inputs_are_not_modified(
    sample_predictions: tuple[FloatArray, FloatArray],
) -> None:
    """Evaluation must not modify reference or predictions."""
    y_true, y_pred = sample_predictions

    original_true = y_true.copy()
    original_pred = y_pred.copy()

    evaluate_fields(
        y_true,
        y_pred,
        volume_weights=np.ones(3),
    )

    np.testing.assert_array_equal(y_true, original_true)
    np.testing.assert_array_equal(y_pred, original_pred)