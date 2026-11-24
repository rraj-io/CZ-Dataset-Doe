"""
File Description: Unit tests for single case CFD evaluation.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-24
Version: 1.0
"""


from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from czmeltbench.evaluation.evaluator import (
    CaseEvaluation,
    CaseEvaluator,
)


FloatArray: TypeAlias = NDArray[np.float64]


class SyntheticReader:
    """Minimal reader with the same interface as CZDataReader."""

    def __init__(self) -> None:
        """Construct a synthetic 3x3 axisymmetric mesh."""
        r: FloatArray = np.array(
            [0.0, 0.15, 0.30],
            dtype=np.float64,
        )

        z: FloatArray = np.array(
            [0.0, 0.075, 0.15],
            dtype=np.float64,
        )

        R, Z = np.meshgrid(r, z)

        self.pos: FloatArray = np.column_stack((
            R.ravel(),
            Z.ravel(),
        ))

        self.mu: FloatArray = np.array(
            [12.0, -5.5, 1750.0],
            dtype=np.float64,
        )

        self.y: FloatArray = np.zeros(
            (9, 5), dtype=np.float64
        )

        self.y[:, 4] = 1700.0 + 100.0 * self.pos[:, 0]

    def load_arrays(
        self,
        case_id: str,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return original coordinates, parameters and CFD fields."""
        if case_id != "synthetic_000":
            raise KeyError(f"Unknown case: {case_id}")

        return (
            self.pos.copy(),
            self.mu.copy(),
            self.y.copy(),
        )


@pytest.fixture
def synthetic_reader() -> SyntheticReader:
    """Provide the synthetic CFD reader."""
    return SyntheticReader()


def test_perfect_prediction(
    synthetic_reader: SyntheticReader,
) -> None:
    """Perfect predictions must produce zero error."""
    evaluator = CaseEvaluator(synthetic_reader)

    result = evaluator.evaluate(
        case_id="synthetic_000",
        y_pred=synthetic_reader.y.copy(),
        prediction_pos=synthetic_reader.pos.copy(),
    )

    assert isinstance(result, CaseEvaluation)
    assert result.num_nodes == 9
    assert result.num_evaluated_nodes == 9
    assert result.region == "all"

    np.testing.assert_allclose(
        result.metrics["mae"].to_numpy(),
        0.0,
    )

    np.testing.assert_allclose(
        result.metrics["volume_rmse"].to_numpy(),
        0.0,
    )

    assert result.total_volume == pytest.approx(
        np.pi * 0.30**2 * 0.15
    )


def test_known_prediction_error(
    synthetic_reader: SyntheticReader,
) -> None:
    """Constant prediction errors must be recovered."""
    evaluator = CaseEvaluator(synthetic_reader)

    y_pred: FloatArray = synthetic_reader.y.copy()

    y_pred[:, 0] += 0.01
    y_pred[:, 4] += 5.0

    result = evaluator.evaluate(
        case_id="synthetic_000",
        y_pred=y_pred,
        prediction_pos=synthetic_reader.pos,
    )

    assert result.metrics.loc["u_r", "mae"] == pytest.approx(
        0.01
    )

    assert result.metrics.loc["u_r", "volume_rmse"] == (
        pytest.approx(0.01)
    )

    assert result.metrics.loc["T", "mae"] == pytest.approx(
        5.0
    )

    assert result.metrics.loc["T", "volume_mae"] == (
        pytest.approx(5.0)
    )


def test_interior_only(
    synthetic_reader: SyntheticReader,
) -> None:
    """Only the central node should be interior."""
    evaluator = CaseEvaluator(synthetic_reader)

    y_pred: FloatArray = synthetic_reader.y.copy()

    # One deliberately wrong boundary value.
    y_pred[0, 4] += 100.0

    result = evaluator.evaluate(
        case_id="synthetic_000",
        y_pred=y_pred,
        prediction_pos=synthetic_reader.pos,
        region="interior",
    )

    assert result.num_nodes == 9
    assert result.num_evaluated_nodes == 1

    # Boundary-only error must not affect interior metrics.
    assert result.metrics.loc["T", "mae"] == pytest.approx(
        0.0
    )


def test_shuffled_prediction_nodes_rejected(
    synthetic_reader: SyntheticReader,
) -> None:
    """Predictions in a different node order must be rejected."""
    evaluator = CaseEvaluator(synthetic_reader)

    order = np.array(
        [4, 0, 2, 3, 1, 5, 6, 7, 8],
        dtype=np.int64,
    )

    shuffled_pos: FloatArray = synthetic_reader.pos[order]
    shuffled_y: FloatArray = synthetic_reader.y[order]

    with pytest.raises(
        ValueError,
        match="node ordering",
    ):
        evaluator.evaluate(
            case_id="synthetic_000",
            y_pred=shuffled_y,
            prediction_pos=shuffled_pos,
        )


def test_incorrect_prediction_shape(
    synthetic_reader: SyntheticReader,
) -> None:
    """Predictions must contain all five CFD fields."""
    evaluator = CaseEvaluator(synthetic_reader)

    with pytest.raises(ValueError):
        evaluator.evaluate(
            case_id="synthetic_000",
            y_pred=np.zeros((9, 4)),
            prediction_pos=synthetic_reader.pos,
        )


def test_invalid_region(
    synthetic_reader: SyntheticReader,
) -> None:
    """Unsupported evaluation regions must be rejected."""
    evaluator = CaseEvaluator(synthetic_reader)

    with pytest.raises(ValueError, match="region"):
        evaluator.evaluate(
            case_id="synthetic_000",
            y_pred=synthetic_reader.y,
            prediction_pos=synthetic_reader.pos,
            region="boundary",  # type: ignore[arg-type]
        )


def test_inputs_not_modified(
    synthetic_reader: SyntheticReader,
) -> None:
    """Evaluation must preserve model outputs and coordinates."""
    evaluator = CaseEvaluator(synthetic_reader)

    pred: FloatArray = synthetic_reader.y.copy()
    pred_pos: FloatArray = synthetic_reader.pos.copy()

    original_pred: FloatArray = pred.copy()
    original_pos: FloatArray = pred_pos.copy()

    evaluator.evaluate(
        case_id="synthetic_000",
        y_pred=pred,
        prediction_pos=pred_pos,
    )

    np.testing.assert_array_equal(pred, original_pred)
    np.testing.assert_array_equal(pred_pos, original_pos)
