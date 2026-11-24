"""
File Description: Unit test for multi-case benchmark evalautor.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-24
Version: 1.0
"""

"""

Tests:
    - Perfect predictions
    - Correct case-specific prediction lookup
    - Equal-case-weighted aggregation
    - Interior-only evaluation
    - Coordinate ordering validation
    - Missing and extra cases
    - Invalid case identifiers
    - Empty predictions
    - Input preservation
    - Undefined relative errors

The tests use a synthetic CFD reader and do not require
the original simulation archives.
"""

from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from czmeltbench.evaluation.benchmark import (
    BenchmarkEvaluator,
    BenchmarkResult,
    CasePrediction,
)
from czmeltbench.evaluation.evaluator import CaseEvaluator


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class SyntheticReader:
    """Synthetic reader implementing CZDataReader's load_arrays API."""

    def __init__(self) -> None:
        """Create a structured 3x3 axisymmetric mesh.

        The physical domain has:
            R = 0.30 m
            H = 0.15 m

        Nodes follow the original CFD field ordering.
        """
        radial: FloatArray = np.array(
            [0.0, 0.15, 0.30],
            dtype=np.float64,
        )

        axial: FloatArray = np.array(
            [0.0, 0.075, 0.15],
            dtype=np.float64,
        )

        r_grid, z_grid = np.meshgrid(radial, axial)

        self.pos: FloatArray = np.column_stack((
            r_grid.ravel(),
            z_grid.ravel(),
        ))

    def load_arrays(
        self,
        case_id: str,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Load one synthetic simulation.

        Parameters
        ----------
        case_id : str
            Either 'case_000' or 'case_001'.

        Returns
        -------
        tuple[FloatArray, FloatArray, FloatArray]
            pos : shape (9, 2)
            mu : shape (3,)
            y : shape (9, 5)
        """
        if case_id not in ("case_000", "case_001"):
            raise KeyError(f"Unknown case: {case_id}")

        case_number: int = (
            0 if case_id == "case_000" else 1
        )

        mu: FloatArray = np.array(
            [10.0 + case_number, -5.0, 1750.0],
            dtype=np.float64,
        )

        y: FloatArray = np.zeros(
            (len(self.pos), 5),
            dtype=np.float64,
        )

        # [u_r, u_z, u_swirl, p, T]
        y[:, 0] = self.pos[:, 0]
        y[:, 1] = self.pos[:, 1]
        y[:, 2] = 0.1 + 0.01 * case_number
        y[:, 3] = 1.0 + case_number
        y[:, 4] = 1700.0 + 10.0 * case_number

        return self.pos.copy(), mu, y


@pytest.fixture
def benchmark_setup(
) -> tuple[
    BenchmarkEvaluator,
    dict[str, CasePrediction],
]:
    """Create evaluator and perfect predictions for both cases."""
    reader = SyntheticReader()

    benchmark = BenchmarkEvaluator(
        CaseEvaluator(reader)
    )

    predictions: dict[str, CasePrediction] = {}

    for case_id in ("case_000", "case_001"):
        pos, _, y = reader.load_arrays(case_id)

        predictions[case_id] = CasePrediction(
            pos=pos.copy(),
            y=y.copy(),
        )

    return benchmark, predictions


def test_perfect_predictions(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Perfect predictions must yield zero absolute errors."""
    benchmark, predictions = benchmark_setup

    result: BenchmarkResult = benchmark.evaluate(
        predictions=predictions,
        expected_case_ids=["case_000", "case_001"],
    )

    assert result.num_cases == 2
    assert result.region == "all"

    # Two cases, each with five physical fields.
    assert result.case_scores.shape == (10, 6)

    np.testing.assert_allclose(
        result.case_scores["mae"].to_numpy(),
        0.0,
    )

    np.testing.assert_allclose(
        result.case_scores["rmse"].to_numpy(),
        0.0,
    )

    np.testing.assert_allclose(
        result.case_scores["volume_rmse"].to_numpy(),
        0.0,
    )

    assert result.summary.loc[
        ("T", "mae"), "valid_cases"
    ] == 2


def test_valid_case_ids_accepted(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Valid string IDs must pass benchmark validation."""
    benchmark, predictions = benchmark_setup

    result = benchmark.evaluate(
        predictions=predictions,
        expected_case_ids=["case_000", "case_001"],
    )

    assert result.num_cases == 2

    assert list(result.case_metadata.index) == [
        "case_000",
        "case_001",
    ]


def test_each_case_uses_its_own_prediction(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Predictions must be retrieved using individual case IDs."""
    benchmark, predictions = benchmark_setup

    # Modify only the second case.
    predictions["case_001"].y[:, 4] += 7.0

    result = benchmark.evaluate(
        predictions=predictions,
        expected_case_ids=["case_000", "case_001"],
    )

    case_0_mae = result.case_scores.loc[
        ("case_000", "T"), "mae"
    ]

    case_1_mae = result.case_scores.loc[
        ("case_001", "T"), "mae"
    ]

    assert case_0_mae == pytest.approx(0.0)
    assert case_1_mae == pytest.approx(7.0)


def test_equal_case_aggregation(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Aggregate metrics with equal weight per CFD case."""
    benchmark, predictions = benchmark_setup

    predictions["case_000"].y[:, 4] += 1.0
    predictions["case_001"].y[:, 4] += 3.0

    result = benchmark.evaluate(
        predictions=predictions,
        expected_case_ids=["case_000", "case_001"],
    )

    temperature_mae = result.summary.loc[
        ("T", "mae")
    ]

    assert temperature_mae["valid_cases"] == 2

    assert temperature_mae["mean"] == pytest.approx(
        2.0
    )

    assert temperature_mae["std"] == pytest.approx(
        1.0
    )

    assert temperature_mae["min"] == pytest.approx(
        1.0
    )

    assert temperature_mae["max"] == pytest.approx(
        3.0
    )


def test_case_metadata(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Benchmark metadata must retain physical case parameters."""
    benchmark, predictions = benchmark_setup

    result = benchmark.evaluate(predictions)

    assert (result.case_metadata["num_nodes"] == 9).all()

    assert result.case_metadata.loc[
        "case_000", "crystal_rpm"
    ] == pytest.approx(10.0)

    assert result.case_metadata.loc[
        "case_001", "crystal_rpm"
    ] == pytest.approx(11.0)

    expected_volume: float = (
        np.pi * 0.30**2 * 0.15
    )

    np.testing.assert_allclose(
        result.case_metadata["total_volume"].to_numpy(),
        expected_volume,
    )


def test_interior_only_evaluation(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Only interior nodes must contribute to interior metrics."""
    benchmark, predictions = benchmark_setup

    # Node zero belongs to the bottom-axis junction.
    predictions["case_000"].y[0, 4] += 100.0

    result = benchmark.evaluate(
        predictions=predictions,
        region="interior",
    )

    assert result.region == "interior"

    assert (
        result.case_metadata["num_evaluated_nodes"] == 1
    ).all()

    # Boundary-only error should be excluded.
    assert result.case_scores.loc[
        ("case_000", "T"), "mae"
    ] == pytest.approx(0.0)


def test_shuffled_prediction_coordinates_rejected(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Shuffled node ordering must be detected."""
    benchmark, predictions = benchmark_setup

    original = predictions["case_000"]

    order: IntArray = np.array(
        [4, 0, 2, 3, 1, 5, 6, 7, 8],
        dtype=np.int64,
    )

    predictions["case_000"] = CasePrediction(
        pos=original.pos[order],
        y=original.y[order],
    )

    with pytest.raises(ValueError, match="node ordering"):
        benchmark.evaluate(predictions)


def test_missing_case_rejected(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """All explicitly required cases must be present."""
    benchmark, predictions = benchmark_setup

    incomplete = {
        "case_000": predictions["case_000"]
    }

    with pytest.raises(ValueError, match="Missing"):
        benchmark.evaluate(
            predictions=incomplete,
            expected_case_ids=["case_000", "case_001"],
        )


def test_extra_case_rejected(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Extra predictions must be rejected when case list is fixed."""
    benchmark, predictions = benchmark_setup

    with pytest.raises(ValueError, match="Extra"):
        benchmark.evaluate(
            predictions=predictions,
            expected_case_ids=["case_000"],
        )


def test_empty_case_id_rejected(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Empty case identifiers must not be accepted."""
    benchmark, predictions = benchmark_setup

    invalid = {
        "": predictions["case_000"]
    }

    with pytest.raises(
        ValueError,
        match="non-empty strings",
    ):
        benchmark.evaluate(predictions=invalid)


def test_whitespace_case_id_rejected(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Whitespace-only case identifiers must be rejected."""
    benchmark, predictions = benchmark_setup

    invalid = {
        "   ": predictions["case_000"]
    }

    with pytest.raises(
        ValueError,
        match="non-empty strings",
    ):
        benchmark.evaluate(predictions=invalid)


def test_empty_predictions_rejected(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Empty prediction collections must be rejected."""
    benchmark, _ = benchmark_setup

    with pytest.raises(ValueError):
        benchmark.evaluate(predictions={})


def test_duplicate_expected_case_ids_rejected(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Required case lists cannot contain duplicate IDs."""
    benchmark, predictions = benchmark_setup

    with pytest.raises(ValueError, match="duplicates"):
        benchmark.evaluate(
            predictions=predictions,
            expected_case_ids=["case_000", "case_000"],
        )


def test_invalid_region_rejected(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Unsupported spatial evaluation regions must be rejected."""
    benchmark, predictions = benchmark_setup

    with pytest.raises(ValueError, match="region"):
        benchmark.evaluate(
            predictions=predictions,
            region="boundary",  # type: ignore[arg-type]
        )


def test_input_predictions_unchanged(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Benchmark evaluation must not modify predictions."""
    benchmark, predictions = benchmark_setup

    original: dict[str, tuple[FloatArray, FloatArray]] = {
        case_id: (
            prediction.pos.copy(),
            prediction.y.copy(),
        )
        for case_id, prediction in predictions.items()
    }

    benchmark.evaluate(predictions)

    for case_id, prediction in predictions.items():
        original_pos, original_y = original[case_id]

        np.testing.assert_array_equal(
            prediction.pos,
            original_pos,
        )

        np.testing.assert_array_equal(
            prediction.y,
            original_y,
        )


def test_undefined_relative_error_handling(
    benchmark_setup: tuple[
        BenchmarkEvaluator,
        dict[str, CasePrediction],
    ],
) -> None:
    """Undefined relative errors must not become false zeros."""
    benchmark, predictions = benchmark_setup

    # The synthetic reference fields are nonzero, so make
    # a dedicated all-zero field through a custom reader.
    class ZeroFieldReader(SyntheticReader):
        """Reader containing an all-zero pressure field."""

        def load_arrays(
            self,
            case_id: str,
        ) -> tuple[FloatArray, FloatArray, FloatArray]:
            pos, mu, y = super().load_arrays(case_id)
            y[:, 3] = 0.0
            return pos, mu, y

    reader = ZeroFieldReader()
    evaluator = BenchmarkEvaluator(CaseEvaluator(reader))

    zero_predictions: dict[str, CasePrediction] = {}

    for case_id in ("case_000", "case_001"):
        pos, _, y = reader.load_arrays(case_id)

        zero_predictions[case_id] = CasePrediction(
            pos=pos,
            y=y,
        )

    result = evaluator.evaluate(zero_predictions)

    assert np.isnan(
        result.case_scores.loc[
            ("case_000", "p"), "relative_l2"
        ]
    )

    assert result.summary.loc[
        ("p", "relative_l2"), "valid_cases"
    ] == 0

    assert np.isnan(
        result.summary.loc[
            ("p", "relative_l2"), "mean"
        ]
    )


