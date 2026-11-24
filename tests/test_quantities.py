"""
File Description: Unit tests for CFD quantities of interest
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-24
Version: 1.0
"""

import numpy as np
import pytest

from numpy.typing import NDArray

from czmeltbench.physics.quantities import (
    calculate_quantities,
)

FloatArray = NDArray[np.float64]


@pytest.fixture
def sample_fields() -> FloatArray:
    """Create a small synthetic CFD solution

    columns: [u_r, u_z, u_swirl, p, T].
    """
    return np.array([
        [3.0, 4.0, 0.0, 0.0, 1700.0],
        [0.0, 0.0, -2.0, 1.0, 1800.0],
        [0.0, 5.0, 0.0, 2.0, 1750.0],
    ], dtype=np.float64)

def test_unweighted_quantities(sample_fields: FloatArray) -> None:
    """Check nodal quantites against analytical values.
    """

    qoi = calculate_quantities(sample_fields)

    assert len(qoi) == 8
    assert qoi["max_meridional"] == pytest.approx(5.0)

    assert qoi["rms_meridional"] == pytest.approx(np.sqrt(50.0 / 3.0))

    assert qoi["max_swirl"] == pytest.approx(2.0)
    assert qoi["max_total"] == pytest.approx(5.0)

    assert qoi["mean_T"] == pytest.approx(1750.0)
    assert qoi["min_T"] == pytest.approx(1700.0)
    assert qoi["max_T"] == pytest.approx(1800.0)
    assert qoi["delta_T"] == pytest.approx(100.0)

def test_volume_weighted_quantities(
    sample_fields: FloatArray,
) -> None:
    """Check weighted statistics with known node weights."""

    weights: FloatArray = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    qoi = calculate_quantities(
        sample_fields,
        volume_weights=weights,
    )

    # (1700*1 + 1800*2 + 1750*3) / 6
    expected_T: float = 1758.3333333333333

    # Meridional speeds: [5, 0, 5]
    expected_rms_meridional: float = np.sqrt(
        (25.0 * 1.0 + 0.0 * 2.0 + 25.0 * 3.0) / 6.0
    )

    # Total speeds: [5, 2, 5]
    expected_rms_total: float = np.sqrt(
        (25.0 * 1.0 + 4.0 * 2.0 + 25.0 * 3.0) / 6.0
    )

    assert qoi["volume_mean_T"] == pytest.approx(
        expected_T
    )

    assert qoi["volume_rms_meridional"] == pytest.approx(
        expected_rms_meridional
    )

    assert qoi["volume_rms_total"] == pytest.approx(
        expected_rms_total
    )

    assert len(qoi) == 11


def test_uniform_weights_match_nodal_statistics(
    sample_fields: FloatArray,
) -> None:
    """Equal weights must reproduce unweighted averages."""

    weights: FloatArray = np.ones(
        sample_fields.shape[0],
        dtype=np.float64,
    )

    qoi = calculate_quantities(
        sample_fields,
        volume_weights=weights,
    )

    assert qoi["volume_mean_T"] == pytest.approx(
        qoi["mean_T"]
    )

    assert qoi["volume_rms_meridional"] == pytest.approx(
        qoi["rms_meridional"]
    )


def test_negative_swirl_uses_absolute_magnitude() -> None:
    """Maximum swirl speed must be nonnegative."""

    fields: FloatArray = np.array([
        [0.0, 0.0, -7.0, 0.0, 1700.0],
        [0.0, 0.0, -2.0, 0.0, 1710.0],
    ], dtype=np.float64)

    qoi = calculate_quantities(fields)

    assert qoi["max_swirl"] == pytest.approx(7.0)
    assert qoi["max_total"] == pytest.approx(7.0)


def test_invalid_field_shape() -> None:
    """A field matrix must have exactly five columns."""

    with pytest.raises(ValueError):
        calculate_quantities(
            np.zeros((10, 4), dtype=np.float64)
        )

    with pytest.raises(ValueError):
        calculate_quantities(
            np.empty((0, 5), dtype=np.float64)
        )


def test_invalid_weights(
    sample_fields: FloatArray,
) -> None:
    """Invalid integration weights must be rejected."""

    with pytest.raises(ValueError):
        calculate_quantities(
            sample_fields,
            volume_weights=np.ones(2),
        )

    with pytest.raises(ValueError):
        calculate_quantities(
            sample_fields,
            volume_weights=np.array([1.0, -1.0, 1.0]),
        )

    with pytest.raises(ValueError):
        calculate_quantities(
            sample_fields,
            volume_weights=np.zeros(3),
        )


def test_inputs_are_not_modified(
    sample_fields: FloatArray,
) -> None:
    """QoI calculation must not modify the CFD solution."""

    original: FloatArray = sample_fields.copy()

    calculate_quantities(
        sample_fields,
        volume_weights=np.ones(3),
    )

    np.testing.assert_array_equal(
        sample_fields,
        original,
    )

def test_nonfinite_fields() -> None:
    """Reject CFD fields containing NaN or infinity."""

    fields: FloatArray = np.zeros(
        (3, 5),
        dtype=np.float64,
    )

    fields[1, 4] = np.nan

    with pytest.raises(ValueError):
        calculate_quantities(fields)

    fields[1, 4] = np.inf

    with pytest.raises(ValueError):
        calculate_quantities(fields)
