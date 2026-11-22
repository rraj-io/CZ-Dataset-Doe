"""
File Description: tests for analytical BC calculations.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-22
Version: 1.0
"""

import numpy as np
import pytest

from czmeltbench.physics.boundary_conditions import (
    BoundaryConditions,
    compute_boundary_conditions,
    rpm_to_rad_per_sec,
)

@pytest.fixture
def sample_positions() -> np.ndarray:
    """Representative mesh nodes and bounndary junctions.
    """
    return np.array(
        [
            [0.10, 0.08],       # Interior
            [0.00, 0.08],       # Axis
            [0.10, 0.00],       # Bottom
            [0.30, 0.08],       # Sidewall
            [0.10, 0.15],       # Crystal
            [0.25, 0.15],       # Outer lid
            [0.18, 0.15],       # Crystal/lid junctin
            [0.00, 0.00],       # Origin
            [0.30, 0.00],       # Bottom/wall junction
            [0.30, 0.15],       # Wall/lid junction
        ],
        dtype=np.float64
    )

@pytest.fixture
def sample_parameters() -> np.ndarray:
    """Example operating condition.
    """
    return np.array(
        [12.0, -5.5, 1757.5],
        dtype=np.float64,
    )

def test_angular_velocity_conversion() -> None:
    """60 rpm should equal 2*pi rad/s.
    """
    assert rpm_to_rad_per_sec(60.0) == pytest.approx(
        2.0 * np.pi
    )

def test_output_shapes(
        sample_positions: np.ndarray,
        sample_parameters: np.ndarray,
) -> None:
    """BC values and masks must match the field schema.
    """
    bc = compute_boundary_conditions(sample_positions,
                                     sample_parameters,)

    assert isinstance(bc, BoundaryConditions)
    assert bc.values.shape == (10, 5)
    assert bc.mask.shape == (10, 5)
    assert bc.mask.dtype == np.bool_

def test_interior_has_no_dirichlet_values(
        sample_positions: np.ndarray,
        sample_parameters: np.ndarray,
) -> None:
    """Interior nodes have no prescribed field values."""
    bc = compute_boundary_conditions(
        sample_positions,
        sample_parameters,
    )

    assert not bc.mask[0].any()

def test_crystal_interface(
        sample_positions: np.ndarray,
        sample_parameters: np.ndarray,
) -> None:
    """Crystal interface must have prescribed T and swirl
    """
    bc = compute_boundary_conditions(
        sample_positions,
        sample_parameters
    )

    omega = rpm_to_rad_per_sec(12.0)

    assert bc.values[4, 2] == pytest.approx(omega * 0.10)
    assert bc.values[4, 4] == pytest.approx(1685.0)

    assert bc.mask[4, 2]
    assert bc.mask[4, 4]

def test_crucible_wall(
        sample_positions: np.ndarray,
        sample_parameters: np.ndarray,
) -> None:
    """Crucible wall must use its prescribed rotation and T.
    """
    bc = compute_boundary_conditions(sample_positions,
                                     sample_parameters)

    omega = rpm_to_rad_per_sec(-5.5)

    assert bc.values[3, 2] == pytest.approx(omega * 0.30)
    assert bc.values[3, 4] == pytest.approx(1757.5)

def test_axis_conditions(
        sample_positions: np.ndarray,
        sample_parameters: np.ndarray,
) -> None:
    """Axis has zero radial/swirl velocity, but no axial BC."""
    bc = compute_boundary_conditions(sample_positions,
                                     sample_parameters)

    axis_index = 1

    assert bc.mask[axis_index, 0]
    assert not bc.mask[axis_index, 1]
    assert bc.mask[axis_index, 2]
    assert not bc.mask[axis_index, 4]

    assert bc.values[axis_index, 0] == 0.0
    assert bc.values[axis_index, 2] == 0.0

def test_shared_boundary_junnctions(
        sample_positions: np.ndarray,
        sample_parameters: np.ndarray,
) -> None:
    """Analytical BCs should be consistent at teh corners."""

    bc = compute_boundary_conditions(sample_positions,
                                     sample_parameters)

    # crystal/lid junction remianns at crystal temp.
    assert bc.values[6, 4] == pytest.approx(1685.0)

    # Bottom/wall junctionn remains at hot-wall temp.
    assert bc.values[8, 4] == pytest.approx(1757.5)

    # lid/wall junction also remainns at hot-wall temp.
    assert bc.values[9, 4] == pytest.approx(1757.5)

def test_pressure_reference(
        sample_positions: np.ndarray,
        sample_parameters: np.ndarray,
) -> None:
    """ONly the origin receives a pressure reference."""
    bc = compute_boundary_conditions(sample_positions,
                                     sample_parameters)

    assert bc.mask[:, 3].sum() == 1
    assert bc.mask[7, 3] 
    assert bc.values[7, 3] == 0.0

def test_invalid_parameters(sample_positions: np.ndarray) -> None:
    """Invalid operating paramters must be rejected."""
    with pytest.raises(ValueError):
        compute_boundary_conditions(
            sample_positions,
            np.array([12.0, -5.5])
        )

    with pytest.raises(ValueError):
        compute_boundary_conditions(
            sample_positions,
            np.array([12.0, np.nan, 1757.5])
        )