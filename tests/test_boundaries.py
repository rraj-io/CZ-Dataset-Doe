"""
File Description: Unnint tests for geometric Boundary functions
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-21
Version: 1.0
"""


import numpy as np
import pytest

from czmeltbench.geometry.boundaries import (
    identify_boundaries,
    boundary_features,
    distance_features,
)


@pytest.fixture
def sample_positions() -> np.ndarray:
    """Representative coordinates from the CFD geometry."""
    return np.array([
        [0.10, 0.08],  # Interior
        [0.00, 0.08],  # Axis
        [0.10, 0.00],  # Bottom
        [0.30, 0.08],  # Wall
        [0.10, 0.15],  # Crystal
        [0.25, 0.15],  # Lid
        [0.18, 0.15],  # Crystal/lid junction
        [0.00, 0.00],  # Axis/bottom junction
    ], dtype=np.float64)


def test_boundary_classification(
    sample_positions: np.ndarray,
) -> None:
    """Verify basic classification and shared junctions."""
    masks = identify_boundaries(sample_positions)

    assert masks["interior"][0]
    assert masks["axis"][1]
    assert masks["bottom"][2]
    assert masks["wall"][3]
    assert masks["crystal"][4]
    assert masks["lid"][5]

    # Top junction belongs to two segments.
    assert masks["crystal"][6]
    assert masks["lid"][6]

    # Origin belongs to axis and bottom.
    assert masks["axis"][7]
    assert masks["bottom"][7]


def test_boundary_feature_shape(
    sample_positions: np.ndarray,
) -> None:
    """Boundary features must have five columns."""
    features = boundary_features(sample_positions)

    assert features.shape == (8, 5)
    assert np.isin(features, [0.0, 1.0]).all()

    np.testing.assert_array_equal(
        features[0],
        np.zeros(5),
    )

    np.testing.assert_array_equal(
        features[6],
        np.array([0, 0, 0, 1, 1]),
    )


def test_distance_features(
    sample_positions: np.ndarray,
) -> None:
    """Distances must agree with known geometry."""
    distances = distance_features(sample_positions)

    assert distances.shape == (8, 5)
    assert np.isfinite(distances).all()
    assert np.all(distances >= 0)

    # The origin lies on the axis and bottom.
    assert distances[7, 0] == pytest.approx(0.0)
    assert distances[7, 1] == pytest.approx(0.0)

    # The crystal-edge node lies on both top segments.
    assert distances[6, 3] == pytest.approx(0.0)
    assert distances[6, 4] == pytest.approx(0.0)


def test_invalid_positions() -> None:
    """Malformed coordinate arrays must be rejected."""
    with pytest.raises(ValueError):
        identify_boundaries(np.array([0.0, 0.1]))

    with pytest.raises(ValueError):
        distance_features(
            np.array([[np.nan, 0.1]])
        )


def test_outside_domain() -> None:
    """Distance features reject coordinates outside the domain."""
    with pytest.raises(ValueError):
        distance_features(
            np.array([[0.40, 0.10]])
        )
