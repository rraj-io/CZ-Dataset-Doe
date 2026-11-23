"""
File Description: Unit test for ML node features construction.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-23
Version: 1.0
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from czmeltbench.physics.features import (
    FeatureConfig,
    build_node_features,
)

FloatArray = NDArray[np.float64]

@pytest.fixture
def sample_data() -> tuple[FloatArray, FloatArray]:
    """Create represtative geoemtry and parameters.
    """
    pos = np.array(
        [
            [0.00, 0.00], # origin
            [0.10, 0.08], # interior
            [0.30, 0.08], # wall
            [0.10, 0.15], # crytasstal
            [0.25, 0.15], # lid
        ],
        dtype=np.float64,
    )

    mu = np.array(
        [12.0, -5.5, 1757.5],
        dtype=np.float64,
    )

    return pos, mu

@pytest.mark.parametrize(
    ("config", "expected_features"),
    [
        (
            FeatureConfig(include_boundaries=False),
            5,
        ),
        (
            FeatureConfig(),
            10,
        ),
        (
            FeatureConfig(include_bc_values=True),
            20,
        ),
        (
            FeatureConfig(
                include_bc_values=True,
                include_distances=True,
            ),
            25,
        ),

    ],
)

def test_feature_dimensions(
    sample_data: tuple[FloatArray, FloatArray],
    config: FeatureConfig,
    expected_features: int,
) -> None:
    """Feature configurations must have consistent widths.
    """
    pos, mu = sample_data

    result = build_node_features(
        pos,
        mu,
        config=config,
    )

    assert result.x.shape == (5, expected_features)
    assert len(result.columns) == expected_features
    assert result.num_nodes == 5
    assert result.num_features == expected_features

def test_parameter_broadcast(
    sample_data: tuple[FloatArray, FloatArray]
) -> None:
    """Operating paramters must be identical at all nodes.
    """
    pos, mu = sample_data

    result = build_node_features(pos, mu)

    for name, expected in zip(
        ("crystal_rpm", "crucible_rpm", "t_hot"),
        mu,
    ):
        np.testing.assert_allclose(
            result.get_column(name),
            expected
        )

def test_boundary_encoding(
    sample_data: tuple[FloatArray, FloatArray]
) -> None:
    """Wall membership must be encoded correctly.
    """
    pos, mu = sample_data

    result = build_node_features(pos, mu)

    wall = result.get_column("boundary_wall")

    np.testing.assert_array_equal(
        wall,
        np.array([0, 0, 1, 0, 0]),
    )

def test_boundary_values_and_masks(
        sample_data: tuple[FloatArray, FloatArray],
) -> None:
    """Only prescribed values should have activve BC masks.
    """
    pos, mu = sample_data

    result = build_node_features(
        pos, mu, config=FeatureConfig(
            include_bc_values=True,
        ),
    )

    interior_index = 1
    crystal_index = 3

    assert result.get_column("bc_mask_T")[interior_index] == 0.0
    assert result.get_column("bc_mask_T")[crystal_index] == 1.0
    assert result.get_column("bc_value_T")[crystal_index] == pytest.approx(1685.0)

def test_distance_features(
        sample_data: tuple[FloatArray, FloatArray]
) -> None:
    """Distance features should reflecct the geometry.
    """
    pos, mu = sample_data

    result = build_node_features(
        pos, mu, config=FeatureConfig(
            include_distances=True,
        ),
    )

    wall_distance = result.get_column("distance_wall")

    assert wall_distance[2] == pytest.approx(0.0)
    assert wall_distance[1] == pytest.approx(0.20)

def test_invalid_inputs() -> None:
    """Invalid inputs must be rejected..
    """

    pos = np.array([[0.1, 0.2]], dtype=np.float64)

    with pytest.raises(ValueError):
        build_node_features(
            pos,
            np.array([12.0, -5.5]),
        )

def test_inputs_are_not_modified(
        sample_data: tuple[FloatArray, FloatArray],
) -> None:
    """Feature construction must preserve the source array
    """

    pos, mu = sample_data

    pos_before = pos.copy()
    mu_before = mu.copy()

    build_node_features(
        pos=pos,
        mu=mu,
        config=FeatureConfig(
            include_boundaries=True,
            include_distances=True,
        ),
    )

    np.testing.assert_array_equal(pos, pos_before)
    np.testing.assert_array_equal(mu, mu_before)