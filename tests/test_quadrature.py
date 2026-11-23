"""
File Description: unit tests for quatrature axisymmetric integration.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-23
Version: 1.0
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from czmeltbench.geometry.quadrature import (
    compute_mesh_quadrature,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def test_single_rectangle_exact_integrals() -> None:
    """A rectangle revolved around the axis has known volume."""

    pos: FloatArray = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ], dtype=np.float64)

    cells: IntArray = np.array(
        [[0, 1, 2, 3]],
        dtype=np.int64,
    )

    q = compute_mesh_quadrature(
        pos=pos,
        cells=cells,
    )

    assert q.total_area == pytest.approx(2.0)

    assert q.total_volume == pytest.approx(
        2.0 * np.pi
    )

    np.testing.assert_allclose(
        q.node_area_weights,
        0.5,
    )

    np.testing.assert_allclose(
        q.node_volume_weights,
        np.array([
            np.pi / 3,
            2.0 * np.pi / 3,
            2.0 * np.pi / 3,
            np.pi / 3,
        ]),
    )

    assert q.node_volume_weights.sum() == pytest.approx(
        q.total_volume
    )


def test_shuffled_node_order() -> None:
    """Reordering original rows must preserve physical geometry."""

    pos: FloatArray = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ], dtype=np.float64)

    # Deliberately shuffle the original node ordering.
    order: IntArray = np.array(
        [2, 0, 3, 1],
        dtype=np.int64,
    )

    shuffled: FloatArray = pos[order]

    # Map the original indices to the shuffled indices.
    inverse: IntArray = np.argsort(order)

    cells: IntArray = inverse[
        np.array([[0, 1, 2, 3]], dtype=np.int64)
    ]

    q = compute_mesh_quadrature(
        pos=shuffled,
        cells=cells,
    )

    assert q.total_volume == pytest.approx(
        2.0 * np.pi
    )

    np.testing.assert_allclose(
        q.node_volume_weights[inverse],
        np.array([
            np.pi / 3,
            2.0 * np.pi / 3,
            2.0 * np.pi / 3,
            np.pi / 3,
        ]),
    )


def test_bad_cell_orientation() -> None:
    """Clockwise quadrilateral cells must be rejected."""

    pos: FloatArray = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ], dtype=np.float64)

    clockwise_cells: IntArray = np.array(
        [[0, 3, 2, 1]],
        dtype=np.int64,
    )

    with pytest.raises(ValueError, match="Jacobian"):
        compute_mesh_quadrature(
            pos=pos,
            cells=clockwise_cells,
        )


def test_repeated_nodes() -> None:
    """Quadrilaterals with repeated node IDs must be rejected."""

    pos: FloatArray = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ], dtype=np.float64)

    invalid_cells: IntArray = np.array(
        [[0, 1, 1, 3]],
        dtype=np.int64,
    )

    with pytest.raises(ValueError, match="repeated"):
        compute_mesh_quadrature(
            pos=pos,
            cells=invalid_cells,
        )


def test_shape_function_properties() -> None:
    """Verify bilinear shape-function consistency."""

    from czmeltbench.geometry.quadrature import (
        _shape_function,
    )

    gauss_points = (
        -1.0 / np.sqrt(3.0),
         1.0 / np.sqrt(3.0),
    )

    for xi in gauss_points:
        for eta in gauss_points:

            n, dxi, deta = _shape_function(
                xi=xi,
                eta=eta,
            )

            # Partition of unity
            np.testing.assert_allclose(
                np.sum(n),
                1.0,
                atol=1e-14,
            )

            # Derivatives of the partition of unity
            np.testing.assert_allclose(
                np.sum(dxi),
                0.0,
                atol=1e-14,
            )

            np.testing.assert_allclose(
                np.sum(deta),
                0.0,
                atol=1e-14,
            )


