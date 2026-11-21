"""
File Description: Tests for logical mesh reconstruction.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-21
Version: 1.0
"""

import numpy as np
import pytest

from czmeltbench.geometry.mesh import build_logical_mesh

def test_logical_mesh_shape() -> None:
    """A small structured mesh shoudl be reconstructed
    """

    r = np.array([0.0, 0.5, 1.0])
    z = np.array([0.0, 0.25])

    R, Z = np.meshgrid(r, z)

    pos = np.column_stack([
        R.ravel(),
        Z.ravel(),
    ])

    mesh = build_logical_mesh(pos)

    assert mesh.shape == (2, 3)
    assert mesh.num_nodes == 6

def test_original_node_order_preserved() -> None:
    """Reconstruction must preserve original node indices
    """

    pos = np.array([
        [1.0, 0.5],
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 0.0],
        [0.0, 0.5],
        [0.5, 0.0],
    ])

    mesh = build_logical_mesh(pos)

    recovered = mesh.original_coordinates()
    np.testing.assert_array_equal(
        recovered,
        pos,
    )

def test_field_reordering() -> None:
    """Nodal vlaues should follow logical node positions.
    """

    pos = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.5],
        [0.0, 0.5],
        [0.5, 0.5],
    ])

    values = np.arange(6, dtype=np.float64)

    mesh = build_logical_mesh(pos)
    reordered = mesh.reshape_field(values)

    np.testing.assert_array_equal(
        reordered,
        np.array([
            [1.0, 2.0, 0.0],
            [4.0, 5.0, 3.0],
        ]),
    )

def test_invalid_coordinates() -> None:
    """Coordinates with incorrect dimensions must fail.
    """

    pos = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        build_logical_mesh(pos)