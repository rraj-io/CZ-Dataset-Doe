"""
File Description: Tests for original-node mesh connectivity
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-23
Version: 1.0
"""

import numpy as np
from numpy.typing import NDArray

from czmeltbench.geometry.mesh import build_logical_mesh
from czmeltbench.geometry.connectivity import build_connectivity

FloatArray = NDArray[np.float64]

def test_shuffled_nodes_preserve_physical_connectivity() -> None:
    """Reordering input rows must not change physical neighbour pairs.
    """

    ordered: FloatArray = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.4], [0.5, 0.4], [1.0, 0.4],
    ], dtype=np.float64)

    pos: FloatArray = ordered[[4, 0, 5, 2, 1, 3]]
    mesh = build_logical_mesh(pos)
    graph = build_connectivity(mesh)

    assert mesh.shape == (2, 3)
    assert graph.edge_index.shape == (2, 14)
    assert graph.edge_attr.shape == (14, 3)
    assert graph.cells.shape == (2, 4)

    # Every graph edge must join nodes differing in exactly one grid axis
    delta = np.abs(pos[graph.edge_index[1]] - pos[graph.edge_index[0]])

    valid = (
        (np.isclose(delta[:, 0], 0.5) & np.isclose(delta[:, 1], 0.0)) |
        (np.isclose(delta[:, 0], 0.0) & np.isclose(delta[:, 1], 0.4))
    )

    assert valid.all()
    # Expected displacement from source to destination
    expected_displacement = (
        pos[graph.edge_index[1]]
        - pos[graph.edge_index[0]]
    )

    # Check radial and axial edge displacements
    np.testing.assert_allclose(
        graph.edge_attr[:, :2],
        expected_displacement,
    )

    # Check Euclidean edge lengths
    expected_lengths = np.linalg.norm(
        expected_displacement,
        axis=1,
    )

    np.testing.assert_allclose(
        graph.edge_attr[:, 2],
        expected_lengths,
    )
    np.testing.assert_allclose(
        graph.edge_attr[:, 2],
        np.linalg.norm(graph.edge_attr[:, :2], axis=1),
    )

    # Every directed edge has an exact reverse.
    edges = set(map(tuple, graph.edge_index.T.tolist()))

    assert all((dst, src) in edges for src, dst in edges)
    assert not np.any(graph.edge_index[0] == graph.edge_index[1])

    # Each quad is counterclockwise in hte r-z plane
    quad_pos = pos[graph.cells]
    r = quad_pos[:, :, 0]
    z = quad_pos[:, :, 1]

    twice_area = np.sum(
        r * np.roll(z, -1, axis=1) - np.roll(r, -1, axis=1) * z,
        axis=1,
    )

    assert np.all(twice_area > 0)

def test_feature_alignment_is_preserved() -> None:
    """Graph IDs must refer directly to the unchanged original CFD rows.
    """
    pos: FloatArray = np.array([
        [1.0, 0.5], [0.0, 0.0], [0.5, 0.5],
        [1.0, 0.0], [0.0, 0.5], [0.5, 0.0],
    ])

    mesh = build_logical_mesh(pos)
    graph = build_connectivity(mesh)

    np.testing.assert_array_equal(mesh.original_coordinates(), pos)
    np.testing.assert_array_equal(
        pos[graph.edge_index[1]] - pos[graph.edge_index[0]],
        graph.edge_attr[:, :2],
    )
