"""
File Description: Reconstruct native mesh adjacency and quadrilateral cells.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-23
Version: 1.0
"""
"""
All indices refer to the original CFD node order.
No interpolation or modification of CFD field values is performed.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from czmeltbench.geometry.mesh import LogicalMesh

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

@dataclass(frozen=True)
class MeshConnectivity:
    """Native logical-grid connectivity in original CFD indexing.
    
    Attributes
    -----------
    edge_index : IntArray
        Directed edges, shape (2, E).
        Row 0: source node IDs.
        Row 1: destination node ID2.

    edge_attr : FloatArray
        Edge attributes, shape (E, 3).
        Columns: [delta_r, delta_z, length], in [m]
        Displacements point from source to destination.

    cells : IntArray
        Quadrilateral node IDs, shape (C, 4).
        Counterclockwise order:
        [bottom_left, bottom-right, top-right, top-left].
    """

    edge_index: IntArray
    edge_attr: FloatArray
    cells: IntArray

    @property
    def num_edges(self) -> int:
        """Returns the number of directed graph edges.
        """
        return int(self.edge_index.shape[1])

    @property
    def num_cells(self) -> int:
        """Return the number of reconstructed quadrilateral cells.
        """
        return int(self.cells.shape[0])


def build_connectivity(mesh: LogicalMesh) -> MeshConnectivity:
    """Build bidirectional four-neighbour graph connnectivity.

    Parameters
    ----------
    mesh : Logicalmesh
        Logical mesh of shape (nz, nr).
        mesh.node_ids contains original CFD node indices.

    Returns
    -------
    MeshConnectivity
        Directed edges, geometric edge attributes and reconstructed
        quadrilateral cells.

    Notes
    -----
    No diagonal eges are added.
    No coordinates or physical fields are interpolated.
    """
    ids: IntArray = np.asarray(
        mesh.node_ids, dtype=np.int64
    )

    r: FloatArray = np.asarray(
        mesh.r, dtype=np.float64
    )

    z: FloatArray = np.asarray(
        mesh.z, dtype=np.float64
    )

    # 1. validate logical shape
    if (
        ids.ndim != 2 or ids.shape != r.shape or r.shape != z.shape
    ):
        raise ValueError(
            "node_ids, r and z must share shape (nz, nr)."
        )

    nz, nr = ids.shape

    if nz < 2 or nr < 2:
        raise ValueError(
            "At lest two axial and radial nodes are required."
        )

    n_nodes: int = nz * nr

    # 2. Check if all original node IDs occur exactly once.
    if not np.array_equal(
        np.sort(ids.ravel()),
        np.arange(n_nodes),
    ):
        raise ValueError(
            "node_ids must be a permutation of 0,...,N-1."
        )

    if not np.isfinite(r).all() or not np.isfinite(z).all():
        raise ValueError("Mesh coordinates must be finite.")

    # Rows corresponds to axial levels.
    # Columns corrrespon to radial positions.
    if np.any(np.diff(r, axis=1) <= 0.0):
        raise ValueError(
            "Radial coordinates must increase along columns."
        )

    
    if np.any(np.diff(z, axis=0) <= 0.0):
        raise ValueError(
            "Axial coordinates must increase along rows."
        )

    # 3. Connect adjacent radial nodes.
    radial_src: IntArray = ids[:, :-1].ravel()
    radial_dst: IntArray = ids[:, 1:].ravel()

    
    # 4. Connect adjacent axial nodes.
    axial_src: IntArray = ids[:-1, :].ravel()
    axial_dst: IntArray = ids[1:, :].ravel()

    # each neighbour pair appears once at this point.
    src: IntArray = np.concatenate((
        radial_src, axial_src
    ))

    # each neighbour pair appears once at this point.
    dst: IntArray = np.concatenate((
        radial_dst, axial_dst
    ))

    # 5. Add reverse edges
    edge_index: IntArray = np.stack((
        np.concatenate((src, dst)),
        np.concatenate((dst, src))
    ))

    # 6. Calculate edge attributes using original coordinates.
    pos: FloatArray = mesh.original_coordinates()

    displacement: FloatArray = (
        pos[edge_index[1]] - pos[edge_index[0]]
    )

    lengths: FloatArray = np.linalg.norm(
        displacement,
        axis=1,
        keepdims=True,
    )

    if np.any(lengths <= 0.0):
        raise ValueError("Mesh contains zero-length edges.")

    edge_attr: FloatArray = np.concatenate(
        (displacement, lengths),
        axis=1,
    )

    # 7. Reconstruct quadrilateral cells.
    # counterclockwise ordering in the r-z plane.
    cells: IntArray = np.stack((
        ids[:-1, :-1],
        ids[:-1, 1:],
        ids[1:, 1:],
        ids[1:, :-1],
    ), axis=-1).reshape(-1, 4)

    return MeshConnectivity(
        edge_index=edge_index,
        edge_attr=edge_attr,
        cells=cells,
    )