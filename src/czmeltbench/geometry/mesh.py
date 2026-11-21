"""
File Description: Logical mesh construction for CZ CFD Data
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-21
Version: 1.0
"""

import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

@dataclass(frozen=True)
class LogicalMesh:
    """Logically structured representation of CFD mesh.

    The original node ordering is conserved through `node_ids`.

    Attributes
    ----------
    r : FlaotArray
        Radial coordinates, shape [nz, nr], in metres.

    z : FlaotArray
        Axial coordinates, shape (nz, nr) in meters

    node_ids : IntArray
        Original node indices, shape (nz, nr)

    Notes
    ----
    This representtion does not interpolate the mesh or change 
    the physical coordinates.
    """

    r: FloatArray
    z: FloatArray
    node_ids: IntArray

    @property
    def shape(self) -> tuple[int, int]:
        """Return logical mesh dimensions (nz, nr).
        """
        return self.r.shape

    @property
    def num_nodes(self) -> int:
        """Return hte number of mesh nodes
        """
        return int(self.r.size)

    @property
    def nr(self) -> int:
        """Return the number of radial nodes per axial level.
        """
        return int(self.r.shape[1])

    @property
    def nz(self) -> int:
        """Return hte number of axial levels.
        """
        return int(self.r.shape[0])

    def reshape_field(self,
                      values: FloatArray) -> FloatArray:
        """Reorder nodal values into logicla mesh order.

        Parameters
        ----------
        values : FloatArray
            Field vlaues in original CFD node order.
            Shape (N,) or (N, C).

        Returns
        -------
        FlaotArray
            Reordered values with sahpe (nz, nr)
            or (nz, nr, C).
        """

        values = np.asarray(values, dtype=np.float64)

        if values.ndim not in (1, 2):
            raise ValueError(
                "Expected values with shape (N,) or (N, C)"
            )
        if values.shape[0] != self.num_nodes:
            raise ValueError(
                "Number of field values does not match mesh nodes."
            )

        return values[self.node_ids]

    def original_coordinates(self) -> FloatArray:
        """Return coordinates in original CFD node order.

        Returns
        -------
        FlaotArray
            Coordinates with shape (N, 2)
        """

        coordinates = np.empty(
            (self.num_nodes, 2),
            dtype=np.float64,
        )

        coordinates[self.node_ids.ravel(), 0] = self.r.ravel()
        coordinates[self.node_ids.ravel(), 1] = self.z.ravel()

        return coordinates

def build_logical_mesh(pos: FloatArray,
                       *,
                       axial_tolerance: float = 1e-10) -> LogicalMesh:
    """Build a logical mesh from CFD coordinates.

    Parameters
    ----------
    pos : FlaotArray
        Original node coordinates with shape (N, 2).
        Columns: [r, z] in meters.

    axial_tolerance : flaot, default=1e-10
        Maximum allowed difference in z within  an axial level, in [m]
    
    Returns
    -------
    LogicalMesh
        Logical mesh with preserved original node indices.
    
    Raises
    ------
    ValueError
        If the coordinates are invalid or cannnot be organized into 
        rectangular logical grid.

    Notes
    -----
    The algo:
    1. Sort nodes by axial coordinaes
    2. Identify axial levels.
    3. Sort nodes radially within each level.
    4. Preserve original node iondices.

    It assumes that hte dataset has a logically rectangular mesh
    with constant nodes per level.
    """

    coordinates = np.asarray(pos, dtype=np.float64)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "Expected coordinnates with shape (N ,2)."
        )

    if not np.isfinite(coordinates).all():
        raise ValueError(
            "mesh coordinates contain nonfinite values."
        )

    if axial_tolerance < 0:
        raise ValueError(
            "axial_tolerance must be non-negative."
        )

    n_nodes = coordinates.shape[0]

    if n_nodes == 0:
        raise ValueError("Mesh contains no nodes.")

    # Sort primarily by axial coordinates
    axial_order = np.argsort(
        coordinates[:, 1],
        kind="stable",
    )

    sorted_z = coordinates[axial_order, 1]

    # identify axial levels using consecutive z differences.
    level_starts = np.concatenate((
        np.array([0], dtype=np.int64),
        np.flatnonzero(
            np.diff(sorted_z) > axial_tolerance
        ) + 1,
    ))

    level_ends = np.concatenate((
        level_starts[1:],
        np.array([n_nodes], dtype=np.int64),
    ))

    counts = level_ends - level_starts

    if not np.all(counts == counts[0]):
        raise ValueError(
            "Mesh does not have a constant number"
            "of radial nodes per axial level."
        )

    nz = len(counts)
    nr = int(counts[0])

    if nr * nz != n_nodes:
        raise ValueError(
            "Logical mesh dimensions do not match node count."
        )

    node_ids = np.empty(
        (nz, nr),
        dtype=np.int64,
    )

    for level, (start, end) in enumerate(zip(level_starts, level_ends)):
        ids = axial_order[start:end]

        radial_order = np.argsort(
            coordinates[ids, 0],
            kind="stable",
        )

        node_ids[level] = ids[radial_order]

    r = coordinates[node_ids, 0]
    z = coordinates[node_ids, 1]

    # Validate that radial coordinate increase
    if np.any(np.diff(r, axis=1) <= 0):
        raise ValueError(
            "Axial coordinates must increase"
            "between logicla levels."
        )

    return LogicalMesh(
        r=r,
        z=z,
        node_ids=node_ids,
    )
