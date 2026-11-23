"""
File Description: Area and axisymmetric volume integration on native quadrilateral meshes.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-23
Version: 1.0
"""
"""
Uses two-point Gauss-Legendre quadrature in each reference coordinate.
Cells must have counterclockwise node order [BL, BR, TR, TL].
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class MeshQuadrature:
    """geometry measures in the axisymmetric (r, z) domain.

    Parameters
    ----------
    cell_area : FloatArray
        Meridional cell areas of shape (C,) in m^2.
    
    cell_volume : FloatArray
        Revolved cell volumes of shape (C,) in m^3.

    node_area_weights : FloatArray
        Integrated bilinear shape-function weights of shape
        (N,), in m^2.

    node_volume_weights : FloatArray
        Axisymmetric integrated shape-function weights of 
        shape (N,), in m^3.
    """

    cell_area: FloatArray
    cell_volume: FloatArray
    node_area_weights: FloatArray
    node_volume_weights: FloatArray

    @property
    def total_area(self) -> float:
        """Returns meriodinal domain area in m^3.
        """
        return float(self.cell_area.sum())

    @property
    def total_volume(self) -> float:
        """Returns revolved melt volume in m^3.
        """
        return float(self.cell_volume.sum())

def _shape_function(xi: float,
                    eta: float,
                    ) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return bilinear shape functions and reference derivatives.

    Vertex order:
        [bottom-left, bottom-right, top-right, top-left]

    Parameters
    ----------
    xi : float
        First reference coordinate in [-1, 1].

    eta : float
        Second reference coordinate in [-1, 1].

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]
        Shape functions N, derivatives dN/dxi and dN/deta.
        Each array has shape (4,).
    """

    n: FloatArray = 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ], dtype=np.float64)

    dxi: FloatArray = 0.25 * np.array([
        -(1.0 - eta),
         (1.0 - eta),
         (1.0 + eta),
        -(1.0 + eta),
    ], dtype=np.float64)

    deta: FloatArray = 0.25 * np.array([
        -(1.0 - xi),
        -(1.0 + xi),
        (1.0 + xi),
        (1.0 - xi),
    ], dtype=np.float64)

    return n, dxi, deta


def compute_mesh_quadrature(pos: FloatArray,
                            cells: IntArray) -> MeshQuadrature:
    """Compute geometric integration weights for axisymmetric quads.

    Parameters
    ----------
    pos : FloatArray
        Original CFD node coordinates, shape (N, 2).
        Columns: [r, z], in [m]

    cells : IntArray
        Counterclockwise quasdrilateral connectivity
        shape (C, 4).

        Each entry indexes the original node of pos.

    Returns
    -------
    MeshQuadrature
        Cell areas, revolved cell volumes and corresponding nodal
        shape-function integrationn weights

    Notes
    -----
    Uses a tensor-product 2x2 gauss rule with a bilinear maping.

    The physical axisymmetric volume element is:

        dV = 2*pi*r*dr*dz

    For nodal values f_i, the approximate volume integral is:
        
        integral(f dV) = sum_i node_volume_weights[i] * f_i

    Dividing by totla_volume produces a volume_weighted mean.

    This routine does not read or modify CFD solution fields.
    """

    xy: FloatArray = np.asarray(
        pos,
        dtype=np.float64,
    )

    cell_ids: NDArray[np.generic] = np.asarray(cells)

    # -------------------------------------------------
    # 1. Validate coordinates
    # -------------------------------------------------

    if (
        xy.ndim != 2 or 
        xy.shape[1] != 2 or
        len(xy) == 0
    ):
        raise ValueError(
            "pos must have non-empty shape (N, 2)."
        )

    if not np.isfinite(xy).all():
        raise ValueError(
            "post must contain only finite coordinates."
        )

    if np.any(xy[:, 0] < 0.0):
        raise ValueError(
            "Axisymmetric radial coordinates must be >= 0."
        )

    # -------------------------------------------------
    # 2. Validate cell connectivity
    # -------------------------------------------------

    if (
        cell_ids.ndim != 2 or 
        cell_ids.shape[1] != 4 or
        len(cell_ids) == 0
    ):
        raise ValueError(
            "cells must have non-empty shape (C, 4)."
        )

    if not np.issubdtype(cell_ids.dtype, np.integer):
        raise ValueError(
            "cells must contain integer node indices."
        )

    indices: IntArray = np.asarray(
        cell_ids, dtype=np.int64,
    )

    if (
        np.any(indices < 0) or 
        np.any(indices >= len(xy))
    ):
        raise ValueError(
            "cells contain node indices outside pos."
        )

    if np.any(
        np.diff(
            np.sort(indices, axis=1),
            axis=1,
        ) == 0
    ):
        raise ValueError(
            "A quadrilateral contains repeated node indices."
        )

    # -------------------------------------------------
    # 3. Extract quadrilateral vertices
    # -------------------------------------------------
    vertices: FloatArray = xy[indices]

    r_vertices: FloatArray = vertices[:, :, 0]
    z_vertices: FloatArray = vertices[:, :, 1]

    n_cells: int = indices.shape[0]
    n_nodes: int = xy.shape[0]

    # -------------------------------------------------
    # 4. Initialize integration arrays
    # -------------------------------------------------

    cell_area: FloatArray = np.zeros(
        n_cells,
        dtype=np.float64,
    )

    cell_volume: FloatArray = np.zeros(
        n_cells,
        dtype=np.float64,
    )

    local_area_weights: FloatArray = np.zeros(
        (n_cells, 4),
        dtype=np.float64,
    )

    local_volume_weights: FloatArray = np.zeros(
        (n_cells, 4),
        dtype=np.float64,
    )

    # Gauss-Legendre integration points
    points: tuple[float, float] = (
        -1.0 / np.sqrt(3.0),
        1.0 / np.sqrt(3.0),
    )

    # -------------------------------------------------
    # 5. Integrate over each quadrilateral
    # -------------------------------------------------

    for xi in points:
        for eta in points:

            n, dxi, deta = _shape_function(
                xi,
                eta,
            )

            # radius at quadrature points
            r_q: FloatArray = r_vertices @ n

            # geometric mmapping derivatives
            dr_dxi: FloatArray = r_vertices @ dxi
            dr_deta: FloatArray = r_vertices @ deta

            dz_dxi: FloatArray = z_vertices @ dxi 
            dz_deta: FloatArray = z_vertices @ deta

            # Jacobian determinant
            det_j: FloatArray = (
                dr_dxi * dz_deta - dr_deta * dz_dxi
            )

            if np.any(det_j <= 0.0):
                raise ValueError(
                    "Cell has a non-positive Jacobian: "
                    "check geoemtry and counterclockwise "
                    "vertex order."
                )

            # Meriodinal area element
            area_element: FloatArray = det_j

            # axisymmetric physical volume element
            volume_element: FloatArray = (
                2.0 * np.pi * r_q * det_j
            )

            # Integrate cell geoemtric measures
            cell_area += area_element
            cell_volume += volume_element

            # Integrate local nodal shape-function weights
            local_area_weights += (
                area_element[:, None] * n[None, :]
            )

            local_volume_weights += (
                volume_element[:, None] * n[None, :]
            )

    # -------------------------------------------------
    # 6. Accumulate local weights at original node IDs
    # -------------------------------------------------

    node_area_weights: FloatArray = np.zeros(
        n_nodes,
        dtype=np.float64,
    )

    node_volume_weights: FloatArray = np.zeros(
        n_nodes,
        dtype=np.float64,
    )

    np.add.at(
        node_area_weights,
        indices.ravel(),
        local_area_weights.ravel(),
    )

    np.add.at(
        node_volume_weights,
        indices.ravel(),
        local_volume_weights.ravel(),
    )

    return MeshQuadrature(
        cell_area=cell_area,
        cell_volume=cell_volume,
        node_area_weights=node_area_weights,
        node_volume_weights=node_volume_weights,
    )