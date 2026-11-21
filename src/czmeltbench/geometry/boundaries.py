"""
File Description: Geometric boundary identification and distance features.
This module works on physical node coordinates in the r-z plane.
It does not apply physical boundary conditions or modify CFD fields.

Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-21
Version: 1.0
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from czmeltbench.constants import (
    BOUNDARY_NAMES,
    DEFAULT_GEOMETRY,
    Geometry,
)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
BoundaryMasks: TypeAlias = dict[str, BoolArray]

def _validate_positions(pos: FloatArray) -> FloatArray:
    """Validate an array of mesh-node coordinates.

    Parameters
    ----------
    pos: FlaotArray
        Coordinates with shape (N, 2).
        Columns are [r, z] in meters.

    Returns
    -------
    FlaotArray
        Validated float64 coordinates.

    Raises
    ------
    ValueError
        if coordinates have an invalid shape, are empty or
        contain nonfinnite values.
    """

    arr = np.asarray(pos, dtype=np.float64)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "pos must have shape (N, 2)."
        )

    if len(arr) == 0:
        raise ValueError(
            "Coordinates array cannot be empty."
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            "Coordinates contain nonn-finite values."
        )

    return arr

def identify_boundaries(pos: FloatArray,
                        geometry: Geometry = DEFAULT_GEOMETRY,
                        *,
                        tol: float = 1e-8,) -> BoundaryMasks:
    """Identify geometruic boundary nodes.
    
    Boundary membership is represented using Boolean masks.
    Junction nodes may belong to mul;tiple boundaries.

    Parameters
    ----------
    pos : FlaotArray
        Node coordinates, shape (N, 2), in meters.

    geometry : Geometry, optional
        Domain dimensions and physical geometry.

    tol : flaot, default=1e-8
        Absolute geometric comparison tolerance in meters

    Returns
    -------
    BoundaryMasks
        Dictionary containing Boolean arrays of shape (N,).

        Keys:
            axis
            bottom
            wall
            crystal
            lid
            interior

    Notes
    -----
    Crystal interface:
        z = H and r <= R_crystal

    Outer lid:
        z = H and r >= R_crystal

    The junction at r = R_crystal belong sto both crystal and lid masks.
    """
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(
            "tol must be finite andd non-negative."
        )

    if not (0.0 < geometry.r_crystal < geometry.r_crucible
        and geometry.h_melt > 0.0):
        raise ValueError(
            "Invalid geometry dimensions."
        )
    arr = _validate_positions(pos)

    r: FloatArray = arr[:, 0]
    z: FloatArray = arr[:, 1]

    Rb: float = geometry.r_crucible
    Rc: float = geometry.r_crystal
    H : float = geometry.h_melt

    axis: BoolArray = np.isclose(
        r, 0.0, atol=tol, rtol=0.0
    )

    bottom: BoolArray = np.isclose(
        z, 0.0, atol=tol, rtol=0.0
    )

    wall: BoolArray = np.isclose(
        r, Rb, atol=tol, rtol=0.0
    )

    top: BoolArray = np.isclose(
        z, H, atol=tol, rtol=0.0
    )

    crystal: BoolArray = top & (r <= Rc + tol)

    lid: BoolArray = top & (r >= Rc - tol)

    on_boundary: BoolArray = (
        axis | bottom | wall | crystal | lid
    )

    interior: BoolArray = ~on_boundary

    return {
        "axis": axis,
        "bottom": bottom,
        "wall": wall,
        "crystal": crystal,
        "lid": lid,
        "interior": interior, 
    }

def boundary_features(pos: FloatArray,
                      geometry: Geometry = DEFAULT_GEOMETRY,
                      *,
                      tol: float = 1e-8) -> FloatArray:
    """Convert boundary memberships into multi-hot features.

    Parameters
    ----------
    pos : FloatArray
        Node coordinates, shape (N, 2).

    geometry : Geometry, optional 
        Physical domain geometry

    tol : flaot, default=1e-8
        Boundary detection tolerances in mters.

    Returns
    -------
    FlaotArray
        Boundary features with shape (N, 5)

        columns:
            [axis, bottom, wall,  crystal, lid]

        Values are 0.0 or 1.0

    Notes
    -----
    Interior nodes have all-zero features.
    Junction nodes may have multiple active features.
    """

    masks: BoundaryMasks = identify_boundaries(pos=pos,
                                               geometry=geometry,
                                               tol=tol)

    features: FloatArray = np.column_stack([
        masks[name] for name in BOUNDARY_NAMES
    ]).astype(np.float64)

    return features

def distance_features(pos: FloatArray,
                      geometry: Geometry = DEFAULT_GEOMETRY) -> FloatArray:
    """CCalcualte distances to the five bounndary segments.

    Parameters
    ----------
    pos : FloatArray
        Node coordinates with shape (N, 2), in [m].

    geometry : geometry, optioinal
        Physical domain geometry.

    Returns
    -------
    FloatArray
        Distance features with shape (N, 5), in [m]

        columns:
            [d_axis, d_bottom, d_wall, d_crystal, d_lid]

    Notes
    -----
    The distances are Euclidean distances in  teh 2D r-z plane.
    Crystal and lid distances are calculated to their respective finite 
    top boundary segments.
    """

    arr = _validate_positions(pos)

    r: FloatArray = arr[:, 0]
    z: FloatArray = arr[:, 1]

    Rb: float = geometry.r_crucible
    Rc: float = geometry.r_crystal
    H: float = geometry.h_melt

    if not (0.0 < Rc < Rb and H > 0.0):
        raise ValueError("Invalid geometry dimensions.")

    if np.any(
        (r < -1e-8) |
        (r > Rb + 1e-8) |
        (z < -1e-8) |
        (z > H + 1e-8)
    ):

        raise ValueError(
            "Coordinates fall outside the physical domain."
        )

     # Distance to the symmetry axis.
    d_axis: FloatArray = np.abs(r)

    # Distance to the bottom.
    d_bottom: FloatArray = np.abs(z)

    # Distance to the crucible sidewall.
    d_wall: FloatArray = np.abs(Rb - r)

    # Distance to crystal top-boundary segment:
    # r in [0, Rc], z = H.
    d_crystal: FloatArray = np.hypot(
        np.maximum(r - Rc, 0.0),
        H - z,
    )

    # Distance to outer-lid segment:
    # r in [Rc, Rb], z = H.
    d_lid: FloatArray = np.hypot(
        np.maximum(Rc - r, 0.0),
        H - z,
    )

    features: FloatArray = np.column_stack([
        d_axis,
        d_bottom,
        d_wall,
        d_crystal,
        d_lid
    ])

    return features

