"""
File Description: Analytical BC for CZ melt flow.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-22
Version: 1.0
"""
"""
This module calculates prescribed boundary values from geometry nad
operating parameters.

The CFD target fields are never modified.
"""

import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass
from typing import TypeAlias

from czmeltbench.constants import (
    DEFAULT_GEOMETRY,
    FIELD_COLUMNS,
    Geometry,
)

from czmeltbench.geometry.boundaries import (
    identify_boundaries,
)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class BoundaryConditions:
    """Prescribed bounndary values and associated masks.

    Attributes
    ----------
    values : FlaotArray
        Prescribed field vlaues with shape (N, 5).

        Columns:
            [u_r, u_z, u_swirl, p, T]

        Units:
            [m/s, m/s, m/s, Pa, K]

        Val;ues outside prescribed locations are zero placeholders and 
        must not be trated as a known physical field values.

    mask : BoolArray
        Boolean array with sahpe (N, 5).

        True indicates an explicitly prescribed vlaue. 
        False indicates that the value is not prescribed.
    """

    values: FloatArray
    mask: BoolArray

    @property
    def num_nodes(self) -> int:
        """REturn the number of nodes
        """
        return int(self.values.shape[0])

    @property
    def num_fields(self) -> int:
        """Return the number of physical fields.
        """
        return int(self.values.shape[1])

    def prescribed_values(self,
                          field: str,) -> FloatArray:
        """Return prescribed valeus of a given field

        Parameters
        ----------
        field : str
            One out of [u_r, u_z, u_swirl, p, T]

        Returns
        -------
        FlaotArray
            1D array containing only the prescribed vlaues of field.
        """

        if field not in FIELD_COLUMNS:
            raise ValueError(
                f"Unnknown field: {field}"
            )

        index: int = FIELD_COLUMNS.index(field)

        return self.values[self.mask[:, index], index].copy()

    def prescribed_indices(self,
                           field: str) -> NDArray[np.int64]:
        """Return original node indices with prescribed values.
        """
        if field not in FIELD_COLUMNS:
            raise ValueError(
                f"Unknown field: {field}"
            )

        index: int = FIELD_COLUMNS.index(field)

        return np.flatnonzero(self.mask[:, index])


def rpm_to_rad_per_sec(rpm: float) -> float:
    """Converts rotational spped from rpm to rad/sec.

    Parameters
    ----------
    rpm : float
        Signed rotational speed in rev/min

    Returns
    -------
    float
        Angular velocity in rad/sec.
    """

    if not np.isfinite(rpm):
        raise ValueError("Rotational speed must be finite.")

    return float(rpm * (2.0 * np.pi / 60.0))


def _validate_inputs(pos: FloatArray,
                     mu: FloatArray,
                     geometry: Geometry,) -> tuple[FloatArray, FloatArray]:
    """Validate node coordinates and operating conditions
    """
    coordinates = np.asarray(pos, dtype=np.float64)
    parameters = np.asarray(mu, dtype=np.float64)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "pos must have shape (N, 2)."
        )

    if coordinates.shape[0] == 0:
        raise ValueError(
            "Atleast one mesh node is required."
        )

    if parameters.shape != (3,):
        raise ValueError(
            "mu must have sahpe (3,)."
        )

    if not np.isfinite(coordinates).all():
        raise ValueError(
            "Coordinates contain non-finite vlaues."
        )

    if not np.isfinite(parameters).all():
        raise ValueError(
            "Operating paramters contain non-finite values."
        )

    if not (0.0 < geometry.r_crystal < geometry.r_crucible
            and geometry.h_melt > 0.0):
        raise ValueError(
            "Invalid geometric dimensions."
        )

    r = coordinates[0]
    z = coordinates[1]

    tol = 1e-8

    if np.any(
        (r < -tol) |
        (r > geometry.r_crucible + tol) |
        (z < -tol) |
        (z > geometry.h_melt + tol)
    ):
        raise ValueError(
            "Some nodes lie outside the physical domain."
        )

    return coordinates, parameters

def compute_boundary_conditions(pos: FloatArray,
                                mu: FloatArray,
                                geometry: Geometry = DEFAULT_GEOMETRY,
                                ) -> BoundaryConditions:
    """ Calculate analytical BCs.

    Parameters
    ----------
    pos: FlaotArray
        originnal  node coordinates with shape (N, 2).
        Columns: [r, z] in [m].

    mu : FlaotArray
        Opeating parameters with shape (3,).

        Order:
            [crystal_rpm, crucible_rpm, t_hot]

        Unnits:
            [rpm, rpm, K]

    geometry : Geometry, optiional
        Domain geometry and thermal reference values.

    Returns
    -------
    BoundaryConditions
        Oject containing:
        values : FloatArray
            prescribed values with shape (N, 5).

        mask : BoolArray
            prescription masks with shape (N ,5).

    Notes
    -----
    The field ordering is :
        [u_r, r_z, u_swirl, p, T]

    The symmetry-axis derivative conditions are not represented 
    in the Dirichlet-value array.

    A pressure reference p=0 is enncoded at the origin.

    At shared boundary junctions, analytical profiles have 
    matchinng endpoint values.
    """

    coordinates, parameters = _validate_inputs(pos=pos,
                                               mu=mu,
                                               geometry=geometry)

    r: FloatArray = coordinates[:, 0]

    Rb: float = geometry.r_crucible
    Rc: float = geometry.r_crystal

    Tc: float = geometry.t_crystal
    T0: float = geometry.t_bottom_axis

    crystal_rpm: float = float(parameters[0])
    crucible_rpm: float = float(parameters[1])
    T_hot: float = float(parameters[2])

    omega_c: float = rpm_to_rad_per_sec(crystal_rpm)
    omega_b: float = rpm_to_rad_per_sec(crucible_rpm)

    masks = identify_boundaries(pos=coordinates,
                                geometry=geometry,)

    N: int = coordinates.shape[0]

    bc_values: FloatArray = np.zeros(
        (N, len(FIELD_COLUMNS)),
        dtype=np.float64,
    )

    bc_mask: BoolArray = np.zeros(
        (N, len(FIELD_COLUMNS)),
        dtype=np.bool_,
    )

    #-------------------------------------------
    # 1. None-penetrating and nonn-slip BCs
    #--------------------------
    solid: BoolArray = (
        masks["bottom"] |
        masks["wall"] |
        masks["crystal"] |
        masks["lid"]
    )

    # prescribed radial and axial velocities = 0
    bc_mask[solid, 0] = True
    bc_mask[solid, 1] = True

    #-------------------------------------------
    # 2. Crucible bottom
    #--------------------------
    m: BoolArray = masks["bottom"]

    bc_values[m, 2] = omega_b * r[m]

    bc_values[m, 4] = T0 + (T_hot - T0) * r[m] / Rb

    bc_mask[m, 2] = True
    bc_mask[m, 4] = True

    #-------------------------------------------
    # 3. Crucible sidewall
    #--------------------------
    m = masks["wall"]

    bc_values[m, 2] = omega_b * Rb
    bc_values[m, 4] = T_hot

    bc_mask[m, 2] = True
    bc_mask[m, 4] = True

    #-------------------------------------------
    # 4. Crystal-melt interface
    #--------------------------
    m = masks["crystal"]

    bc_values[m, 2] = omega_c * r[m]
    bc_values[m, 4] = Tc

    bc_mask[m, 2] = True
    bc_mask[m, 4] = True    

    #------------------------------------
    # 5. Outer melt lid
    #-----------------------------------------
    m = masks["lid"]

    s: FloatArray = (
        (r[m] - Rc) / (Rb - Rc)
    )

    s = np.clip(s, 0.0, 1.0)

    bc_values[m, 2] = (
        (1.0 - s) * omega_c * Tc + s * omega_b * Rb
    )

    bc_values[m, 4] = (
        (1.0 - s) * Tc + s * T_hot
    )

    bc_mask[m, 2] = True
    bc_mask[m, 4] = True

    #------------------------------------
    # 6. Symmetry Axis
    #------------------------------------

    m = masks["axis"]

    bc_values[m, 0] = 0.0
    bc_values[m, 2] = 0.0

    bc_mask[m, 0] = True
    bc_mask[m, 2] = True

    # Axial velocity and Temp are not assigned
    # Dirichlet values on teh axis.

    #------------------------------------
    # 7. Pressure guage reference
    #------------------------------------

    origin: BoolArray = (
        masks["axis"] & masks["bottom"]
    )

    bc_values[origin, 3] = 0.0
    bc_mask[origin, 3] = True

    return BoundaryConditions(
        values=bc_values,
        mask=bc_mask
    )