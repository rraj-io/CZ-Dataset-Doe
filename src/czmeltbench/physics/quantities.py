"""
File Description: Physical Quantities of interest for CZ CFD fields.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-24
Version: 1.0
"""
"""
The input field ordering is:
    [u_r, u_z, u_swirl, p, T]

All calculations use the original CFD nodal values.
No interpolation or boundary-value replacement is performed.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]


def _validate_fields(y: FloatArray) -> FloatArray:
    """Validate CFD nodal solution fields.

    Parameters
    ----------
    y : FloatArray
        CFD fields array wit hshape (N ,5).

    Returns
    -------
    FloatArray
        Validated float64 array.

    
    """

    fields: FloatArray = np.asarray(
        y, dtype=np.float64,
    )

    if fields.ndim != 2 or fields.shape[1] != 5:
        raise ValueError("y must have shape (N, 5).")
    
    if fields.shape[0] == 0:
        raise ValueError("y must contain at least one node.")

    if not np.isfinite(fields).all():
        raise ValueError("CFD fields contain non-finite values.")

    return fields


def _validate_volume_weights(
        volume_weights: FloatArray,
        num_nodes: int,
) -> FloatArray:
    """Validate node-wise azisymmetric integration weights.

    Parameters
    ----------
    volume_weights : FloatArray
        Nodal volume weights with shape (N,), in  m^3.

    num_nodes : int
        Expected number of CFD nodes.

    Returns
    -------
    FloatArray
        Validated float64 weights.

    """
    weights: FloatArray = np.asarray(
        volume_weights,
        dtype=np.float64,
    )

    if weights.shape != (num_nodes,):
        raise ValueError(f"volume_weights must have shape ({num_nodes},).")

    if not np.isfinite(weights).all():
        raise ValueError("Volume weights contain non-finite values.")

    if np.any(weights < 0.0):
        raise ValueError("volume weights cannot be negative.")

    if not np.isfinite(weights.sum()) or weights.sum() <= 0.0:
        raise ValueError("Total volume weight must be positive and finite.")

    return weights

def calculate_quantities(y: FloatArray,
                         *,
                         volume_weights: FloatArray | None = None
                         ) -> dict[str, float]:
    """Calculate physical quyantities of interest from CFD fields.

    Parameters
    ----------
    y : FloatArray
        Original CFD nodal fields with shape (N, 5).

        Column order:
            [u_r, u_z, u_swirl, p, T]

        Units:
            [m/s, m/s, m/s, Pa, K]

    volume_weights : FloatArray | None, optional
        nodal axisymmetric integration weights from 
        compute_mesh_quadrature(), shape (N,), in m^3.

        If None, only unweighted nodal quantities are rreturned.

    Returns
    -------
    dict[str, float]
        Dictionary of nodal quantities:
            max_mariodinal
            rms_meriodinal
            max_swirl
            max_total
            mean_T
            min_T
            max_T
            delta_T

        When Volume_weights is provided, also returns
            volume_mean_T
            volume_rms_meriodinal
            volume_rms_total

    Notes
    -----
    The maximum and minimum quantites are evaluated at the 
    original nodes.
    Volume-weighted statistics use the lumped nodal integration
    weights from the mesh quadrature.
    Volume-weighted RMS quantities approximate the integral of
    squared nodal vlaues. They are not identical to the exact
    integral of a bilinearly interpolated squared field.

    The exported Fluent node values are never modified.
    """

    fields: FloatArray = _validate_fields(y)

    u_r: FloatArray = fields[:, 0]
    u_z: FloatArray = fields[:, 1]
    u_swirl: FloatArray = fields[:, 2]
    temperature: FloatArray = fields[:, 4]
    #pressure: FloatArray = fields[:, 3]

    # 1. Velocity magnitudes
    meridional: FloatArray = np.hypot(u_r, u_z)

    total_speed: FloatArray = np.hypot(meridional, u_swirl)

    # 2. Unweighted descriptive qunatities
    quantities: dict[str, float] = {
        "max_meridional": float(np.max(meridional)),
        "rms_meridional": float(np.sqrt(np.mean(meridional**2))),
        "max_swirl": float(np.max(np.abs(u_swirl))),
        "max_total": float(np.max(total_speed)),
        "mean_T": float(np.mean(temperature)),
        "min_T": float(np.min(temperature)),
        "max_T": float(np.max(temperature)),
        "delta_T": float(np.max(temperature) - np.min(temperature)),
    }

    # 3. Optinal axisymmetric volume-weighted quantities
    if volume_weights is not None:

        weights: FloatArray = _validate_volume_weights(
            volume_weights=volume_weights,
            num_nodes=fields.shape[0],
        )

        quantities["volume_mean_T"] = float(
            np.average(temperature, weights=weights)
        )

        quantities["volume_rms_meridional"] = float(
            np.sqrt(
                np.average(
                    meridional**2, 
                    weights=weights,
                )
            )
        )

        quantities["volume_rms_total"] = float(
            np.sqrt(
                np.average(
                    total_speed**2, 
                    weights=weights,
                )
            )
        )

    return quantities