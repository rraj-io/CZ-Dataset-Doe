"""
File Description: Native-mesh CFD Field Visualization for CZmeltBench
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-24
Version: 1.0
"""

"""
Uses Logicalmesh to preserve original CFD node orderinng.

No interolation, smoothing, bounddary-value replacement, or pressure-gauge
adjustment is performed.
"""

from typing import Literal, Protocol, Sequence, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.contour import QuadContourSet
from numpy.typing import NDArray

from czmeltbench.constants import FIELD_COLUMNS
from czmeltbench.geometry.mesh import (
    LogicalMesh,
    build_logical_mesh,
)


FloatArray: TypeAlias = NDArray[np.float64]

FieldName: TypeAlias = Literal[
    "u_r",
    "u_z",
    "u_swirl",
    "p",
    "T",
    "meridional_speed",
    "total_speed",
]


FIELD_LABELS: dict[str, str] = {
    "u_r": "Radial velocity [m/s]",
    "u_z": "Axial velocity [m/s]",
    "u_swirl": "Swirl velocity [m/s]",
    "p": "Pressure [Pa]",
    "T": "Temperature [K]",
    "meridional_speed": "Meridional speed [m/s]",
    "total_speed": "Total speed [m/s]",
}


class SimulationReader(Protocol):
    """Required interfacce for loading original CFD arrays.
    """

    def load_arrays(self,
                    case_id: str
                    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Returns pos (N, 2), mu (3,) and y (N, 5).
        """
        ...


def _validate_fields(pos: FloatArray,
                     y: FloatArray,
                     ) -> tuple[FloatArray, FloatArray]:
    """Vallidate original nodal coordinates and CFD fields.
    """ 

    coordinates: FloatArray = np.asarray(
        pos, dtype=np.float64,
    )

    fields: FloatArray = np.asarray(
        y, dtype=np.float64,
    )

    if (
        coordinates.ndim != 2 or
        coordinates.shape[1] != 2 or
        coordinates.shape[0] == 0
    ):
        raise ValueError("pos must be a non-empty shape (N, 2).")

    if fields.shape != (len(coordinates), 5):
        raise ValueError("y must have shape (N, 5).")

    if not (
        np.isfinite(coordinates).all() and np.isfinite(fields).all()
    ):
        raise ValueError("Coordinates and fields must be finite.")

    return coordinates, fields


def _extract_field(y: FloatArray,
                   field: FieldName,
                   ) -> FloatArray:
    """Extract a scalar field in original CFD node ordering.

    Parameters
    ----------
    y : FloatArray
        CFD fields, shape (N, 5).

    field : FieldName
        original field name or derived velocity magnitude.

    Returns
    -------
    FloatArray
        Scalar nodal values with shape (N,).
    """

    if field in FIELD_COLUMNS:
        index: int = FIELD_COLUMNS.index(field)
        return y[:, index].copy()

    if field == "meridional_speed":
        return np.hypot(y[:, 0], y[:, 1])

    if field == "total_speed":
        return np.sqrt(
            y[:, 0]**2 + y[:, 1]**2 + y[:, 2]**2
        )

    raise ValueError("Unknown CFD field: {field}")


def _color_limits(values: Sequence[FloatArray],
                  limits: tuple[float, float] | None,
                  ) -> tuple[float, float]:
    """Returns valid color limits across one or more arrays.
    """
    if limits is not None:
        low, high = limits

        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise ValueError("Color limits must be finite with min < max.")

        return float(low), float(high)

    low = min(float(np.min(v)) for v in values)
    high = max(float(np.max(v)) for v in values)

    # matplotlib contourf needs distinct levels
    if low == high:
        padding = max(abs(low) * 1e-6, 1e-9)
        low -= padding
        high += padding


    return low, high


def _draw_field(ax: Axes,
                mesh: LogicalMesh,
                values: FloatArray,
                *,
                cmap: str,
                limits: tuple[float, float],
                levels: int,
                show_colorbar: bool = True,
                ) -> QuadContourSet:
    """Draw a field on its original non-uniform logical mesh.
    """

    if levels < 2:
        raise ValueError("levels must be at least 2.")

    field_grids: FloatArray = mesh.reshape_field(values)

    color_levels: FloatArray = np.linspace(
        limits[0],
        limits[1],
        levels, dtype=np.float64,
    )

    contour = ax.contourf(mesh.r,
                          mesh.z,
                          field_grids,
                          levels=color_levels,
                          cmap=cmap,
                          extend="neither",)

    if show_colorbar:
        ax.figure.colorbar(
            contour,
            ax=ax,
            shrink=0.85,
        )

    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="box")

    return contour


def plot_case(reader: SimulationReader,
              case_id: str,
              field: FieldName = "T",
              *,
              cmap: str = "viridis",
              limits: tuple[float, float] | None  = None,
              levels: int = 40,
              ) -> tuple[Figure, Axes]:
    """Visualize one original CFD simulation field.

    Parameters
    ----------
    reader : SimulationReader
        Reader for original CFd simualtions
    
    case_id : str
        CFD simulation identifier

    field : FieldName, default='T'
        Field to visualize
    
    cmap : str, default='viridis'
        Matplotlib colormap.

    limits : tuple[float, float] | None
        Optional color scale limits

    levels : int, default=40
        Number of contour levels.

    Returns
    -------
    tuple[Figure, Axes]
        matplotlib figure and plot axes.
    """

    pos, _, y = reader.load_arrays(case_id)
    coordinates, fields = _validate_fields(pos, y)

    mesh: LogicalMesh = build_logical_mesh(coordinates)
    values: FloatArray = _extract_field(fields, field)

    color_limits = _color_limits([values], limits)

    fig, ax = plt.subplots(figsize=(8, 5))

    _draw_field(ax=ax,
                mesh=mesh,
                values=values,
                cmap=cmap,
                limits=color_limits,
                levels=levels,)

    ax.set_title(f"{case_id} - {FIELD_LABELS[field]}")
    #fig.tight_layout()

    return fig, ax



def compare_cases(reader: SimulationReader,
                  case_ids: Sequence[str],
                  field: FieldName = "T",
                  *,
                  cmap: str = "viridis",
                  limits: tuple[float, float] | None = None,
                  levels: int = 40,
                  ) -> tuple[Figure, NDArray[np.object_]]:
    """Compare a scalar field across multiple simulations.

    Each case is plotted on its own original CFD mesh. All panes use
    identical color limits.

    Parameters
    ----------
    reader : SimulationReader
        CFD simulation reader.

    case_ids : Sequence[str]
        Two or more simulation IDs.

    field : FieldName, default='T'
        Field to compare
    
    cmap : str
        Shared matplotlib colormap.
    
    limits : tuple[float, float] | None
        Shared color limits. If None, determin limits from all selected
        simulations.

    levels: int
        Number of contour levels.

    Returns
    -------
    tuple[Figure, NDArray[np.object_]]
        Figure and 1D array of axes.
    
    Notes
    -----
    Comparisons are visual only, no difference is calculated between 
    operating conditions.

    Different mesh resolutions are allowed because every panel uses its 
    own native mesh without interpolation.
    
    """

    ids: list[str] = list(case_ids)

    if len(ids) < 2:
        raise ValueError("Provide at least two simulation IDs.")

    if len(ids) != len(set(ids)):
        raise ValueError("Simulation IDs must be unique.")

    meshes: list[LogicalMesh] = []
    field_values: list[FloatArray] = []

    for case_id in ids:
        pos, _, y = reader.load_arrays(case_id)
        coordinates, fields = _validate_fields(pos, y)

        meshes.append(build_logical_mesh(coordinates))
        field_values.append(_extract_field(fields, field))

    color_limits = _color_limits(field_values,
                                 limits)

    fig, axes = plt.subplots(1, 
                             len(ids),
                             figsize=(5 * len(ids), 4.5),
                             squeeze=False,)

    flat_axes: NDArray[np.object_] = axes.ravel()

    contour = None

    for ax, case_id, mesh, values in zip(
        flat_axes, ids, meshes, field_values,
    ):
        contour = _draw_field(ax=ax,
                              mesh=mesh,
                              values=values,
                              cmap=cmap,
                              limits=color_limits,
                              levels=levels,
                              show_colorbar=False,
                              )

        ax.set_title(case_id)

    # single colorbar for all simulations
    if contour is not None:
        colorbar = fig.colorbar(
            contour,
            ax=list(flat_axes),
            orientation="vertical",
            fraction=0.025,
            pad=0.03,
            aspect=30,
        )

        colorbar.set_label(FIELD_LABELS[field])

    fig.suptitle(
        f"CFD simulation comparison - {FIELD_LABELS[field]}",
        fontsize=14,
    )

    # reserve sapcce for figure titles
    fig.subplots_adjust(top=0.85,
                        wspace=0.25,)

    #fig.tight_layout()

    return fig, flat_axes


def plot_prediction_errors(reader: SimulationReader,
                          case_id: str,
                          y_pred: FloatArray,
                          prediction_pos: FloatArray,
                          field: FieldName = 'T',
                          *,
                          coordinate_tolerance: float = 1e-12,
                          cmap: str = "viridis",
                          error_cmap: str = "magma",
                          levels: int = 40,
                          ) -> tuple[Figure, NDArray[np.object_]]:
    """Compare CFD reference, prediction and absolute field error.

    Parameters
    ----------
    reader : SimulationReader
        CFD reference datta reader.

    case_id : str
        Original CFDD simulation ID.

    y_pred : FloatArray
        Model predictions, shape (N, 5), in original CFD fields order.

    prediction_pos : FloatArray
        Coordinates corresponding to predicted rows, shape (N, 2).

    field : FieldName, default='T'
        Field to compare.

    ccoordinate_tolerance : float
        Absolute coordinate matching tolerance in meters.

    cmap : str
        Shared reference and predition colormap.

    error_cmap : str
        Colormap for non-negative absolute error.

    levels : int
        Numer of contour levels.

    Returns
    -------
    tuple[Figure, NDArray[np.object_]]
        Figure and three plotting axes.

    Notes
    -----
    Predictins must have the same physical coordinates, node order
    and field order as the CFD reference.

    For meridional_speed and total_speed, the error is teh absolute 
    difference of teh two speed magnitudes, not the magnitude of the 
    vector velocity errror.

    Original Fluent target values are never changed.
    """

    pos, _, y_true = reader.load_arrays(case_id)

    coordinates, reference = _validate_fields(pos, y_true)

    predicted_pos, prediction = _validate_fields(prediction_pos, y_pred)

    if (
        not np.isfinite(coordinate_tolerance) 
        or coordinate_tolerance < 0.0
    ):
        raise ValueError(
            "coordinate_tolerance must be non-negative and finite."
        )

    if (
        predicted_pos.shape != coordinates.shape 
        or not np.allclose(
            coordinates,
            predicted_pos,
            rtol=0.0, atol=coordinate_tolerance,
        )
    ):
        raise ValueError("Prediction coordinates do not match.")

    mesh: LogicalMesh = build_logical_mesh(coordinates)

    reference_values = _extract_field(reference, field)

    predicted_values = _extract_field(prediction, field)

    absolute_error: FloatArray = np.abs(
        predicted_values - reference_values
    )

    shared_limits = _color_limits(
        [reference_values, predicted_values],
        None,
    )

    error_max: float = float(np.max(absolute_error))

    error_limits: tuple[float, float] = (
        0.0,
        error_max if error_max > 0.0 else 1e-9,
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4.5),
        squeeze=False,
    )

    flat_axes: NDArray[np.object_] = axes.ravel()

    panels = (
        ("CFD reference", reference_values, cmap, shared_limits),
        ("ML prediction", predicted_values, cmap, shared_limits),
        ("Absolute error", absolute_error, error_cmap, error_limits),
    )

    contours = []

    for ax, (title, values, panel_cmap, panel_limits) in zip(
        flat_axes, panels,
    ):
        contour = _draw_field(
            ax,
            mesh,
            values,
            cmap=panel_cmap,
            limits=panel_limits,
            levels=levels,
            show_colorbar=False,
        )

        contours.append(contour)
        ax.set_title(title)

    # One shared colorbar for CFD reference and ML prediction.
    field_colorbar = fig.colorbar(
        contours[0],
        ax=list(flat_axes[:2]),
        fraction=0.035,
        pad=0.03,
    )

    field_colorbar.set_label(FIELD_LABELS[field])

    # separate colorbar for absolute error.
    error_colorbar = fig.colorbar(
        contours[2],
        ax=flat_axes[2],
        fraction=0.035,
        pad=0.03,
    )

    error_colorbar.set_label(f"Absolute error - {FIELD_LABELS[field]}")

    fig.suptitle(f"{case_id} - {FIELD_LABELS[field]}", fontsize=14)

    fig.subplots_adjust(top=0.85,
                        wspace=0.25)

    return fig, flat_axes