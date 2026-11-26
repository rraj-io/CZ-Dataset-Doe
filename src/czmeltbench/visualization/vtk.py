"""
File Description: VTK support and ParaView export for CZMeltBench
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-25
Version: 1.0
"""
"""
Creates a 2D quadrilateral mesh in the r-z plane.
All point data use the original CFD node ordering.

No interpolation or modificationn of physical fields occur.
"""

from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from czmeltbench.constants import FIELD_COLUMNS
from czmeltbench.geometry.mesh import build_logical_mesh
from czmeltbench.geometry.connectivity import build_connectivity


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class SimulationReader(Protocol):
    """Reader interface used by VTK export.
    """

    def load_arrays(self,
                    case_id: str,
                    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return coordinates, paramters, and CFd fields.
        """
        ...



def create_vtk_grid(pos: FloatArray,
                    y: FloatArray,
                    ) -> pv.UnstructuredGrid:
    """Create a native quadrilateral VTK mesh.

    Parameters
    ----------
    pos : FloatArray
        Node coordinates, shape (N, 2).
        Columns: [r, z], in meters.

    y : FloatArray
        Nodal solution fields, shape (N ,5).
        Columns : [u_r, u_z, u_swirl, p, T].

    Returns
    -------
    pv.UnstructuredGrid
        2D quadrilateral VTK grid containing all original CFD feilds
        and derived velocity magnitude.

    Notes
    -----
    VTK uses 3D point coordinates. We embed the 2D mesh into the XY 
    plane:
        X = r
        Y = z
        Z = 0

    The resulting geometry is an axisymmetric meridional cross-section,
    not a reconstructed 3D melt volume.
    """

    coordinates: FloatArray = np.asarray(
        pos, dtype=np.float64,
    )

    fields: FloatArray = np.asarray(
        y, dtype=np.float64,
    )

    if (coordinates.ndim != 2 or coordinates.shape[1] != 2
        or coordinates.shape[0] == 0):
        raise ValueError("pos must have shape (N, 2).")

    if fields.shape != (len(coordinates), 5):
        raise ValueError("y must have shape (N, 5).")

    if not (
        np.isfinite(coordinates).all()
        and np.isfinite(fields).all()
    ):
        raise ValueError("Coordinates and fields must be finite.")

    # Use the already validated native logical mesh.
    mesh = build_logical_mesh(coordinates)
    connectivity = build_connectivity(mesh)

    cell_ids: IntArray = connectivity.cells

    # VTK connectivity format:
    # [4, id0, id1, id2, id3] for each quad.
    vtk_cells: IntArray = np.column_stack((
        np.full(len(cell_ids), 4, dtype=np.int64),
        cell_ids,
    )).ravel()

    cell_types = np.full(len(cell_ids),
                         pv.CellType.QUAD,
                         dtype=np.uint8,)

    points: FloatArray = np.column_stack((
        coordinates[:, 0],
        coordinates[:, 1],
        np.zeros(len(coordinates), dtype=np.float64),
    ))

    grid = pv.UnstructuredGrid(vtk_cells,
                               cell_types,
                               points,)

    # preserve the original CFD point IDs
    grid.point_data["original_node_id"] = np.arange(
        len(coordinates), dtype=np.int64
    )

    # original scalar fields
    for index, field_name in enumerate(FIELD_COLUMNS):
        grid.point_data[field_name] = fields[:, index].copy()

    # Derived velocity magnitudes
    meridional_speed: FloatArray = np.hypot(
        fields[:, 0], fields[:, 1],
    )

    total_speed: FloatArray = np.hypot(
        meridional_speed, fields[:, 2],
    )

    grid.point_data["meridional_speed"] = meridional_speed
    grid.point_data["total_speed"] = total_speed

    # Velocity compoonents in physical cylindrical coordinates.
    # Stored as data vector, not a cartesian 3DD velocity.
    grid.point_data["velocity_cylindrical"] = fields[:, :3].copy()

    # meridional vector embeded into teh visualization XY plane
    grid.point_data["velocity_meridional"] = np.column_stack((
        fields[:, 0],
        fields[:, 1],
        np.zeros(len(coordinates), dtype=np.float64),
    ))

    grid.set_active_scalars("T")

    return grid



def export_case_vtu(reader: SimulationReader,
                    case_id: str,
                    output_path: str | Path,
                    ) -> Path:
    """Export one CFD simualtion to Paraview-compatible VTU.

    Parameters
    ----------
    reader : SimulationReader
        CFD daatset reader.

    case_id : str
        Simulation identifier, eg. 'sobol_000'

    output_path : str | Path
        Destination .vtu file.

    Retruns
    -------
    Path
        Path to the saved VTU file.
    
    """

    destination: Path = Path(output_path)

    if destination.suffix.lower() != "vtu":
        raise ValueError("Output file must have a .vtu extension.")

    pos, _, y = reader.load_arrays(case_id)

    grid = create_vtk_grid(pos, y)

    destination.parent.mkdir(parents=True, exist_ok=True,)

    grid.save(destination)

    return destination



def export_prediction_comparison(
    reader: SimulationReader,
    case_id: str,
    y_pred: FloatArray,
    prediction_pos: FloatArray,
    output_dir: str | Path,
    *,
    coordinate_tolerance: float = 1e-12,
) -> tuple[Path, Path, Path]:
    """Export reference, prediction and error meshes to ParaView.

    Parameters
    ----------
    reader : SimulationReader
        Original CFD simulation reader.

    case_id : str
        Simulation identifier.

    y_pred : FloatArray
        Predicted fields, shape (N, 5).

    prediction_pos : FloatArray
        Coordinates associated with predictions, shape (N, 2).

    output_dir : str | Path
        Directory for output VTU files.

    coordinate_tolerance : float, default=1e-12
        Maximum absolute coordinate difference in metres.

    Returns
    -------
    tuple[Path, Path, Path]
        Reference, prediction and absolute-error VTU paths.

    Notes
    -----
    The error grid stores absolute scalar errors.
    Velocity-magnitude errors compare speed magnitudes
    and are not vector-error magnitudes.

    No Fluent nodal values are overwritten.
    """
    pos, _, y_true = reader.load_arrays(case_id)

    coordinates: FloatArray = np.asarray(
        prediction_pos, dtype=np.float64
    )
    prediction: FloatArray = np.asarray(
        y_pred, dtype=np.float64
    )

    if (
        not np.isfinite(coordinate_tolerance)
        or coordinate_tolerance < 0.0
    ):
        raise ValueError(
            "coordinate_tolerance must be finite and nonnegative."
        )

    if (
        coordinates.shape != pos.shape
        or not np.isfinite(coordinates).all()
        or not np.allclose(
            coordinates,
            pos,
            rtol=0.0,
            atol=coordinate_tolerance,
        )
    ):
        raise ValueError(
            "Prediction coordinates do not match original node order."
        )

    if (
        prediction.shape != y_true.shape
        or not np.isfinite(prediction).all()
    ):
        raise ValueError(
            "Predicted fields must be finite with shape (N, 5)."
        )

    reference_grid = create_vtk_grid(pos, y_true)
    prediction_grid = create_vtk_grid(pos, prediction)

    error_grid = reference_grid.copy(deep=True)

    # Remove reference solution arrays from the error grid.
    error_grid.point_data.clear()
    error_grid.point_data["original_node_id"] = np.arange(
        len(pos), dtype=np.int64
    )

    for index, name in enumerate(FIELD_COLUMNS):
        error_grid.point_data[f"error_{name}"] = np.abs(
            prediction[:, index] - y_true[:, index]
        )

    for speed_name in ("meridional_speed", "total_speed"):
        error_grid.point_data[f"error_{speed_name}"] = np.abs(
            prediction_grid.point_data[speed_name]
            - reference_grid.point_data[speed_name]
        )

    error_grid.set_active_scalars("error_T")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    ref_path = destination / f"{case_id}_reference.vtu"
    pred_path = destination / f"{case_id}_prediction.vtu"
    error_path = destination / f"{case_id}_error.vtu"

    reference_grid.save(ref_path)
    prediction_grid.save(pred_path)
    error_grid.save(error_path)

    return ref_path, pred_path, error_path
