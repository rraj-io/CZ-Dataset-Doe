"""
File Description: Tests for PyVista mesh conversion and ParaView export.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-26
Version: 1.0
"""


from pathlib import Path

import numpy as np
import pyvista as pv
import pytest
from numpy.typing import NDArray

from czmeltbench.visualization.vtk import (
    create_vtk_grid,
    export_case_vtu,
    export_prediction_comparison,
)


FloatArray = NDArray[np.float64]


class SyntheticReader:
    """Minimal CFD reader for VTK export tests."""

    def load_arrays(
        self,
        case_id: str,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return a structured synthetic CFD simulation."""
        if case_id != "synthetic_000":
            raise KeyError(case_id)

        r = np.linspace(0.0, 0.30, 4)
        z = np.linspace(0.0, 0.15, 3)
        R, Z = np.meshgrid(r, z)

        pos: FloatArray = np.column_stack((
            R.ravel(), Z.ravel()
        ))

        mu: FloatArray = np.array(
            [10.0, -5.0, 1750.0],
            dtype=np.float64,
        )

        y: FloatArray = np.zeros(
            (len(pos), 5),
            dtype=np.float64,
        )

        y[:, 0] = pos[:, 0]
        y[:, 1] = pos[:, 1]
        y[:, 2] = 0.1
        y[:, 4] = 1700.0 + 100.0 * pos[:, 0]

        return pos, mu, y


@pytest.fixture
def reader() -> SyntheticReader:
    """Return the synthetic reader."""
    return SyntheticReader()


def test_grid_preserves_native_mesh(
    reader: SyntheticReader,
) -> None:
    """VTK conversion preserves node IDs and quad connectivity."""
    pos, _, y = reader.load_arrays("synthetic_000")

    grid = create_vtk_grid(pos, y)

    assert grid.n_points == 12
    assert grid.n_cells == 6

    assert np.all(
        grid.celltypes == pv.CellType.QUAD
    )

    np.testing.assert_array_equal(
        grid.points[:, :2],
        pos,
    )

    np.testing.assert_array_equal(
        grid.point_data["T"],
        y[:, 4],
    )

    np.testing.assert_array_equal(
        grid.point_data["original_node_id"],
        np.arange(len(pos)),
    )


def test_paraview_export(
    reader: SyntheticReader,
    tmp_path: Path,
) -> None:
    """Exported VTU can be opened again without data loss."""
    path = export_case_vtu(
        reader,
        "synthetic_000",
        tmp_path / "synthetic_000.vtu",
    )

    assert path.is_file()

    grid = pv.read(path)

    assert grid.n_points == 12
    assert grid.n_cells == 6
    assert "T" in grid.point_data
    assert "meridional_speed" in grid.point_data


def test_prediction_error_export(
    reader: SyntheticReader,
    tmp_path: Path,
) -> None:
    """Export predicted and original fields with known errors."""
    pos, _, y = reader.load_arrays("synthetic_000")

    predicted: FloatArray = y.copy()
    predicted[:, 4] += 2.0

    paths = export_prediction_comparison(
        reader,
        "synthetic_000",
        y_pred=predicted,
        prediction_pos=pos,
        output_dir=tmp_path,
    )

    assert len(paths) == 3
    assert all(path.is_file() for path in paths)

    error_grid = pv.read(paths[2])

    np.testing.assert_allclose(
        error_grid.point_data["error_T"],
        2.0,
    )


def test_shuffled_prediction_rejected(
    reader: SyntheticReader,
    tmp_path: Path,
) -> None:
    """Wrong node ordering must not pass silently."""
    pos, _, y = reader.load_arrays("synthetic_000")

    order = np.arange(len(pos))[::-1]

    with pytest.raises(ValueError, match="node order"):
        export_prediction_comparison(
            reader,
            "synthetic_000",
            y_pred=y[order],
            prediction_pos=pos[order],
            output_dir=tmp_path,
        )
