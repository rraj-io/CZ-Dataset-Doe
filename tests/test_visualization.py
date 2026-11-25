"""
File Description: Unit test fro CZMeltBench visualization tool.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-25
Version: 1.0
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from czmeltbench.visualization.fields import (
    compare_cases,
    plot_case,
    plot_prediction_errors,
)


FloatArray = NDArray[np.float64]


class SyntheticReader:
    """Small structured CFD dataset used for plot tests."""

    def __init__(self) -> None:
        """Construct a 3x4 meridional mesh."""
        r = np.linspace(0.0, 0.30, 4)
        z = np.linspace(0.0, 0.15, 3)

        R, Z = np.meshgrid(r, z)

        self.pos: FloatArray = np.column_stack((
            R.ravel(),
            Z.ravel(),
        ))

    def load_arrays(
        self,
        case_id: str,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return a synthetic CFD case."""
        if case_id not in ("case_000", "case_001"):
            raise KeyError(case_id)

        offset = 0.0 if case_id == "case_000" else 10.0

        mu: FloatArray = np.array(
            [10.0, -5.0, 1750.0 + offset],
            dtype=np.float64,
        )

        y: FloatArray = np.zeros(
            (len(self.pos), 5),
            dtype=np.float64,
        )

        y[:, 0] = self.pos[:, 0]
        y[:, 1] = self.pos[:, 1]
        y[:, 2] = -0.1
        y[:, 3] = 5.0
        y[:, 4] = (
            1700.0
            + 100.0 * self.pos[:, 0]
            + offset
        )

        return self.pos.copy(), mu, y


@pytest.fixture
def reader() -> SyntheticReader:
    """Provide a synthetic CFD reader."""
    return SyntheticReader()


def test_plot_case(
    reader: SyntheticReader,
) -> None:
    """One simulation should produce a valid plot."""
    fig, ax = plot_case(
        reader,
        "case_000",
        field="T",
    )

    assert fig is not None
    assert ax is not None
    assert "case_000" in ax.get_title()

    plt.close(fig)


def test_compare_cases(
    reader: SyntheticReader,
) -> None:
    """Two cases should produce two plotting axes."""
    fig, axes = compare_cases(
        reader,
        ["case_000", "case_001"],
        field="T",
    )

    assert len(axes) == 2
    assert axes[0].get_title() == "case_000"
    assert axes[1].get_title() == "case_001"

    plt.close(fig)


def test_prediction_error_plot(
    reader: SyntheticReader,
) -> None:
    """Reference, prediction and error panels should be created."""
    pos, _, y_true = reader.load_arrays("case_000")

    y_pred = y_true.copy()
    y_pred[:, 4] += 2.0

    fig, axes = plot_prediction_errors(
        reader,
        "case_000",
        y_pred=y_pred,
        prediction_pos=pos,
        field="T",
    )

    assert len(axes) == 3
    assert axes[0].get_title() == "CFD reference"
    assert axes[1].get_title() == "ML prediction"
    assert axes[2].get_title() == "Absolute error"

    plt.close(fig)


def test_shuffled_prediction_rejected(
    reader: SyntheticReader,
) -> None:
    """Prediction ordering must match original CFD coordinates."""
    pos, _, y = reader.load_arrays("case_000")

    order = np.arange(len(pos))[::-1]

    with pytest.raises(ValueError, match="node ordering"):
        plot_prediction_errors(
            reader,
            "case_000",
            y_pred=y[order],
            prediction_pos=pos[order],
            field="T",
        )


def test_unknown_field_rejected(
    reader: SyntheticReader,
) -> None:
    """Unknown physical field names must be rejected."""
    with pytest.raises(ValueError, match="Unknown CFD field"):
        plot_case(
            reader,
            "case_000",
            field="invalid",  # type: ignore[arg-type]
        )


def test_duplicate_cases_rejected(
    reader: SyntheticReader,
) -> None:
    """Repeated cases should not appear in a comparison."""
    with pytest.raises(ValueError, match="unique"):
        compare_cases(
            reader,
            ["case_000", "case_000"],
        )


def test_original_fields_unchanged(
    reader: SyntheticReader,
) -> None:
    """Visualization should never mutate source CFD arrays."""
    pos, _, y = reader.load_arrays("case_000")

    original = y.copy()

    fig, _ = plot_prediction_errors(
        reader,
        "case_000",
        y_pred=y,
        prediction_pos=pos,
    )

    np.testing.assert_array_equal(y, original)

    plt.close(fig)
