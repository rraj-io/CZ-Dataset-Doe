"""
File Description: Constant values
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-21
Version: 1.0
"""

"""Constants and physical geometry definitions for CZMeltBench."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    """Physical geometry and thermal reference values.

    All lengths are in metres and temperatures in kelvin.
    """

    r_crucible: float = 0.30
    r_crystal: float = 0.18
    h_melt: float = 0.15
    t_crystal: float = 1685.0
    t_bottom_axis: float = 1700.0


DEFAULT_GEOMETRY = Geometry()


COORDINATE_COLUMNS: tuple[str, ...] = (
    "r",
    "z",
)

PARAMETER_COLUMNS: tuple[str, ...] = (
    "crystal_rpm",
    "crucible_rpm",
    "t_hot",
)

FIELD_COLUMNS: tuple[str, ...] = (
    "u_r",
    "u_z",
    "u_swirl",
    "p",
    "T",
)

BOUNDARY_NAMES: tuple[str, ...] = (
    "axis",
    "bottom",
    "wall",
    "crystal",
    "lid",
)

MESH_SIZES: dict[str, int] = {
    "coarse": 2091,
    "base": 8181,
    "fine": 32200,
}

FIELD_UNITS: dict[str, str] = {
    "u_r": "m/s",
    "u_z": "m/s",
    "u_swirl": "m/s",
    "p": "Pa",
    "T": "K",
}