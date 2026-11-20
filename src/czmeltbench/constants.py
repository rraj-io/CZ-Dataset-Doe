"""
Physical constants and dataset definitions
"""

GEOMETRY = {
    "R_crucible": 0.30,
    "R_crystal": 0.18,
    "H_melt": 0.15,
    "T_crystal": 1685.0,
    "T_bottom_axis": 1700.0,
}

COORDINATE_COLUMNS = (
    "r",
    "z",
)

PARAMETER_COLUMNS = (
    "crystal_rpm",
    "crucible_rpm",
    "t_hot",
)

FIELD_COLUMNS = (
    "u_r", 
    "u_z", 
    "u_swirl", 
    "p", 
    "T",
    )

BOUNDARY_NAMES = (
    "axis",
    "bottom",
    "wall",
    "crystal",
    "lid",
)

MESH_SIZES = {
    "coarse": 2091,
    "base": 8181,
    "fine": 32200,
}

FIELD_UNNITS = {
    "u_r": "m/s",
    "u_z": "m/s",
    "u_swirl": "m/s",
    "T": "K",
    "p": "Pa",
}