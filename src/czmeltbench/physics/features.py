"""
File Description: Architecture independent node feature connstruction
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-22
Version: 1.0
"""
"""
Features are derived only from:
    - spatial coordinates
    - simulation operating paramters
    - known geometry
    -prescribed analytical boundary conditions.

    Unknown CFD solution fields are never used as inputs.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from czmeltbench.constants import (
    BOUNDARY_NAMES,
    COORDINATE_COLUMNS,
    DEFAULT_GEOMETRY,
    FIELD_COLUMNS,
    PARAMETER_COLUMNS,
    Geometry,
)

from czmeltbench.geometry.boundaries import (
    boundary_features,
    distance_features,
)

from czmeltbench.physics.boundary_conditions import (
    compute_boundary_conditions
)

FloatArray: TypeAlias = NDArray[np.float64]

@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for constructing node-level ML features.

    Parameters
    ----------
    include_parameters : bool
        Repeat the golbal operating parameters at each node.

    include_boundaries : bool
        Innclude geometric boundary multi-hot indicators.

    include_bc_values : bool
        Include analytical boundary valeus and prescription masks.

    include_distances : bool
        Include distances to the five physical boundaries.
    """

    include_parameters: bool = True
    include_boundaries: bool = True
    include_bc_values: bool = False
    include_distances: bool = False

@dataclass(frozen=True)
class NodeFeatures:
    """Container for generated node features.

    Attributes
    ----------
    x : FloatAray
        Feature matrix with shape (N, F).

    columns : tuple[str, ...]
        Ordered names of all F features

    pos : FloatArray
        Originnal node coordinates, shape (N, 2).

    mu : FloatArray
        Global operating parameters, shape (3,).
    """

    x: FloatArray
    columns: tuple[str, ...]
    pos: FloatArray
    mu: FloatArray

    @property
    def num_nodes(self) -> int:
        """Returns teh number of nodes.
        """
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        """Return the number of node features.
        """
        return int(self.x.shape[1])

    def get_column(self,
                   name: str) -> FloatArray:
        """Retrieve one feature column by its name.

        Parameters
        ----------
        name : str
            Feature name

        Returns
        -------
        FlaotArray
            !D array of shape (N,).

        """
        if name not in self.columns:
            raise KeyError(
                f"Unknown feature: {name}"
            )

        index: int = self.columns.index(name)

        return self.x[:, index].copy()

    
def _validate_feature_inputs(pos: FloatArray,
                             mu: FloatArray,
                            ) -> tuple[FloatArray, FloatArray]:
    """Validate coordinates and operating parameters
    
    Parameters
    ----------
    pos : FlaotArray
        Node coordinates, shape (N, 2), in [m]
    
    mu : FloatArray
        Operating parameters, shape (3,).
        Order: crystal_rpm, crucible_rpm, t_hot.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Validated flaot64 coordinates and parameters.
    """
    coordinates: FloatArray = np.asarray(
        pos, dtype=np.float64
    )

    parameters: FloatArray = np.asarray(
        mu, dtype=np.float64
    )

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "pos must have shape (N, 2)."
        )

    if coordinates.shape[0] == 0:
        raise ValueError(
            "At least one node is required."
        )

    if parameters.shape != (3,):
        raise ValueError(
            "mu must have shape (3,)."
        )

    if not np.isfinite(coordinates).all():
        raise ValueError(
            "Coordinates contain nonn-finite values."
        )

    if not np.isfinite(parameters).all():
        raise ValueError(
            "Operating parameters conntainn non-finite values."
        )

    return coordinates, parameters


def build_node_features(pos: FloatArray,
                        mu: FloatArray,
                        *,
                        config: FeatureConfig = FeatureConfig(),
                        geometry: Geometry = DEFAULT_GEOMETRY,
                        ) -> NodeFeatures:
    """Construct ML input features for a CFD simulation.

    Parameters
    ----------
    pos : FlaotArray
        Original node coordinates, shape (N, 2).
        Columns: [r, z] in [m]
    
    mu : FloatArray
        Global parameters, shape (N,).
        Columns: [crystal_rpm, crucible_rpm, t_hot]
        Units: [rpom, rpm, K].

    config : FeatureConfig, optional 
        Selection of input feature groups.

    geometry : Geometry, optional
        Physical donain geometry.

    Returns
    -------
    NodeFeatures
        Node input matrix, feature names, coordinates and global operating
        paramters.

    Notes
    -----
    Feature remain in their original physical units.

    No CFD target files are used yet.
    No normalization or interpolationn is performed.

    Analytical BC values have zeros ar positions where their 
    associalted prescription mask is False.
    """

    coordinates, parameters = _validate_feature_inputs(
        pos=pos,
        mu=mu,
    )

    n_nodes: int = coordinates.shape[0]

    arrays: list[FloatArray] = [coordinates.copy()]

    names: list[str] = list(COORDINATE_COLUMNS)

    # 1. Global parameters, broadcast to all nodes.
    if config.include_parameters:
        repeated_parameters: FloatArray = np.broadcast_to(
            parameters,
            (n_nodes, 3),
        ).copy()

        arrays.append(repeated_parameters)
        names.extend(PARAMETER_COLUMNS)

    # 2. Geometric boundary indicators.
    if config.include_boundaries:
        boundaries: FloatArray = boundary_features(
            pos=coordinates,
            geometry=geometry,
        )

        arrays.append(boundaries)
        names.extend(
            f"boundary_{name}" for name in BOUNDARY_NAMES
        )

    # 3. Analyticla boundary values and prescription masks.
    if config.include_bc_values:
        bc = compute_boundary_conditions(
            pos=coordinates,
            mu=parameters,
            geometry=geometry,
        )

        arrays.append(bc.values.copy())
        names.extend(
            f"bc_value_{name}" for name in FIELD_COLUMNS
        )

        arrays.append(bc.mask.astype(np.float64))

        names.extend(
            f"bc_mask_{name}" for name in FIELD_COLUMNS
        )

    # 4. Distance to physical boundaries
    if config.include_distances:
        distances: FloatArray = distance_features(
            pos=coordinates,
            geometry=geometry,
        )

        arrays.append(distances)
        names.extend(
            f"distance_{name}" for name in BOUNDARY_NAMES
        )

    x: FloatArray = np.concatenate(arrays,
                                   axis=1)

    columns: tuple[str, ...] = tuple(names)

    if x.shape[1] != len(columns):
        raise RuntimeError(
            "Feature matrix and column schema are inconsistent."
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Generated features contain non-finite values."
        )
    
    return NodeFeatures(
        x=x,
        columns=columns,
        pos=coordinates.copy(),
        mu=parameters.copy(),
    )

