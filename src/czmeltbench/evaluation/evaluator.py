"""
File Description: Single case evaluator for CZMeltBench CFD predictions 
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-24
Version: 1.0
"""
"""
This module evaluates predictionns against original CFD nodal fields
using native mesh geometry and axisymmetry integration weights.

No interpolation, field replacement or pressure adjustemnt is applied.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np
import pandas as pd 
from numpy.typing import NDArray

from czmeltbench.geometry.boundaries import identify_boundaries
from czmeltbench.geometry.connectivity import build_connectivity
from czmeltbench.geometry.mesh import build_logical_mesh
from czmeltbench.geometry.quadrature import compute_mesh_quadrature
from czmeltbench.evaluation.metrics import evaluate_fields


FloatArray: TypeAlias = NDArray[np.float64]
EvaluationRegion: TypeAlias = Literal["all", "interior"]


class CaseReader(Protocol):
    """INterface required by teh evaluator's data reader.
    """

    def load_arrays(self,
                    case_id: str
                    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Load positions, paramters and fields for one case.
        """
        ...


@dataclass(frozen=True)
class CaseEvaluation:
    """results of evaluating one CFD simulation.

    Attributes
    ----------
    case_id : str
        Unique simulation identifier.

    region : EvaluationRegion
        Evalauted node region: "all" or "interior".

    num_nodes :int
        Total number of nodes in the reference mesh.

    num_evaluated_nodes : int
        Number of nodes included in the metrics.

    total_volume : float
        Reconstructed physical melt volume, in m^3.
    
    mu : FloatArray
        Operating parameters, shape (3,).

    metrics : pd.DataFrame
        Field-wise prediction errors.

        Rows:
            [u_r, u_z, u_swirl, p, T]
        Columns:
            [mae, rmse, relativ_l2, volume_mae,
            volume_rmse, volume_relative_l2]
    """

    case_id: str
    region: EvaluationRegion
    num_nodes: int
    num_evaluated_nodes: int
    total_volume: float
    mu: FloatArray
    metrics: pd.DataFrame



class CaseEvaluator:
    """Evaluate model predictions against original CFD simulations.

    Parameters
    ----------
    reader : CaseReader
        Reader implimenting load_arrays(case_id).

        The existing CZDataReader satisfeis this interface.
    """

    def __init__(self,
                 reader: CaseReader) -> None:
        """Initialize the evaluator with a simulation reader.
        """
        self.reader: CaseReader = reader

    @staticmethod
    def _validate_prediction(
        reference_pos: FloatArray,
        prediction_pos: FloatArray,
        y_true: FloatArray,
        y_pred: FloatArray,
        *,
        coordinate_tolerance: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Validate reference/prediction alignnment and field shapes.

        Parameters
        ----------
        reference_pos : FloatArray
            Original CFd coordinates, shape (N, 2).

        prediction_pos : FloatArray
            Coordinates corresponding to predicted field rows, shape (N,2).

        y_true : FloatArray
            Reference CFD solution, shape (N, 5).

        y_pred : FloatArray
            Predicted CFD solution, shape (N, 5).

        coordinate_tolerance : float
            Absolute coordinate tolerance in [m].

        Returns
        -------
        tuple[FloatArray, FloatArray]
            validated reference and predicted field arrays
        """

        ref_pos: FloatArray = np.asarray(
            reference_pos, dtype=np.float64
        )

        pred_pos: FloatArray = np.asarray(
            prediction_pos, dtype=np.float64
        )

        reference: FloatArray = np.asarray(
            y_true, dtype=np.float64
        )

        prediction: FloatArray = np.asarray(
            y_pred, dtype=np.float64
        )

        if (ref_pos.ndim != 2 or ref_pos.shape[1] != 2 or ref_pos.shape[0] == 0):
            raise ValueError("Reference coordinates must have shape (N, 2).")

        if pred_pos.shape != ref_pos.shape:
            raise ValueError("Prediction coordinates must match reference shape.")

        if reference.shape != (len(ref_pos), 5):
            raise ValueError("Reference CFD fields must have shape (N, 5).")

        if prediction.shape != reference.shape:
            raise ValueError("Prediccted fields must match reference shape (N, 5).")

        if (
            not np.isfinite(coordinate_tolerance)
            or coordinate_tolerance < 0.0
        ):
            raise ValueError(
                "coordinate_tolerance must be finite and non-negative."
            )

        if not (
            np.isfinite(ref_pos).all() and
            np.isfinite(pred_pos).all() and
            np.isfinite(reference).all() and
            np.isfinite(prediction).all()
        ):
            raise ValueError(
                "Coordinates and CFD fields must be finite."
            )

        # Compare row-by-row, Never silently sort predictions.
        if not np.allclose(
            ref_pos,
            pred_pos,
            rtol=0.0,
            atol=coordinate_tolerance,
        ):
            raise ValueError(
                "Prediction node coordinates do not mathc teh"
                "referencce node ordering. Aliign predictions "
                "before evalaution."
            )

        return reference, prediction

    def evaluate(self,
                 case_id: str,
                 y_pred: FloatArray,
                 prediction_pos: FloatArray,
                 *,
                 region: EvaluationRegion = "all",
                 coordinate_tolerance: float = 1e-12,
                 ) -> CaseEvaluation:
        """Evalaute a model prediction for one CFD simualtion.

        Parameters
        ----------
        case_id : str
            Reference simualtion identifier, e.g. 'sobol_000'.

        y_pred : FloatArray
            Predicted fields, shape (N, 5).
            Columns: [u_r, u_z, u_swirl, p, T].

        prediction_pos : FloatArray
            Node coordinates corresponding to y_pred rows, shape (N, 2).

        region : {'all', 'interior'}, default='all'
            which nodes to ionclude iun the evaluation

        coordinate_tolerance : float, default=1e-12
            Absolute tolerance when matching coordinates in [m].
            relative tolerance is zero.

        Returns
        -------
        CaseEvaluation
            Field-wise metrics and evaluation metadata.

        Notes
        ------
        Prediction coordinates must be akligned with the original CFD
        node ordering.

        The evaluator does not reorder, interpolate, normalize or modify 
        either field array.

        'interior' uses teh geometric boundary masks.
        The weighted interior metrics use selected nodal quadrature weights,
        not an exact truncated domain inetgration.

        Analytically prescribed BCs must be evaluated separately from 
        supervised CFD errors.
        """

        if region not in ("all", "interior"):
            raise ValueError(
                "region must be in 'all' or 'interior'."
            )

        pos, mu, y_true = self.reader.load_arrays(case_id)

        reference, prediction = self._validate_prediction(
            reference_pos=pos,
            prediction_pos=prediction_pos,
            y_true=y_true,
            y_pred=y_pred,
            coordinate_tolerance=coordinate_tolerance,
        )

        parameters: FloatArray = np.asarray(
            mu, dtype=np.float64
        )

        if parameters.shape != (3,):
            raise ValueError(
                "Operating paramters must be finite."
            )

        # Reconstruct native logical mesh and connectivity
        mesh = build_logical_mesh(pos)
        graph = build_connectivity(mesh)

        # Compute integration weights on original CFD indices
        quadrature = compute_mesh_quadrature(
            pos=pos,
            cells=graph.cells,
        )

        selected_nodes: NDArray[np.bool_]

        if region == "all":
            selected_nodes = np.ones(
                len(pos), dtype=np.bool_
            )

        else:
            masks = identify_boundaries(pos)
            selected_nodes = masks["interior"]

        if not np.any(selected_nodes):
            raise ValueError(
                f"No nodes available for region '{region}'."
            )

        metrics: pd.DataFrame = evaluate_fields(
            y_true=reference,
            y_pred=prediction,
            volume_weights=quadrature.node_volume_weights,
            node_mask=selected_nodes,
        )

        return CaseEvaluation(
            case_id=case_id,
            region=region,
            num_nodes=int(len(pos)),
            num_evaluated_nodes=int(selected_nodes.sum()),
            total_volume=quadrature.total_volume,
            mu=parameters.copy(),
            metrics=metrics,
        )