"""
File Description: Multi-case benchmark evaluation for CZMeltBench
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-24
Version: 1.0
"""
"""
Uses a single-case evaluator to compute prediction errros.

Aggregation is performed across simulation cases, not by pooling nodes 
from diferent meshes.

No normalization, interpolation or modification of CFD targets is performed.
"""

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, TypeAlias

import numpy as np
import pandas as pd 
from numpy.typing import NDArray

from czmeltbench.evaluation.evaluator import (
    CaseEvaluation,
    EvaluationRegion,
)


FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class CasePrediction:
    """Prediction for one CFD simulation.

    Attributes
    ----------
    pos : FloatArray
        Coordinates corresponding to prediction rows.
        Shape (N, 2) in original CFD node order.

    y : FloatArray
        Predicted CFD fields, shape (N, 5).
        Columns: [u_r, u_z, u_swirl, p, T].
    """

    pos: FloatArray
    y:FloatArray


@dataclass(frozen=True)
class BenchmarkResult:
    """results from evaluating a collection of CFD cases

    Attributes
    ----------
    region : EvaluationRegion
        Evaluation region, either 'all' or 'interior'.

    case_metadata : pd.DataFrame
        One row per evaluated case. 
        Index: case_id
        Includes node counts, physical volume and mu.

    case_scores : pd.DataFrame
        Individual metrics for every case and field.
        MultiIndex: [case_id, field].

    summary : pd.DataFrame
        Aggregate benchmark statistics.
        MultiIndex: [field, metric]
        Columns: [valid_cases, mean, std, min, max]
    
    Notes
    -----
    Each case contributes equally to the aggregation.

    Undefined realtive errors (NaN) are excluded from that metric's
    mean and counted through valid_cases.
    """

    region: EvaluationRegion
    case_metadata: pd.DataFrame
    case_scores: pd.DataFrame
    summary: pd.DataFrame

    @property
    def num_cases(self) -> int:
        """Return the numebr of evaluated simulations.
        """
        return int(len(self.case_metadata))


class SingleCaseEvaluator(Protocol):
    """Required interface for a single-case evaluator.
    """

    def evaluate(self,
                 case_id: str,
                 y_pred: FloatArray,
                 prediction_pos: FloatArray,
                 *,
                 region: EvaluationRegion = "all",
                 coordinate_tolerance: float = 1e-12,
                 ) -> CaseEvaluation:
        """Evaluate one simualtion prediction.
        """
        ...


class BenchmarkEvaluator:
    """Evaluate multiple cases with a shaored evaluation protocol.

    Parameters
    ----------
    case_evaluator : SingleCaseEvaluator
        Anexisting CaseEvalautor instance.
    """

    def __init__(self,
                 case_evaluator: SingleCaseEvaluator,
                 ) -> None:
        """Initialize the benchmark evaluator.
        """
        self.case_evaluator: SingleCaseEvaluator = case_evaluator


    @staticmethod
    def _aggregate_scores(case_scores: pd.DataFrame,
                          ) -> pd.DataFrame:
        """Aggregate per-case metrics with equal case weights.

        Parameters
        ----------
        case_scores : pd.DataFrame
            MultiIndex dataframe indexed by [case_id, fields]

        Returns
        -------
        pd.DataFrame
            MultiIndex dataframe indexed by [field, metric].
            Columns: valid_cases, mean, std, min, max.

        Notes
        -----
        Uses population standard deviation (ddof=0).

        Undefined (NaN) metrics are excluded from the corresponding
        aggregation. They are not replaced by zero.
        """

        records: list[dist[str, str | int | float]] = []

        metric_name: list[str] = list(case_scores.columns)

        fields: list[str] = list(
            dict.fromkeys(
                case_scores.index.get_level_values("field")
            )
        )

        for field in fields:
            field_scores: pd.DataFrame = (
                case_scores.xs(field, level="field")
            )

            for metric in metric_name:
                values: FloatArray = (
                    field_scores[metric].to_numpy(dtype=np.float64)
                )

                valid: FloatArray = values[np.isfinite(values)]

                count: int = int(len(valid))

                records.append({
                    "field": field,
                    "metric":metric,
                    "valid_cases": count,
                    "mean": (
                        float(np.mean(valid))
                        if count > 0 else float("nan")
                    ),
                    "std": (
                        float(np.std(valid, ddof=0))
                        if count > 0 else float("nan")
                    ),
                    "min": (
                        float(np.min(valid))
                        if count > 0 else float("nan")
                    ),
                    "max": (
                        float(np.max(valid))
                        if count > 0 else float("nan")
                    ),

                })
        return pd.DataFrame.from_records(
            records
        ).set_index(["field", "metric"])

    
    def evaluate(self,
                 predictions: Mapping[str, CasePrediction],
                 *,
                 region: EvaluationRegion = "all",
                 expected_case_ids: Sequence[str] | None = None,
                 coordinate_tolerance: float = 1e-12,
                 ) -> BenchmarkResult:
        """Evaluate a collection of simulation predictions.

        Parameters
        ----------
        predictions : Mapping[str, CasePrediction]
            mapping from case ID to predicted coordinates and CFD fields.

        region : {'all", 'interior'}, default='all'
            Spatial evaluation region for every case.

        expected_case_ids : Sequence[str] | None, optional
            Explicit list of cases required for this benchmark evalaution.

            If provided, missing or extra  predictions cause an error.

            use this to prevent accidently reporting results for only 
            part of a test split.

        coordinate_tolerance : float, default=1e-12
            Absolute coordinate matching tolerance, in [m].

        Returns
        -------
        BenchmarkResult
            Case metadata, case scores and aggregated metrics.

        Notes
        ------
        Each case id evaluated independently using the existing
        CaseEvaluator.

        The benchmark never reorders prediction nodes or alters
        original CFD targets.

        Aggregation is performed across cases after calculating 
        per-case metrics.
        """

        if not predictions:
            raise ValueError("Attleast one case prediction is required.")

        if region not in ("all", "interior"):
            raise ValueError("region must be 'all' or 'interior'.")

        if not np.isfinite(coordinate_tolerance):
            raise ValueError("Coordinate tolerance must be finite.")

        if coordinate_tolerance < 0.0:
            raise ValueError("Coordinate tolerance cannot be negative.")

        if any(
            not isinstance(case_id, str) or not case_id for case_id in predictions
        ):
            raise ValueError("Case IDs must be non-empty strings.")

        if expected_case_ids is None:
            case_ids: list[str] = list(predictions)
        else:
            case_ids = list(expected_case_ids)

            if len(case_ids) != len(set(case_ids)):
                raise ValueError(
                    "expected_case_ids contains duplicates."
                )

            missing = set(case_ids) - set(predictions)
            extra = set(predictions) - set(case_ids)

            if missing or extra:
                raise ValueError(
                    "Prediction case IDs do not match the required case list. "
                    f"Missing: {sorted(missing)}; "
                    f"Extra: {sorted(extra)}" 
                )

            if not case_ids:
                raise ValueError(
                    "Expected case list cannot be empty."
                )

        metadata_records: list[
            dict[str, str | int | float]
        ] = []

        score_frames: list[pd.DataFrame] = []
        metric_schema: tuple[str, ...] | None = None

        for case_id in case_ids:
            prediction: CasePrediction = predictions[case_id]

            result: CaseEvaluation = (
                self.case_evaluator.evaluate(
                    case_id=case_id,
                    y_pred=prediction.y,
                    prediction_pos=prediction.pos,
                    region=region,
                    coordinate_tolerance=coordinate_tolerance,
                )
            )

            if result.case_id != case_id:
                raise ValueError(
                    "Single-case evaluator retruned an incorrect case IS."
                )

            current_schema = tuple(result.metrics.columns)

            if metric_schema is None:
                metric_schema = current_schema
            elif current_schema != metric_schema:
                raise ValueError(
                    "Inconsistent metric columns across cases."
                )

            if (
                result.mu.shape != (3,) or not np.isfinite(result.mu).all()
            ):
                raise ValueError(f"Invalid parameters for {case_id}.")

            metadata_records.append({
                "case_id": case_id,
                "num_nodes": result.num_nodes,
                "num_evaluated_nodes": (result.num_evaluated_nodes),
                "total_volume": result.total_volume,
                "crystal_rpm": float(result.mu[0]),
                "crucible_rpm": float(result.mu[1]),
                "t_hot": float(result.mu[2]),
            })

            frame: pd.DataFrame = (result.metrics.copy())
            frame.index.name = "field"

            frame = pd.concat(
                {case_id: frame}, names=["case_id"],
            )

            score_frames.append(frame)

        case_metadata: pd.DataFrame = (
            pd.DataFrame.from_records(
                metadata_records
            ).set_index("case_id")
        )

        case_scores: pd.DataFrame = pd.concat(score_frames)

        case_scores.index.names = [
            "case_id",
            "field",
        ]

        summary: pd.DataFrame = self._aggregate_scores(case_scores)

        return BenchmarkResult(
            region=region,
            case_metadata=case_metadata,
            case_scores=case_scores,
            summary=summary,
        )
        