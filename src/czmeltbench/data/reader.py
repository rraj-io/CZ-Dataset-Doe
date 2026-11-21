"""
File Description: Utiliteis for reading raw CZ Crystal growth CFD dataset.
Author: Rohit Raj
Email: rajrohit403@gmail.com
Date: 2026-11-20
Version: 1.0
"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray 

from pathlib import Path
from zipfile import ZipFile

from czmeltbench.constants import (
    COORDINATE_COLUMNS,
    FIELD_COLUMNS,
    PARAMETER_COLUMNS,
    MESH_SIZES,
)


FloatArray = NDArray[np.float64]

class CZDataReader:
    """Reader for the original CZ CFD Dataset.

    This class provides access to simulation metadaata, 
    spatial coordinates, operating parameters and CFD Fields.

    No preprocessing, normalization or interpolation is applied here.

    Paramters
    ---------
    data_dir : str | Path
        Directory containing the original extracted dataset
    
    Attributes
    ----------
    data_sir : Path
        Absolute path to the dataset directory
    doe_zip : Path
        Path to the DOE simulation archive
    cases : pd.DataFrame
        DOE operating condition metadata
    status : pd.DataFrame
        DOE simulation convergence information.
    metadata : pd.DataFrame
        Combined case metadata indexed by case_id.    
    """

    def __init__(self,
                 data_dir: str | Path) -> None:
        """
        Initialize the CFD dataset reader.
        """
        self.data_dir: Path = (
            Path(data_dir).expanduser().resolve()
        )

        self.doe_zip: Path = (
            self.data_dir / "cz_v2_doe.zip"
        )

        required_files: tuple[str, ...] = (
            "case_matrix.csv",
            "doe_run_status.csv",
            "cz_v2_doe.zip",
        )

        for filename in required_files:
            filepath = self.data_dir / filename

            if not filepath.is_file():
                raise FileNotFoundError(
                    f"Required dataset file not found: {filenamel}"
                )

        self.cases: pd.DataFrame = pd.read_csv(
            self.data_dir / "case_matrix.csv"
        )

        self.status: pd.DataFrame = pd.read_csv(
            self.data_dir / "doe_run_status.csv"
        )

        self.metadata: pd.DataFrame = (
            self.cases.merge(
                self.status[
                    ["case_id", "status", "iterations"]
                ],
                on="case_id",
                how="left",
                validate="one_to_one",
            ).set_index(
                "case_id",
                verify_integrity=True,
            )
        )

        if self.metadata["status"].isna().any():
            raise ValueError(
                "Some cases have no matching run-status record"
            )

    def list_cases(self,
                   accepted_only: bool = True) -> list[str]:
        """Return available DOE simulation IDs

        Parameters
        ----------
        accepted only : bool, default=True
            If True, return only successfully convergede simulations
            marked as 'kept'.
        
        Returns
        --------
        list[str]
            Simulation case identifiers
        """

        metadata: pd.DataFrame = self.metadata

        if accepted_only:
            metadata = metadata.loc[
                metadata["status"].eq("kept")
            ]

        return metadata.index.astype(str).tolist()

    def get_case_info(self,
                      case_id: str) -> pd.Series:
        """Return metadata for a simulation
        
        Parameters
        ----------
        case_id : str
            Simulations identifier, e.g. 'sobol_000'
            
        Returns
        -------
        pd.Series
            Operatinng conditions, group, convergencce status
            and iteration counts
        
        Raises
        ------
        KeyError
            If teh case ID is not present in the metadata
        """

        if case_id not in self.metadata.index:
            raise KeyError(
                f"Unknown DOE simulation: {case_id}"
            )

        return self.metadata.loc[case_id].copy()

    def load_case(self,
                  case_id: str) -> tuple[pd.DataFrame, pd.Series]:
        """Load one accepted DOE CFD simulation.

        Parameters
        ----------
        case_id : str
            Simulaiton identifier.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            data:
                CFD nodal field dataframe.

        Raises
        ------
        ValueError
            If the simulation is excluded or invalid.
        FileNotFoundError
            If the simualtion is absent from the archive.
    
        """
        info: pd.Series = self.get_case_info(case_id)

        if info["status"] != "kept":
            raise ValueError(
                f"Simualation {case_id} is not acepted."
            )

        member_name: str = f"doe/{case_id}.csv"

        with ZipFile(self.doe_zip, mode="r") as archive:
            if member_name not in archive.namelist():
                raise FileNotFoundError(
                    f"{member_name} not found in DOE archive."
                )

            with archive.open(member_name) as file:
                data:pd.DataFrame = pd.read_csv(file)

        required_columns: set[str] = set(
            COORDINATE_COLUMNS + FIELD_COLUMNS
        )

        missing_columns: set[str] = (
            required_columns - set(data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing CFD fields: {sorted(missing_columns)}"
            )

        return data, info

    def load_arrays(self,
                    case_id: str) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Load CFD data as numerical arrays
        
        Parameters
        ----------
        case_id : str
            Simulation identifier.
        
        Returns
        -------
        pos : FlaotArray
            Spatial coordinates with shape (N, 2).
            Columns [r, z].
            Units: metres
        
        mu: FlaotArray
            Operating parameters with shape (3,).
            Columns: [crystal_rpm, crucible_rpm, t_hot].
            Units: [rpm, rpm, K].

        y : FlaotAray
            CFD fields with shape (N, 5).
            Columnns: [u_r, u_z, u_swirl, p, T].
            Units: [m/s, m/s, m/s, Pa, K]

        """

        data, info = self.load_case(case_id)

        pos: FloatArray = data.loc[
            :, list(COORDINATE_COLUMNS)
        ].to_numpy(dtype=np.float64)

        mu: FloatArray = info.loc[
            list(PARAMETER_COLUMNS)
        ].to_numpy(dtype=np.float64)

        y: FloatArray = data.loc[
            :, list(FIELD_COLUMNS)
        ].to_numpy(dtype=np.float64)

        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("Expected coordinates of shape (N, 2).")

        if mu.shape != (3,):
            raise ValueError("Expected parameters of shape (3,).")

        if y.shape != (len(pos), 5):
            raise ValueError("Expected fields of shape (N, 5).")

        if not (
            np.isfinite(pos).all() |
            np.isfinite(mu).all() |
            np.isfinite(y).all()
        ):
            raise ValueError(
                f"Simulation {case_id} contains non-finite values."
            )

        return pos, mu, y

    def load_reference_positions(self,
                                 mesh: str = "base") -> FloatArray:
        """Load reference node coordinates for a mesh resolution.

        Parameters
        ----------
        mesh : str, default="base"
            One of "coarse", "base", or "fine"

        Returns
        -------
        FlaotArray
            Reference node coordinates, shape (N, 2).
            columns: [r, z].
        """

        if mesh not in MESH_SIZES:
            raise ValueError(
                f"Unknown mesh '{mesh}'."
                f"Expected one of {list(MESH_SIZES)}"
            )

        num_nodes: int = MESH_SIZES[mesh]

        filename: str = (
            f"node_order_{mesh}_{num_nodes}.csv"
        )

        filepath: Path = self.data_dir / filename

        if not filepath.is_file():
            raise FileNotFoundError(filepath)

        data: pd.DataFrame = pd.read_csv(filepath)

        pos: FloatArray = data.loc[
            :, list(COORDINATE_COLUMNS)
        ].to_numpy(dtype=np.float64)

        if pos.shape != (num_nodes, 2):
            raise ValueError(
                f"Unexpected reference mesh shape: {pos.shape}"
            )

        if not np.isfinite(pos).all():
            raise ValueError(
                f"Reference mesh '{mesh}' contains nnonfinite coordinates."
            )

        return pos
