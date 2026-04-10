"""
Data I/O Module
================

Provides utilities for loading PK data from CSV/Excel files and
exporting analysis results.
"""

import pandas as pd
import numpy as np
import io
from typing import Optional


def load_pk_data(
    source,
    time_col: Optional[str] = None,
    conc_col: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load pharmacokinetic data from a file path, uploaded file, or DataFrame.

    Parameters
    ----------
    source : str, file-like, or pd.DataFrame
        Path to a CSV or Excel file, a file-like object, or an existing
        DataFrame.
    time_col : str, optional
        Name of the time column. If None, auto-detected from common names.
    conc_col : str, optional
        Name of the concentration column. If None, auto-detected.
    sheet_name : str or int, optional
        Sheet name/index for Excel files.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least 'Time' and 'Concentration' columns.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (str, io.IOBase)):
        if isinstance(source, str) and (
            source.endswith(".xlsx") or source.endswith(".xls")
        ):
            df = pd.read_excel(
                source,
                sheet_name=sheet_name if sheet_name is not None else 0,
            )
        else:
            df = pd.read_csv(source)
    else:
        # Assume file-like object (e.g., Streamlit UploadedFile)
        name = getattr(source, "name", "")
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(
                source,
                sheet_name=sheet_name if sheet_name is not None else 0,
            )
        else:
            df = pd.read_csv(source)

    # Auto-detect columns
    if time_col is None:
        time_col = _detect_column(
            df,
            ["time", "t", "time_h", "time_hr", "time_hours", "hours", "hr"],
        )
    if conc_col is None:
        conc_col = _detect_column(
            df,
            [
                "concentration",
                "conc",
                "cp",
                "plasma_concentration",
                "dv",
                "conc_ng_ml",
                "c",
            ],
        )

    if time_col is None:
        raise ValueError(
            "Could not auto-detect time column. "
            f"Available columns: {list(df.columns)}. "
            "Please specify time_col."
        )
    if conc_col is None:
        raise ValueError(
            "Could not auto-detect concentration column. "
            f"Available columns: {list(df.columns)}. "
            "Please specify conc_col."
        )

    # Standardize column names
    result = pd.DataFrame(
        {
            "Time": pd.to_numeric(df[time_col], errors="coerce"),
            "Concentration": pd.to_numeric(df[conc_col], errors="coerce"),
        }
    )

    # Carry over any additional columns (e.g., Subject, Dose)
    for col in df.columns:
        if col not in (time_col, conc_col):
            result[col] = df[col].values

    # Drop rows with NaN in Time or Concentration
    result = result.dropna(subset=["Time", "Concentration"]).reset_index(drop=True)

    return result


def _detect_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    col_lower = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in col_lower:
            return col_lower[candidate.lower()]
    return None


def export_results(
    results_dict: dict,
    filepath: str,
    format: str = "csv",
) -> None:
    """
    Export analysis results to a file.

    Parameters
    ----------
    results_dict : dict
        Dictionary of parameter name -> value.
    filepath : str
        Output file path.
    format : str
        'csv' or 'excel'.
    """
    df = pd.DataFrame(
        {"Parameter": list(results_dict.keys()), "Value": list(results_dict.values())}
    )
    if format.lower() == "excel":
        df.to_excel(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)


def results_to_csv_bytes(results_dict: dict) -> bytes:
    """Convert results dictionary to CSV bytes for download."""
    df = pd.DataFrame(
        {"Parameter": list(results_dict.keys()), "Value": list(results_dict.values())}
    )
    return df.to_csv(index=False).encode("utf-8")
