"""Tests for the data I/O module."""

import os
import tempfile
import pandas as pd
import pytest
from pk_analysis.data_io import load_pk_data, export_results, results_to_csv_bytes


SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "sample_data"
)


class TestLoadPKData:
    """Tests for load_pk_data function."""

    def test_load_csv_file(self):
        filepath = os.path.join(SAMPLE_DIR, "iv_bolus_1comp.csv")
        df = load_pk_data(filepath)
        assert "Time" in df.columns
        assert "Concentration" in df.columns
        assert len(df) == 10

    def test_load_oral_csv(self):
        filepath = os.path.join(SAMPLE_DIR, "oral_1comp.csv")
        df = load_pk_data(filepath)
        assert len(df) == 11

    def test_load_from_dataframe(self):
        input_df = pd.DataFrame({"Time": [0, 1, 2], "Concentration": [0, 10, 5]})
        df = load_pk_data(input_df)
        assert len(df) == 3
        assert df["Time"].iloc[1] == 1.0

    def test_auto_detect_columns_case_insensitive(self):
        input_df = pd.DataFrame({"TIME": [0, 1, 2], "CONC": [0, 10, 5]})
        df = load_pk_data(input_df)
        assert "Time" in df.columns
        assert "Concentration" in df.columns

    def test_explicit_column_names(self):
        input_df = pd.DataFrame({"t_h": [0, 1, 2], "plasma_ng": [0, 10, 5]})
        df = load_pk_data(input_df, time_col="t_h", conc_col="plasma_ng")
        assert len(df) == 3

    def test_missing_time_column_raises(self):
        input_df = pd.DataFrame({"x": [0, 1, 2], "Concentration": [0, 10, 5]})
        with pytest.raises(ValueError, match="time column"):
            load_pk_data(input_df)

    def test_missing_conc_column_raises(self):
        input_df = pd.DataFrame({"Time": [0, 1, 2], "y": [0, 10, 5]})
        with pytest.raises(ValueError, match="concentration column"):
            load_pk_data(input_df)

    def test_nan_rows_dropped(self):
        input_df = pd.DataFrame(
            {"Time": [0, 1, None, 3], "Concentration": [0, 10, 5, None]}
        )
        df = load_pk_data(input_df)
        assert len(df) == 2

    def test_additional_columns_preserved(self):
        input_df = pd.DataFrame(
            {
                "Time": [0, 1, 2],
                "Concentration": [0, 10, 5],
                "Subject": [1, 1, 1],
            }
        )
        df = load_pk_data(input_df)
        assert "Subject" in df.columns


class TestExportResults:
    """Tests for export_results function."""

    def test_export_csv(self):
        results = {"Cmax": 100.0, "Tmax": 2.0, "AUC": 500.0}
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_results(results, path, format="csv")
            df = pd.read_csv(path)
            assert len(df) == 3
            assert "Parameter" in df.columns
        finally:
            os.unlink(path)

    def test_export_excel(self):
        results = {"Cmax": 100.0, "Tmax": 2.0}
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            path = f.name
        try:
            export_results(results, path, format="excel")
            df = pd.read_excel(path)
            assert len(df) == 2
        finally:
            os.unlink(path)


class TestResultsToCsvBytes:
    """Tests for results_to_csv_bytes function."""

    def test_returns_bytes(self):
        results = {"Cmax": 100.0, "Tmax": 2.0}
        b = results_to_csv_bytes(results)
        assert isinstance(b, bytes)
        assert b"Cmax" in b
