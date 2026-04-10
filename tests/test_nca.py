"""Tests for the NCA module."""

import numpy as np
import pytest
from pk_analysis.nca import NCAAnalysis, NCAResult


class TestNCAResult:
    """Tests for NCAResult dataclass."""

    def test_to_dict(self):
        result = NCAResult(cmax=100.0, tmax=2.0, auc_last=500.0, t_half=4.0)
        d = result.to_dict()
        assert "Cmax (ng/mL)" in d
        assert d["Cmax (ng/mL)"] == 100.0
        assert d["Tmax (h)"] == 2.0

    def test_to_dataframe(self):
        result = NCAResult(cmax=100.0, tmax=2.0)
        df = result.to_dataframe()
        assert "Parameter" in df.columns
        assert "Value" in df.columns
        assert len(df) == 16


class TestNCAAnalysis:
    """Tests for NCA analysis calculations."""

    @pytest.fixture
    def iv_data(self):
        """Simulated IV bolus 1-compartment data."""
        time = np.array([0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24])
        conc = np.array([0, 450, 380, 300, 210, 100, 48, 23, 5.2, 0.12])
        return time, conc

    @pytest.fixture
    def oral_data(self):
        """Simulated oral absorption data."""
        time = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24])
        conc = np.array([0, 85, 170, 200, 190, 140, 95, 40, 16, 2.5, 0.02])
        return time, conc

    def test_cmax_tmax_iv(self, iv_data):
        time, conc = iv_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.cmax == 450.0
        assert result.tmax == 0.25

    def test_cmax_tmax_oral(self, oral_data):
        time, conc = oral_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.cmax == 200.0
        assert result.tmax == 1.5

    def test_auc_last_positive(self, iv_data):
        time, conc = iv_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.auc_last > 0

    def test_auc_inf_greater_than_auc_last(self, iv_data):
        time, conc = iv_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.auc_inf >= result.auc_last

    def test_lambda_z_positive(self, iv_data):
        time, conc = iv_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.lambda_z > 0

    def test_half_life_positive(self, iv_data):
        time, conc = iv_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.t_half > 0

    def test_clearance_positive(self, iv_data):
        time, conc = iv_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.clearance > 0

    def test_vd_positive(self, iv_data):
        time, conc = iv_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.vd > 0

    def test_mrt_positive(self, oral_data):
        time, conc = oral_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.mrt > 0

    def test_r_squared_high(self, iv_data):
        time, conc = iv_data
        nca = NCAAnalysis(time, conc, dose=100e6)
        result = nca.run()
        assert result.r_squared > 0.9

    def test_input_validation_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            NCAAnalysis([1, 2, 3], [1, 2], dose=100)

    def test_input_validation_too_few_points(self):
        with pytest.raises(ValueError, match="At least 3"):
            NCAAnalysis([1, 2], [1, 2], dose=100)

    def test_all_zero_concentrations(self):
        nca = NCAAnalysis([0, 1, 2, 3], [0, 0, 0, 0], dose=100)
        result = nca.run()
        assert "All concentrations are zero" in result.warnings[0]

    def test_unsorted_data_is_sorted(self):
        """Verify that unsorted input is automatically sorted by time."""
        time = np.array([2, 0, 1, 4, 3])
        conc = np.array([210, 0, 300, 100, 140])
        nca = NCAAnalysis(time, conc, dose=100e6)
        assert np.all(nca.time == np.array([0, 1, 2, 3, 4]))

    def test_known_auc_simple_trapezoid(self):
        """Test AUC with simple increasing data (should use linear trap)."""
        time = np.array([0, 1, 2])
        conc = np.array([0, 10, 20])
        nca = NCAAnalysis(time, conc, dose=1000)
        result = nca.run()
        # Linear trapezoidal: (0+10)/2*1 + (10+20)/2*1 = 5 + 15 = 20
        assert abs(result.auc_last - 20.0) < 0.01
