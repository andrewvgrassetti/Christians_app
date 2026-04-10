"""Tests for the compartmental modeling module."""

import numpy as np
import pytest
from pk_analysis.compartmental import (
    OneCompartmentIV,
    OneCompartmentOral,
    TwoCompartmentIV,
    _compute_aic,
    _compute_r_squared,
)


class TestHelperFunctions:
    """Tests for utility functions."""

    def test_compute_aic(self):
        aic = _compute_aic(n=10, rss=1.0, k=2)
        assert isinstance(aic, float)
        assert np.isfinite(aic)

    def test_compute_aic_zero_rss(self):
        aic = _compute_aic(n=10, rss=0.0, k=2)
        assert aic == np.inf

    def test_compute_r_squared_perfect(self):
        obs = np.array([1, 2, 3, 4, 5], dtype=float)
        pred = np.array([1, 2, 3, 4, 5], dtype=float)
        assert _compute_r_squared(obs, pred) == pytest.approx(1.0)

    def test_compute_r_squared_zero(self):
        obs = np.array([1, 2, 3, 4, 5], dtype=float)
        pred = np.array([3, 3, 3, 3, 3], dtype=float)
        assert _compute_r_squared(obs, pred) == pytest.approx(0.0)


class TestOneCompartmentIV:
    """Tests for the 1-compartment IV bolus model."""

    def test_fit_synthetic_data(self):
        """Fit to data generated from the model itself."""
        dose = 1e6  # ng
        vd_true = 50.0  # L
        ke_true = 0.3  # 1/h
        t = np.array([0.1, 0.5, 1, 2, 4, 6, 8, 12])
        c = (dose / vd_true) * np.exp(-ke_true * t)

        model = OneCompartmentIV(dose=dose)
        result = model.fit(t, c)

        assert result.converged
        assert result.r_squared > 0.99
        assert abs(result.parameters["Vd (L)"] - vd_true) / vd_true < 0.05
        assert abs(result.parameters["ke (1/h)"] - ke_true) / ke_true < 0.05

    def test_predict_after_fit(self):
        dose = 1e6
        t = np.array([0.1, 0.5, 1, 2, 4, 6, 8])
        c = (dose / 50.0) * np.exp(-0.3 * t)

        model = OneCompartmentIV(dose=dose)
        model.fit(t, c)
        pred = model.predict(np.array([0.5, 1.0, 2.0]))
        assert len(pred) == 3
        assert all(p > 0 for p in pred)

    def test_predict_before_fit_raises(self):
        model = OneCompartmentIV(dose=1000)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.array([1, 2, 3]))


class TestOneCompartmentOral:
    """Tests for the 1-compartment oral model."""

    def test_fit_synthetic_data(self):
        """Fit to Bateman function data."""
        dose = 1e6
        vd_f = 40.0
        ka = 1.5
        ke = 0.3
        t = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12])
        c = dose * (ka / (vd_f * (ka - ke))) * (np.exp(-ke * t) - np.exp(-ka * t))

        model = OneCompartmentOral(dose=dose)
        result = model.fit(t, c)

        assert result.converged
        assert result.r_squared > 0.99

    def test_predict_after_fit(self):
        dose = 1e6
        vd_f = 40.0
        ka = 1.5
        ke = 0.3
        t = np.array([0.25, 0.5, 1, 2, 4, 6, 8])
        c = dose * (ka / (vd_f * (ka - ke))) * (np.exp(-ke * t) - np.exp(-ka * t))

        model = OneCompartmentOral(dose=dose)
        model.fit(t, c)
        pred = model.predict(np.array([1.0, 2.0]))
        assert len(pred) == 2

    def test_predict_before_fit_raises(self):
        model = OneCompartmentOral(dose=1000)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.array([1, 2]))


class TestTwoCompartmentIV:
    """Tests for the 2-compartment IV bolus model."""

    def test_fit_synthetic_data(self):
        """Fit to bi-exponential data."""
        a_true, alpha_true = 800.0, 2.0
        b_true, beta_true = 200.0, 0.1
        t = np.array([0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24])
        c = a_true * np.exp(-alpha_true * t) + b_true * np.exp(-beta_true * t)

        dose = (a_true + b_true) * 50.0  # Vc approximately 50 L
        model = TwoCompartmentIV(dose=dose)
        result = model.fit(t, c)

        assert result.converged
        assert result.r_squared > 0.99

    def test_predict_before_fit_raises(self):
        model = TwoCompartmentIV(dose=1000)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.array([1, 2]))

    def test_alpha_beta_ordering(self):
        """Ensure alpha >= beta after fitting."""
        a, alpha = 500.0, 1.5
        b, beta = 300.0, 0.05
        t = np.array([0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 24, 48])
        c = a * np.exp(-alpha * t) + b * np.exp(-beta * t)

        model = TwoCompartmentIV(dose=50000)
        result = model.fit(t, c)

        if result.converged:
            assert result.parameters["alpha (1/h)"] >= result.parameters["beta (1/h)"]
