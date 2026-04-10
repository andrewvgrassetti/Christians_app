"""
Compartmental PK Modeling Module
=================================

Provides curve-fitting to standard PK compartmental models:
- One-compartment IV bolus
- One-compartment extravascular (oral / SC)
- Two-compartment IV bolus

Each model class exposes:
- fit(time, concentration): fit the model and return optimized parameters
- predict(time): predict concentrations at given time points
- parameters: dictionary of fitted PK parameters
"""

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FitResult:
    """Container for compartmental model fitting results."""

    model_name: str = ""
    parameters: dict = field(default_factory=dict)
    predicted: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    aic: float = 0.0
    r_squared: float = 0.0
    converged: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        """Return parameters as a rounded dictionary."""
        d = {"Model": self.model_name}
        for k, v in self.parameters.items():
            d[k] = round(v, 6) if isinstance(v, float) else v
        d["AIC"] = round(self.aic, 2)
        d["R²"] = round(self.r_squared, 6)
        d["Converged"] = self.converged
        return d


def _compute_aic(n: int, rss: float, k: int) -> float:
    """Compute Akaike Information Criterion."""
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def _compute_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute coefficient of determination (R²)."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# One-Compartment IV Bolus
# ---------------------------------------------------------------------------


class OneCompartmentIV:
    """
    One-compartment model for IV bolus administration.

    C(t) = (Dose / Vd) * exp(-ke * t)

    where:
        Vd = volume of distribution
        ke = elimination rate constant
        CL = ke * Vd
        t_half = ln(2) / ke
    """

    def __init__(self, dose: float):
        self.dose = float(dose)
        self._params = {}
        self._popt = None

    @staticmethod
    def _model(t, vd, ke):
        """Model function (dose passed via closure)."""
        return np.maximum((1.0 / vd) * np.exp(-ke * t), 0)

    def fit(
        self,
        time: np.ndarray,
        concentration: np.ndarray,
    ) -> FitResult:
        """Fit the model to observed data."""
        time = np.asarray(time, dtype=float)
        concentration = np.asarray(concentration, dtype=float)

        def model_with_dose(t, vd, ke):
            return self.dose * self._model(t, vd, ke)

        result = FitResult(model_name="One-Compartment IV Bolus")

        try:
            c0_est = concentration[0] if concentration[0] > 0 else np.max(concentration)
            vd_est = self.dose / c0_est if c0_est > 0 else 1.0
            ke_est = 0.1

            popt, _ = curve_fit(
                model_with_dose,
                time,
                concentration,
                p0=[vd_est, ke_est],
                bounds=([1e-10, 1e-10], [np.inf, np.inf]),
                maxfev=10000,
            )
            self._popt = popt
            vd, ke = popt

            predicted = model_with_dose(time, vd, ke)
            residuals = concentration - predicted

            self._params = {
                "Vd (L)": vd,
                "ke (1/h)": ke,
                "CL (L/h)": ke * vd,
                "t½ (h)": np.log(2) / ke,
                "C0 (ng/mL)": self.dose / vd,
            }

            rss = np.sum(residuals**2)
            result.parameters = self._params
            result.predicted = predicted
            result.residuals = residuals
            result.aic = _compute_aic(len(time), rss, 2)
            result.r_squared = _compute_r_squared(concentration, predicted)
            result.converged = True

        except (RuntimeError, ValueError) as e:
            result.converged = False
            result.message = str(e)

        return result

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predict concentrations at new time points."""
        if self._popt is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        vd, ke = self._popt
        return self.dose * self._model(np.asarray(time, dtype=float), vd, ke)


# ---------------------------------------------------------------------------
# One-Compartment Extravascular (Oral / SC)
# ---------------------------------------------------------------------------


class OneCompartmentOral:
    """
    One-compartment model with first-order absorption (extravascular).

    C(t) = (F * Dose * ka) / (Vd * (ka - ke)) * (exp(-ke * t) - exp(-ka * t))

    Parameters fitted: Vd/F, ka, ke  (F is assumed = 1 unless provided).
    """

    def __init__(self, dose: float, bioavailability: float = 1.0):
        self.dose = float(dose)
        self.f = float(bioavailability)
        self._params = {}
        self._popt = None

    @staticmethod
    def _model(t, vd_f, ka, ke):
        """Bateman function."""
        if abs(ka - ke) < 1e-10:
            # Degenerate case
            return (1.0 / vd_f) * ka * t * np.exp(-ke * t)
        return (ka / (vd_f * (ka - ke))) * (np.exp(-ke * t) - np.exp(-ka * t))

    def fit(
        self,
        time: np.ndarray,
        concentration: np.ndarray,
    ) -> FitResult:
        """Fit the model to observed data."""
        time = np.asarray(time, dtype=float)
        concentration = np.asarray(concentration, dtype=float)
        effective_dose = self.dose * self.f

        def model_with_dose(t, vd_f, ka, ke):
            return effective_dose * self._model(t, vd_f, ka, ke)

        result = FitResult(model_name="One-Compartment Oral (Extravascular)")

        try:
            cmax_idx = np.argmax(concentration)
            tmax_obs = time[cmax_idx] if time[cmax_idx] > 0 else 1.0
            cmax_obs = concentration[cmax_idx] if concentration[cmax_idx] > 0 else 1.0

            ka_est = 2.0 / tmax_obs
            ke_est = 0.5 / tmax_obs
            vd_f_est = effective_dose / (cmax_obs * 2)

            popt, _ = curve_fit(
                model_with_dose,
                time,
                concentration,
                p0=[vd_f_est, ka_est, ke_est],
                bounds=([1e-10, 1e-10, 1e-10], [np.inf, np.inf, np.inf]),
                maxfev=10000,
            )
            self._popt = popt
            vd_f, ka, ke = popt

            predicted = model_with_dose(time, vd_f, ka, ke)
            residuals = concentration - predicted

            self._params = {
                "Vd/F (L)": vd_f,
                "ka (1/h)": ka,
                "ke (1/h)": ke,
                "CL/F (L/h)": ke * vd_f,
                "t½ (h)": np.log(2) / ke,
                "Tmax_pred (h)": np.log(ka / ke) / (ka - ke) if ka != ke else 0,
            }

            rss = np.sum(residuals**2)
            result.parameters = self._params
            result.predicted = predicted
            result.residuals = residuals
            result.aic = _compute_aic(len(time), rss, 3)
            result.r_squared = _compute_r_squared(concentration, predicted)
            result.converged = True

        except (RuntimeError, ValueError) as e:
            result.converged = False
            result.message = str(e)

        return result

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predict concentrations at new time points."""
        if self._popt is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        vd_f, ka, ke = self._popt
        return (
            self.dose * self.f * self._model(np.asarray(time, dtype=float), vd_f, ka, ke)
        )


# ---------------------------------------------------------------------------
# Two-Compartment IV Bolus
# ---------------------------------------------------------------------------


class TwoCompartmentIV:
    """
    Two-compartment model for IV bolus administration.

    C(t) = A * exp(-alpha * t) + B * exp(-beta * t)

    Macro-parameters: A, B, alpha, beta
    Micro-parameters derived: k10, k12, k21, Vc, Vss, CL
    """

    def __init__(self, dose: float):
        self.dose = float(dose)
        self._params = {}
        self._popt = None

    @staticmethod
    def _model(t, a, alpha, b, beta):
        return a * np.exp(-alpha * t) + b * np.exp(-beta * t)

    def fit(
        self,
        time: np.ndarray,
        concentration: np.ndarray,
    ) -> FitResult:
        time = np.asarray(time, dtype=float)
        concentration = np.asarray(concentration, dtype=float)

        result = FitResult(model_name="Two-Compartment IV Bolus")

        try:
            c0_est = concentration[0] if concentration[0] > 0 else np.max(concentration)
            a_est = c0_est * 0.7
            b_est = c0_est * 0.3
            alpha_est = 1.0
            beta_est = 0.1

            popt, _ = curve_fit(
                self._model,
                time,
                concentration,
                p0=[a_est, alpha_est, b_est, beta_est],
                bounds=([0, 1e-10, 0, 1e-10], [np.inf, np.inf, np.inf, np.inf]),
                maxfev=10000,
            )
            self._popt = popt
            a, alpha, b, beta = popt

            # Ensure alpha > beta (alpha is the faster phase)
            if beta > alpha:
                a, b = b, a
                alpha, beta = beta, alpha
                self._popt = np.array([a, alpha, b, beta])

            predicted = self._model(time, a, alpha, b, beta)
            residuals = concentration - predicted

            # Derive micro-constants
            c0 = a + b
            vc = self.dose / c0 if c0 > 0 else 0
            k21 = (a * beta + b * alpha) / c0 if c0 > 0 else 0
            k10 = alpha * beta / k21 if k21 > 0 else 0
            k12 = alpha + beta - k21 - k10

            self._params = {
                "A (ng/mL)": a,
                "alpha (1/h)": alpha,
                "B (ng/mL)": b,
                "beta (1/h)": beta,
                "C0 (ng/mL)": c0,
                "Vc (L)": vc,
                "k10 (1/h)": k10,
                "k12 (1/h)": k12,
                "k21 (1/h)": k21,
                "CL (L/h)": k10 * vc,
                "t½_alpha (h)": np.log(2) / alpha,
                "t½_beta (h)": np.log(2) / beta,
                "Vss (L)": vc * (1 + k12 / k21) if k21 > 0 else 0,
            }

            rss = np.sum(residuals**2)
            result.parameters = self._params
            result.predicted = predicted
            result.residuals = residuals
            result.aic = _compute_aic(len(time), rss, 4)
            result.r_squared = _compute_r_squared(concentration, predicted)
            result.converged = True

        except (RuntimeError, ValueError) as e:
            result.converged = False
            result.message = str(e)

        return result

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predict concentrations at new time points."""
        if self._popt is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        a, alpha, b, beta = self._popt
        return self._model(np.asarray(time, dtype=float), a, alpha, b, beta)
