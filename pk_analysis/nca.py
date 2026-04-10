"""
Non-Compartmental Analysis (NCA) Module
========================================

Calculates standard PK parameters from concentration-time data
without assuming a specific compartmental model.

Parameters computed:
- Cmax: Maximum observed concentration
- Tmax: Time of maximum observed concentration
- AUC_last: Area under the curve to last measurable concentration (linear-log trapezoidal)
- AUC_inf: Area under the curve extrapolated to infinity
- Lambda_z: Terminal elimination rate constant
- t_half: Terminal half-life
- CL (or CL/F): Clearance (or apparent clearance for extravascular)
- Vd (or Vd/F): Volume of distribution (or apparent Vd for extravascular)
- MRT: Mean residence time
- AUC_%extrap: Percent of AUC extrapolated beyond last observation
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NCAResult:
    """Container for NCA analysis results."""

    cmax: float = 0.0
    tmax: float = 0.0
    clast: float = 0.0
    tlast: float = 0.0
    auc_last: float = 0.0
    auc_inf: float = 0.0
    auc_pct_extrap: float = 0.0
    lambda_z: float = 0.0
    t_half: float = 0.0
    clearance: float = 0.0
    vd: float = 0.0
    mrt: float = 0.0
    aumc_last: float = 0.0
    aumc_inf: float = 0.0
    r_squared: float = 0.0
    n_points_terminal: int = 0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert results to dictionary."""
        return {
            "Cmax (ng/mL)": round(self.cmax, 4),
            "Tmax (h)": round(self.tmax, 4),
            "Clast (ng/mL)": round(self.clast, 4),
            "Tlast (h)": round(self.tlast, 4),
            "AUC_last (ng·h/mL)": round(self.auc_last, 4),
            "AUC_inf (ng·h/mL)": round(self.auc_inf, 4),
            "AUC_%extrap": round(self.auc_pct_extrap, 2),
            "Lambda_z (1/h)": round(self.lambda_z, 6),
            "t½ (h)": round(self.t_half, 4),
            "CL or CL/F (L/h)": round(self.clearance, 4),
            "Vd or Vd/F (L)": round(self.vd, 4),
            "MRT (h)": round(self.mrt, 4),
            "AUMC_last (ng·h²/mL)": round(self.aumc_last, 4),
            "AUMC_inf (ng·h²/mL)": round(self.aumc_inf, 4),
            "R² (terminal phase)": round(self.r_squared, 6),
            "N points (terminal)": self.n_points_terminal,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        d = self.to_dict()
        return pd.DataFrame(
            {"Parameter": list(d.keys()), "Value": list(d.values())}
        )


class NCAAnalysis:
    """
    Non-Compartmental Analysis of pharmacokinetic data.

    Parameters
    ----------
    time : array-like
        Time points (e.g., hours).
    concentration : array-like
        Observed plasma concentrations (e.g., ng/mL).
    dose : float
        Administered dose (e.g., mg). Converted to same mass unit as
        concentration numerator for CL/Vd (e.g., if conc is ng/mL,
        dose should be provided in ng, or specify dose_units).
    route : str, optional
        'iv' for intravenous bolus, 'ev' for extravascular (oral, SC, etc.).
        Default is 'ev'.
    """

    def __init__(
        self,
        time: np.ndarray,
        concentration: np.ndarray,
        dose: float,
        route: str = "ev",
    ):
        self.time = np.asarray(time, dtype=float)
        self.concentration = np.asarray(concentration, dtype=float)
        self.dose = float(dose)
        self.route = route.lower()

        if len(self.time) != len(self.concentration):
            raise ValueError(
                "time and concentration arrays must have the same length."
            )
        if len(self.time) < 3:
            raise ValueError("At least 3 data points are required for NCA.")

        # Sort by time
        order = np.argsort(self.time)
        self.time = self.time[order]
        self.concentration = self.concentration[order]

    def run(self) -> NCAResult:
        """Execute the full NCA and return an NCAResult."""
        result = NCAResult()

        # --- Cmax and Tmax ---
        idx_max = np.argmax(self.concentration)
        result.cmax = float(self.concentration[idx_max])
        result.tmax = float(self.time[idx_max])

        # --- Clast and Tlast (last measurable, non-zero concentration) ---
        nonzero = np.where(self.concentration > 0)[0]
        if len(nonzero) == 0:
            result.warnings.append("All concentrations are zero.")
            return result

        idx_last = nonzero[-1]
        result.clast = float(self.concentration[idx_last])
        result.tlast = float(self.time[idx_last])

        # --- AUC_last using linear-log trapezoidal method ---
        result.auc_last = self._auc_linear_log(
            self.time[: idx_last + 1], self.concentration[: idx_last + 1]
        )

        # --- AUMC_last ---
        result.aumc_last = self._aumc_linear_log(
            self.time[: idx_last + 1], self.concentration[: idx_last + 1]
        )

        # --- Terminal phase estimation (lambda_z) ---
        lambda_z, r_sq, n_pts = self._estimate_lambda_z(idx_last)
        result.lambda_z = lambda_z
        result.r_squared = r_sq
        result.n_points_terminal = n_pts

        if lambda_z > 0:
            result.t_half = np.log(2) / lambda_z

            # AUC_inf = AUC_last + Clast / lambda_z
            auc_extrap = result.clast / lambda_z
            result.auc_inf = result.auc_last + auc_extrap
            if result.auc_inf > 0:
                result.auc_pct_extrap = (auc_extrap / result.auc_inf) * 100.0

            # AUMC_inf
            aumc_extrap = (
                result.clast * result.tlast / lambda_z
                + result.clast / (lambda_z**2)
            )
            result.aumc_inf = result.aumc_last + aumc_extrap

            # MRT
            if result.auc_inf > 0:
                result.mrt = result.aumc_inf / result.auc_inf

            # Clearance and Volume of distribution
            if result.auc_inf > 0:
                result.clearance = self.dose / result.auc_inf
                result.vd = result.clearance / lambda_z
        else:
            result.warnings.append(
                "Could not estimate terminal elimination rate constant (lambda_z)."
            )

        if result.auc_pct_extrap > 20:
            result.warnings.append(
                f"AUC extrapolated beyond last observation is {result.auc_pct_extrap:.1f}% "
                "(>20%), which may indicate insufficient sampling duration."
            )

        return result

    def _auc_linear_log(self, t: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate AUC using the linear-log trapezoidal method.

        - Linear trapezoidal rule when concentrations are increasing.
        - Log trapezoidal rule when concentrations are decreasing.
        """
        auc = 0.0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            c1, c2 = c[i - 1], c[i]
            if dt <= 0:
                continue
            if c1 <= 0 or c2 <= 0 or c2 >= c1:
                # Linear trapezoidal
                auc += 0.5 * (c1 + c2) * dt
            else:
                # Log trapezoidal
                auc += (c1 - c2) * dt / np.log(c1 / c2)
        return auc

    def _aumc_linear_log(self, t: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate AUMC (area under the first moment curve) using the
        linear-log trapezoidal method.
        """
        aumc = 0.0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            tc1, tc2 = t[i - 1] * c[i - 1], t[i] * c[i]
            if dt <= 0:
                continue
            c1, c2 = c[i - 1], c[i]
            if c1 <= 0 or c2 <= 0 or c2 >= c1:
                aumc += 0.5 * (tc1 + tc2) * dt
            else:
                aumc += (tc1 - tc2) * dt / np.log(c1 / c2)
        return aumc

    def _estimate_lambda_z(self, idx_last: int):
        """
        Estimate the terminal elimination rate constant (lambda_z) using
        log-linear regression on the terminal phase.

        Uses the best-fit approach: tests regressions with the last 3, 4, ..., n
        points and selects the one with the highest adjusted R².

        Returns
        -------
        lambda_z : float
            Terminal elimination rate constant.
        r_squared : float
            R² of the best fit.
        n_points : int
            Number of points used.
        """
        # Work with data after Tmax only
        idx_max = np.argmax(self.concentration[: idx_last + 1])
        t_term = self.time[idx_max : idx_last + 1]
        c_term = self.concentration[idx_max : idx_last + 1]

        # Keep only positive concentrations
        mask = c_term > 0
        t_term = t_term[mask]
        c_term = c_term[mask]
        ln_c = np.log(c_term)

        if len(t_term) < 3:
            return 0.0, 0.0, 0

        best_r2_adj = -np.inf
        best_slope = 0.0
        best_r2 = 0.0
        best_n = 0

        # Test regressions from last 3 points up to all terminal points
        for n in range(3, len(t_term) + 1):
            t_subset = t_term[-n:]
            ln_c_subset = ln_c[-n:]

            slope, _intercept, r_value, _p_value, _std_err = stats.linregress(
                t_subset, ln_c_subset
            )
            r2 = r_value**2
            # Adjusted R²
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else r2

            if slope < 0 and r2_adj > best_r2_adj:
                best_r2_adj = r2_adj
                best_slope = slope
                best_r2 = r2
                best_n = n

        lambda_z = -best_slope
        return lambda_z, best_r2, best_n
