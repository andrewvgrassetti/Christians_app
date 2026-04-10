"""
PK Plotting Module
===================

Provides visualization utilities for pharmacokinetic data and model fits
using Plotly for interactive plots and Matplotlib for static exports.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


class PKPlotter:
    """Plotting utilities for PK concentration-time data."""

    @staticmethod
    def concentration_time(
        time: np.ndarray,
        concentration: np.ndarray,
        predicted: Optional[np.ndarray] = None,
        title: str = "Concentration-Time Profile",
        time_label: str = "Time (h)",
        conc_label: str = "Concentration (ng/mL)",
        log_y: bool = False,
    ) -> go.Figure:
        """
        Plot observed concentration-time data with optional model prediction.

        Parameters
        ----------
        time : array-like
            Time points.
        concentration : array-like
            Observed concentrations.
        predicted : array-like, optional
            Model-predicted concentrations.
        title : str
            Plot title.
        time_label, conc_label : str
            Axis labels.
        log_y : bool
            If True, use logarithmic y-axis.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        fig = go.Figure()

        # Observed data
        fig.add_trace(
            go.Scatter(
                x=time,
                y=concentration,
                mode="markers",
                name="Observed",
                marker=dict(size=8, color="#2196F3"),
            )
        )

        # Predicted curve
        if predicted is not None:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=predicted,
                    mode="lines",
                    name="Predicted",
                    line=dict(color="#F44336", width=2),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=time_label,
            yaxis_title=conc_label,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        if log_y:
            fig.update_yaxes(type="log")

        return fig

    @staticmethod
    def semi_log_plot(
        time: np.ndarray,
        concentration: np.ndarray,
        title: str = "Semi-Log Concentration-Time Profile",
    ) -> go.Figure:
        """Plot concentration-time data on a semi-logarithmic scale."""
        return PKPlotter.concentration_time(
            time, concentration, title=title, log_y=True
        )

    @staticmethod
    def residual_plot(
        time: np.ndarray,
        residuals: np.ndarray,
        title: str = "Residual Plot",
    ) -> go.Figure:
        """Plot residuals vs time."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(size=8, color="#4CAF50"),
            )
        )

        # Zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title=title,
            xaxis_title="Time (h)",
            yaxis_title="Residual (Observed − Predicted)",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def comparison_plot(
        time: np.ndarray,
        concentration: np.ndarray,
        fits: dict,
        title: str = "Model Comparison",
    ) -> go.Figure:
        """
        Compare multiple model fits on the same plot.

        Parameters
        ----------
        time : array-like
            Observed time points.
        concentration : array-like
            Observed concentrations.
        fits : dict
            Dictionary of {model_name: predicted_array}.
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time,
                y=concentration,
                mode="markers",
                name="Observed",
                marker=dict(size=8, color="#212121"),
            )
        )

        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, (name, pred) in enumerate(fits.items()):
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=pred,
                    mode="lines",
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time (h)",
            yaxis_title="Concentration (ng/mL)",
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    @staticmethod
    def multi_subject_plot(
        data: pd.DataFrame,
        time_col: str = "Time",
        conc_col: str = "Concentration",
        subject_col: str = "Subject",
        title: str = "Multi-Subject PK Profiles",
        log_y: bool = False,
    ) -> go.Figure:
        """
        Plot concentration-time profiles for multiple subjects.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with Time, Concentration, and Subject columns.
        """
        fig = go.Figure()

        subjects = data[subject_col].unique()
        for subj in subjects:
            subj_data = data[data[subject_col] == subj]
            fig.add_trace(
                go.Scatter(
                    x=subj_data[time_col],
                    y=subj_data[conc_col],
                    mode="lines+markers",
                    name=f"Subject {subj}",
                    marker=dict(size=5),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time (h)",
            yaxis_title="Concentration (ng/mL)",
            template="plotly_white",
            hovermode="x unified",
        )

        if log_y:
            fig.update_yaxes(type="log")

        return fig

    @staticmethod
    def goodness_of_fit(
        observed: np.ndarray,
        predicted: np.ndarray,
        title: str = "Goodness of Fit",
    ) -> go.Figure:
        """Plot observed vs predicted with identity line."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=observed,
                mode="markers",
                name="Data",
                marker=dict(size=8, color="#2196F3"),
            )
        )

        # Identity line
        all_vals = np.concatenate([observed, predicted])
        min_val, max_val = np.min(all_vals), np.max(all_vals)
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Identity",
                line=dict(color="gray", dash="dash"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Predicted Concentration",
            yaxis_title="Observed Concentration",
            template="plotly_white",
        )

        return fig
