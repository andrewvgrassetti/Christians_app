"""Tests for the plotting module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pk_analysis.plotting import PKPlotter


class TestPKPlotter:
    """Tests for PKPlotter methods."""

    def test_concentration_time_returns_figure(self):
        t = np.array([0, 1, 2, 3, 4])
        c = np.array([0, 100, 80, 40, 10])
        fig = PKPlotter.concentration_time(t, c)
        assert isinstance(fig, go.Figure)

    def test_concentration_time_with_predicted(self):
        t = np.array([0, 1, 2, 3, 4])
        c = np.array([0, 100, 80, 40, 10])
        pred = np.array([0, 95, 78, 42, 12])
        fig = PKPlotter.concentration_time(t, c, predicted=pred)
        assert len(fig.data) == 2  # observed + predicted

    def test_semi_log_plot(self):
        t = np.array([0, 1, 2, 3, 4])
        c = np.array([1, 100, 80, 40, 10])
        fig = PKPlotter.semi_log_plot(t, c)
        assert isinstance(fig, go.Figure)
        assert fig.layout.yaxis.type == "log"

    def test_residual_plot(self):
        t = np.array([0, 1, 2, 3, 4])
        r = np.array([0.5, -1.0, 2.0, -0.5, 0.1])
        fig = PKPlotter.residual_plot(t, r)
        assert isinstance(fig, go.Figure)

    def test_comparison_plot(self):
        t = np.array([0, 1, 2, 3])
        c = np.array([0, 100, 50, 10])
        fits = {
            "Model A": np.array([0, 95, 52, 12]),
            "Model B": np.array([0, 98, 48, 11]),
        }
        fig = PKPlotter.comparison_plot(t, c, fits)
        assert len(fig.data) == 3  # observed + 2 models

    def test_multi_subject_plot(self):
        data = pd.DataFrame(
            {
                "Time": [0, 1, 2, 0, 1, 2],
                "Concentration": [0, 100, 50, 0, 90, 45],
                "Subject": [1, 1, 1, 2, 2, 2],
            }
        )
        fig = PKPlotter.multi_subject_plot(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # 2 subjects

    def test_goodness_of_fit(self):
        obs = np.array([10, 20, 30, 40])
        pred = np.array([12, 19, 31, 38])
        fig = PKPlotter.goodness_of_fit(obs, pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # data + identity line
