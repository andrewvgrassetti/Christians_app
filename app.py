"""
PK Analysis Tool — Streamlit Web Application
==============================================

Interactive web interface for pharmacokinetic analysis.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

from pk_analysis.nca import NCAAnalysis
from pk_analysis.compartmental import (
    OneCompartmentIV,
    OneCompartmentOral,
    TwoCompartmentIV,
)
from pk_analysis.plotting import PKPlotter
from pk_analysis.data_io import load_pk_data, results_to_csv_bytes

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PK Analysis Tool",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — Data input
# ---------------------------------------------------------------------------
st.sidebar.title("💊 PK Analysis Tool")
st.sidebar.markdown("**Pharmacokinetic Analysis for Biotechnology**")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data Source",
    ["Upload File", "Sample Data", "Manual Entry"],
)

df = None

if data_source == "Upload File":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded:
        try:
            df = load_pk_data(uploaded)
            st.sidebar.success(f"Loaded {len(df)} data points.")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

elif data_source == "Sample Data":
    sample_choice = st.sidebar.selectbox(
        "Select sample dataset",
        ["IV Bolus (1-compartment)", "Oral (1-compartment)", "IV Bolus (2-compartment)"],
    )
    if sample_choice == "IV Bolus (1-compartment)":
        df = pd.DataFrame(
            {
                "Time": [0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24],
                "Concentration": [
                    0,
                    450.0,
                    380.0,
                    300.0,
                    210.0,
                    100.0,
                    48.0,
                    23.0,
                    5.2,
                    0.12,
                ],
            }
        )
    elif sample_choice == "Oral (1-compartment)":
        df = pd.DataFrame(
            {
                "Time": [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24],
                "Concentration": [
                    0,
                    85.0,
                    170.0,
                    200.0,
                    190.0,
                    140.0,
                    95.0,
                    40.0,
                    16.0,
                    2.5,
                    0.02,
                ],
            }
        )
    else:
        df = pd.DataFrame(
            {
                "Time": [0, 0.083, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24, 48],
                "Concentration": [
                    0,
                    900.0,
                    700.0,
                    520.0,
                    350.0,
                    200.0,
                    100.0,
                    65.0,
                    45.0,
                    22.0,
                    5.0,
                    0.3,
                ],
            }
        )
    st.sidebar.success(f"Loaded sample: {sample_choice}")

elif data_source == "Manual Entry":
    st.sidebar.markdown("Enter data as comma-separated values:")
    time_input = st.sidebar.text_area(
        "Time points (h)",
        value="0, 0.5, 1, 2, 4, 6, 8, 12, 24",
    )
    conc_input = st.sidebar.text_area(
        "Concentrations (ng/mL)",
        value="0, 85, 170, 190, 140, 95, 40, 16, 2.5",
    )
    if time_input and conc_input:
        try:
            times = [float(x.strip()) for x in time_input.split(",")]
            concs = [float(x.strip()) for x in conc_input.split(",")]
            if len(times) == len(concs):
                df = pd.DataFrame({"Time": times, "Concentration": concs})
                st.sidebar.success(f"Loaded {len(df)} data points.")
            else:
                st.sidebar.error("Time and concentration arrays must have the same length.")
        except ValueError:
            st.sidebar.error("Invalid input. Please enter numeric values separated by commas.")

# ---------------------------------------------------------------------------
# Sidebar — Analysis parameters
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Parameters")

dose = st.sidebar.number_input("Dose (mg)", min_value=0.001, value=100.0, step=1.0)
dose_units = st.sidebar.selectbox("Dose Units", ["mg", "µg", "ng"])

# Convert dose to match concentration units (ng/mL)
dose_conversion = {"mg": 1e6, "µg": 1e3, "ng": 1.0}
dose_in_ng = dose * dose_conversion[dose_units]

route = st.sidebar.selectbox("Route of Administration", ["Extravascular (Oral/SC)", "IV Bolus"])
route_code = "iv" if "IV" in route else "ev"

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("💊 Pharmacokinetic Analysis Tool")
st.markdown(
    """
    A comprehensive PK analysis tool for biotechnology applications.
    Upload your concentration-time data or use sample datasets to perform
    **Non-Compartmental Analysis (NCA)** and **Compartmental Modeling**.
    """
)

if df is not None:
    # --- Data preview ---
    st.header("📊 Data Preview")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(df, use_container_width=True, height=300)

    with col2:
        fig_raw = PKPlotter.concentration_time(
            df["Time"].values,
            df["Concentration"].values,
            title="Raw Concentration-Time Data",
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    # --- Semi-log plot ---
    with st.expander("📈 Semi-Log Plot"):
        positive_mask = df["Concentration"] > 0
        if positive_mask.any():
            fig_log = PKPlotter.semi_log_plot(
                df.loc[positive_mask, "Time"].values,
                df.loc[positive_mask, "Concentration"].values,
            )
            st.plotly_chart(fig_log, use_container_width=True)

    # ===================================================================
    # Tab-based analysis
    # ===================================================================
    tab_nca, tab_comp, tab_compare = st.tabs(
        ["🧪 Non-Compartmental Analysis", "📐 Compartmental Modeling", "⚖️ Model Comparison"]
    )

    # --- NCA ---
    with tab_nca:
        st.header("Non-Compartmental Analysis (NCA)")
        st.markdown(
            "NCA calculates PK parameters directly from the observed data "
            "without assuming a specific compartmental model."
        )

        try:
            nca = NCAAnalysis(
                df["Time"].values,
                df["Concentration"].values,
                dose=dose_in_ng,
                route=route_code,
            )
            nca_result = nca.run()

            col_params, col_info = st.columns([2, 1])

            with col_params:
                st.subheader("PK Parameters")
                result_df = nca_result.to_dataframe()
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                # Download button
                csv_bytes = results_to_csv_bytes(nca_result.to_dict())
                st.download_button(
                    "⬇️ Download NCA Results (CSV)",
                    data=csv_bytes,
                    file_name="nca_results.csv",
                    mime="text/csv",
                )

            with col_info:
                st.subheader("Key Metrics")
                st.metric("Cmax", f"{nca_result.cmax:.2f} ng/mL")
                st.metric("Tmax", f"{nca_result.tmax:.2f} h")
                st.metric("AUC∞", f"{nca_result.auc_inf:.2f} ng·h/mL")
                st.metric("t½", f"{nca_result.t_half:.2f} h")

                if nca_result.warnings:
                    st.warning("\n".join(nca_result.warnings))

        except Exception as e:
            st.error(f"NCA analysis failed: {e}")

    # --- Compartmental Modeling ---
    with tab_comp:
        st.header("Compartmental Modeling")

        model_choice = st.selectbox(
            "Select Model",
            [
                "One-Compartment IV Bolus",
                "One-Compartment Oral (Extravascular)",
                "Two-Compartment IV Bolus",
            ],
        )

        if st.button("🔬 Fit Model", type="primary"):
            time_arr = df["Time"].values
            conc_arr = df["Concentration"].values

            with st.spinner("Fitting model..."):
                try:
                    if model_choice == "One-Compartment IV Bolus":
                        model = OneCompartmentIV(dose=dose_in_ng)
                    elif model_choice == "One-Compartment Oral (Extravascular)":
                        model = OneCompartmentOral(dose=dose_in_ng)
                    else:
                        model = TwoCompartmentIV(dose=dose_in_ng)

                    fit_result = model.fit(time_arr, conc_arr)

                    if fit_result.converged:
                        st.success(f"✅ Model converged! R² = {fit_result.r_squared:.6f}")

                        # Parameters table
                        st.subheader("Fitted Parameters")
                        param_df = pd.DataFrame(
                            {
                                "Parameter": list(fit_result.parameters.keys()),
                                "Value": [
                                    f"{v:.6f}" for v in fit_result.parameters.values()
                                ],
                            }
                        )
                        st.dataframe(param_df, use_container_width=True, hide_index=True)
                        st.markdown(f"**AIC:** {fit_result.aic:.2f}")

                        # Plots
                        col_fit, col_resid = st.columns(2)
                        with col_fit:
                            fig_fit = PKPlotter.concentration_time(
                                time_arr,
                                conc_arr,
                                predicted=fit_result.predicted,
                                title=f"{model_choice} — Fit",
                            )
                            st.plotly_chart(fig_fit, use_container_width=True)

                        with col_resid:
                            fig_resid = PKPlotter.residual_plot(
                                time_arr, fit_result.residuals
                            )
                            st.plotly_chart(fig_resid, use_container_width=True)

                        # Goodness of fit
                        fig_gof = PKPlotter.goodness_of_fit(
                            conc_arr, fit_result.predicted
                        )
                        st.plotly_chart(fig_gof, use_container_width=True)

                        # Download
                        csv_bytes = results_to_csv_bytes(fit_result.to_dict())
                        st.download_button(
                            "⬇️ Download Model Results (CSV)",
                            data=csv_bytes,
                            file_name="compartmental_results.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error(
                            f"❌ Model did not converge: {fit_result.message}"
                        )
                except Exception as e:
                    st.error(f"Fitting failed: {e}")

    # --- Model Comparison ---
    with tab_compare:
        st.header("Model Comparison")
        st.markdown("Compare all applicable models on the same dataset.")

        if st.button("🔄 Run All Models", type="primary"):
            time_arr = df["Time"].values
            conc_arr = df["Concentration"].values
            comparison_results = []
            fits_for_plot = {}

            with st.spinner("Fitting all models..."):
                # 1-Compartment IV
                try:
                    m = OneCompartmentIV(dose=dose_in_ng)
                    r = m.fit(time_arr, conc_arr)
                    if r.converged:
                        comparison_results.append(r.to_dict())
                        fits_for_plot[r.model_name] = r.predicted
                except Exception:
                    pass

                # 1-Compartment Oral
                try:
                    m = OneCompartmentOral(dose=dose_in_ng)
                    r = m.fit(time_arr, conc_arr)
                    if r.converged:
                        comparison_results.append(r.to_dict())
                        fits_for_plot[r.model_name] = r.predicted
                except Exception:
                    pass

                # 2-Compartment IV
                try:
                    m = TwoCompartmentIV(dose=dose_in_ng)
                    r = m.fit(time_arr, conc_arr)
                    if r.converged:
                        comparison_results.append(r.to_dict())
                        fits_for_plot[r.model_name] = r.predicted
                except Exception:
                    pass

            if comparison_results:
                # Summary table
                summary_data = []
                for cr in comparison_results:
                    summary_data.append(
                        {
                            "Model": cr["Model"],
                            "AIC": cr["AIC"],
                            "R²": cr["R²"],
                            "Converged": cr["Converged"],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True,
                    hide_index=True,
                )

                # Comparison plot
                fig_comp = PKPlotter.comparison_plot(
                    time_arr, conc_arr, fits_for_plot
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Best model
                best = min(comparison_results, key=lambda x: x["AIC"])
                st.info(
                    f"🏆 Best model by AIC: **{best['Model']}** (AIC = {best['AIC']:.2f})"
                )
            else:
                st.warning("No models converged successfully.")

else:
    # No data loaded — show instructions
    st.info(
        "👈 Use the sidebar to upload your data, select a sample dataset, "
        "or manually enter concentration-time values to get started."
    )

    st.markdown("---")
    st.header("About This Tool")
    st.markdown(
        """
        ### Features
        - **Non-Compartmental Analysis (NCA)**: Cmax, Tmax, AUC, half-life,
          clearance, volume of distribution, MRT
        - **Compartmental Modeling**: One-compartment (IV & oral) and
          two-compartment (IV) models with curve fitting
        - **Model Comparison**: Compare models side-by-side using AIC
        - **Interactive Visualization**: Concentration-time profiles, semi-log
          plots, residuals, goodness-of-fit
        - **Data Import/Export**: CSV and Excel support, downloadable results

        ### Supported Routes
        - Intravenous (IV) bolus
        - Extravascular (oral, subcutaneous)

        ### Getting Started
        1. Upload a CSV or Excel file with **Time** and **Concentration** columns
        2. Or select a built-in sample dataset
        3. Set your dose and route of administration
        4. Navigate the analysis tabs for NCA and compartmental results
        """
    )

# Footer
st.markdown("---")
st.caption("PK Analysis Tool v1.0.0 | Built for Biotechnology Applications")
