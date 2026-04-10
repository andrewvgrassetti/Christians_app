# 💊 PK Analysis Tool for Biotechnology

A comprehensive pharmacokinetic (PK) analysis tool built for biotechnology applications. Perform non-compartmental analysis (NCA) and compartmental modeling through an interactive web interface.

## Features

- **Non-Compartmental Analysis (NCA)**
  - Cmax, Tmax, Clast, Tlast
  - AUC (linear-log trapezoidal method): AUC_last, AUC_inf, % extrapolated
  - Terminal half-life (t½) via best-fit log-linear regression
  - Clearance (CL or CL/F) and Volume of distribution (Vd or Vd/F)
  - Mean Residence Time (MRT)
  - AUMC (first moment)

- **Compartmental Modeling**
  - One-compartment IV bolus
  - One-compartment extravascular (oral/SC) with first-order absorption
  - Two-compartment IV bolus
  - Nonlinear least squares curve fitting with automatic initial estimates
  - AIC-based model comparison

- **Interactive Visualization** (Plotly)
  - Concentration-time profiles (linear and semi-log)
  - Model fit overlays
  - Residual plots
  - Goodness-of-fit (observed vs predicted)
  - Multi-subject overlay plots
  - Model comparison charts

- **Data Import/Export**
  - CSV and Excel file upload
  - Auto-detection of Time and Concentration columns
  - Downloadable results (CSV)
  - Sample datasets included

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/andrewvgrassetti/Christians_app.git
cd Christians_app

# Install dependencies
pip install -r requirements.txt
```

### Run the Web Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Use as a Python Library

```python
import numpy as np
from pk_analysis import NCAAnalysis, OneCompartmentIV, PKPlotter

# Example: NCA
time = np.array([0, 0.5, 1, 2, 4, 6, 8, 12, 24])
conc = np.array([0, 450, 380, 300, 210, 100, 48, 23, 5.2])

nca = NCAAnalysis(time, conc, dose=100e6)  # dose in ng
result = nca.run()
print(result.to_dict())

# Example: Compartmental fitting
model = OneCompartmentIV(dose=100e6)
fit = model.fit(time, conc)
print(fit.parameters)

# Example: Plotting
fig = PKPlotter.concentration_time(time, conc, predicted=fit.predicted)
fig.show()
```

## Project Structure

```
Christians_app/
├── app.py                      # Streamlit web application
├── pk_analysis/                # Core analysis library
│   ├── __init__.py
│   ├── nca.py                  # Non-compartmental analysis
│   ├── compartmental.py        # Compartmental PK models
│   ├── data_io.py              # Data import/export utilities
│   └── plotting.py             # Visualization module
├── sample_data/                # Example datasets
│   ├── iv_bolus_1comp.csv
│   ├── iv_bolus_2comp.csv
│   ├── oral_1comp.csv
│   └── multi_subject_oral.csv
├── tests/                      # Test suite
│   ├── test_nca.py
│   ├── test_compartmental.py
│   ├── test_data_io.py
│   └── test_plotting.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Supported Routes of Administration

| Route | Models Available |
|-------|-----------------|
| IV Bolus | 1-compartment, 2-compartment |
| Extravascular (Oral/SC) | 1-compartment with first-order absorption |

## PK Parameters Calculated

| Parameter | Description |
|-----------|-------------|
| Cmax | Maximum observed concentration |
| Tmax | Time of maximum concentration |
| AUC_last | AUC to last measurable concentration |
| AUC_inf | AUC extrapolated to infinity |
| λz | Terminal elimination rate constant |
| t½ | Terminal half-life |
| CL (or CL/F) | Clearance (or apparent clearance) |
| Vd (or Vd/F) | Volume of distribution (or apparent Vd) |
| MRT | Mean residence time |

## Technology Stack

- **Python 3.9+**
- **NumPy / SciPy** — numerical computation and curve fitting
- **Pandas** — data handling
- **Plotly** — interactive visualizations
- **Streamlit** — web application framework