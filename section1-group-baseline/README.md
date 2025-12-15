# Section 1: Group Baseline - XGBoost Prediction Models

**Authors:** All (Maria Teresa Daffan, Runa Maria Kleppek, Marvin Schumann, Raquel Santos)

## Research Question

Can we accurately predict call handling metrics (TMC, FTR, OT) using XGBoost models to enable cost-optimized call routing?

## Project Overview

This section contains a Python-based modelling pipeline designed to predict telecom customer interaction metrics and simulate optimal resource allocation. The pipeline processes raw data through stages of access, cleaning, merging, and modelling, ultimately training models for:

- **TMC (Tempo Médio de Chamada)**: Call duration - Regression model (R² = 0.12)
- **FTR (First Time Resolution)**: Issue resolved on first contact - Classification model (AUC = 0.72)
- **OT (Ordem de Trabalho)**: Technician visit required - Classification model (AUC = 0.65)

The pipeline is structured into modular sub-libraries: `data_access`, `data_cleaning`, and `modelling`.

## Cost Formula

The cost of each call is defined as:

```
Cost = (TMC/60) × 0.35 + (1-FTR) × (TMC/60) × 0.35 + OT × 22
```

## Dependencies

This section is the **foundation** for:
- **Section 3**: Uses the same data methodology
- **Section 5**: Uses the trained XGBoost models for the RL simulator

## Directory Structure

```
section1-group-baseline/
├── config/                          # Configuration files
│   ├── __init__.py
│   └── settings.py                  # Pipeline settings, paths, column names
├── libPBL2425NovaNOS/               # Core library for the pipeline
│   ├── __init__.py
│   ├── data_access/                 # Module for data loading
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── data_cleaning/               # Module for data cleaning tasks
│   │   ├── __init__.py
│   │   ├── call_cleaner.py
│   │   ├── client_cleaner.py
│   │   ├── gc_cleaner.py
│   │   └── merger.py
│   └── modelling/                   # Module for model training and simulation
│       ├── __init__.py
│       └── modelling.py
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── 01_Cleaning_and_Modelling.ipynb
│   ├── 02_Cleaning_and_Pre-Processing.ipynb
│   ├── 03_Modelling.ipynb
│   └── 04_Best_Models.ipynb
├── tests/                           # Unit tests for pipeline components
│   ├── __init__.py
│   ├── data_access/
│   └── data_cleaning/
├── run_pipeline.py                  # Main script to execute the pipeline
├── check_feature_importance.py      # Feature importance analysis
├── compute_gc_residuals.py          # GC residual computation
├── residual_adjustments.py          # Residual adjustment utilities
└── README.md                        # This file
```

## Pipeline Stages

The pipeline is orchestrated by `run_pipeline.py` and consists of the following stages:

### 1. Data Access

- **Module**: `libPBL2425NovaNOS.data_access.loader`
- **Functionality**:
  - Connects to a PostgreSQL database using credentials from `config/settings.py`
  - Fetches raw data tables
  - Concatenates different periods of client and call data
  - Outputs: `clients_df`, `calls_df`, `gcs_df`

### 2. Data Cleaning

**Call Data Cleaning** (`call_cleaner.py`):
- Adds `FTR_dependent` column
- Drops irrelevant columns
- Removes rows with NaN values

**Client Data Cleaning** (`client_cleaner.py`):
- Age cleaning and imputation
- Rolling window metrics processing
- Call-related feature processing
- Forward-filling and imputation with flags

**GC Data Cleaning** (`gc_cleaner.py`):
- Removes median columns and specified columns
- Calculates `DAYS_ACTIVE` for each GC
- Aggregates GC data per `RESOURCE_KEY`
- Creates binary flags for call categories

### 3. Data Merging

- **Module**: `libPBL2425NovaNOS.data_cleaning.merger`
- **Functionality**:
  - Prefixes columns to avoid name clashes
  - Performs `merge_asof` between call and client data
  - Left merge with GC data on `RESOURCE_KEY`
  - Outputs: `final_modelling_df`

### 4. Modelling

- **Module**: `libPBL2425NovaNOS.modelling.modelling`
- **Functionality**:
  - **Feature Engineering**: Creates target variables and features (time-based, topic interactions, cyclical)
  - **Preprocessing**: One-Hot Encoding for categorical, StandardScaler for numerical
  - **Model Training**: XGBoost models with SMOTE for class imbalance
  - **Model Evaluation**: R², RMSE, ROC AUC, F1-score, Precision, Recall
  - **GC Assignment Simulation**: Predicts costs for all GC-call combinations

## Configuration

**`config/settings.py`**: Central configuration file defining:
- Database credentials (loaded from `.env`)
- Database schema and table names
- Column lists for cleaning and feature engineering
- Logging configuration

**Database credentials**: Create a `.env` file in the `config/` directory:
```
DB_HOST="your_db_host"
DB_PORT="your_db_port"
DB_NAME="your_db_name"
DB_USER="your_db_user"
DB_PASSWORD="your_db_password"
```

## How to Run

### Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

3. Configure database credentials in `config/.env`

### Running the Pipeline

```bash
cd section1-group-baseline
python run_pipeline.py
```

The script will log progress to the console and to a log file.

## Output

The pipeline execution will:
- Train three XGBoost models (TMC, FTR, OT)
- Print model evaluation metrics
- Generate GC assignment simulation results

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_Cleaning_and_Modelling.ipynb` | Full pipeline from data access to model creation |
| `02_Cleaning_and_Pre-Processing.ipynb` | Data cleaning and preprocessing details |
| `03_Modelling.ipynb` | Modelling exploration |
| `04_Best_Models.ipynb` | Best performing model analysis |

## Requirements

- Python 3.8+
- PostgreSQL database access
- See root `requirements.txt` for dependencies

## Notes

- Data is not included (NOS confidential)
- Database credentials should be kept confidential
- Models can be saved to disk using `joblib`
