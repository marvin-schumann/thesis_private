# NOS-PBL-DELIVERABLE: Modelling Pipeline

This repository contains a Python-based modelling pipeline designed to predict telecom customer interaction metrics and simulate optimal resource allocation. The pipeline processes raw data through stages of access, cleaning, merging, and modelling, ultimately training models for Call Time (TMC), First Contact Resolution (FTR), and Work Order (OT) which is an Onsite Technician dispatch, and includes a simulation for cost-optimized Contact Manager (GC) assignment.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Directory Structure](#directory-structure)
3.  [Pipeline Stages](#pipeline-stages)
    *   [Data Access](#data-access)
    *   [Data Cleaning](#data-cleaning)
    *   [Data Merging](#data-merging)
    *   [Modelling](#modelling)
4.  [Configuration](#configuration)
5.  [Key Files](#key-files)
6.  [Setup Instructions](#setup-instructions)
7.  [Running the Pipeline](#running-the-pipeline)
8.  [Output](#output)
9.  [Notebooks](#notebooks)
10. [License](#license)

## Project Overview

The core objective of this project is to build an end-to-end data pipeline that:
*   Ingests raw data related to customer calls, client information, and guidance counselor (GC) performance.
*   Cleans and preprocesses this data.
*   Merges disparate data sources into a unified dataset suitable for modelling.
*   Trains machine learning models to predict:
    *   **TMC (Call Duration)**: Regression model.
    *   **FTR (First Contact Resolution)**: Classification model.
    *   **OT (Work Order)**: Classification model.
*   Simulates the assignment of incoming calls to GCs to minimize a defined cost function, leveraging the trained models.

The pipeline is structured into modular sub-libraries: `data_access`, `data_cleaning`, and `modelling`.

## Directory Structure
```
NOS-PBL-DELIVERABLE/
├── config/ # Configuration files
│ ├── .env # Environment variables (e.g., DB credentials - GITIGNORED)
│ ├── settings.py # Pipeline settings, paths, column names, etc.
│ └── interim/ # (As per tree; settings.py defines data/interim for outputs)
├── libPBL2425NovaNOS/ # Core library for the pipeline
│ ├── init.py
│ ├── data_access/ # Module for data loading
│ │ ├── init.py
│ │ └── loader.py
│ ├── data_cleaning/ # Module for data cleaning tasks
│ │ ├── init.py
│ │ ├── call_cleaner.py
│ │ ├── client_cleaner.py
│ │ ├── gc_cleaner.py
│ │ └── merger.py # Script to merge cleaned datasets
│ └── modelling/ # Module for model training and simulation
│ ├── init.py
│ └── modelling.py
├── notebooks/ # Jupyter notebooks for exploration, analysis, and testing
│ ├── 01_Cleaning_and_Modelling.ipynb
│ ├── 02_Cleaning_and_Pre-Processing.ipynb
│ ├── 03_Runa_Model_No_10.ipynb
│ └── 04_Best_Models.ipynb
├── tests/ # Unit tests for pipeline components
│ ├── data_access/
│ ├── data_cleaning/
│ └── init.py
├── .gitignore # Specifies intentionally untracked files
├── run_pipeline.py # Main script to execute the entire pipeline
├── requirements.txt # Project dependencies
└── LICENSE # Project license information
```


## Pipeline Stages

The pipeline is orchestrated by `run_pipeline.py` and consists of the following stages:

### Data Access
*   **Module**: `libPBL2425NovaNOS.data_access.loader`
*   **Script**: `loader.py`
*   **Functionality**:
    *   Connects to a PostgreSQL database using credentials and connection details from `config/settings.py` (sourced from `.env`).
    *   Fetches raw data tables as specified in `settings.RAW_TABLE_NAMES`.
    *   Concatenates different periods of client data (`settings.MASTER_CLIENT_TABLES_FOR_CONCAT`) and call data (`settings.MASTER_CALL_TABLES_FOR_CONCAT`).
    *   Outputs three main DataFrames: `clients_df`, `calls_df`, and `gcs_df`.

### Data Cleaning
This stage involves cleaning the three raw DataFrames.

1.  **Call Data Cleaning**:
    *   **Module**: `libPBL2425NovaNOS.data_cleaning.call_cleaner`
    *   **Script**: `call_cleaner.py`
    *   **Functionality**:
        *   Adds an `FTR_dependent` column.
        *   Drops specified irrelevant columns.
        *   Removes rows with any NaN values (Note: To ensure the final dataset is ready for modelling without requiring further imputation, any rows with remaining NaN values after the merges are dropped).
    *   **Input**: `calls_raw_df`
    *   **Output**: `calls_cleaned_df`

2.  **Client Data Cleaning**:
    *   **Module**: `libPBL2425NovaNOS.data_cleaning.client_cleaner`
    *   **Script**: `client_cleaner.py`
    *   **Functionality**: A comprehensive cleaning process involving:
        *   Age cleaning and imputation.
        *   Filling NaNs in rolling window metrics.
        *   Processing call-related features (call count flags, subscription categories, issue proportions).
        *   Forward-filling characteristic columns and imputing remaining NaNs with -1, adding imputation flags.
    *   **Input**: `clients_raw_df`
    *   **Output**: `client_cleaned_df`

3.  **GC (Call Manager) Data Cleaning**:
    *   **Module**: `libPBL2425NovaNOS.data_cleaning.gc_cleaner`
    *   **Script**: `gc_cleaner.py`
    *   **Functionality**:
        *   Removes columns with 'MEDIAN' in their names and other specified columns.
        *   Calculates `DAYS_ACTIVE` for each GC.
        *   Aggregates GC data per `RESOURCE_KEY` using mode for defined columns, creating a `gcs_unique` table.
        *   Creates binary flags for call categories associated with GCs.
        *   Fills remaining NaNs with 0.
    *   **Input**: `gcs_raw_df`
    *   **Output**: `gcs_unique_cleaned_df`

### Data Merging
*   **Module**: `libPBL2425NovaNOS.data_cleaning.merger`
*   **Script**: `merger.py`
*   **Functionality**:
    *   Takes the three cleaned DataFrames (`calls_cleaned_df`, `client_cleaned_df`, `gcs_unique_cleaned_df`).
    *   Prefixes columns of each DataFrame (e.g., `call_`, `client_`, `gcs_`) to avoid name clashes, excluding key columns.
    *   Performs a `merge_asof` between call data and client data based on `PERSON_SK` and normalized dates, associating each call with the most recent client snapshot.
    *   Performs a left merge of the result with GC data on `RESOURCE_KEY`.
    *   Drops rows with any remaining NaN values (Note: To ensure the final dataset is ready for modelling without requiring further imputation, any rows with remaining NaN values after the merges are dropped).
    *   Cleans up temporary date columns.
*   **Input**: `calls_cleaned_df`, `client_cleaned_df`, `gcs_unique_cleaned_df`
*   **Output**: `final_modelling_df`

### Modelling
*   **Module**: `libPBL2425NovaNOS.modelling.modelling`
*   **Script**: `modelling.py`
*   **Functionality**:
    *   **Feature Engineering**: Creates target variables (`TMC_dependent`, `FTR_dependent`, `OT_dependent`) and various features from `final_modelling_df` (e.g., time-based features, Level 2 topic interaction features, cyclical features).
    *   **Preprocessing**: Handles categorical features via One-Hot Encoding and scales numerical features using `StandardScaler`.
    *   **Model Training**:
        *   Splits data into training and testing sets.
        *   It applies SMOTE (from \imblearn`) to handle class imbalance** in the FTR and OT classification tasks.
        *   Trains three separate XGBoost models:
            *   `XGBRegressor` for TMC (Call Duration).
            *   `XGBClassifier` for FTR (First Contact Resolution).
            *   `XGBClassifier` for OT (Work Order).
    *   **Model Evaluation**: Calculates relevant metrics (R2, RMSE for TMC; ROC AUC, F1-score, Precision, Recall for FTR & OT).
    *   **GC Assignment Simulation**:
        *   Takes a sample of calls from the dataset.
        *   For each call, it iterates through all available GCs (from `gcs_unique_cleaned_df`).
        *   Predicts TMC, FTR probability, and OT probability for the call if handled by that specific GC.
        *   Calculates an estimated total cost for handling the call by that GC, considering call duration cost, potential technician visit cost, and potential repeat call cost.
        *   Identifies the GCs that would result in the lowest predicted cost for each sampled call.
*   **Input**: `final_modelling_df` (from merger), `gcs_unique_cleaned_df` (from gc_cleaner, for simulation)
*   **Output**: A dictionary containing trained models, scalers, evaluation metrics, and a DataFrame with simulation results. These are currently kept in memory but can be saved to disk (e.g., using `joblib`).

## Configuration

*   **`config/settings.py`**: This is the central configuration file. It defines:
    *   Database credentials (loaded from `.env`).
    *   Database schema and table names.
    *   Lists of columns for various cleaning and feature engineering steps.
    *   Logging configuration.
*   **`config/.env`**: This file (which should be in your `.gitignore`) stores sensitive information like database credentials. Create this file in the `config/` directory based on `.env.example` (if provided, otherwise manually). Example structure:
    ```
    DB_HOST="your_db_host"
    DB_PORT="your_db_port"
    DB_NAME="your_db_name"
    DB_USER="your_db_user"
    DB_PASSWORD="your_db_password"
    ```

## Key Files

*   **`run_pipeline.py`**: The main entry point to execute the entire data pipeline from data extraction to model training and simulation.
*   **`requirements.txt`**: Lists all Python dependencies required for the project.
*   **`libPBL2425NovaNOS/`**: The core Python library containing all pipeline logic.
*   **`notebooks/`**: Contains Jupyter notebooks used for development, experimentation, and detailed analysis.
*   **`tests/`**: Contains unit tests for various components of the pipeline.

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd NOS-PBL-DELIVERABLE
    ```

2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
   

4.  **Configure Environment Variables**:
    *   Create a `.env` file inside the `config/` directory (e.g., `config/.env`).
    *   Add your database credentials and any other required environment variables as shown in the [Configuration](#configuration) section.

## Running the Pipeline

To run the entire pipeline, execute the `run_pipeline.py` script from the project root directory:

```bash
python run_pipeline.py
```
The script will log its progress to the console and to a file specified in config/settings.py (default: data/logs/pipeline.log).
### Output
The pipeline execution will:
* Create/populate directories defined in config/settings.py:
* Train three XGBoost models (TMC, FTR, OT). These models and their associated scalers are currently stored in memory within the modelling_results dictionary returned by modelling.run_modelling().
* Print model evaluation metrics to the console.
* Print a sample of the GC assignment simulation results to the console.



## Notebooks
The notebooks/ directory contains Jupyter notebooks for various stages of analysis and development. These should act as resources, but are not directly necessary for the pipeline:
* 01_Cleaning_and_Modelling.ipynb: The full notebook from data access to model creation.
* 02_Cleaning_and_Pre-Processing.ipynb: Focuses on data cleaning and preprocessing details.
* 03_Modelling.ipynb: Modelling Exploration.
* 04_Best_Models.ipynb: Summarizes or showcases the best performing models

These notebooks can be used to understand the data, experiment with different approaches, and visualize results. Ensure you have Jupyter installed (pip install notebook) and run it from the project root.

## License
This project is licensed under the terms specified in the LICENSE file.
