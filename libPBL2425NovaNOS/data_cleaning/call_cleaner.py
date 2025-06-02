# data_pipeline_nos/data_cleaning/call_cleaner.py
import pandas as pd
import numpy as np
# from ..config import settings # Or use relative imports
# from ..utils import helpers # If you have helpers

def add_ftr_dependent_column(calls_df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'FTR_depen' column: 1 if FTR_1_SUM > 1, else 0."""
    df = calls_df.copy()
    # Ensure FTR_1_SUM is numeric, coercing errors for safety if not already
    df['FTR_1_SUM'] = pd.to_numeric(df['FTR_1_SUM'], errors='coerce')
    df['FTR_depen'] = np.where(df['FTR_1_SUM'] >= 1, 1, 0)
    # Handle NaNs that might result from coercion or original NaNs in FTR_1_SUM
    # If FTR_1_SUM was NaN, FTR_depen will be 0. Decide if this is correct or if it needs specific handling.
    return df

def rename_call_columns(calls_df: pd.DataFrame) -> pd.DataFrame:
    """Renames call_df columns by prefixing with 'call_'."""
    df = calls_df.copy()
    df = df.rename(columns={col: f"call_{col}" for col in df.columns})
    return df

def initial_call_cleaning(calls_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial cleaning on the calls_df.
    - Adds FTR_depen
    - Renames columns
    - Drops specified columns (original PERSON_ID etc. *after* renaming if still needed)
    - (Consider adding dropna logic here if call_df_dropped is the target)
    """
    print("Performing initial call data cleaning...")
    df = calls_df.copy()
    df = add_ftr_dependent_column(df)
    # df = rename_call_columns(df) # Do this LATER, just before merging, or handle name clashes more carefully.
                                # Renaming too early makes subsequent steps harder to map from the notebook.

    