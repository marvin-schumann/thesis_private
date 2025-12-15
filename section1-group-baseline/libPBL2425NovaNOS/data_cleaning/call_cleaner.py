# libPBL2425NovaNOS/data_cleaning/call_cleaner.py
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add project root to Python path to locate the 'config' module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import settings # settings might be used for column names in future

logger = logging.getLogger(__name__)

def add_ftr_dependent_column(calls_df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'FTR_depen' column: 1 if FTR_1_SUM > 0, else 0."""
    df = calls_df.copy()
    
    # Assuming 'FTR_1_SUM' is the relevant column name from the raw calls data.
    # This could be parameterized via settings.py if it varies.
    ftr_sum_col = 'FTR_1_SUM' 

    if ftr_sum_col not in df.columns:
        logger.warning(f"'{ftr_sum_col}' column not found in calls_df. Cannot create 'FTR_depen'. 'FTR_depen' will be set to 0.")
        df['FTR_depen'] = 0 
        return df

    df[ftr_sum_col] = pd.to_numeric(df[ftr_sum_col], errors='coerce')
    df['FTR_depen'] = np.where(df[ftr_sum_col] > 0, 1, 0)
    
    # Log if coercion introduced NaNs that were not already NaNs in the input
    if df[ftr_sum_col].isnull().any() and not calls_df[ftr_sum_col].isnull().all(): # Check if new NaNs appeared
        # Check if original column was already numeric. If not, coercion is expected.
        if not pd.api.types.is_numeric_dtype(calls_df[ftr_sum_col]):
             logger.info(f"Coercion of '{ftr_sum_col}' to numeric resulted in NaNs. This might be expected if original column was not purely numeric.")
        else: # Original was numeric, but still Nans appeared - this should not happen unless errors='coerce' changed existing numbers to NaN, which is unlikely
             logger.warning(f"Unexpected NaNs appeared in '{ftr_sum_col}' after to_numeric, though original was numeric.")
    return df

def initial_call_cleaning(calls_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial cleaning on the calls_df.
    - Adds FTR_depen (if FTR_1_SUM exists)
    - Drops specified columns (could be moved to settings.py)
    - Drops rows with any NaNs (this is an aggressive step)
    """
    logger.info("Performing initial call data cleaning...")
    if calls_df.empty:
        logger.warning("Input calls_df is empty. Skipping cleaning.")
        return calls_df
        
    df = calls_df.copy()
    df = add_ftr_dependent_column(df)
    
    # Columns to drop: ideally, these would be listed in settings.py
    # For example: settings.CALL_CLEANER_COLUMNS_TO_DROP
    cols_to_drop_config_key = "CALL_CLEANER_COLUMNS_TO_DROP" # Example key for settings
    if hasattr(settings, cols_to_drop_config_key) and isinstance(getattr(settings, cols_to_drop_config_key), list):
        cols_to_drop = getattr(settings, cols_to_drop_config_key)
    else:
        cols_to_drop = ['PERSON_ID', 'GROUP_KEY', 'Column1'] # Fallback to original list
        logger.debug(f"'{cols_to_drop_config_key}' not found or not a list in settings. Using default: {cols_to_drop}")
        
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        logger.info(f"Dropped columns: {existing_cols_to_drop}")
    else:
        logger.info("No specified columns for dropping were found in calls_df.")

    # Aggressive dropna - be cautious as this can remove a lot of data.
    # Consider whether this step is appropriate or if NaNs should be handled differently.
    rows_before_dropna = len(df)
    df = df.dropna()
    rows_after_dropna = len(df)
    
    if rows_before_dropna > 0 and rows_after_dropna == 0:
        logger.warning("dropna() removed all rows from calls_df. This indicates widespread NaNs. Review data quality and previous cleaning steps.")
    elif rows_before_dropna > rows_after_dropna:
        logger.info(f"Dropped {rows_before_dropna - rows_after_dropna} rows due to NaNs.")
    else:
        logger.info("No rows dropped by dropna().")
    
    logger.info(f"Initial call cleaning complete. Output DataFrame shape: {df.shape}")
    return df