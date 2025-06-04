# libPBL2425NovaNOS/data_cleaning/gc_cleaner.py
import pandas as pd
import numpy as np
import sys
import os
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import settings

logger = logging.getLogger(__name__)

def create_gcs_unique_with_mode_aggregation(masterdatagcs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the master GCS data and creates the gcs_unique table by:
    1. Filtering out 'MEDIAN' columns and columns specified in settings.GC_COLUMNS_TO_DROP_INITIAL.
    2. Reverting FTR values based on settings.GC_FTR_COLUMNS_PATTERN.
    3. Calculating DAYS_ACTIVE.
    4. Aggregating data per RESOURCE_KEY using mode for columns in settings.GCS_UNIQUE_AGG_COLUMNS_TO_KEEP.
    5. Creating binary flags for call categories based on settings.GCS_UNIQUE_CATEGORY_COLUMNS_FOR_BINARY_FLAGS.
    6. Filling all remaining NaNs with 0.
    """
    logger.info("Starting creation of gcs_unique table with mode aggregation...")
    if masterdatagcs_df.empty:
        logger.warning("Input masterdatagcs_df is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df = masterdatagcs_df.copy()
    logger.info(f"Initial shape of GCS data: {df.shape}")
    
    # Remove columns with 'MEDIAN' in the name
    cols_before_median_removal = df.columns.tolist()
    df = df[[col for col in df.columns if 'MEDIAN' not in col]] # Case-sensitive
    cols_after_median_removal = df.columns.tolist()
    num_median_cols_removed = len(cols_before_median_removal) - len(cols_after_median_removal)
    if num_median_cols_removed > 0:
        logger.info(f"Removed {num_median_cols_removed} columns containing 'MEDIAN'.")
    else:
        logger.info("No columns containing 'MEDIAN' found to remove.")

    # Drop initial columns (e.g., 'RESOURCE_ID' if it's the non-key one)
    # Using settings.GC_COLUMNS_TO_DROP_INITIAL
    cols_to_drop_initial = [
        col for col in settings.GC_COLUMNS_TO_DROP_INITIAL 
        if col in df.columns and col != settings.KEY_RESOURCE_KEY # Ensure we don't drop the main key
    ]
    if cols_to_drop_initial:
        df = df.drop(columns=cols_to_drop_initial)
        logger.info(f"Dropped initial GC columns based on settings.GC_COLUMNS_TO_DROP_INITIAL: {cols_to_drop_initial}")
    
    # Commented out FTR reversion logic 
    '''
    # Revert FTR values (FTR = 1 - FTR) using settings.GC_FTR_COLUMNS_PATTERN
    ftr_columns_to_revert = [col for col in df.columns if settings.GC_FTR_COLUMNS_PATTERN in col]
    if ftr_columns_to_revert:
        for col in ftr_columns_to_revert:
            df[col] = 1 - pd.to_numeric(df[col], errors='coerce') # Ensure numeric for calculation
            if df[col].isnull().any():
                logger.warning(f"NaNs produced during FTR reversion for column '{col}'. Check original data type and values.")
        logger.info(f"Reverted FTR values for columns: {ftr_columns_to_revert}")
    else:
        logger.info(f"No FTR columns found matching pattern '{settings.GC_FTR_COLUMNS_PATTERN}' for reversion.")
    '''

    # Ensure LEG_START_TIME is datetime and extract date part
    # This column name should ideally be in settings if it can vary.
    leg_start_time_col = 'LEG_START_TIME' 
    if leg_start_time_col not in df.columns:
        logger.error(f"'{leg_start_time_col}' column is required but not found in masterdatagcs_df.")
        raise ValueError(f"'{leg_start_time_col}' column is required for GCS cleaning.")
    df[leg_start_time_col] = pd.to_datetime(df[leg_start_time_col], errors='coerce')
    df['LEG_START_DATE_for_active_days'] = df[leg_start_time_col].dt.date # Use .date for nunique count of days
    
    # Calculate DAYS_ACTIVE per RESOURCE_KEY
    if settings.KEY_RESOURCE_KEY not in df.columns:
        logger.error(f"Key column '{settings.KEY_RESOURCE_KEY}' not found. Cannot calculate DAYS_ACTIVE.")
        raise ValueError(f"'{settings.KEY_RESOURCE_KEY}' not found in GCS data.")
        
    df['DAYS_ACTIVE'] = df.groupby(settings.KEY_RESOURCE_KEY)['LEG_START_DATE_for_active_days'].transform('nunique')
    df = df.drop(columns=['LEG_START_DATE_for_active_days']) # Clean up temp column
    logger.info("DAYS_ACTIVE calculated.")
    
    # Use settings.GCS_UNIQUE_AGG_COLUMNS_TO_KEEP for aggregation
    columns_for_aggregation = [col for col in settings.GCS_UNIQUE_AGG_COLUMNS_TO_KEEP if col in df.columns]
    if not columns_for_aggregation:
        logger.warning("No specified columns from settings.GCS_UNIQUE_AGG_COLUMNS_TO_KEEP exist in the GCS DataFrame after pre-processing. Aggregation might be empty.")
        # Create a DataFrame with just RESOURCE_KEY if no agg columns, or handle as error
        if settings.KEY_RESOURCE_KEY in df.columns:
            return df[[settings.KEY_RESOURCE_KEY]].drop_duplicates().reset_index(drop=True).fillna(0) # fillna(0) if any key is NaN (unlikely)
        else:
            return pd.DataFrame() # Should have failed earlier if KEY_RESOURCE_KEY missing

    def mode_agg_fn(series):
        if series.empty or series.dropna().empty:
            return np.nan
        modes = series.mode()
        # If multiple modes, pick the first. If no mode (all unique values), pandas mode() returns empty.
        # In such case, original code took series.iloc[0] if series not allna.
        # For numeric data where mode fails (e.g. all unique floats), picking first is arbitrary.
        # Returning NaN might be safer if mode is truly the desired aggregation.
        return modes.iloc[0] if not modes.empty else (series.dropna().iloc[0] if not series.dropna().empty else np.nan)


    logger.info(f"Aggregating data by '{settings.KEY_RESOURCE_KEY}' using mode for columns: {columns_for_aggregation}")
    
    gcs_unique = df.groupby(settings.KEY_RESOURCE_KEY, as_index=False)[columns_for_aggregation].agg(mode_agg_fn)
    logger.info(f"Shape after aggregation: {gcs_unique.shape}")

    # Create binary flags for call categories using settings.GCS_UNIQUE_CATEGORY_COLUMNS_FOR_BINARY_FLAGS
    logger.info("Creating binary flags for call categories based on settings.GCS_UNIQUE_CATEGORY_COLUMNS_FOR_BINARY_FLAGS...")
    for cat, count_cols_for_flag_check in settings.GCS_UNIQUE_CATEGORY_COLUMNS_FOR_BINARY_FLAGS.items():
        # Original logic: flag is 1 if ANY of the listed columns for that category are not NaN *before* fillna(0)
        # More robust: check if the main count column for the category (e.g. COUNT_CALLS_IF) is > 0
        # This assumes that if COUNT_CALLS_CAT > 0, then the category is active.
        
        # Check if the *aggregated* count columns (e.g., 'COUNT_CALLS_IF') exist in gcs_unique
        # and have a value greater than 0 (after mode aggregation and potential NaNs).
        # The list in settings.GCS_UNIQUE_CATEGORY_COLUMNS_FOR_BINARY_FLAGS refers to columns in gcs_unique.
        
        flag_col_name = f"CALLS_{cat}_BINARY"
        # Check if any of the specified indicator columns for this category exist in gcs_unique and are non-zero
        # These columns are post-aggregation.
        relevant_gcs_unique_cols_for_cat = [col for col in count_cols_for_flag_check if col in gcs_unique.columns]
        if relevant_gcs_unique_cols_for_cat:
            # Flag is 1 if any of these columns have a value > 0 (assuming counts or similar metrics)
            # Need to handle NaNs from mode aggregation before this check if they mean "no activity"
            # The final fillna(0) comes later. Here, a NaN from mode means mode couldn't be determined.
            # Let's assume NaN here means "no data for mode", so effectively 0 for count-like things.
            gcs_unique[flag_col_name] = (gcs_unique[relevant_gcs_unique_cols_for_cat].fillna(0) > 0).any(axis=1).astype(int)
            logger.debug(f"Created binary flag '{flag_col_name}' based on columns: {relevant_gcs_unique_cols_for_cat}")
        else:
            gcs_unique[flag_col_name] = 0 
            logger.warning(f"No columns found in gcs_unique for category '{cat}' based on settings to create binary flag '{flag_col_name}'. Set to 0.")
            
    logger.info("Filling remaining NaNs with 0 in gcs_unique...")
    gcs_unique = gcs_unique.fillna(0)
    
    if not gcs_unique.empty:
        nan_counts = gcs_unique.isna().sum()
        nan_counts_filtered = nan_counts[nan_counts > 0]
        if not nan_counts_filtered.empty:
            logger.warning(f"Unexpected NaNs remain in gcs_unique after fillna(0):\n{nan_counts_filtered}")
        else:
            logger.debug("No NaNs remaining in gcs_unique after fillna(0).")
    else:
        logger.warning("gcs_unique is empty after processing.")
        
    logger.info(f"Creation of gcs_unique table complete. Final shape: {gcs_unique.shape}")
    return gcs_unique