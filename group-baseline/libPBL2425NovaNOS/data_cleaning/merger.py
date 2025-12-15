# libPBL2425NovaNOS/data_cleaning/merger.py
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

def prefix_columns_except_keys(df: pd.DataFrame, prefix: str, key_cols: list) -> pd.DataFrame:
    """Adds a prefix to DataFrame columns, excluding specified key columns."""
    df_copy = df.copy()
    cols_to_rename = [col for col in df_copy.columns if col not in key_cols]
    renaming_map = {col: f"{prefix}{col}" for col in cols_to_rename}
    df_copy.rename(columns=renaming_map, inplace=True)
    return df_copy

def create_modeling_dataset(
    calls_cleaned_df: pd.DataFrame,
    client_cleaned_df: pd.DataFrame,
    gcs_unique_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges cleaned calls, client, and GCS unique data to create the final modeling dataset.
    """
    logger.info("--- Starting Final Merge Process to Create Modeling Dataset ---")

    if calls_cleaned_df.empty or client_cleaned_df.empty: # gcs_unique_df can sometimes be sparse
        logger.warning("One or more input DataFrames (calls_cleaned_df, client_cleaned_df) are empty. Merge will likely result in an empty DataFrame.")
        # Depending on requirements, either return empty or raise error
        if calls_cleaned_df.empty: return pd.DataFrame()

    calls_df = calls_cleaned_df.copy()
    client_df = client_cleaned_df.copy()
    gcs_df = gcs_unique_df.copy()

    # --- 1. Normalize and Prepare Date Columns for merge_asof ---
    logger.info("Normalizing and preparing date columns for merge_asof...")

    # For calls_df (using settings.DATE_COL_CALLS_RAW and settings.NORMALIZED_DATE_COL_CALLS)
    if settings.DATE_COL_CALLS_RAW not in calls_df.columns:
        logger.error(f"'{settings.DATE_COL_CALLS_RAW}' not found in calls_df.")
        raise ValueError(f"Date column '{settings.DATE_COL_CALLS_RAW}' missing in calls data.")
    calls_df[settings.NORMALIZED_DATE_COL_CALLS] = pd.to_datetime(calls_df[settings.DATE_COL_CALLS_RAW], errors='coerce').dt.normalize()
    calls_df.dropna(subset=[settings.NORMALIZED_DATE_COL_CALLS, settings.KEY_PERSON_SK], inplace=True)
    calls_df.sort_values([settings.KEY_PERSON_SK, settings.NORMALIZED_DATE_COL_CALLS], inplace=True)

    # For client_df (using settings.DATE_COL_CLIENT_RAW and settings.NORMALIZED_DATE_COL_CLIENT)
    # Note: settings.DATE_COL_CLIENT_RAW refers to "START_DATE" in current settings.
    # The client_cleaner might have already created/standardized this.
    client_date_col = settings.DATE_COL_CLIENT_RAW 
    if client_date_col not in client_df.columns:
        # Fallback if primary date column is missing (as in original merger)
        alt_client_date_col = 'LEG_START_TIME_DAY' # This was a fallback in original client_cleaner.py
        if alt_client_date_col in client_df.columns:
            logger.warning(f"'{client_date_col}' not found in client_df, using '{alt_client_date_col}'.")
            client_date_col = alt_client_date_col
        else:
            logger.error(f"A valid date column ('{settings.DATE_COL_CLIENT_RAW}' or fallback) not found in client_df.")
            raise ValueError("Client date column missing.")
            
    client_df[settings.NORMALIZED_DATE_COL_CLIENT] = pd.to_datetime(client_df[client_date_col], errors='coerce').dt.normalize()
    client_df.dropna(subset=[settings.NORMALIZED_DATE_COL_CLIENT, settings.KEY_PERSON_SK], inplace=True)
    client_df.sort_values([settings.KEY_PERSON_SK, settings.NORMALIZED_DATE_COL_CLIENT], inplace=True)
    
    # --- 2. Prefix Columns ---
    logger.info("Prefixing columns...")
    # Keys that should NOT be prefixed (normalized dates are used for merge, original dates might be kept or dropped)
    call_keys_to_exclude = [settings.KEY_PERSON_SK, settings.KEY_RESOURCE_KEY, 
                              settings.NORMALIZED_DATE_COL_CALLS, settings.DATE_COL_CALLS_RAW]
    client_keys_to_exclude = [settings.KEY_PERSON_SK, 
                                settings.NORMALIZED_DATE_COL_CLIENT, client_date_col] # client_date_col is the original one used
    gcs_keys_to_exclude = [settings.KEY_RESOURCE_KEY]

    calls_df_prefixed = prefix_columns_except_keys(calls_df, "call_", call_keys_to_exclude)
    client_df_prefixed = prefix_columns_except_keys(client_df, "client_", client_keys_to_exclude)
    gcs_df_prefixed = prefix_columns_except_keys(gcs_df, "gcs_", gcs_keys_to_exclude)

    # --- 3. Merge calls_df with client_df_clean using merge_asof ---
    logger.info("Merging calls data with client data (merge_asof)...")
    if not all(k in calls_df_prefixed.columns and k in client_df_prefixed.columns for k in [settings.KEY_PERSON_SK]):
         logger.error(f"'{settings.KEY_PERSON_SK}' missing in prefixed DFs for merge_asof.")
         raise KeyError(f"'{settings.KEY_PERSON_SK}' missing for merge_asof.")
    if not settings.NORMALIZED_DATE_COL_CALLS in calls_df_prefixed.columns or not settings.NORMALIZED_DATE_COL_CLIENT in client_df_prefixed.columns:
        logger.error("Normalized date columns missing in prefixed DFs for merge_asof.")
        raise KeyError("Normalized date columns missing for merge_asof.")


    merged_call_client_df = pd.merge_asof(
        calls_df_prefixed,
        client_df_prefixed,
        left_on=settings.NORMALIZED_DATE_COL_CALLS, # Use unprefixed normalized date from settings
        right_on=settings.NORMALIZED_DATE_COL_CLIENT, # Use unprefixed normalized date from settings
        by=settings.KEY_PERSON_SK, # Unprefixed key
        direction='backward'
    )
    logger.info(f"Shape after calls-client merge_asof: {merged_call_client_df.shape}")
    initial_rows = len(merged_call_client_df)
    # Drop rows where client data couldn't be matched (the right key 'START_DATE_normalized' will be NaN)
    merged_call_client_df.dropna(subset=[settings.NORMALIZED_DATE_COL_CLIENT], inplace=True)
    logger.info(f"Rows dropped due to no matching client record (merge_asof): {initial_rows - len(merged_call_client_df)}")

    # --- 4. Merge the result with gcs_unique_df ---
    logger.info("Merging with GCS data...")
    if gcs_df_prefixed.empty:
        logger.warning("gcs_df_prefixed is empty. Merge with GCS data will add no GCS columns or might behave unexpectedly depending on `how`.")
        # If gcs_df_prefixed is empty, a left merge will keep all from merged_call_client_df and add NaNs for gcs columns
        # To avoid issues, ensure RESOURCE_KEY exists for merge, even if GCS is empty
        if settings.KEY_RESOURCE_KEY not in merged_call_client_df:
             logger.error(f"'{settings.KEY_RESOURCE_KEY}' missing for GCS merge.")
             raise KeyError(f"'{settings.KEY_RESOURCE_KEY}' missing for GCS merge.")
        # if gcs_df_prefixed is empty, we can skip the merge or add empty columns
        # For now, let it proceed, merge will handle empty right DF.
    elif settings.KEY_RESOURCE_KEY not in merged_call_client_df or settings.KEY_RESOURCE_KEY not in gcs_df_prefixed:
        logger.error(f"'{settings.KEY_RESOURCE_KEY}' missing for GCS merge.")
        raise KeyError(f"'{settings.KEY_RESOURCE_KEY}' missing for GCS merge.")


    full_merged_df = pd.merge(
        merged_call_client_df,
        gcs_df_prefixed,
        on=settings.KEY_RESOURCE_KEY, # Unprefixed key
        how='left' 
    )
    logger.info(f"Shape after GCS merge: {full_merged_df.shape}")

    # --- 5. Drop rows with any remaining NaNs (Aggressive Step) ---
    # This step can significantly reduce dataset size. Consider if this is always desirable.
    # It might be better to handle NaNs strategically in the modelling phase.
    logger.info("Dropping rows with any remaining NaNs...")
    rows_before_final_dropna = full_merged_df.shape[0]
    full_merged_df.dropna(inplace=True)
    rows_after_final_dropna = full_merged_df.shape[0]
    logger.info(f"Shape after final dropna: {full_merged_df.shape}. Rows dropped: {rows_before_final_dropna - rows_after_final_dropna}")
    if full_merged_df.empty and rows_before_final_dropna > 0:
        logger.warning("All rows were dropped by the final dropna(). Check for widespread NaNs.")


    # --- 6. Final Column Cleanup ---
    # Drop normalized date columns used for merging, and potentially original client date if redundant.
    # The original unprefixed date columns (settings.DATE_COL_CALLS_RAW, client_date_col)
    # should still be present as they were in the exclude list for prefixing.
    # We want to keep call_CALL_START_TIME_DAT (settings.DATE_COL_CALLS_RAW) as the primary event time.
    cols_to_drop_final = [
        settings.NORMALIZED_DATE_COL_CALLS, 
        settings.NORMALIZED_DATE_COL_CLIENT
    ]
    # If the original client date column (client_date_col) is different from the main call date
    # and is not desired in the final output, it can be dropped.
    # For now, assume settings.DATE_COL_CALLS_RAW (from calls_df) is the main one to keep.
    if client_date_col != settings.DATE_COL_CALLS_RAW and client_date_col in full_merged_df.columns:
        cols_to_drop_final.append(client_date_col)
    
    # Drop only existing columns
    cols_to_drop_final_existing = [col for col in cols_to_drop_final if col in full_merged_df.columns]
    if cols_to_drop_final_existing:
        full_merged_df = full_merged_df.drop(columns=cols_to_drop_final_existing)
        logger.info(f"Dropped final columns: {cols_to_drop_final_existing}")

    logger.info(f"--- Final Modeling Dataset Created. Shape: {full_merged_df.shape} ---")
    return full_merged_df