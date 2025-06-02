# data_pipeline_nos/data_cleaning/gc_cleaner.py
import pandas as pd
import numpy as np
# from ..config import settings # For package structure
# from config import settings # For standalone testing if config is at root

def create_gcs_unique_with_mode_aggregation(masterdatagcs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the master GCS data and creates the gcs_unique table by:
    1. Filtering out 'MEDIAN' columns and 'RESOURCE_ID' (non-key).
    2. Reverting FTR values.
    3. Calculating DAYS_ACTIVE.
    4. Aggregating data per RESOURCE_KEY using mode for specified columns.
    5. Creating binary flags for call categories based on the aggregated data.
    6. Filling all remaining NaNs with 0.
    """
    print("Starting creation of gcs_unique table with mode aggregation...")

    # Step 1: Start from original masterdatagcs & initial filtering
    df = masterdatagcs_df.copy()
    print(f"Shape before removing 'MEDIAN' columns: {df.shape}")

    # Step 2: Remove columns with 'MEDIAN' in the name
    columns_to_keep_after_median_removal = [col for col in df.columns if 'MEDIAN' not in col]
    df = df[columns_to_keep_after_median_removal]
    print(f"Shape after removing 'MEDIAN' columns: {df.shape}")

    # Step 3: Drop irrelevant column RESOURCE_ID (if it's the non-key one)

    if 'RESOURCE_ID' in df.columns and 'RESOURCE_KEY' in df.columns :
        df = df.drop(columns=['RESOURCE_ID'])
        print(f"Shape after dropping 'RESOURCE_ID' (non-key): {df.shape}")
    

    # Step 4: Revert FTR values (FTR = 1 - FTR)
    ftr_columns = [col for col in df.columns if 'FTR' in col]
    for col in ftr_columns:
        df[col] = 1 - df[col]
    print("FTR values reverted.")

    
    # Ensure LEG_START_TIME is datetime and extract date part
    if 'LEG_START_TIME' not in df.columns:
        raise ValueError("LEG_START_TIME column is required but not found in masterdatagcs_df.")
    df['LEG_START_TIME'] = pd.to_datetime(df['LEG_START_TIME'], errors='coerce')
    df['LEG_START_DATE'] = df['LEG_START_TIME'].dt.date

    # Calculate DAYS_ACTIVE per RESOURCE_KEY
    grouping_key_for_days_active = 'RESOURCE_KEY'

    df['DAYS_ACTIVE'] = df.groupby(grouping_key_for_days_active)['LEG_START_DATE'].transform('nunique')
    print("DAYS_ACTIVE calculated.")

    # Define the columns to keep/aggregate (from notebook)
    columns_for_aggregation = [ # These are the columns to be aggregated using mode
        "COUNT_CALLS", "DAYS_ACTIVE",
        "MEAN_RPC", "MEAN_FTR", "MEAN_TMC", "TOTAL_OTS", "OTS_BY_CALL",
        "COUNT_CALLS_IF", "MEAN_RPC_IF", "MEAN_FTR_IF", "MEAN_TMC_IF", "TOTAL_OTS_IF", "OTS_BY_CALL_IF",
        "COUNT_CALLS_OTHER", "MEAN_RPC_OTHER", "MEAN_FTR_OTHER", "MEAN_TMC_OTHER", "TOTAL_OTS_OTHER", "OTS_BY_CALL_OTHER",
        "COUNT_CALLS_VF", "MEAN_RPC_VF", "MEAN_FTR_VF", "MEAN_TMC_VF", "TOTAL_OTS_VF", "OTS_BY_CALL_VF",
        "COUNT_CALLS_TV", "MEAN_RPC_TV", "MEAN_FTR_TV", "MEAN_TMC_TV", "TOTAL_OTS_TV", "OTS_BY_CALL_TV",
        "COUNT_CALLS_MOVEL", "MEAN_RPC_MOVEL", "MEAN_FTR_MOVEL", "MEAN_TMC_MOVEL", "TOTAL_OTS_MOVEL", "OTS_BY_CALL_MOVEL"
    ]
    # Ensure these columns actually exist in the DataFrame 'df' before trying to aggregate them
    existing_columns_for_aggregation = [col for col in columns_for_aggregation if col in df.columns]
    if not existing_columns_for_aggregation:
        print("Warning: None of the specified columns for aggregation exist in the DataFrame.")
        

    # Helper for mode aggregation
    def mode_agg_fn(series):
        if series.empty:
            return np.nan
        modes = series.mode()
        return modes.iloc[0] if not modes.empty else (series.iloc[0] if not series.dropna().empty else np.nan)

    print(f"Aggregating data by '{grouping_key_for_days_active}' using mode...")
    # Group and aggregate
    gcs_unique = (
        df.groupby(grouping_key_for_days_active, as_index=False) # Keep key as column
        [existing_columns_for_aggregation] # Select only existing columns
        .agg(mode_agg_fn) # Apply the refined mode_agg_fn
    )
    # The .reset_index() is not needed if as_index=False is used in groupby
    print(f"Shape after aggregation: {gcs_unique.shape}")


    # Define columns grouped by category for binary flags
    # These refer to columns *in the aggregated gcs_unique* DataFrame
    category_columns_for_binary_flags = {
        "IF": ["COUNT_CALLS_IF", "MEAN_RPC_IF", "MEAN_FTR_IF", "MEAN_TMC_IF", "TOTAL_OTS_IF", "OTS_BY_CALL_IF"],
        "OTHER": ["COUNT_CALLS_OTHER", "MEAN_RPC_OTHER", "MEAN_FTR_OTHER", "MEAN_TMC_OTHER", "TOTAL_OTS_OTHER", "OTS_BY_CALL_OTHER"],
        "VF": ["COUNT_CALLS_VF", "MEAN_RPC_VF", "MEAN_FTR_VF", "MEAN_TMC_VF", "TOTAL_OTS_VF", "OTS_BY_CALL_VF"],
        "TV": ["COUNT_CALLS_TV", "MEAN_RPC_TV", "MEAN_FTR_TV", "MEAN_TMC_TV", "TOTAL_OTS_TV", "OTS_BY_CALL_TV"],
        "MOVEL": ["COUNT_CALLS_MOVEL", "MEAN_RPC_MOVEL", "MEAN_FTR_MOVEL", "MEAN_TMC_MOVEL", "TOTAL_OTS_MOVEL", "OTS_BY_CALL_MOVEL"]
    }

    print("Creating binary flags for call categories...")
    for cat, cols_in_category in category_columns_for_binary_flags.items():
        # Filter to only columns that actually exist in gcs_unique
        existing_cols_for_flag = [col for col in cols_in_category if col in gcs_unique.columns]
        if existing_cols_for_flag:
            # Create binary column: 1 if any of the category's columns are not NaN, 0 otherwise
            # This check happens *before* the final fillna(0) on gcs_unique
            gcs_unique[f"CALLS_{cat}_BINARY"] = gcs_unique[existing_cols_for_flag].notna().any(axis=1).astype(int)
        else:
            gcs_unique[f"CALLS_{cat}_BINARY"] = 0 # Default if no relevant columns found for this category
            print(f"Warning: No columns found for category {cat} to create binary flag. CALLS_{cat}_BINARY set to 0.")

    # Replace all NaNs in the DataFrame with 0
    print("Filling remaining NaNs with 0...")
    gcs_unique = gcs_unique.fillna(0)

    
    print("Final shape of gcs_unique:", gcs_unique.shape)
    if not gcs_unique.empty:
        print("Remaining NaNs:\n", gcs_unique.isna().sum()[gcs_unique.isna().sum() > 0])
    else:
        print("gcs_unique is empty.")
    print("Creation of gcs_unique table complete.")
    return gcs_unique

