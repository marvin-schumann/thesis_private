# libPBL2425NovaNOS/data_cleaning/client_cleaner.py
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

def _check_cols_exist(df: pd.DataFrame, cols_to_check: list, function_name: str, df_name:str = "DataFrame", level=logging.WARNING) -> bool:
    """Helper to check if all specified columns exist in the DataFrame and log if not."""
    missing_cols = [col for col in cols_to_check if col not in df.columns]
    if missing_cols:
        logger.log(level, f"In {function_name}: The following required columns are missing from {df_name}: {missing_cols}. Functionality may be affected.")
        return False
    return True

def clean_client_age(client_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and imputes CLIENT_AGE_YEARS."""
    logger.info("Starting client age cleaning...")
    df = client_df.copy()
    
    age_col = 'CLIENT_AGE_YEARS' # This should be a defined name, possibly from settings if it varies
    if not _check_cols_exist(df, [age_col], "clean_client_age", level=logging.ERROR):
        logger.error(f"'{age_col}' not found. Skipping age cleaning.")
        return df # Or raise error

    df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
    df['Age_missing'] = ((df[age_col] == -1) | df[age_col].isna()).astype(int)
    df[age_col] = df[age_col].replace(-1, np.nan)

    if settings.KEY_PERSON_SK in df.columns:
        df[age_col] = df.groupby(settings.KEY_PERSON_SK)[age_col].transform(lambda x: x.fillna(x.median()))
    else:
        logger.warning(f"'{settings.KEY_PERSON_SK}' not found. Cannot impute age by person median.")

    # Impute by FLG_EMPRESARIAL median (assuming 'FLG_EMPRESARIAL' is the column name)
    flg_empresarial_col = 'FLG_EMPRESARIAL' 
    if flg_empresarial_col in df.columns:
        df[flg_empresarial_col] = pd.to_numeric(df[flg_empresarial_col], errors='coerce').fillna(0) # fillna(0) assuming 0 is consumer
        # Ensure there are actually 1s and 0s to calculate medians
        if 1 in df[flg_empresarial_col].unique():
            median_age_b2b = df[df[flg_empresarial_col] == 1][age_col].median()
            df.loc[(df[age_col].isna()) & (df[flg_empresarial_col] == 1), age_col] = median_age_b2b
        if 0 in df[flg_empresarial_col].unique():
            median_age_consumer = df[df[flg_empresarial_col] == 0][age_col].median()
            df.loc[(df[age_col].isna()) & (df[flg_empresarial_col] == 0), age_col] = median_age_consumer
    else:
        logger.warning(f"'{flg_empresarial_col}' not found. Cannot impute age by business flag.")

    global_median_age = df[age_col].median()
    if pd.isna(global_median_age):
        logger.warning(f"Global median for '{age_col}' is NaN. Filling remaining NaNs with 0.")
        global_median_age = 0 
    df[age_col].fillna(global_median_age, inplace=True)
    
    logger.info(f"Client age cleaning complete. Remaining NaNs in {age_col}: {df[age_col].isna().sum()}")
    return df

def _fill_metric_vectorized_ffill(df, metric_col, count_col, person_id_col, date_col):
    """Helper to ffill metrics within each person_id_col group where count_col is 0."""
    # This helper is complex due to ensuring correct sorting and application.
    # The original code used df.groupby().transform(lambda x: x.ffill()) which is generally robust.
    # Let's simplify to that if it meets the need for conditional fill.
    # The condition is (df[count_col] == 0) & (df[metric_col].isna())
    
    # Ensure DataFrame is sorted for ffill to work correctly within groups
    df_sorted_indices = df.sort_values([person_id_col, date_col]).index
    
    # Create a temporary series for ffill result
    # Ensure metric_col is numeric
    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
        logger.debug(f"Coerced metric {metric_col} to numeric for ffill.")

    ffilled_series = df.loc[df_sorted_indices].groupby(person_id_col)[metric_col].ffill()
    
    # Apply ffill only where condition is met
    mask = (df[count_col] == 0) & (df[metric_col].isna())
    df.loc[mask, metric_col] = ffilled_series[mask] # Apply on original df using the mask
    
    return df


def fill_rolling_window_metrics(client_df: pd.DataFrame) -> pd.DataFrame:
    """Fills NaNs in rolling window metrics based on call counts being 0."""
    logger.info("Starting fill_rolling_window_metrics...")
    df = client_df.copy()

    date_col = settings.DATE_COL_CLIENT_RAW # e.g., 'START_DATE'
    if not _check_cols_exist(df, [date_col], "fill_rolling_window_metrics", level=logging.ERROR):
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # These base lists and suffixes should ideally come from settings.py for flexibility
    # Example: settings.CLIENT_ROLLING_METRIC_BASES = {'180': [...], '365': [...]}
    # settings.CLIENT_ROLLING_METRIC_SUFFIXES = ['IF', ...]
    metrics_180_base = ['LEG_DURATION_MEAN_180_ASSUNTO_{}', 'RPC_MEAN_180_ASSUNTO_{}', 'FTR_CALCULATED_MEAN_180_ASSUNTO_{}']
    metrics_365_base = ['LEG_DURATION_MEAN_365_ASSUNTO_{}', 'RPC_MEAN_365_ASSUNTO_{}', 'FTR_CALCULATED_MEAN_365_ASSUNTO_{}']
    metric_suffixes = ['IF', 'MOVEL', 'OTHER', 'TV', 'VF', 'TOTAL'] # Should be settings.CLIENT_PROP_CALLS_SUFFIXES + ['TOTAL']
    
    metrics_180_to_fill = [m_template.format(s) for m_template in metrics_180_base for s in metric_suffixes]
    metrics_365_to_fill = [m_template.format(s) for m_template in metrics_365_base for s in metric_suffixes]

    count_col_180 = settings.CLIENT_CALL_COUNT_180_TOTAL_COL
    count_col_365 = settings.CLIENT_CALL_COUNT_365_TOTAL_COL
    person_id_col = settings.KEY_PERSON_SK

    if not _check_cols_exist(df, [count_col_180, count_col_365, person_id_col], "fill_rolling_window_metrics", level=logging.ERROR):
        return df

    for metric in metrics_180_to_fill:
        if metric in df.columns:
            df = _fill_metric_vectorized_ffill(df, metric, count_col_180, person_id_col, date_col)
        else:
            logger.debug(f"Metric column {metric} for 180-day window not found. Skipping fill.")
            
    for metric in metrics_365_to_fill:
        if metric in df.columns:
            df = _fill_metric_vectorized_ffill(df, metric, count_col_365, person_id_col, date_col)
        else:
            logger.debug(f"Metric column {metric} for 365-day window not found. Skipping fill.")
            
    logger.info("Rolling window metrics filling complete.")
    return df

def process_call_related_features(client_df: pd.DataFrame) -> pd.DataFrame:
    """Processes features related to call counts and proportions by issue."""
    logger.info("Starting process_call_related_features...")
    df = client_df.copy()
    
    date_col_for_sort = 'LEG_START_TIME_DAY' # As per notebook logic for this specific function
    if date_col_for_sort not in df.columns:
        date_col_for_sort = settings.DATE_COL_CLIENT_RAW 
        logger.warning(f"'LEG_START_TIME_DAY' not found, using '{date_col_for_sort}' for sorting in process_call_related_features.")
    
    person_id_col = settings.KEY_PERSON_SK
    if _check_cols_exist(df, [person_id_col, date_col_for_sort], "process_call_related_features (sorting)"):
        df[date_col_for_sort] = pd.to_datetime(df[date_col_for_sort], errors='coerce')
        df = df.sort_values(by=[person_id_col, date_col_for_sort], ascending=[True, True])
    
    # --- Create Flag columns for NUNIQUE call counts ---
    # Using settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE and settings.CLIENT_TIME_FRAMES_FOR_PROP_CALLS
    time_periods = settings.CLIENT_TIME_FRAMES_FOR_PROP_CALLS # e.g., ["180", "365"]
    
    all_call_count_cols_to_flag = []
    for base_template in settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE: # e.g., "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_IF"
        for tp in time_periods:
            all_call_count_cols_to_flag.append(base_template.format(tp))
            
    for col in all_call_count_cols_to_flag:
        flag_col_name = f"Flag_{col}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[flag_col_name] = np.where((df[col].isnull()) | (df[col] == 0), 0, 1)
        else:
            df[flag_col_name] = 0
            logger.debug(f"Call count column '{col}' not found for flag creation. '{flag_col_name}' set to 0.")

    # --- Calculate Num_Subscribed_Service_Categories ---
    # Using settings.CLIENT_SUBSCRIPTION_COLS_INDIVIDUAL and settings.CLIENT_SUBSCRIPTION_COLS_MOBILE
    all_subscription_cols = settings.CLIENT_SUBSCRIPTION_COLS_INDIVIDUAL + settings.CLIENT_SUBSCRIPTION_COLS_MOBILE
    for col in all_subscription_cols:
        if col in df.columns:
            # Convert boolean-like strings/values to 0/1
            df[col] = df[col].replace({'True': 1, 'False': 0, True: 1, False: 0})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0 # Create as 0 if missing
            logger.debug(f"Subscription column '{col}' not found. Initialized as 0.")
            
    # Sum individual subscriptions
    existing_individual_sub_cols = [col for col in settings.CLIENT_SUBSCRIPTION_COLS_INDIVIDUAL if col in df.columns]
    df['Num_Subscribed_Service_Categories'] = df[existing_individual_sub_cols].sum(axis=1)
    
    # Add 1 if subscribed to any mobile service
    existing_mobile_sub_cols = [col for col in settings.CLIENT_SUBSCRIPTION_COLS_MOBILE if col in df.columns]
    subscribes_to_any_mobile = (df[existing_mobile_sub_cols].sum(axis=1) > 0) if existing_mobile_sub_cols else pd.Series(0, index=df.index)
    df['Num_Subscribed_Service_Categories'] += subscribes_to_any_mobile.astype(int)

    # --- Calculate Num_Issue_Service_Categories_180/365 ---
    # This needs a mapping from subscription column (e.g., 'IF_PROD_FLG') to issue suffix (e.g., 'IF')
    # Settings has CLIENT_SERVICE_MAP_INDIVIDUAL_180/365 but maps to full flag name.
    # We need a simpler map: {'IF_PROD_FLG': 'IF', 'TV_PROD_FLG': 'TV', ...}
    # Let's assume such a map exists or create a simplified one here.
    # This map should be in settings.py: e.g. settings.CLIENT_SUB_COL_TO_ISSUE_SUFFIX_MAP
    sub_col_to_issue_suffix_map = {
        'IF_PROD_FLG': 'IF', 'TV_PROD_FLG': 'TV', 'VF_PROD_FLG': 'VF'
        # This map might need to be expanded or made more robust via settings
    }

    for tp in time_periods: # "180", "365"
        num_issue_col_name = f'Num_Issue_Service_Categories_{tp}'
        df[num_issue_col_name] = 0
        
        # Individual services
        for sub_col, issue_suffix in sub_col_to_issue_suffix_map.items():
            # Construct the flag column name from settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE
            # e.g., find "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_{issue_suffix}" in settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE
            # and format it with tp
            base_flag_template_found = None
            for base_template in settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE:
                if f"ASSUNTO_{issue_suffix}" in base_template:
                    base_flag_template_found = base_template
                    break
            
            if base_flag_template_found:
                flag_call_col = f"Flag_{base_flag_template_found.format(tp)}" # Name of the flag created earlier
                if sub_col in df.columns and flag_call_col in df.columns:
                     df[num_issue_col_name] += ((df[sub_col] == 1) & (df[flag_call_col] == 1)).astype(int)
            else:
                logger.debug(f"Could not find base template for issue suffix '{issue_suffix}' in settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE.")


        # Mobile category
        # Find base template for MOVEL
        base_flag_template_movel = None
        for base_template in settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE:
            if "ASSUNTO_MOVEL" in base_template: # Assuming 'MOVEL' is the suffix for mobile
                base_flag_template_movel = base_template
                break
        
        if base_flag_template_movel:
            flag_mobile_call_col = f"Flag_{base_flag_template_movel.format(tp)}"
            if flag_mobile_call_col in df.columns:
                df[num_issue_col_name] += (subscribes_to_any_mobile & (df[flag_mobile_call_col] == 1)).astype(int)
        else:
            logger.debug("Could not find base template for 'ASSUNTO_MOVEL' in settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE.")

        df[num_issue_col_name] = df[num_issue_col_name].astype(int)

    # --- Calculate Prop_Issue_Services_Subscribed_180/365 ---
    for tp in time_periods:
        num_issue_col = f'Num_Issue_Service_Categories_{tp}'
        prop_issue_col = f'Prop_Issue_Services_Subscribed_{tp}'
        if num_issue_col in df.columns : # Check if it was created
            df[prop_issue_col] = np.where(
                df['Num_Subscribed_Service_Categories'] > 0,
                df[num_issue_col] / df['Num_Subscribed_Service_Categories'],
                0 # Or np.nan if preferred when no subscriptions
            )
            df[prop_issue_col] = df[prop_issue_col].fillna(0) # Fill NaNs from division (e.g. 0/0)
        else:
            df[prop_issue_col] = 0 # Default if num_issue_col wasn't created
            logger.debug(f"Column '{num_issue_col}' not found for calculating '{prop_issue_col}'. Set to 0.")


    # --- Impute PROP_CALLS_* ---
    # Using settings.CLIENT_PROP_CALLS_SUFFIXES and settings.CLIENT_TIME_FRAMES_FOR_PROP_CALLS
    for tf in settings.CLIENT_TIME_FRAMES_FOR_PROP_CALLS: # "180", "365"
        # Find the total calls column, e.g., "CALL_IF_ID_NUNIQUE_{tf}_TOTAL"
        total_calls_col_template = next((b for b in settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE if "_TOTAL" in b), None)
        if not total_calls_col_template:
            logger.warning(f"Could not find a 'TOTAL' call count column template in settings. Skipping PROP_CALLS for {tf} days.")
            continue
        
        total_calls_col = total_calls_col_template.format(tf)

        if total_calls_col not in df.columns:
            logger.warning(f"Total call count column '{total_calls_col}' not found. PROP_CALLS for {tf} days may be inaccurate or skipped.")
            # Attempt to compute it if missing (as in original notebook)
            temp_cat_call_cols = []
            for suffix in settings.CLIENT_PROP_CALLS_SUFFIXES: # e.g. IF, MOVEL, TV...
                cat_call_col_template = next((b for b in settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE if f"_ASSUNTO_{suffix}" in b), None)
                if cat_call_col_template:
                    cat_call_col = cat_call_col_template.format(tf)
                    if cat_call_col in df.columns:
                        temp_cat_call_cols.append(cat_call_col)
            if temp_cat_call_cols:
                df[total_calls_col] = df[temp_cat_call_cols].sum(axis=1, skipna=False) # skipna=False to propagate NaN if any component is NaN
                logger.info(f"Computed missing total calls column '{total_calls_col}'.")
            else:
                logger.warning(f"Could not compute missing '{total_calls_col}' as category call columns are also missing.")
                continue # Skip PROP_CALLS for this time frame if total cannot be determined

        for suffix in settings.CLIENT_PROP_CALLS_SUFFIXES: # 'IF', 'MOVEL', 'OTHER', 'TV', 'VF'
            prop_col = f'PROP_CALLS_{suffix}_{tf}'
            # Find the category-specific call count column, e.g., "CALL_IF_ID_NUNIQUE_{tf}_ASSUNTO_{suffix}"
            call_col_template = next((b for b in settings.CLIENT_CALL_COUNT_FLAG_COLUMNS_BASE if f"_ASSUNTO_{suffix}" in b), None)
            if not call_col_template:
                logger.debug(f"No call count template for suffix '{suffix}'. Skipping {prop_col}.")
                if prop_col not in df.columns: df[prop_col] = 0 # Ensure column exists and is 0
                continue
                
            call_col = call_col_template.format(tf)

            if call_col in df.columns:
                # Ensure prop_col exists, create if not
                if prop_col not in df.columns:
                    df[prop_col] = 0.0 # Initialize

                # Fill PROP_CALLS where it's NaN or 0 (if 0 was from initial fillna)
                # and the specific call_col > 0 and total_calls_col > 0
                # Original logic: ((df[prop_col].isna()) | (df[prop_col] == 0))
                # This can overwrite legitimate 0s if a component was 0.
                # More direct: calculate if total > 0, else 0.
                df[prop_col] = np.where(
                    (df[total_calls_col].notna()) & (df[total_calls_col] > 0) & (df[call_col].notna()),
                    df[call_col] / df[total_calls_col],
                    0 # If total_calls is 0 or NaN, or call_col is NaN, prop is 0
                )
                df[prop_col] = df[prop_col].fillna(0) # Final fill for this prop col if any division by NaN occurred
            else:
                logger.debug(f"Call count column '{call_col}' for suffix '{suffix}' not found. Cannot calculate '{prop_col}'.")
                if prop_col not in df.columns: df[prop_col] = 0 # Ensure column exists and is 0


    logger.info("Call related features processing complete.")
    return df

def fill_remaining_na_and_flags(client_df: pd.DataFrame, fill_value_main=0, fill_value_char=-1) -> pd.DataFrame:
    """Fills remaining NaNs in client_df based on column types/patterns."""
    logger.info(f"Starting fill_remaining_na_and_flags (main with {fill_value_main}, characteristics with {fill_value_char})...")
    df = client_df.copy()

    # --- Define metric columns to fill with `fill_value_main` ---
    # These are typically rolling means, proportions, etc.
    # The original script dynamically builds these lists. Using settings for patterns or explicit lists is better.
    # Example: (This should be more systematically derived from settings)
    main_metrics_to_fill_zero = []
    # Rolling window metrics already handled by fill_rolling_window_metrics, but ensure they are 0 if still NaN
    metrics_180_base = ['LEG_DURATION_MEAN_180_ASSUNTO_{}', 'RPC_MEAN_180_ASSUNTO_{}', 'FTR_CALCULATED_MEAN_180_ASSUNTO_{}']
    metrics_365_base = ['LEG_DURATION_MEAN_365_ASSUNTO_{}', 'RPC_MEAN_365_ASSUNTO_{}', 'FTR_CALCULATED_MEAN_365_ASSUNTO_{}']
    prop_ots_base = ['PROP_OTS_{}_{}'] # {category} {timeframe}
    metric_suffixes = ['IF', 'MOVEL', 'OTHER', 'TV', 'VF', 'TOTAL'] # Should be from settings
    time_frames_prop = settings.CLIENT_TIME_FRAMES_FOR_PROP_CALLS # e.g., ['180', '365']

    for base_list in [metrics_180_base, metrics_365_base]:
        for m_template in base_list:
            for s in metric_suffixes:
                main_metrics_to_fill_zero.append(m_template.format(s))
    
    for p_template in [prop_ots_base[0]]:
        for s in metric_suffixes: 
            for tf in time_frames_prop:
                main_metrics_to_fill_zero.append(p_template.format(s, tf))
    
    # PROP_CALLS (already handled in process_call_related_features, but ensure filled)
    prop_calls_base_template = 'PROP_CALLS_{}_{}' # Assume one template
    for s in settings.CLIENT_PROP_CALLS_SUFFIXES:
         for tf in time_frames_prop:
            main_metrics_to_fill_zero.append(prop_calls_base_template.format(s, tf))
    
    existing_main_metrics_to_fill = [col for col in main_metrics_to_fill_zero if col in df.columns]
    if existing_main_metrics_to_fill:
        df[existing_main_metrics_to_fill] = df[existing_main_metrics_to_fill].fillna(fill_value_main)
        logger.info(f"Filled {len(existing_main_metrics_to_fill)} main metric columns with {fill_value_main}.")

    # --- Fill "TOTAL" count columns based on sum of category columns if still NaN ---
    # The 'total_mapping' from original notebook should be defined in settings.py for robustness.
    # Example: settings.CLIENT_TOTAL_COUNT_COLUMN_MAPPING
    # For now, hardcoding it as in original script, but this is a key area for settings integration.
    total_mapping = {
        "LEG_IF_ID_COUNT_180_TOTAL": ["LEG_IF_ID_COUNT_180_ASSUNTO_MOVEL", "LEG_IF_ID_COUNT_180_ASSUNTO_OTHER", "LEG_IF_ID_COUNT_180_ASSUNTO_IF", "LEG_IF_ID_COUNT_180_ASSUNTO_VF", "LEG_IF_ID_COUNT_180_ASSUNTO_TV"],
        "LEG_IF_ID_COUNT_365_TOTAL": ["LEG_IF_ID_COUNT_365_ASSUNTO_MOVEL", "LEG_IF_ID_COUNT_365_ASSUNTO_OTHER", "LEG_IF_ID_COUNT_365_ASSUNTO_IF", "LEG_IF_ID_COUNT_365_ASSUNTO_VF", "LEG_IF_ID_COUNT_365_ASSUNTO_TV"],
        # ... other total columns from original script
    }
    for total_col, component_cols_list in total_mapping.items():
        if total_col in df.columns:
            existing_component_cols = [c for c in component_cols_list if c in df.columns]
            if existing_component_cols:
                # Fill NaNs in component columns with 0 before summing for total
                df[existing_component_cols] = df[existing_component_cols].fillna(0)
                missing_mask = df[total_col].isna()
                if missing_mask.any():
                    df.loc[missing_mask, total_col] = df.loc[missing_mask, existing_component_cols].sum(axis=1)
                    logger.info(f"Recalculated/filled NaNs in total column '{total_col}'.")
    logger.info("Total count columns re-summed where NaN (if applicable).")

    # --- Forward fill characteristic columns, then fill remaining NaNs with `fill_value_char` and add flags ---
    # Using settings.CLIENT_CHARACTERISTIC_COLS_FFILL
    char_cols_for_ffill = settings.CLIENT_CHARACTERISTIC_COLS_FFILL
    
    person_id_col = settings.KEY_PERSON_SK
    date_col_for_sort = settings.DATE_COL_CLIENT_RAW # Primary date for client characteristics
    if date_col_for_sort not in df.columns:
        date_col_for_sort = 'LEG_START_TIME_DAY' # Fallback from original
    
    if _check_cols_exist(df, [person_id_col, date_col_for_sort], "fill_remaining_na_and_flags (ffill sort)"):
        df[date_col_for_sort] = pd.to_datetime(df[date_col_for_sort], errors='coerce')
        df = df.sort_values([person_id_col, date_col_for_sort]) # Sort for ffill
        
        for col in char_cols_for_ffill:
            if col in df.columns:
                # Ensure column is suitable for ffill (e.g. not all NaN in a group)
                 df[col] = df.groupby(person_id_col)[col].ffill()
            else:
                # Create column if missing so it gets filled with fill_value_char later (original behavior)
                df[col] = np.nan 
                logger.debug(f"Characteristic column '{col}' not found for ffill, created as NaN. Will be filled with {fill_value_char}.")
    else:
        logger.warning(f"Cannot ffill characteristics due to missing '{person_id_col}' or date column '{date_col_for_sort}'.")

    # Flag for all characteristics empty (based on a representative column like 'FLG_NO_BOX' missing after ffill)
    # This representative column should be in settings, e.g., settings.CLIENT_REPR_CHAR_COL_FOR_EMPTY_FLAG
    repr_char_col_for_empty_check = 'FLG_NO_BOX' 
    if repr_char_col_for_empty_check in df.columns:
        df["Flag_Charachteristics_Empty"] = np.where(df[repr_char_col_for_empty_check].isna(), 1, 0)
    else:
        # If the representative column itself is missing, assume characteristics are empty
        df["Flag_Charachteristics_Empty"] = 1 
        logger.debug(f"Representative characteristic column '{repr_char_col_for_empty_check}' not found. 'Flag_Charachteristics_Empty' set to 1.")

    # Fill remaining NaNs in these characteristic columns with `fill_value_char`
    # And create specific flags based on settings.CLIENT_COLS_FLAG_AND_FILL_MINUS_ONE
    # This dict maps col_name -> True (create flag) or False (no specific flag beyond general fill)
    
    # Iterate over all characteristic columns that underwent ffill
    for col in char_cols_for_ffill: 
        if col in df.columns: # Should exist due to ffill logic or prior creation
            # Check if this column needs a specific "Is_Imputed" flag
            if col in settings.CLIENT_COLS_FLAG_AND_FILL_MINUS_ONE and settings.CLIENT_COLS_FLAG_AND_FILL_MINUS_ONE[col]:
                df[f"Flag_{col}_Is_Imputed"] = np.where(df[col].isna(), 1, 0) # Flag before filling
            
            df[col] = df[col].fillna(fill_value_char)
        # No else needed, columns should exist.
        
    logger.info(f"Characteristic columns processed (ffill, then filled with {fill_value_char}).")
    return df

def full_client_cleaning(client_df_raw: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates all client data cleaning steps."""
    logger.info("--- Starting Full Client Cleaning Process ---")
    if client_df_raw.empty:
        logger.warning("Input client_df_raw is empty. Skipping full client cleaning.")
        return client_df_raw
        
    df = client_df_raw.copy()

    # Standardize on settings.DATE_COL_CLIENT_RAW for client history date
    primary_date_col = settings.DATE_COL_CLIENT_RAW
    fallback_date_col = 'LEG_START_TIME_DAY' # From original notebook logic

    if primary_date_col not in df.columns:
        if fallback_date_col in df.columns:
            df[primary_date_col] = df[fallback_date_col] # Assign, will be converted to datetime next
            logger.info(f"Created primary date column '{primary_date_col}' from fallback '{fallback_date_col}'.")
        else:
            msg = f"Neither primary date column '{primary_date_col}' nor fallback '{fallback_date_col}' found in client_df."
            logger.error(msg)
            raise ValueError(msg)
            
    df[primary_date_col] = pd.to_datetime(df[primary_date_col], errors='coerce')
    if df[primary_date_col].isnull().all() and not client_df_raw.empty : # Check if all NaT only if df not empty
         logger.warning(f"All values in primary date column '{primary_date_col}' are NaT after conversion. This may affect date-dependent operations.")

    df = clean_client_age(df)
    df = fill_rolling_window_metrics(df) # Uses settings.DATE_COL_CLIENT_RAW
    df = process_call_related_features(df) # Internally manages its date column choice
    df = fill_remaining_na_and_flags(df, fill_value_main=0, fill_value_char=-1)
    
    logger.info(f"--- Full Client Cleaning Process Complete. Output shape: {df.shape} ---")
    return df