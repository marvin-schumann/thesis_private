# data_pipeline_nos/data_cleaning/client_cleaner.py
import pandas as pd
import numpy as np
# from ..config import settings # For package structure
# from config import settings # For standalone testing if config is at root

def clean_client_age(client_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and imputes CLIENT_AGE_YEARS.
    - Creates Age_missing flag.
    - Replaces -1 with NaN.
    - Imputes NaNs using PERSON_ID median, then FLG_EMPRESARIAL median, then global median.
    """
    print("Cleaning client age...")
    df = client_df.copy()

    # Ensure CLIENT_AGE_YEARS is numeric, coercing errors
    df['CLIENT_AGE_YEARS'] = pd.to_numeric(df['CLIENT_AGE_YEARS'], errors='coerce')

    df['Age_missing'] = ((df['CLIENT_AGE_YEARS'] == -1) | df['CLIENT_AGE_YEARS'].isna()).astype(int)
    df['CLIENT_AGE_YEARS'] = df['CLIENT_AGE_YEARS'].replace(-1, np.nan)

    # Impute by PERSON_ID median
    if 'PERSON_ID' in df.columns:
        df['CLIENT_AGE_YEARS'] = df.groupby('PERSON_ID')['CLIENT_AGE_YEARS']\
                                   .transform(lambda x: x.fillna(x.median()))
    else:
        print("Warning: PERSON_ID not found for age imputation by person median.")

    # Impute by FLG_EMPRESARIAL median
    if 'FLG_EMPRESARIAL' in df.columns:
        # Ensure FLG_EMPRESARIAL is numeric (0 or 1)
        df['FLG_EMPRESARIAL'] = pd.to_numeric(df['FLG_EMPRESARIAL'], errors='coerce')
        median_age_b2b = df[df['FLG_EMPRESARIAL'] == 1]['CLIENT_AGE_YEARS'].median()
        median_age_consumer = df[df['FLG_EMPRESARIAL'] == 0]['CLIENT_AGE_YEARS'].median()

        df.loc[(df['CLIENT_AGE_YEARS'].isna()) & (df['FLG_EMPRESARIAL'] == 1), 'CLIENT_AGE_YEARS'] = median_age_b2b
        df.loc[(df['CLIENT_AGE_YEARS'].isna()) & (df['FLG_EMPRESARIAL'] == 0), 'CLIENT_AGE_YEARS'] = median_age_consumer
    else:
        print("Warning: FLG_EMPRESARIAL not found for age imputation by business flag.")

    # Impute remaining with global median
    global_median_age = df['CLIENT_AGE_YEARS'].median()
    df['CLIENT_AGE_YEARS'].fillna(global_median_age, inplace=True) # Fill any remaining NaNs

    print(f"Client age cleaning complete. Remaining NaNs in CLIENT_AGE_YEARS: {df['CLIENT_AGE_YEARS'].isna().sum()}")
    return df

def _fill_metric_vectorized_ffill(df, metric, count_col, person_id_col='PERSON_ID', date_col='START_DATE'):
    """Helper to ffill then bfill metrics within each PERSON_ID group where count_col is 0."""
    if not all(c in df.columns for c in [metric, count_col, person_id_col, date_col]):
        print(f"Warning: Missing one or more required columns for _fill_metric_vectorized: {metric}, {count_col}, {person_id_col}, {date_col}")
        return df

    # Ensure DataFrame is sorted for ffill to work correctly within groups
    df_sorted = df.sort_values([person_id_col, date_col])

    mask = (df_sorted[count_col] == 0) & (df_sorted[metric].isna())
    if mask.any():
        # Apply ffill within each person's group
        filled_series = df_sorted.groupby(person_id_col)[metric].transform(lambda x: x.ffill())
        # Update the original DataFrame (or its copy if this function operates on a copy)
        # To update the original df passed to the parent, we need to use its index
        df.loc[df_sorted[mask].index, metric] = filled_series[mask]
    return df


def fill_rolling_window_metrics(client_df: pd.DataFrame) -> pd.DataFrame:
    """Fills NaNs in rolling window metrics (180 & 365 day versions) based on call counts being 0."""
    print("Filling rolling window metrics...")
    df = client_df.copy()

    if 'START_DATE' not in df.columns:
        print("Warning: START_DATE column not found. Cannot perform rolling window metric filling accurately.")
        return df
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')


    # Define metrics lists (ideally from config/column_definitions.py)
    metrics_180_base = ['LEG_DURATION_MEAN_180_ASSUNTO_{}', 'RPC_MEAN_180_ASSUNTO_{}', 'FTR_CALCULATED_MEAN_180_ASSUNTO_{}']
    metrics_365_base = ['LEG_DURATION_MEAN_365_ASSUNTO_{}', 'RPC_MEAN_365_ASSUNTO_{}', 'FTR_CALCULATED_MEAN_365_ASSUNTO_{}']
    suffixes = ['IF', 'MOVEL', 'OTHER', 'TV', 'VF', 'TOTAL']

    metrics_180 = [m.format(s) for m in metrics_180_base for s in suffixes]
    metrics_365 = [m.format(s) for m in metrics_365_base for s in suffixes]

    count_col_180 = 'LEG_IF_ID_COUNT_180_TOTAL'
    count_col_365 = 'LEG_IF_ID_COUNT_365_TOTAL'

    for metric in metrics_180:
        df = _fill_metric_vectorized_ffill(df, metric, count_col_180)
    for metric in metrics_365:
        df = _fill_metric_vectorized_ffill(df, metric, count_col_365)

    print("Rolling window metrics filling complete.")
    return df


def process_call_related_features(client_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes features related to call counts and proportions by issue.
    - Creates flag columns for NUNIQUE call counts.
    - Calculates Num_Subscribed_Service_Categories.
    - Calculates Num_Issue_Service_Categories_180/365.
    - Calculates Prop_Issue_Services_Subscribed_180/365.
    - Imputes PROP_CALLS_* where possible.
    """
    print("Processing call related features (counts by issue, proportions)...")
    df = client_df.copy()

    # --- Sort by PERSON_ID and LEG_START_TIME_DAY (if exists) ---
    # This date column seems to be the snapshot date for client features
    date_col_for_sort = 'LEG_START_TIME_DAY' # As per notebook
    if date_col_for_sort not in df.columns:
        date_col_for_sort = 'START_DATE' # Fallback if the other isn't there
        print(f"Warning: 'LEG_START_TIME_DAY' not found, using '{date_col_for_sort}' for sorting in process_call_related_features.")
    if date_col_for_sort in df.columns and 'PERSON_ID' in df.columns:
        df[date_col_for_sort] = pd.to_datetime(df[date_col_for_sort], errors='coerce')
        df = df.sort_values(by=['PERSON_ID', date_col_for_sort], ascending=[True, True])
    else:
        print(f"Warning: PERSON_ID or {date_col_for_sort} not found, cannot sort for call_related_features processing.")


    # --- Create Flag columns for NUNIQUE call counts ---
    call_count_columns_base = [
        "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_IF", "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_MOVEL",
        "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_OTHER", "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_TV",
        "CALL_IF_ID_NUNIQUE_{}_ASSUNTO_VF", "CALL_IF_ID_NUNIQUE_{}_TOTAL"
    ]
    time_periods = ["180", "365"]
    all_call_count_flag_cols = [f.format(tp) for tp in time_periods for f in call_count_columns_base]

    for col in all_call_count_flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Ensure numeric
            df[f"Flag_{col}"] = np.where((df[col].isnull()) | (df[col] == 0), 0, 1)
        else:
            # Create the flag column as all 0s if the source column doesn't exist
            df[f"Flag_{col}"] = 0
            print(f"Warning: Call count column {col} not found for flag creation. Flag_{col} set to 0.")


    # --- Calculate Num_Subscribed_Service_Categories ---
    subscription_cols_individual = ['IF_PROD_FLG', 'TV_PROD_FLG', 'VF_PROD_FLG']
    subscription_cols_mobile = ['VM_PROD_FLG', 'IM_PROD_FLG'] # Voice Mobile, Internet Mobile

    for col in subscription_cols_individual + subscription_cols_mobile:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace({'True': 1, 'False': 0, True: 1, False: 0}), errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0 # Create as 0 if missing
            print(f"Warning: Subscription column {col} not found. Initialized as 0.")

    df['Num_Subscribed_Service_Categories'] = df[[col for col in subscription_cols_individual if col in df.columns]].sum(axis=1)
    subscribes_to_any_mobile = (df[[col for col in subscription_cols_mobile if col in df.columns]].sum(axis=1) > 0)
    df['Num_Subscribed_Service_Categories'] += subscribes_to_any_mobile.astype(int)


    # --- Calculate Num_Issue_Service_Categories_180/365 ---
    service_map_individual = {'IF_PROD_FLG': 'IF', 'TV_PROD_FLG': 'TV', 'VF_PROD_FLG': 'VF'}

    for tp in time_periods:
        num_issue_col = f'Num_Issue_Service_Categories_{tp}'
        df[num_issue_col] = 0
        # Individual services
        for sub_col, cat_suffix in service_map_individual.items():
            flag_call_col = f'Flag_CALL_IF_ID_NUNIQUE_{tp}_ASSUNTO_{cat_suffix}'
            if sub_col in df.columns and flag_call_col in df.columns:
                 df[num_issue_col] += (df[sub_col] == 1) & (df[flag_call_col] == 1)

        # Mobile category
        flag_mobile_call_col = f'Flag_CALL_IF_ID_NUNIQUE_{tp}_ASSUNTO_MOVEL'
        if flag_mobile_call_col in df.columns:
            df[num_issue_col] += (subscribes_to_any_mobile & (df[flag_mobile_call_col] == 1))
        df[num_issue_col] = df[num_issue_col].astype(int)


    # --- Calculate Prop_Issue_Services_Subscribed_180/365 ---
    for tp in time_periods:
        num_issue_col = f'Num_Issue_Service_Categories_{tp}'
        prop_issue_col = f'Prop_Issue_Services_Subscribed_{tp}'
        df[prop_issue_col] = np.where(
            df['Num_Subscribed_Service_Categories'] > 0, # Denominator check
            df[num_issue_col] / df['Num_Subscribed_Service_Categories'],
            0 # Or np.nan if preferred for no subscriptions
        )
        df[prop_issue_col] = df[prop_issue_col].fillna(0) # Fill NaNs from division (e.g. 0/0)

    # --- Impute PROP_CALLS_* ---
    all_suffixes = ['IF', 'MOVEL', 'OTHER', 'TV', 'VF']
    for tf in time_periods:
        total_calls_col = f'CALL_IF_ID_NUNIQUE_{tf}_TOTAL'
        if total_calls_col not in df.columns:
            # Attempt to compute it if missing, or skip
            temp_cat_call_cols = [f'CALL_IF_ID_NUNIQUE_{tf}_ASSUNTO_{s}' for s in all_suffixes if f'CALL_IF_ID_NUNIQUE_{tf}_ASSUNTO_{s}' in df.columns]
            if temp_cat_call_cols:
                df[total_calls_col] = df[temp_cat_call_cols].sum(axis=1)
                print(f"Computed missing {total_calls_col}")
            else:
                print(f"Warning: {total_calls_col} not found and cannot be computed. PROP_CALLS for {tf} days may be inaccurate.")
                continue # Skip PROP_CALLS for this time frame

        for suffix in all_suffixes:
            call_col = f'CALL_IF_ID_NUNIQUE_{tf}_ASSUNTO_{suffix}'
            prop_col = f'PROP_CALLS_{suffix}_{tf}'

            if call_col in df.columns and prop_col in df.columns: # Ensure prop_col exists to be filled
                # Fill PROP_CALLS where it's NaN or 0 (if 0 was from initial fillna)
                # and the specific call_col > 0 and total_calls_col > 0
                condition_for_recalc = ((df[prop_col].isna()) | (df[prop_col] == 0)) & \
                                       (df[call_col] > 0) & \
                                       (df[total_calls_col] > 0)
                df[prop_col] = np.where(
                    condition_for_recalc,
                    df[call_col] / df[total_calls_col],
                    df[prop_col]
                )
                # Ensure prop_col is 0 if total_calls_col is 0 (after potential recalc)
                df[prop_col] = np.where(df[total_calls_col] == 0, 0, df[prop_col])
                df[prop_col] = df[prop_col].fillna(0) # Final fill for this prop col
            elif prop_col not in df.columns and call_col in df.columns: # Create prop_col if it doesn't exist
                df[prop_col] = np.where(
                    (df[call_col] > 0) & (df[total_calls_col] > 0),
                    df[call_col] / df[total_calls_col],
                    0
                )
                df[prop_col] = df[prop_col].fillna(0)


    print("Call related features processing complete.")
    return df


def fill_remaining_na_and_flags(client_df: pd.DataFrame, fill_value_main=0, fill_value_char=-1) -> pd.DataFrame:
    """
    Fills remaining NaNs in client_df:
    - Fills main metrics (rolling means, PROP_OTS) with `fill_value_main`.
    - Fills specified characteristic columns with `fill_value_char` and adds flags.
    - Performs ffill on characteristic columns.
    - Recalculates 'TOTAL' count columns from their components if NaN.
    """
    print(f"Filling remaining NaNs (main with {fill_value_main}, characteristics with {fill_value_char})...")
    df = client_df.copy()

    # Define metric lists (can be moved to config)
    metrics_180_base = ['LEG_DURATION_MEAN_180_ASSUNTO_{}', 'RPC_MEAN_180_ASSUNTO_{}', 'FTR_CALCULATED_MEAN_180_ASSUNTO_{}']
    metrics_365_base = ['LEG_DURATION_MEAN_365_ASSUNTO_{}', 'RPC_MEAN_365_ASSUNTO_{}', 'FTR_CALCULATED_MEAN_365_ASSUNTO_{}']
    prop_ots_base = ['PROP_OTS_{}_{}'] # {category} {timeframe}
    suffixes = ['IF', 'MOVEL', 'OTHER', 'TV', 'VF', 'TOTAL']
    time_frames = ['180', '365']

    main_metrics_to_fill_zero = []
    for base_list in [metrics_180_base, metrics_365_base]:
        for m_template in base_list:
            for s in suffixes:
                main_metrics_to_fill_zero.append(m_template.format(s))
    for p_template in [prop_ots_base[0]]: # Only one template for PROP_OTS
        for s in suffixes: # 'TOTAL' is also a suffix here
            for tf in time_frames:
                main_metrics_to_fill_zero.append(p_template.format(s, tf))
    # PROP_CALLS were handled in process_call_related_features, but we can ensure they are filled here too
    prop_calls_base = ['PROP_CALLS_{}_{}']
    for p_template in prop_calls_base:
        for s in ['IF', 'MOVEL', 'OTHER', 'TV', 'VF']: # No 'TOTAL' for PROP_CALLS suffix
             for tf in time_frames:
                main_metrics_to_fill_zero.append(p_template.format(s, tf))


    existing_main_metrics = [col for col in main_metrics_to_fill_zero if col in df.columns]
    df[existing_main_metrics] = df[existing_main_metrics].fillna(fill_value_main)
    print(f"Filled {len(existing_main_metrics)} main metric columns with {fill_value_main}.")


    # --- Fill "TOTAL" count columns based on sum of category columns if still NaN ---
    # This step was in the notebook *before* ffill of characteristics
    total_mapping = {
        "LEG_IF_ID_COUNT_180_TOTAL": ["LEG_IF_ID_COUNT_180_ASSUNTO_MOVEL", "LEG_IF_ID_COUNT_180_ASSUNTO_OTHER", "LEG_IF_ID_COUNT_180_ASSUNTO_IF", "LEG_IF_ID_COUNT_180_ASSUNTO_VF", "LEG_IF_ID_COUNT_180_ASSUNTO_TV"], #Added TV
        "LEG_IF_ID_COUNT_365_TOTAL": ["LEG_IF_ID_COUNT_365_ASSUNTO_MOVEL", "LEG_IF_ID_COUNT_365_ASSUNTO_OTHER", "LEG_IF_ID_COUNT_365_ASSUNTO_IF", "LEG_IF_ID_COUNT_365_ASSUNTO_VF", "LEG_IF_ID_COUNT_365_ASSUNTO_TV"], #Added TV
        "NR_INCIDENTS_PER_LEG_SUM_180_TOTAL": ["NR_INCIDENTS_PER_LEG_SUM_180_ASSUNTO_MOVEL", "NR_INCIDENTS_PER_LEG_SUM_180_ASSUNTO_OTHER", "NR_INCIDENTS_PER_LEG_SUM_180_ASSUNTO_IF", "NR_INCIDENTS_PER_LEG_SUM_180_ASSUNTO_VF", "NR_INCIDENTS_PER_LEG_SUM_180_ASSUNTO_TV"],
        "NR_INCIDENTS_PER_LEG_SUM_365_TOTAL": ["NR_INCIDENTS_PER_LEG_SUM_365_ASSUNTO_MOVEL", "NR_INCIDENTS_PER_LEG_SUM_365_ASSUNTO_OTHER", "NR_INCIDENTS_PER_LEG_SUM_365_ASSUNTO_IF", "NR_INCIDENTS_PER_LEG_SUM_365_ASSUNTO_VF", "NR_INCIDENTS_PER_LEG_SUM_365_ASSUNTO_TV"],
        "FLAG_OT_SUM_180_TOTAL": ["FLAG_OT_SUM_180_ASSUNTO_MOVEL", "FLAG_OT_SUM_180_ASSUNTO_OTHER", "FLAG_OT_SUM_180_ASSUNTO_IF", "FLAG_OT_SUM_180_ASSUNTO_TV", "FLAG_OT_SUM_180_ASSUNTO_VF"], #Added VF
        "FLAG_OT_SUM_365_TOTAL": ["FLAG_OT_SUM_365_ASSUNTO_MOVEL", "FLAG_OT_SUM_365_ASSUNTO_OTHER", "FLAG_OT_SUM_365_ASSUNTO_IF", "FLAG_OT_SUM_365_ASSUNTO_TV", "FLAG_OT_SUM_365_ASSUNTO_VF"]  #Added VF
    }
    for total_col, category_cols_list in total_mapping.items():
        if total_col in df.columns:
            existing_cat_cols = [c for c in category_cols_list if c in df.columns]
            if existing_cat_cols:
                # Fill NaNs in category columns with 0 before summing for total
                df[existing_cat_cols] = df[existing_cat_cols].fillna(0)
                missing_mask = df[total_col].isna()
                if missing_mask.any():
                    df.loc[missing_mask, total_col] = df.loc[missing_mask, existing_cat_cols].sum(axis=1)
                    print(f"Recalculated/filled NaNs in {total_col}.")
    print("Total count columns re-summed where NaN.")


    # --- Forward fill characteristic columns, then fill remaining NaNs with -1 and add flags ---
    characteristic_cols_for_ffill_then_minus_one = [ # From notebook's `missing_cols` list
        'IF_DOWNLOAD_SPEED_MBPS_QTY', 'PF_CLIENT_MONTHS', 'NR_SAS',
        'FLG_EMPRESARIAL', 'FLG_CONSUMER', 'FLG_UNKNOWN', 'FLG_NO_BOX',
        'FLG_BOX_3_ULTRA_HD', 'FLG_BOX_2_HD_PLUS_DVR_CABO', 'FLG_BOX_2_HD_CABO',
        'FLG_BOX_1_HD_PLUS_DVR_SAT_TDT', 'FLG_BOX_1_HD_SAT',
        'IF_UPLOAD_SPEED_KBPS_QTY', 'TV_TVCINE_FLG', 'FLG_INDEFINIDO',
        'TV_11SPORTS_FLG', 'TV_BTV_FLG', 'TV_SPTV_FLG', 'TECH_GSM_FLG',
        'TECH_DTH_FLG', 'TECH_FTTH_FLG', 'TECH_CABLE_FLG', 'TECH_COPPER_FLG',
        'SERVICES_QTY', 'ARPU_CALCULATED', 'FLG_ILHAS', 'CLIENT_ANTIQUITY_YEARS', 'ARPU_CLIENT'
    ]
    date_col_for_sort = 'START_DATE' # Primary date for client characteristics
    if date_col_for_sort not in df.columns:
        date_col_for_sort = 'LEG_START_TIME_DAY' # Fallback
    if date_col_for_sort in df.columns and 'PERSON_ID' in df.columns:
        df[date_col_for_sort] = pd.to_datetime(df[date_col_for_sort], errors='coerce')
        df = df.sort_values(['PERSON_ID', date_col_for_sort])
        for col in characteristic_cols_for_ffill_then_minus_one:
            if col in df.columns:
                df[col] = df.groupby('PERSON_ID')[col].ffill()
            else:
                df[col] = np.nan # Create column if missing, so it gets filled with -1
                print(f"Warning: Characteristic column {col} not found for ffill, will be filled with {fill_value_char}.")
    else:
        print(f"Warning: PERSON_ID or date column for sorting ({date_col_for_sort}) missing. Cannot ffill characteristics.")


    # Flag for all characteristics empty (based on FLG_NO_BOX missing after ffill)
    if 'FLG_NO_BOX' in df.columns:
        df["Flag_Charachteristics_Empty"] = np.where(df['FLG_NO_BOX'].isna(), 1, 0)
    else:
        df["Flag_Charachteristics_Empty"] = 1 # Assume all empty if FLG_NO_BOX doesn't exist

    # Fill remaining NaNs in these characteristic columns with -1 and create specific flags
    cols_needing_specific_flags = ['PF_CLIENT_MONTHS', 'ARPU_CLIENT', 'CLIENT_ANTIQUITY_YEARS', 'ARPU_CALCULATED']
    for col in characteristic_cols_for_ffill_then_minus_one: # Iterate over all characteristic cols
        if col in df.columns: # Check if column exists (it should, due to ffill or creation above)
            if col in cols_needing_specific_flags:
                df[f"Flag_{col}_Is_Imputed"] = np.where(df[col].isna(), 1, 0) # Flag before filling
            df[col] = df[col].fillna(fill_value_char)
        # No else needed, as columns should exist by now

    print(f"Characteristic columns processed (ffill, then filled with {fill_value_char}).")
    return df


def full_client_cleaning(client_df_raw: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates all client data cleaning steps."""
    print("--- Starting Full Client Cleaning Process ---")
    df = client_df_raw.copy()

    # Ensure primary date column 'START_DATE' exists and is datetime
    # The notebook uses 'START_DATE' for some operations and 'LEG_START_TIME_DAY' for others.
    # Let's try to standardize on 'START_DATE' for client history.
    if 'START_DATE' not in df.columns:
        if 'LEG_START_TIME_DAY' in df.columns:
            df['START_DATE'] = pd.to_datetime(df['LEG_START_TIME_DAY']).dt.date
            print("Created 'START_DATE' from 'LEG_START_TIME_DAY'.")
        else:
            raise ValueError("'START_DATE' or 'LEG_START_TIME_DAY' must be present in client_df.")
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')


    df = clean_client_age(df)
    df = fill_rolling_window_metrics(df) # Uses 'START_DATE'
    df = process_call_related_features(df) # Uses 'LEG_START_TIME_DAY' or 'START_DATE'
    df = fill_remaining_na_and_flags(df, fill_value_main=0, fill_value_char=-1)

    print("--- Full Client Cleaning Process Complete ---")
    return df

