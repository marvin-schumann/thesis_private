# libPBL2425NovaNOS/modelling/modelling.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, roc_auc_score, precision_recall_curve, f1_score,
    precision_score, recall_score 
)
import sys
import os
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import settings # For KEY_PERSON_SK, KEY_RESOURCE_KEY, DATE_COL_CALLS_RAW etc.

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None 
    logging.getLogger(__name__).warning(
        "imblearn.over_sampling.SMOTE not found. Resampling will not be performed for FTR/OT. "
        "Install with: pip install imbalanced-learn"
    )

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_time_of_day(hour: int) -> str:
    """Categorizes hour of the day."""
    if not isinstance(hour, (int, float, np.number)) or pd.isna(hour): # Handle NaN hours
        return 'Unknown'
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 18: return 'Afternoon'
    elif 18 <= hour < 22: return 'Evening'
    else: return 'Night' # Covers 22, 23, 0, 1, 2, 3, 4

def enrich_with_lvl2_topic_features(df: pd.DataFrame, person_col: str) -> pd.DataFrame:
    """Adds historical Level 2 topic features. Adapted from notebook cell 20."""
    logger.debug(f"Enriching with Lvl2 topic features, using person_col: {person_col}")
    df_enriched = df.copy() # Work on a copy

    # Required columns for this function
    # 'call_CALL_START_TIME_DAT' is expected by original logic for cumcount and shift
    # It should be the main call timestamp column after prefixing.
    call_start_time_col_for_enrich = 'call_CALL_START_TIME_DAT' # Assumed name
    topic_lvl2_col = "call_TOPIC_TIPIFICATION_LVL_2_DSC" # Assumed name

    required_cols_for_enrich = [person_col, call_start_time_col_for_enrich]
    if not all(c in df_enriched.columns for c in required_cols_for_enrich):
        logger.error(f"Missing one or more required columns for Lvl2 topic enrichment: {required_cols_for_enrich}. Aborting enrichment.")
        # Add IS_FIRST_CALL as a fallback if models expect it
        df_enriched["IS_FIRST_CALL"] = 1 # Default to first call if person/time info missing
        return df_enriched

    # Fallback if topic column is missing
    if topic_lvl2_col not in df_enriched.columns:
        logger.warning(f"Column '{topic_lvl2_col}' not found. Adding basic CUM_TOTAL_CALLS and IS_FIRST_CALL only for Lvl2 features.")
        df_enriched["CUM_TOTAL_CALLS"] = df_enriched.groupby(person_col, observed=False)[call_start_time_col_for_enrich].cumcount()
        df_enriched["IS_FIRST_CALL"] = (df_enriched["CUM_TOTAL_CALLS"] == 0).astype(int)
        
        # Add dummy LAG1_LVL2_ columns for feature consistency if expected by models
        # This list should ideally come from settings or be derived if features are dynamic.
        # Using the list from original modelling.py
        dummy_lag_cols = [
            "LAG1_LVL2_ADESÃO/ALTERAÇÃO", "LAG1_LVL2_BOX", "LAG1_LVL2_CARTÃO SIM/ESIM/TWIN",
            "LAG1_LVL2_INTERNET FIXA", "LAG1_LVL2_MAIS INFORMAÇÕES", "LAG1_LVL2_NO_PREV_CALL",
            "LAG1_LVL2_OUTROS", "LAG1_LVL2_SEM VOZ DE CLIENTE", "LAG1_LVL2_TELEVISÃO", "LAG1_LVL2_VOZ FIXA"
        ]
        for col_name in dummy_lag_cols:
            if col_name not in df_enriched.columns: df_enriched[col_name] = 0
        return df_enriched

    # Proceed with full Lvl2 topic enrichment
    df_enriched[topic_lvl2_col] = df_enriched[topic_lvl2_col].replace("TV", "TELEVISÃO") # Specific replacement
    
    # topics_to_keep_lvl2: This set should be defined in settings.py for maintainability
    # e.g., settings.LVL2_TOPICS_TO_KEEP
    topics_to_keep_lvl2 = {
        "INTERNET FIXA", "TELEVISÃO", "BOX", "VOZ FIXA",
        "ADESÃO/ALTERAÇÃO", "SEM VOZ DE CLIENTE", "MAIS INFORMAÇÕES", "CARTÃO SIM/ESIM/TWIN"
    }
    df_enriched["TOPIC_LVL_2_GROUPED"] = df_enriched[topic_lvl2_col].apply(
        lambda x: x if isinstance(x, str) and x in topics_to_keep_lvl2 else "OUTROS"
    )
    
    # Ensure person_col does not have NaNs before groupby.shift, or handle them.
    if df_enriched[person_col].isnull().any():
        logger.warning(f"NaNs found in person_col ('{person_col}') during Lvl2 topic enrichment. Grouping NaNs together for shift/cumsum.")
        # Fill NaNs with a placeholder string for groupby operations, then revert if necessary or accept mixed type.
        # This example proceeds with a placeholder, which might create an unwanted category.
        # A better approach might be to drop rows with NaN person_col or impute if appropriate.
        df_enriched[person_col] = df_enriched[person_col].fillna("UNKNOWN_PERSON_FOR_LVL2_ENRICH")

    df_enriched["TOPIC_LVL_2_GROUPED_LAG1_STR"] = df_enriched.groupby(person_col, observed=False)["TOPIC_LVL_2_GROUPED"].shift(1).fillna("NO_PREV_CALL")
    
    # --- Cumulative Counts (CNT_LVL2_) and Proportions (PROP_LVL2_) ---
    # Create dummies for TOPIC_LVL_2_GROUPED_LAG1_STR for cumulative sum (original notebook used LAG1 for cumsum)
    # This means we count how many times each *previous* topic occurred for the person.
    lvl2_lag1_dummies_for_cumsum = pd.get_dummies(df_enriched["TOPIC_LVL_2_GROUPED_LAG1_STR"], prefix="CNT_LVL2", dtype=int)
    
    # Drop the "NO_PREV_CALL" dummy if created, as it's not a real topic count
    no_prev_call_dummy_col = "CNT_LVL2_NO_PREV_CALL"
    if no_prev_call_dummy_col in lvl2_lag1_dummies_for_cumsum.columns:
        lvl2_lag1_dummies_for_cumsum = lvl2_lag1_dummies_for_cumsum.drop(columns=[no_prev_call_dummy_col], errors='ignore')

    # Join dummies and perform cumsum
    # Need to handle index carefully if df_enriched was modified (e.g. NaN fill in person_col)
    # It's safer to perform cumsum on a temporary frame aligned with original df_enriched structure.
    temp_df_for_cumsum = df_enriched[[person_col]].join(lvl2_lag1_dummies_for_cumsum)
    grouped_cumsum = temp_df_for_cumsum.groupby(person_col, observed=False)[lvl2_lag1_dummies_for_cumsum.columns].cumsum()
    
    # Concatenate cumsum columns to df_enriched
    df_enriched = pd.concat([df_enriched, grouped_cumsum], axis=1)
    
    # CUM_TOTAL_CALLS (0 for first call, 1 for second, etc. for this person)
    df_enriched["CUM_TOTAL_CALLS"] = df_enriched.groupby(person_col, observed=False)[call_start_time_col_for_enrich].cumcount()
    
    # Calculate PROP_LVL2_ columns
    for col in lvl2_lag1_dummies_for_cumsum.columns: # Iterate over the CNT_LVL2_ columns created
        prop_col_name = col.replace("CNT_", "PROP_")
        if col in df_enriched.columns: # Ensure the CNT_ column exists (it should from cumsum step)
            # Denominator is CUM_TOTAL_CALLS + 1 (current call number: 1st, 2nd, etc.)
            # If CUM_TOTAL_CALLS is 0 (first call), LAG1 was NO_PREV_CALL, so CNT_ for actual topics is 0. Prop is 0.
            # If CUM_TOTAL_CALLS is 1 (second call), LAG1 was the topic of 1st call. CNT_ for that topic is 1. Prop is 1 / (1+1) = 0.5.
            # This seems correct: proportion of previous calls that were of this topic.
            df_enriched[prop_col_name] = df_enriched[col] / (df_enriched["CUM_TOTAL_CALLS"] + 1) # Use CUM_TOTAL_CALLS for total prior calls
            # Handle division by zero if CUM_TOTAL_CALLS + 1 could be zero (not possible if cumcount starts at 0)
            # Fill NaN results from 0/0 with 0.
            df_enriched[prop_col_name].fillna(0, inplace=True)
        else:
            df_enriched[prop_col_name] = 0 # Should not happen
            logger.warning(f"Count column {col} for PROP calculation not found in df_enriched.")

    # --- Lag 1 Features (LAG1_LVL2_) ---
    # Dummies for the actual TOPIC_LVL_2_GROUPED_LAG1_STR (one-hot encoding of the previous call's topic)
    lag1_l2_ohe_features = pd.get_dummies(df_enriched["TOPIC_LVL_2_GROUPED_LAG1_STR"], prefix="LAG1_LVL2", drop_first=False, dtype=int)
    
    # Concatenate LAG1_LVL2_ OHE features
    cols_to_add_lag1_ohe = [col for col in lag1_l2_ohe_features.columns if col not in df_enriched.columns]
    df_enriched = pd.concat([df_enriched, lag1_l2_ohe_features[cols_to_add_lag1_ohe]], axis=1)
    
    # IS_FIRST_CALL (based on CUM_TOTAL_CALLS)
    df_enriched["IS_FIRST_CALL"] = (df_enriched["CUM_TOTAL_CALLS"] == 0).astype(int)
    
    # Clean up intermediate columns
    temp_cols_to_drop_lvl2 = ["TOPIC_LVL_2_GROUPED", "TOPIC_LVL_2_GROUPED_LAG1_STR"]
    existing_temp_cols_to_drop_lvl2 = [c for c in temp_cols_to_drop_lvl2 if c in df_enriched.columns]
    if existing_temp_cols_to_drop_lvl2:
        df_enriched = df_enriched.drop(columns=existing_temp_cols_to_drop_lvl2, errors='ignore')
    
    logger.debug("Lvl2 topic feature enrichment complete.")
    return df_enriched


def scale_features_df(X_train_data: pd.DataFrame, X_test_data: pd.DataFrame,
                        binary_prefixes: list, specific_binary_cols: list,
                        fit_scaler: bool = True, existing_scaler: StandardScaler = None):
    """
    Scales numeric features in X_train_data and X_test_data.
    Excludes binary/OHE columns based on prefixes and specific names.
    Returns scaled X_train, X_test, the scaler, and list of columns that were scaled.
    Operates on copies of DataFrames to avoid modifying originals passed to it.
    """
    X_train_scaled = X_train_data.copy()
    X_test_scaled = X_test_data.copy()

    # Identify all numeric columns in training data
    numeric_cols_in_train = X_train_scaled.select_dtypes(include=np.number).columns.tolist()
    
    # Determine columns to scale: numeric, not binary/OHE, and not constant
    cols_to_scale_candidates = [
        col for col in numeric_cols_in_train
        if not any(col.startswith(prefix) for prefix in binary_prefixes) and 
           col not in specific_binary_cols
    ]
    
    # Filter out constant columns only if fitting a new scaler
    cols_to_scale_final_train = []
    if fit_scaler:
        for col in cols_to_scale_candidates:
            if X_train_scaled[col].nunique(dropna=False) > 1: # dropna=False to consider presence of NaN as a unique value for nunique purposes
                cols_to_scale_final_train.append(col)
            else:
                logger.debug(f"Column '{col}' in training data is constant or all NaN. Skipping scaling for this column.")
    else: # If using existing scaler, assume columns were already checked for constancy during fit
        cols_to_scale_final_train = [col for col in cols_to_scale_candidates if col in X_train_scaled.columns]


    if not cols_to_scale_final_train:
        logger.warning("No columns identified for scaling in training data after filtering. Returning unscaled data.")
        return X_train_scaled, X_test_scaled, existing_scaler if existing_scaler else StandardScaler(), []

    current_scaler = None
    if fit_scaler:
        current_scaler = StandardScaler()
        X_train_scaled.loc[:, cols_to_scale_final_train] = current_scaler.fit_transform(X_train_scaled[cols_to_scale_final_train])
        logger.info(f"StandardScaler fitted on {len(cols_to_scale_final_train)} training columns.")
    elif existing_scaler:
        current_scaler = existing_scaler
        # Transform training data using the provided scaler (e.g., in cross-validation folds)
        # Ensure columns exist in current X_train_scaled slice
        train_cols_for_transform = [c for c in cols_to_scale_final_train if c in X_train_scaled.columns]
        if train_cols_for_transform:
             X_train_scaled.loc[:, train_cols_for_transform] = current_scaler.transform(X_train_scaled[train_cols_for_transform])
    else: # fit_scaler is False but no existing_scaler provided
        msg = "If fit_scaler is False, an existing_scaler must be provided."
        logger.error(msg)
        raise ValueError(msg)

    # Transform test data using the fitted or provided scaler
    # Only transform columns that were part of the scaler's fit (i.e., cols_to_scale_final_train)
    # And that also exist in the test set
    cols_to_scale_in_test = [col for col in cols_to_scale_final_train if col in X_test_scaled.columns]
    if cols_to_scale_in_test and current_scaler:
        try:
            X_test_scaled.loc[:, cols_to_scale_in_test] = current_scaler.transform(X_test_scaled[cols_to_scale_in_test])
            logger.info(f"Test data transformed for {len(cols_to_scale_in_test)} columns.")
        except Exception as e:
            logger.error(f"Error transforming test data: {e}. Columns involved: {cols_to_scale_in_test}")
            # This might happen if test data has unexpected values (e.g. all NaNs in a column scaler expects numbers for)
            # Or if schema mismatch significantly. For now, error out.
            raise
    elif not cols_to_scale_in_test:
        logger.warning("No matching columns found in test data for scaling transformation based on training set's scaled columns.")
            
    return X_train_scaled, X_test_scaled, current_scaler, cols_to_scale_final_train


def simulate_boosted_calls(
    sample_ids_for_sim: list,
    final_df_for_lookup: pd.DataFrame, # Contains features for the sampled calls (X)
    gcs_unique_features: pd.DataFrame, # Prefixed GCS features, indexed by RESOURCE_KEY
    xgb_reg, xgb_clf_ftr, xgb_clf_ot,
    scaler_tmc, scaler_ftr, scaler_ot,
    feature_cols_tmc: list, feature_cols_ftr: list, feature_cols_ot: list, # Full list of features model expects
    cols_scaled_tmc: list, cols_scaled_ftr: list, cols_scaled_ot: list # List of cols that were scaled for each model
) -> pd.DataFrame:
    logger.info(f"Starting GC selection simulation for {len(sample_ids_for_sim)} sample calls.")
    all_predictions = []

    if gcs_unique_features.empty:
        logger.warning("gcs_unique_features (cleaned GCS data) is empty. Cannot run simulation.")
        return pd.DataFrame()
        
    # Identify non-GC features from one of the model's feature lists (e.g., TMC model)
    # These are features from the 'call' or 'client' part, or engineered features not starting with 'gcs_'
    call_client_engineered_feature_names = [
        col for col in feature_cols_tmc 
        if not col.startswith('gcs_') # GCS features are assumed to be prefixed 'gcs_' by merger
    ]
    
    for call_id in sample_ids_for_sim:
        if call_id not in final_df_for_lookup.index:
            logger.warning(f"call_LEG_IF_ID {call_id} not found in final_df_for_lookup. Skipping this sample for simulation.")
            continue
        
        # Get the call/client/engineered specific part of features for this call_id from X_sampled
        # These features are already OHE, etc., as they come from the model's input X.
        call_specific_part_series = final_df_for_lookup.loc[call_id, call_client_engineered_feature_names]
        
        for gc_resource_key, gc_row_all_features in gcs_unique_features.iterrows():
            # gc_row_all_features contains all *gcs_prefixed* features for this GC.
            
            # Create a combined feature row for prediction: start with call-specific, then add GC features
            # This combined_series needs to match the full feature set expected by the models.
            # Step 1: Create a base series from call_specific_part_series
            # Step 2: Add/Update with GC-specific features from gc_row_all_features
            # Step 3: Reindex to the model's full feature list, filling any missing with 0 (or other strategy)

            # For TMC model:
            current_features_unscaled_tmc = call_specific_part_series.copy()
            for gc_col_name_from_gcs_unique, gc_val in gc_row_all_features.items():
                if gc_col_name_from_gcs_unique in feature_cols_tmc: # Ensure this GC feature is used by the model
                    current_features_unscaled_tmc[gc_col_name_from_gcs_unique] = gc_val
            
            x_tmc_unscaled_series = current_features_unscaled_tmc.reindex(feature_cols_tmc).fillna(0)
            x_tmc_unscaled_df = pd.DataFrame([x_tmc_unscaled_series])
            x_tmc_scaled_df = x_tmc_unscaled_df.copy() # For scaled features
            if scaler_tmc and cols_scaled_tmc:
                cols_to_transform_tmc = [c for c in cols_scaled_tmc if c in x_tmc_scaled_df.columns]
                if cols_to_transform_tmc:
                    x_tmc_scaled_df[cols_to_transform_tmc] = scaler_tmc.transform(x_tmc_unscaled_df[cols_to_transform_tmc])
            pred_tmc = xgb_reg.predict(x_tmc_scaled_df)[0]

            # For FTR model (similar logic):
            current_features_unscaled_ftr = call_specific_part_series.copy()
            for gc_col_name_from_gcs_unique, gc_val in gc_row_all_features.items():
                if gc_col_name_from_gcs_unique in feature_cols_ftr:
                    current_features_unscaled_ftr[gc_col_name_from_gcs_unique] = gc_val
            
            x_ftr_unscaled_series = current_features_unscaled_ftr.reindex(feature_cols_ftr).fillna(0)
            x_ftr_unscaled_df = pd.DataFrame([x_ftr_unscaled_series])
            x_ftr_scaled_df = x_ftr_unscaled_df.copy()
            if scaler_ftr and cols_scaled_ftr:
                cols_to_transform_ftr = [c for c in cols_scaled_ftr if c in x_ftr_scaled_df.columns]
                if cols_to_transform_ftr:
                    x_ftr_scaled_df[cols_to_transform_ftr] = scaler_ftr.transform(x_ftr_unscaled_df[cols_to_transform_ftr])
            p_ftr = xgb_clf_ftr.predict_proba(x_ftr_scaled_df)[0, 1]

            # For OT model (similar logic):
            current_features_unscaled_ot = call_specific_part_series.copy()
            for gc_col_name_from_gcs_unique, gc_val in gc_row_all_features.items():
                if gc_col_name_from_gcs_unique in feature_cols_ot:
                    current_features_unscaled_ot[gc_col_name_from_gcs_unique] = gc_val

            x_ot_unscaled_series = current_features_unscaled_ot.reindex(feature_cols_ot).fillna(0)
            x_ot_unscaled_df = pd.DataFrame([x_ot_unscaled_series])
            x_ot_scaled_df = x_ot_unscaled_df.copy()
            if scaler_ot and cols_scaled_ot:
                cols_to_transform_ot = [c for c in cols_scaled_ot if c in x_ot_scaled_df.columns]
                if cols_to_transform_ot:
                    x_ot_scaled_df[cols_to_transform_ot] = scaler_ot.transform(x_ot_unscaled_df[cols_to_transform_ot])
            p_ot = xgb_clf_ot.predict_proba(x_ot_scaled_df)[0, 1]
            
            # Cost Calculation:
            # Ensure 'client_LEG_DURATION_MEAN_365_TOTAL' is correctly named and available.
            # It should be part of 'call_specific_part_series' if it was selected as a feature in X.
            # Or, it could be looked up from an auxiliary DataFrame if not in X.
            # Assuming it's in X (and thus in x_tmc_unscaled_series which is derived from call_specific_part_series).
            avg_client_duration_col_name = "client_LEG_DURATION_MEAN_365_TOTAL" # As per original notebook
            avg_client_duration = x_tmc_unscaled_series.get(avg_client_duration_col_name, 0) 
                                                            # Default to 0 if not found, though it should be.
            
            # Cost factors (these should be in settings.py)
            COST_PER_MINUTE = 0.35 
            COST_PER_TECH_VISIT = 22
            
            call_cost   = pred_tmc / 60 * COST_PER_MINUTE
            tech_cost   = p_ot * COST_PER_TECH_VISIT # p_ot is probability of OT
            # repeat_cost: if FTR fails (1-p_ftr), cost is avg duration of a call for that client
            repeat_cost = (1 - p_ftr) * (avg_client_duration / 60 * COST_PER_MINUTE)
            total_cost  = call_cost + tech_cost + repeat_cost
            
            all_predictions.append({
                "Model": "BOOSTED", 
                "LEG_IF_ID": call_id, 
                settings.KEY_RESOURCE_KEY: gc_resource_key, # Use key from settings
                "Pred_TMC_s": pred_tmc, 
                "P_FTR_is_1": p_ftr,  # Prob FTR=1 (successful first call resolution)
                "P_OT_is_1": p_ot,    # Prob OT=1 (technician visit needed)
                "Calculated_Total_Cost": total_cost,
                "Cost_CallHandling": call_cost,
                "Cost_TechVisit_Expected": tech_cost,
                "Cost_RepeatCall_Expected": repeat_cost
            })
    
    logger.info(f"GC selection simulation finished. Generated {len(all_predictions)} cost predictions.")
    return pd.DataFrame(all_predictions)


# --- Main Modelling Function ---
def run_modelling(df_input: pd.DataFrame, gcs_unique_input: pd.DataFrame) -> dict:
    """
    Main function for data preparation, model training, evaluation, and simulation.
    Args:
        df_input: DataFrame from the merging step (final_modelling_df).
        gcs_unique_input: DataFrame from gc_cleaner.py (cleaned GCS features, prefixed 'gcs_').
                          Expected to be indexed by settings.KEY_RESOURCE_KEY.
    Returns:
        A dictionary containing trained models, scalers, evaluation results, and simulation_df.
    """
    logger.info(f"--- Starting Modelling Process ---")
    if df_input.empty:
        logger.error("Input DataFrame (df_input) for modelling is empty. Aborting.")
        return {'error': 'Input DataFrame (df_input) for modelling is empty.'}
        
    full_merged_df = df_input.copy() # This is the output of merger.py
    
    # Ensure gcs_unique_input is correctly indexed for simulation
    if gcs_unique_input.index.name != settings.KEY_RESOURCE_KEY:
        if settings.KEY_RESOURCE_KEY in gcs_unique_input.columns:
            gcs_unique_input = gcs_unique_input.set_index(settings.KEY_RESOURCE_KEY)
            logger.info(f"Set index of gcs_unique_input to '{settings.KEY_RESOURCE_KEY}'.")
        else:
            msg = (f"Key '{settings.KEY_RESOURCE_KEY}' not found in gcs_unique_input columns for index setting. "
                   "Simulation may fail or produce incorrect results.")
            logger.error(msg)
            # Depending on severity, either return error or allow to proceed with warning
            return {'error': msg}

    results_dict = {} 
    
    logger.info("Starting Data Preparation for Modelling...")
    
    # --- 1. Target Variable Creation & Feature Engineering ---
    # Define target variable names (these are internal to modelling.py after creation)
    TARGET_TMC = "TMC_dependent"
    TARGET_FTR = "FTR_dependent"
    TARGET_OT  = "OT_dependent"

    # Raw column names for targets (should come from merger.py with prefixes)
    # These need to be robustly identified, possibly from settings or a naming convention.
    raw_col_tmc = 'call_LEG_DURATION_SEC_QTY'
    raw_col_ftr = 'call_FTR_CALCULATED' # Note: Original modelling.py used call_FTR_CALCULATED > 0
                                        # Call_cleaner created 'FTR_depen' from 'FTR_1_SUM'
                                        # Ensure consistency: if merger passes 'call_FTR_depen', use that.
                                        # If merger passes 'call_FTR_CALCULATED', use that.
                                        # Assuming 'call_FTR_CALCULATED' is available and suitable.
    raw_col_ot  = 'call_FLAG_OT'

    # Create TMC_dependent
    if raw_col_tmc not in full_merged_df.columns:
        raise ValueError(f"Required raw target column '{raw_col_tmc}' is missing from merged data.")
    full_merged_df = full_merged_df[full_merged_df[raw_col_tmc] <= 10800].copy() # Filter outliers
    full_merged_df[TARGET_TMC] = full_merged_df[raw_col_tmc]

    # Create FTR_dependent
    if raw_col_ftr not in full_merged_df.columns:
        raise ValueError(f"Required raw target column '{raw_col_ftr}' is missing from merged data.")
    # Ensure FTR is 0 or 1. Original logic: (call_FTR_CALCULATED > 0).astype(int)
    full_merged_df[TARGET_FTR] = (pd.to_numeric(full_merged_df[raw_col_ftr], errors='coerce').fillna(0) > 0).astype(int)

    # Create OT_dependent
    if raw_col_ot not in full_merged_df.columns:
        raise ValueError(f"Required raw target column '{raw_col_ot}' is missing from merged data.")
    full_merged_df[TARGET_OT] = pd.to_numeric(full_merged_df[raw_col_ot], errors='coerce').fillna(0).astype(int)

    # --- Time-based & Other Engineered Features ---
    # Main call time reference (unprefixed, as excluded from prefixing in merger)
    main_call_time_col = settings.DATE_COL_CALLS_RAW # e.g., 'CALL_START_TIME_DAT'
    if main_call_time_col not in full_merged_df.columns:
        # If it got prefixed somehow, check that
        prefixed_main_call_time = f"call_{settings.DATE_COL_CALLS_RAW}"
        if prefixed_main_call_time in full_merged_df.columns:
            main_call_time_col = prefixed_main_call_time
            logger.warning(f"Main call time column found as prefixed: '{main_call_time_col}'. This is unusual if merger excluded it.")
        else:
            raise ValueError(f"Main call time column '{settings.DATE_COL_CALLS_RAW}' (or prefixed) missing.")

    # Convert relevant date columns (already prefixed by merger) to datetime
    date_cols_to_convert_raw = ["CALL_START_TIME_DAT", "CALL_END_TIME_DAT", "LEG_START_TIME_DAT", "LEG_END_TIME_DAT"]
    prefixed_date_cols_to_convert = [f"call_{col}" for col in date_cols_to_convert_raw]
    for col in prefixed_date_cols_to_convert:
        if col in full_merged_df.columns:
            full_merged_df[col] = pd.to_datetime(full_merged_df[col], errors='coerce')
    
    if full_merged_df[main_call_time_col].isnull().any(): # Check the active main_call_time_col
        logger.warning(f"NaNs found in main call time column '{main_call_time_col}' after conversion. This may affect time-based features.")

    # Client identifier (unprefixed, as excluded by merger)
    client_id_col = settings.KEY_PERSON_SK 
    if client_id_col not in full_merged_df.columns:
        raise ValueError(f"Client identifier column '{client_id_col}' not found in merged data.")
    
    # Ensure sorting for shift operations
    full_merged_df = full_merged_df.sort_values(by=[client_id_col, main_call_time_col])
    
    # Days since previous call for client
    full_merged_df['previous_call_time_client'] = full_merged_df.groupby(client_id_col, observed=False)[main_call_time_col].shift(1)
    full_merged_df['days_since_prev_call_for_client'] = (full_merged_df[main_call_time_col] - full_merged_df['previous_call_time_client']).dt.days
    full_merged_df['client_called_in_last_7_days'] = (full_merged_df['days_since_prev_call_for_client'].fillna(999) <= 7).astype(int)
    full_merged_df['days_since_prev_call_for_client'] = full_merged_df['days_since_prev_call_for_client'].fillna(-1).astype(int)
    
    # Enrich with Lvl2 topic features (expects 'call_CALL_START_TIME_DAT' for time, person_col for grouping)
    # Ensure the 'call_CALL_START_TIME_DAT' column used by enrich_with_lvl2_topic_features exists.
    # If main_call_time_col is different, temporarily rename or pass the correct one.
    if 'call_CALL_START_TIME_DAT' in full_merged_df.columns:
         full_merged_df = enrich_with_lvl2_topic_features(full_merged_df, person_col=client_id_col)
    else:
        logger.warning(f"'call_CALL_START_TIME_DAT' (expected by enrich_with_lvl2_topic_features) not found. Lvl2 features might be incomplete.")
        # Basic IS_FIRST_CALL if enrichment is skipped/partial
        if "IS_FIRST_CALL" not in full_merged_df.columns: full_merged_df["IS_FIRST_CALL"] = 1


    # Days since last call for GC (Resource Key is unprefixed from merger)
    gc_id_col = settings.KEY_RESOURCE_KEY
    if gc_id_col in full_merged_df.columns:
        full_merged_df = full_merged_df.sort_values(by=[gc_id_col, main_call_time_col])
        full_merged_df['previous_call_time_gc'] = full_merged_df.groupby(gc_id_col, observed=False)[main_call_time_col].shift(1)
        full_merged_df['days_since_last_call_gc'] = (full_merged_df[main_call_time_col] - full_merged_df['previous_call_time_gc']).dt.days
        full_merged_df['days_since_last_call_gc'] = full_merged_df['days_since_last_call_gc'].fillna(-1).astype(int)
    else:
        logger.warning(f"GC identifier column '{gc_id_col}' not found. Skipping GC time-based features.")
        full_merged_df['days_since_last_call_gc'] = -1 # Default if no GC key
        
    # Time of day features
    full_merged_df['day_of_week_num'] = full_merged_df[main_call_time_col].dt.dayofweek # 0=Mon, 6=Sun
    full_merged_df['is_weekend'] = (full_merged_df['day_of_week_num'] >= 5).astype(int)
    full_merged_df['hour_of_day'] = full_merged_df[main_call_time_col].dt.hour
    full_merged_df['time_of_day_str'] = full_merged_df['hour_of_day'].apply(get_time_of_day)

    # Previous call topic classification (using prefixed column from merger)
    prev_topic_raw_col = 'call_TOPIC_CLASSIFIC_ENTRY_AT_FT' # This is assumed to be the prefixed name
    if prev_topic_raw_col in full_merged_df.columns:
        full_merged_df["PREV_CALL_TOPIC_CLASSIFIC_ENTRY_AT_FT"] = full_merged_df.groupby(client_id_col, observed=False)[prev_topic_raw_col].shift(1).fillna("NO_PREV_TOPIC")
    else:
        logger.warning(f"Raw topic column '{prev_topic_raw_col}' not found. 'PREV_CALL_TOPIC_CLASSIFIC_ENTRY_AT_FT' set to 'NO_PREV_TOPIC'.")
        full_merged_df["PREV_CALL_TOPIC_CLASSIFIC_ENTRY_AT_FT"] = "NO_PREV_TOPIC"

    # --- Column Selection for final_df (Features + Targets + ID for indexing) ---
    # Call LEG_IF_ID column for indexing (should be prefixed by merger)
    call_leg_id_col_for_index = 'call_LEG_IF_ID' # This must be consistent with merger.py output
    if call_leg_id_col_for_index not in full_merged_df.columns:
        raise ValueError(f"Call identifier '{call_leg_id_col_for_index}' missing for setting index.")

    # Dynamically select features:
    # Start with all columns, then exclude non-features and targets.
    potential_feature_cols = full_merged_df.columns.tolist()
    
    # Columns to EXCLUDE from features:
    cols_to_exclude = [
        # IDs
        client_id_col,  # settings.KEY_PERSON_SK
        gc_id_col,      # settings.KEY_RESOURCE_KEY
        call_leg_id_col_for_index, # Used as index
        # Original main date column (if not transformed into a feature like hour_of_day already)
        main_call_time_col, 
        # Other raw date/time columns that were prefixed if not used as features
        f"call_{settings.DATE_COL_CALLS_RAW}", # Redundant if main_call_time_col is this and prefixed
        'call_CALL_END_TIME_DAT', 'call_LEG_START_TIME_DAT', 'call_LEG_END_TIME_DAT',
        # Target variables
        TARGET_TMC, TARGET_FTR, TARGET_OT,
        # Intermediate helper columns for feature engineering if not final features
        'previous_call_time_client', 'previous_call_time_gc',
        'day_of_week_num' # 'is_weekend' and 'hour_of_day' are used instead
    ]
    # Add any other specific non-feature columns here based on your data.
    
    final_feature_list = [col for col in potential_feature_cols if col not in cols_to_exclude]
    final_feature_list = sorted(list(set(final_feature_list))) # Unique and sorted for consistency

    logger.info(f"Identified {len(final_feature_list)} features for modelling.")
    if not final_feature_list:
        logger.error("No features were identified for modelling. Check feature selection logic and input data.")
        return {'error': "No features identified."}

    # Prepare final_df for modelling
    cols_for_final_df = [call_leg_id_col_for_index] + \
                        [TARGET_TMC, TARGET_FTR, TARGET_OT] + \
                        final_feature_list
    cols_for_final_df = [c for c in cols_for_final_df if c in full_merged_df.columns] # Ensure all exist
    cols_for_final_df = list(dict.fromkeys(cols_for_final_df)) # Unique
    
    final_df = full_merged_df[cols_for_final_df].copy()
    
    # Aggressive dropna - review if this is appropriate for your dataset
    rows_before_dropna = len(final_df)
    final_df.dropna(inplace=True) # Drops rows with ANY NaN in selected columns
    logger.info(f"Dropped {rows_before_dropna - len(final_df)} rows from final_df due to NaNs. Current shape: {final_df.shape}")
    
    if final_df.empty:
        logger.error("final_df is empty after dropping NaNs. Modelling cannot proceed. Review data quality, feature engineering, and NaN handling.")
        return {'error': 'final_df became empty after critical NaN drop.'}

    # One-Hot Encode categorical features present in final_df
    # These column names are the *final* names after any engineering.
    categorical_cols_for_ohe = [
        "call_TOPIC_CLASSIFIC_ENTRY_AT_FT", # If this was selected as a feature
        "PREV_CALL_TOPIC_CLASSIFIC_ENTRY_AT_FT", # Engineered
        "time_of_day_str" # Engineered
        # Add any other categorical features that are in `final_feature_list`
    ]
    # Filter to only those present in final_df's columns (which are based on final_feature_list)
    actual_categorical_cols_to_encode = [col for col in categorical_cols_for_ohe if col in final_df.columns]
    if actual_categorical_cols_to_encode:
        logger.info(f"One-hot encoding categorical features: {actual_categorical_cols_to_encode}")
        final_df = pd.get_dummies(final_df, columns=actual_categorical_cols_to_encode, drop_first=True, dtype=int)
    
    final_df.set_index(call_leg_id_col_for_index, inplace=True)
    
    logger.info("Modelling Data Preparation Finished.")

    # --- 2. Define features X and targets y from final_df ---
    X = final_df.drop(columns=[TARGET_TMC, TARGET_FTR, TARGET_OT], errors='ignore')
    y_tmc = final_df[TARGET_TMC]
    y_ftr = final_df[TARGET_FTR]
    y_ot = final_df[TARGET_OT]

    # Define binary prefixes and specific binary columns for scaling exclusion
    # These should align with prefixes from OHE and known binary flags in X.
    binary_prefixes_for_scaling = [
        "call_TOPIC_CLASSIFIC_ENTRY_AT_FT_", 
        "PREV_CALL_TOPIC_CLASSIFIC_ENTRY_AT_FT_",
        "LAG1_LVL2_", # From enrich_with_lvl2_topic_features
        "time_of_day_str_" # From OHE of 'time_of_day_str'
        # Add other OHE prefixes if any
    ]
    specific_binary_cols_for_scaling = [ # Manually specified binary columns that are already 0/1
        'IS_FIRST_CALL', 'is_weekend', 'client_called_in_last_7_days',
        # Add client-level flags like 'client_Age_missing'
    ]
    # Dynamically add GCS binary flags (e.g., gcs_CALLS_IF_BINARY)
    specific_binary_cols_for_scaling.extend([col for col in X.columns if col.startswith('gcs_CALLS_') and col.endswith('_BINARY')])
    # Dynamically add client characteristic flags (e.g., client_Age_missing, client_Flag_PF_CLIENT_MONTHS_Is_Imputed)
    specific_binary_cols_for_scaling.extend([col for col in X.columns if col.startswith('client_') and ('_missing' in col or '_Is_Imputed' in col or col.endswith('_FLG') or col.endswith('_Flg'))])
    # Dynamically add any other known binary flags from feature engineering (e.g. "Flag_Charachteristics_Empty")
    if "Flag_Charachteristics_Empty" in X.columns: specific_binary_cols_for_scaling.append("Flag_Charachteristics_Empty")
    
    specific_binary_cols_for_scaling = sorted(list(set(specific_binary_cols_for_scaling))) # Unique and sorted

    # --- Model Training Parameters (can be moved to settings.py) ---
    xgb_common_params = {'random_state': 42, 'n_jobs': -1, 'objective': 'reg:squarederror'} # Default objective for regressor

    # --- TMC Model (XGBoost Regressor) ---
    logger.info("\n--- Training TMC Model (XGBoost Regressor) ---")
    X_train_tmc, X_test_tmc, y_train_tmc, y_test_tmc = train_test_split(X, y_tmc, test_size=0.2, random_state=42)
    
    # Scale features for TMC model
    X_train_scaled_tmc, X_test_scaled_tmc, scaler_tmc, scaled_cols_tmc_list = scale_features_df(
        X_train_tmc, X_test_tmc, # Pass copies if original X_train/X_test needed later unscaled
        binary_prefixes_for_scaling, specific_binary_cols_for_scaling, fit_scaler=True
    )
        
    xgb_tmc_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=50, **xgb_common_params)
    xgb_tmc_model.fit(X_train_scaled_tmc, y_train_tmc)
    y_pred_tmc = xgb_tmc_model.predict(X_test_scaled_tmc)
    tmc_r2 = r2_score(y_test_tmc, y_pred_tmc)
    tmc_rmse = np.sqrt(mean_squared_error(y_test_tmc, y_pred_tmc))
    logger.info(f"TMC Model - R2: {tmc_r2:.4f}, RMSE: {tmc_rmse:.2f}")
    results_dict['tmc'] = {
        'model': xgb_tmc_model, 
        'scaler': scaler_tmc, 
        'scaled_cols': scaled_cols_tmc_list, 
        'features': X_train_scaled_tmc.columns.tolist(), # Full feature list for this model
        'metrics': {'R2': tmc_r2, 'RMSE': tmc_rmse}
    }

    # --- FTR Model (XGBoost Classifier) ---
    logger.info("\n--- Training FTR Model (XGBoost Classifier) ---")
    # Stratify only if there are at least 2 unique values in y_ftr and each class has enough samples for split
    stratify_ftr = y_ftr if y_ftr.nunique() > 1 and all(y_ftr.value_counts() >= 2) else None
    X_train_ftr, X_test_ftr, y_train_ftr, y_test_ftr = train_test_split(X, y_ftr, test_size=0.2, random_state=42, stratify=stratify_ftr)
    
    X_train_ftr_processed = X_train_ftr.copy() # Operate on copy for SMOTE
    y_train_ftr_processed = y_train_ftr.copy()
    if SMOTE and y_train_ftr_processed.nunique() > 1 and all(y_train_ftr_processed.value_counts() >= (SMOTE().get_params()['k_neighbors'] if hasattr(SMOTE(), 'get_params') and 'k_neighbors' in SMOTE().get_params() else 5)): # Check k_neighbors for SMOTE
        logger.info("Applying SMOTE to FTR training data...")
        smote_ftr_obj = SMOTE(random_state=42)
        try:
            # SMOTE expects NumPy arrays for X
            X_train_ftr_np_res, y_train_ftr_res = smote_ftr_obj.fit_resample(X_train_ftr_processed.values, y_train_ftr_processed.values)
            X_train_ftr_processed = pd.DataFrame(X_train_ftr_np_res, columns=X_train_ftr_processed.columns)
            y_train_ftr_processed = pd.Series(y_train_ftr_res)
        except ValueError as e:
            logger.warning(f"SMOTE for FTR failed: {e}. Using original data for FTR model.")
    
    X_train_scaled_ftr, X_test_scaled_ftr, scaler_ftr, scaled_cols_ftr_list = scale_features_df(
        X_train_ftr_processed, X_test_ftr, # Pass copies if needed
        binary_prefixes_for_scaling, specific_binary_cols_for_scaling, fit_scaler=True
    )
        
    xgb_ftr_model = xgb.XGBClassifier(learning_rate=0.01, max_depth=4, n_estimators=200, 
                                use_label_encoder=False, eval_metric='logloss', 
                                objective='binary:logistic', **xgb_common_params)
    xgb_ftr_model.fit(X_train_scaled_ftr, y_train_ftr_processed)
    y_prob_ftr = xgb_ftr_model.predict_proba(X_test_scaled_ftr)[:, 1]
    
    roc_auc_ftr = roc_auc_score(y_test_ftr, y_prob_ftr) if y_test_ftr.nunique() > 1 else 0.5
    precisions_ftr, recalls_ftr, thresholds_ftr = precision_recall_curve(y_test_ftr, y_prob_ftr)
    f1_scores_ftr = np.divide(2 * precisions_ftr * recalls_ftr, precisions_ftr + recalls_ftr, 
                              out=np.zeros_like(precisions_ftr), where=(precisions_ftr + recalls_ftr) != 0)
    
    best_f1_idx_ftr = np.argmax(f1_scores_ftr[:-1]) if len(f1_scores_ftr) > 1 else 0 # Exclude last P/R for threshold array alignment
    best_thresh_ftr = thresholds_ftr[best_f1_idx_ftr] if len(thresholds_ftr) > 0 and len(f1_scores_ftr) > 1 else 0.5
    
    y_pred_ftr_thresh = (y_prob_ftr >= best_thresh_ftr).astype(int)
    f1_ftr_val = f1_score(y_test_ftr, y_pred_ftr_thresh, zero_division=0)
    precision_ftr_val = precision_score(y_test_ftr, y_pred_ftr_thresh, zero_division=0)
    recall_ftr_val = recall_score(y_test_ftr, y_pred_ftr_thresh, zero_division=0)
    
    logger.info(f"FTR Model - ROC AUC: {roc_auc_ftr:.4f}, Best F1 Thresh: {best_thresh_ftr:.2f}")
    logger.info(f"FTR Model - Metrics @ Best Thresh: Precision: {precision_ftr_val:.4f}, Recall: {recall_ftr_val:.4f}, F1: {f1_ftr_val:.4f}")
    results_dict['ftr'] = {
        'model': xgb_ftr_model, 'scaler': scaler_ftr, 'scaled_cols': scaled_cols_ftr_list,
        'features': X_train_scaled_ftr.columns.tolist(),
        'metrics': {'ROC_AUC': roc_auc_ftr, 'F1': f1_ftr_val, 'Precision': precision_ftr_val, 
                    'Recall': recall_ftr_val, 'Best_Threshold': best_thresh_ftr}
    }

    # --- OT Model (XGBoost Classifier) --- (Similar structure to FTR model)
    logger.info("\n--- Training OT Model (XGBoost Classifier) ---")
    stratify_ot = y_ot if y_ot.nunique() > 1 and all(y_ot.value_counts() >= 2) else None
    X_train_ot, X_test_ot, y_train_ot, y_test_ot = train_test_split(X, y_ot, test_size=0.2, random_state=42, stratify=stratify_ot)
    
    X_train_ot_processed = X_train_ot.copy()
    y_train_ot_processed = y_train_ot.copy()
    if SMOTE and y_train_ot_processed.nunique() > 1 and all(y_train_ot_processed.value_counts() >= (SMOTE().get_params()['k_neighbors'] if hasattr(SMOTE(), 'get_params') and 'k_neighbors' in SMOTE().get_params() else 5)):
        logger.info("Applying SMOTE to OT training data...")
        smote_ot_obj = SMOTE(random_state=42)
        try:
            X_train_ot_np_res, y_train_ot_res = smote_ot_obj.fit_resample(X_train_ot_processed.values, y_train_ot_processed.values)
            X_train_ot_processed = pd.DataFrame(X_train_ot_np_res, columns=X_train_ot_processed.columns)
            y_train_ot_processed = pd.Series(y_train_ot_res)
        except ValueError as e:
            logger.warning(f"SMOTE for OT failed: {e}. Using original data for OT model.")

    X_train_scaled_ot, X_test_scaled_ot, scaler_ot, scaled_cols_ot_list = scale_features_df(
        X_train_ot_processed, X_test_ot, # Pass copies if needed
        binary_prefixes_for_scaling, specific_binary_cols_for_scaling, fit_scaler=True
    )
    xgb_ot_model = xgb.XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=100,
                               use_label_encoder=False, eval_metric='logloss', 
                               objective='binary:logistic', **xgb_common_params)
    xgb_ot_model.fit(X_train_scaled_ot, y_train_ot_processed)
    y_prob_ot = xgb_ot_model.predict_proba(X_test_scaled_ot)[:, 1]
    
    roc_auc_ot = roc_auc_score(y_test_ot, y_prob_ot) if y_test_ot.nunique() > 1 else 0.5
    precisions_ot, recalls_ot, thresholds_ot = precision_recall_curve(y_test_ot, y_prob_ot)
    f1_scores_ot = np.divide(2 * precisions_ot * recalls_ot, precisions_ot + recalls_ot, 
                             out=np.zeros_like(precisions_ot), where=(precisions_ot + recalls_ot) != 0)
    
    best_f1_idx_ot = np.argmax(f1_scores_ot[:-1]) if len(f1_scores_ot) > 1 else 0
    best_thresh_ot = thresholds_ot[best_f1_idx_ot] if len(thresholds_ot) > 0 and len(f1_scores_ot) > 1 else 0.5
    
    y_pred_ot_thresh = (y_prob_ot >= best_thresh_ot).astype(int)
    f1_ot_val = f1_score(y_test_ot, y_pred_ot_thresh, zero_division=0)
    precision_ot_val = precision_score(y_test_ot, y_pred_ot_thresh, zero_division=0)
    recall_ot_val = recall_score(y_test_ot, y_pred_ot_thresh, zero_division=0)
    
    logger.info(f"OT Model - ROC AUC: {roc_auc_ot:.4f}, Best F1 Thresh: {best_thresh_ot:.2f}")
    logger.info(f"OT Model - Metrics @ Best Thresh: Precision: {precision_ot_val:.4f}, Recall: {recall_ot_val:.4f}, F1: {f1_ot_val:.4f}")
    results_dict['ot'] = {
        'model': xgb_ot_model, 'scaler': scaler_ot, 'scaled_cols': scaled_cols_ot_list,
        'features': X_train_scaled_ot.columns.tolist(),
        'metrics': {'ROC_AUC': roc_auc_ot, 'F1': f1_ot_val, 'Precision': precision_ot_val, 
                    'Recall': recall_ot_val, 'Best_Threshold': best_thresh_ot}
    }
    
    # --- Simulation for GC Selection ---
    logger.info("\n--- Running GC Selection Simulation ---")
    # Use gcs_unique_input (already cleaned, prefixed 'gcs_', and indexed by RESOURCE_KEY)
    if not gcs_unique_input.empty and not X.empty:
        sample_n_sim = min(10, len(X)) # Simulate for N calls
        if sample_n_sim > 0:
            sampled_call_ids_for_sim = X.sample(n=sample_n_sim, random_state=42).index.tolist()
            
            # final_df_for_lookup_sim should contain all features from X for these sampled IDs
            final_df_for_lookup_sim = X.loc[sampled_call_ids_for_sim].copy()
            
            # The simulate_boosted_calls function needs the full feature list that each model was trained on.
            # These are stored in results_dict[model_key]['features']
            # And the list of columns that were scaled for each: results_dict[model_key]['scaled_cols']

            boosted_simulation_df = simulate_boosted_calls(
                sample_ids_for_sim=sampled_call_ids_for_sim,
                final_df_for_lookup=final_df_for_lookup_sim, # This is X_sampled
                gcs_unique_features=gcs_unique_input, 
                xgb_reg=xgb_tmc_model, xgb_clf_ftr=xgb_ftr_model, xgb_clf_ot=xgb_ot_model,
                scaler_tmc=scaler_tmc, scaler_ftr=scaler_ftr, scaler_ot=scaler_ot,
                feature_cols_tmc=results_dict['tmc']['features'], 
                feature_cols_ftr=results_dict['ftr']['features'],
                feature_cols_ot=results_dict['ot']['features'],
                cols_scaled_tmc=results_dict['tmc']['scaled_cols'],
                cols_scaled_ftr=results_dict['ftr']['scaled_cols'],
                cols_scaled_ot=results_dict['ot']['scaled_cols']
            )
            results_dict['simulation_df'] = boosted_simulation_df
            logger.info("GC Selection Simulation Finished.")
            
            if not boosted_simulation_df.empty:
                for leg_id_sim in sampled_call_ids_for_sim:
                    top_gcs_for_call = boosted_simulation_df[boosted_simulation_df["LEG_IF_ID"] == leg_id_sim].sort_values("Calculated_Total_Cost").head(3)
                    logger.info(f"\nTop 3 GCs for Call {leg_id_sim} (BOOSTED Cost Model):")
                    if not top_gcs_for_call.empty:
                        # Log relevant columns for conciseness
                        log_sim_cols = [settings.KEY_RESOURCE_KEY, "Pred_TMC_s", "P_FTR_is_1", "P_OT_is_1", "Calculated_Total_Cost"]
                        logger.info(top_gcs_for_call[log_sim_cols].to_string())
                    else:
                        logger.info(f"No simulation results found for call_LEG_IF_ID {leg_id_sim}")
        else: # sample_n_sim is 0
            logger.warning("Not enough data points in X (features) to run GC simulation sample.")
            results_dict['simulation_df'] = pd.DataFrame()
    else:
        logger.warning("Skipping GC simulation as gcs_unique_input (cleaned GCS data) or feature set X is empty.")
        results_dict['simulation_df'] = pd.DataFrame()
        
    logger.info(f"--- Modelling Process Finished ---")
    return results_dict