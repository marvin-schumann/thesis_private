#!/usr/bin/env python3
"""
Simulator Validation Script

This script validates that the RL environment simulator produces accurate predictions
by comparing simulator predictions against actual historical outcomes on a test set.

The validation follows these steps:
1. Load the full merged dataset
2. Recreate the train-test split (80/20, random_state=42)
3. For each call in the test set, use the simulator to predict TMC, FTR, OT
4. Calculate predicted cost vs actual cost
5. Report validation metrics: MAE, RMSE, R², Correlation

Usage:
    python validate_simulator.py [--sample-size N] [--output-path PATH]

Author: Thesis RL Methodology Validation
Date: 2025-11-20
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import environment components
from call_center_env import CallCenterEnv
from residual_adjustments import ResidualAdjuster

# Configuration
DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Validate RL simulator accuracy")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Number of test samples to validate (default: 5000)")
    parser.add_argument("--output-path", type=str, default=os.path.join(ASSETS_DIR, 'simulator_validation_report.csv'),
                        help="Output path for validation results CSV")
    parser.add_argument("--data-path", type=str, default=DATA_PATH,
                        help="Path to merged dataset CSV")
    parser.add_argument("--no-residuals", action="store_true",
                        help="Disable residual bias corrections (test raw model predictions)")
    return parser.parse_args()


def load_models_and_scalers():
    """Load all trained models and scalers."""
    print("Loading models and scalers...")

    models = {
        'tmc': joblib.load(os.path.join(ASSETS_DIR, 'model_tmc.joblib')),
        'ftr': joblib.load(os.path.join(ASSETS_DIR, 'model_ftr.joblib')),
        'ot': joblib.load(os.path.join(ASSETS_DIR, 'model_ot.joblib'))
    }

    scalers = {
        'tmc': joblib.load(os.path.join(ASSETS_DIR, 'scaler_tmc.joblib')),
        'ftr': joblib.load(os.path.join(ASSETS_DIR, 'scaler_ftr.joblib')),
        'ot': joblib.load(os.path.join(ASSETS_DIR, 'scaler_ot.joblib'))
    }

    with open(os.path.join(ASSETS_DIR, 'feature_lists.json'), 'r') as f:
        feature_lists = json.load(f)

    # Load residual adjuster if available
    residuals_path = os.path.join(ASSETS_DIR, 'gc_residuals_summary.csv')
    residual_adjuster = None
    if os.path.exists(residuals_path):
        try:
            residual_adjuster = ResidualAdjuster(residuals_path)
            print(f"✓ Loaded residual adjustments from {residuals_path}")
        except Exception as exc:
            print(f"⚠ Warning: Could not load residual adjustments: {exc}")

    return models, scalers, feature_lists, residual_adjuster


def calculate_cost(tmc, ftr_prob, ot_prob):
    """Calculate cost using the same formula as the environment."""
    cost_per_minute = 0.35
    cost_per_ot = 22.0

    tmc_pop = tmc  # Using TMC as proxy for TMC_pop

    cost_duration = (tmc / 60.0) * cost_per_minute
    cost_repeat = (1.0 - ftr_prob) * (tmc_pop / 60.0) * cost_per_minute
    cost_ot = ot_prob * cost_per_ot

    total_cost = cost_duration + cost_repeat + cost_ot
    return total_cost


def build_feature_vector(call_data, agent_data, feature_list):
    """Build feature vector for model prediction."""
    combined_data = {**call_data, **agent_data}
    combined_series = pd.Series(combined_data)
    feature_values = combined_series.reindex(feature_list, fill_value=0)
    feature_values = pd.to_numeric(feature_values, errors='coerce').fillna(0.0)
    return feature_values.to_frame().T.astype(np.float32)


def predict_with_simulator(call_data, agent_data, models, scalers, feature_lists, residual_adjuster, agent_key, topic_value, apply_residuals=True):
    """Use the simulator to predict TMC, FTR, OT for a given call-agent pair."""

    # TMC prediction
    vec_tmc_df = build_feature_vector(call_data, agent_data, feature_lists['tmc'])
    scaled_cols_tmc = list(getattr(scalers['tmc'], 'feature_names_in_', []))
    vec_tmc_scaled_df = vec_tmc_df.copy()
    if scaled_cols_tmc:
        vec_tmc_scaled_df[scaled_cols_tmc] = scalers['tmc'].transform(vec_tmc_df[scaled_cols_tmc])
    pred_tmc = float(models['tmc'].predict(vec_tmc_scaled_df)[0])
    pred_tmc = max(30.0, pred_tmc)  # Minimum 30 seconds

    # FTR prediction
    vec_ftr_df = build_feature_vector(call_data, agent_data, feature_lists['ftr'])
    scaled_cols_ftr = list(getattr(scalers['ftr'], 'feature_names_in_', []))
    vec_ftr_scaled_df = vec_ftr_df.copy()
    if scaled_cols_ftr:
        vec_ftr_scaled_df[scaled_cols_ftr] = scalers['ftr'].transform(vec_ftr_df[scaled_cols_ftr])
    pred_ftr_prob = float(models['ftr'].predict_proba(vec_ftr_scaled_df)[0][1])

    # OT prediction
    vec_ot_df = build_feature_vector(call_data, agent_data, feature_lists['ot'])
    scaled_cols_ot = list(getattr(scalers['ot'], 'feature_names_in_', []))
    vec_ot_scaled_df = vec_ot_df.copy()
    if scaled_cols_ot:
        vec_ot_scaled_df[scaled_cols_ot] = scalers['ot'].transform(vec_ot_df[scaled_cols_ot])
    ot_proba = models['ot'].predict_proba(vec_ot_scaled_df)[0]
    pred_ot_prob = float(ot_proba[1]) if len(ot_proba) == 2 else float(ot_proba[0])

    # Apply residual bias correction (without stochastic noise for validation)
    if apply_residuals and residual_adjuster is not None:
        adjustments = residual_adjuster.get_adjustments(agent_key, topic_value)
        pred_tmc += adjustments['tmc'].bias
        pred_ftr_prob += adjustments['ftr'].bias
        pred_ot_prob += adjustments['ot'].bias

        # Enforce bounds
        pred_tmc = max(30.0, float(pred_tmc))
        pred_ftr_prob = float(np.clip(pred_ftr_prob, 0.0, 1.0))
        pred_ot_prob = float(np.clip(pred_ot_prob, 0.0, 1.0))

    return pred_tmc, pred_ftr_prob, pred_ot_prob


def validate_simulator(data_path, sample_size, output_path, apply_residuals=True):
    """Main validation function."""

    print("="*60)
    print("SIMULATOR VALIDATION")
    print("="*60)
    print(f"Data path: {data_path}")
    print(f"Sample size: {sample_size}")
    print(f"Output path: {output_path}")
    print(f"Residual adjustments: {'ENABLED' if apply_residuals else 'DISABLED'}")
    print()

    # Load models
    models, scalers, feature_lists, residual_adjuster = load_models_and_scalers()

    # Load full dataset
    print("Loading dataset...")
    full_df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(full_df):,} rows")

    # Identify columns
    call_cols = [col for col in full_df.columns if col.startswith('call_')]
    client_cols = [col for col in full_df.columns if col.startswith('client_')]
    gc_cols = [col for col in full_df.columns if col.startswith('gc_')]

    # Find target columns (actual outcomes)
    tmc_target_col = None
    ftr_target_col = None
    ot_target_col = None

    possible_tmc_cols = ['call_LEG_DURATION_SEC_QTY', 'TMC_dependent', 'call_TMC']
    possible_ftr_cols = ['call_FTR_dependent', 'call_FTR_CALCULATED', 'call_FTR_1_SUM']
    possible_ot_cols = ['call_FLAG_OT', 'OT_dependent']

    for col in possible_tmc_cols:
        if col in full_df.columns:
            tmc_target_col = col
            break

    for col in possible_ftr_cols:
        if col in full_df.columns:
            ftr_target_col = col
            break

    for col in possible_ot_cols:
        if col in full_df.columns:
            ot_target_col = col
            break

    if not all([tmc_target_col, ftr_target_col, ot_target_col]):
        raise ValueError(f"Could not find target columns. Found: TMC={tmc_target_col}, FTR={ftr_target_col}, OT={ot_target_col}")

    print(f"✓ Target columns: TMC={tmc_target_col}, FTR={ftr_target_col}, OT={ot_target_col}")

    # Check for RESOURCE_KEY
    if 'RESOURCE_KEY' not in full_df.columns:
        raise ValueError("RESOURCE_KEY column not found in dataset")

    # Recreate train-test split (same as training)
    print("\nRecreating train-test split (80/20, random_state=42)...")
    indices = np.arange(len(full_df))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)
    print(f"✓ Test set size: {len(test_df):,} rows")

    # Sample if needed
    if sample_size < len(test_df):
        print(f"Sampling {sample_size} rows from test set...")
        test_df = test_df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    # Load GCS data for agent features
    data_dir = os.path.dirname(data_path)
    gcs_path = os.path.join(data_dir, "gcs_unique.csv")

    if not os.path.exists(gcs_path):
        print(f"⚠ Warning: GCS file not found at {gcs_path}. Using empty agent features.")
        gcs_df = pd.DataFrame()
    else:
        gcs_df = pd.read_csv(gcs_path)
        if 'RESOURCE_KEY' in gcs_df.columns:
            gcs_df = gcs_df.set_index('RESOURCE_KEY')
        print(f"✓ Loaded GCS data with {len(gcs_df)} agents")

    # Validate predictions
    print("\nValidating predictions...")
    print("-" * 60)

    results = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Validating"):
        try:
            # Extract features
            call_data = {col: row[col] for col in call_cols if col in row}
            agent_key = str(row['RESOURCE_KEY'])

            # Get agent features
            if len(gcs_df) > 0 and agent_key in gcs_df.index:
                agent_data = gcs_df.loc[agent_key].to_dict()
            else:
                agent_data = {}

            # Get topic for residual adjustment
            topic_value = row.get('call_TOPIC_CLASSIFIC_ENTRY_AT_FT', None)

            # Predict with simulator
            pred_tmc, pred_ftr, pred_ot = predict_with_simulator(
                call_data, agent_data, models, scalers, feature_lists,
                residual_adjuster, agent_key, topic_value, apply_residuals
            )

            # Get actual values
            actual_tmc = float(row[tmc_target_col])
            actual_ftr = float(row[ftr_target_col])
            actual_ot = float(row[ot_target_col])

            # Calculate costs
            pred_cost = calculate_cost(pred_tmc, pred_ftr, pred_ot)
            actual_cost = calculate_cost(actual_tmc, actual_ftr, actual_ot)

            results.append({
                'actual_tmc': actual_tmc,
                'pred_tmc': pred_tmc,
                'actual_ftr': actual_ftr,
                'pred_ftr': pred_ftr,
                'actual_ot': actual_ot,
                'pred_ot': pred_ot,
                'actual_cost': actual_cost,
                'pred_cost': pred_cost,
                'cost_error': abs(pred_cost - actual_cost),
                'agent_key': agent_key
            })

        except Exception as e:
            # Skip rows with errors
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("❌ No valid predictions generated. Check data and model compatibility.")
        return

    print(f"\n✓ Successfully validated {len(results_df):,} predictions")

    # Calculate metrics
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)

    # TMC metrics
    mae_tmc = mean_absolute_error(results_df['actual_tmc'], results_df['pred_tmc'])
    rmse_tmc = np.sqrt(mean_squared_error(results_df['actual_tmc'], results_df['pred_tmc']))
    r2_tmc = r2_score(results_df['actual_tmc'], results_df['pred_tmc'])
    corr_tmc = np.corrcoef(results_df['actual_tmc'], results_df['pred_tmc'])[0, 1]

    print(f"\nTMC (Call Duration) Predictions:")
    print(f"  MAE:         {mae_tmc:.2f} seconds")
    print(f"  RMSE:        {rmse_tmc:.2f} seconds")
    print(f"  R²:          {r2_tmc:.4f}")
    print(f"  Correlation: {corr_tmc:.4f}")

    # FTR metrics
    mae_ftr = mean_absolute_error(results_df['actual_ftr'], results_df['pred_ftr'])
    rmse_ftr = np.sqrt(mean_squared_error(results_df['actual_ftr'], results_df['pred_ftr']))
    corr_ftr = np.corrcoef(results_df['actual_ftr'], results_df['pred_ftr'])[0, 1]

    print(f"\nFTR (First Time Resolution) Predictions:")
    print(f"  MAE:         {mae_ftr:.4f}")
    print(f"  RMSE:        {rmse_ftr:.4f}")
    print(f"  Correlation: {corr_ftr:.4f}")

    # OT metrics
    mae_ot = mean_absolute_error(results_df['actual_ot'], results_df['pred_ot'])
    rmse_ot = np.sqrt(mean_squared_error(results_df['actual_ot'], results_df['pred_ot']))
    corr_ot = np.corrcoef(results_df['actual_ot'], results_df['pred_ot'])[0, 1]

    print(f"\nOT (Onsite Technician) Predictions:")
    print(f"  MAE:         {mae_ot:.4f}")
    print(f"  RMSE:        {rmse_ot:.4f}")
    print(f"  Correlation: {corr_ot:.4f}")

    # Cost metrics (MOST IMPORTANT)
    mae_cost = mean_absolute_error(results_df['actual_cost'], results_df['pred_cost'])
    rmse_cost = np.sqrt(mean_squared_error(results_df['actual_cost'], results_df['pred_cost']))
    r2_cost = r2_score(results_df['actual_cost'], results_df['pred_cost'])
    corr_cost = np.corrcoef(results_df['actual_cost'], results_df['pred_cost'])[0, 1]

    print(f"\n" + "="*60)
    print("COST PREDICTION (Primary Validation Metric)")
    print("="*60)
    print(f"  MAE:         €{mae_cost:.2f}")
    print(f"  RMSE:        €{rmse_cost:.2f}")
    print(f"  R²:          {r2_cost:.4f}")
    print(f"  Correlation: {corr_cost:.4f}")

    # Summary statistics
    print(f"\nCost Summary:")
    print(f"  Mean Actual Cost:    €{results_df['actual_cost'].mean():.2f}")
    print(f"  Mean Predicted Cost: €{results_df['pred_cost'].mean():.2f}")
    print(f"  Mean Error:          €{results_df['cost_error'].mean():.2f}")
    print(f"  Median Error:        €{results_df['cost_error'].median():.2f}")

    # Acceptance criteria check
    print("\n" + "="*60)
    print("ACCEPTANCE CRITERIA")
    print("="*60)

    criteria_met = []

    if mae_cost < 3.0:
        print(f"✓ MAE < €3.00: PASS (€{mae_cost:.2f})")
        criteria_met.append(True)
    else:
        print(f"✗ MAE < €3.00: FAIL (€{mae_cost:.2f})")
        criteria_met.append(False)

    if corr_cost > 0.70:
        print(f"✓ Correlation > 0.70: PASS ({corr_cost:.4f})")
        criteria_met.append(True)
    else:
        print(f"✗ Correlation > 0.70: FAIL ({corr_cost:.4f})")
        criteria_met.append(False)

    if all(criteria_met):
        print("\n🎉 VALIDATION PASSED: Simulator is sufficiently accurate for RL training")
    else:
        print("\n⚠️  VALIDATION CONCERNS: Simulator may have accuracy issues")

    # Save results
    print(f"\nSaving detailed results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved")

    # Save summary metrics
    summary_path = output_path.replace('.csv', '_summary.csv')
    summary_df = pd.DataFrame([{
        'metric': 'MAE_TMC_seconds',
        'value': mae_tmc
    }, {
        'metric': 'RMSE_TMC_seconds',
        'value': rmse_tmc
    }, {
        'metric': 'R2_TMC',
        'value': r2_tmc
    }, {
        'metric': 'Correlation_TMC',
        'value': corr_tmc
    }, {
        'metric': 'MAE_FTR',
        'value': mae_ftr
    }, {
        'metric': 'RMSE_FTR',
        'value': rmse_ftr
    }, {
        'metric': 'Correlation_FTR',
        'value': corr_ftr
    }, {
        'metric': 'MAE_OT',
        'value': mae_ot
    }, {
        'metric': 'RMSE_OT',
        'value': rmse_ot
    }, {
        'metric': 'Correlation_OT',
        'value': corr_ot
    }, {
        'metric': 'MAE_Cost_EUR',
        'value': mae_cost
    }, {
        'metric': 'RMSE_Cost_EUR',
        'value': rmse_cost
    }, {
        'metric': 'R2_Cost',
        'value': r2_cost
    }, {
        'metric': 'Correlation_Cost',
        'value': corr_cost
    }, {
        'metric': 'Mean_Actual_Cost_EUR',
        'value': results_df['actual_cost'].mean()
    }, {
        'metric': 'Mean_Predicted_Cost_EUR',
        'value': results_df['pred_cost'].mean()
    }, {
        'metric': 'Validation_Passed',
        'value': int(all(criteria_met))
    }])

    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary metrics saved to {summary_path}")

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found at {args.data_path}")
        print("Please provide a valid data path with --data-path")
        sys.exit(1)

    validate_simulator(args.data_path, args.sample_size, args.output_path, apply_residuals=not args.no_residuals)


if __name__ == '__main__':
    main()
