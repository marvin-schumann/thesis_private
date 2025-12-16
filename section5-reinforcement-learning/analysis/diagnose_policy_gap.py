#!/usr/bin/env python3
"""
Diagnose Why Greedy XGBoost and Rule-Based Are So Close

Investigates potential sources of error:
1. Are XGBoost models getting the same features as training?
2. Are the models making meaningfully different predictions?
3. Is the rule-based policy accidentally mimicking XGBoost?
4. Feature importance analysis
"""

import numpy as np
import pandas as pd
import joblib
import json
from baseline_policies import BaselinePolicies
from call_center_env import CallCenterEnv

# Configuration - Update DATA_PATH to your local data location
DATA_PATH = os.environ.get('NOS_DATA_PATH', 'data/full_merged_df.csv')
TEST_INDICES_PATH = 'models/test_indices.npy'

def check_feature_consistency():
    """Check if XGBoost models are using the same features as training."""
    print("="*80)
    print("FEATURE CONSISTENCY CHECK")
    print("="*80)

    # Load models and feature lists
    model_tmc = joblib.load('models/model_tmc.joblib')
    model_ftr = joblib.load('models/model_ftr.joblib')
    model_ot = joblib.load('models/model_ot.joblib')

    with open('models/feature_lists.json', 'r') as f:
        feature_lists = json.load(f)

    # Check if models have feature_names_in_ attribute (from sklearn)
    print("\nTMC Model:")
    print(f"  Expected features: {len(feature_lists['tmc'])}")
    if hasattr(model_tmc, 'feature_names_in_'):
        print(f"  Model trained on: {len(model_tmc.feature_names_in_)} features")
        print(f"  Match: {list(feature_lists['tmc']) == list(model_tmc.feature_names_in_)}")
    elif hasattr(model_tmc, 'feature_name'):
        print(f"  XGBoost feature count: {len(model_tmc.feature_name)}")
    else:
        print("  ⚠️  Cannot verify - model doesn't expose feature names")

    print("\nFTR Model:")
    print(f"  Expected features: {len(feature_lists['ftr'])}")
    if hasattr(model_ftr, 'feature_names_in_'):
        print(f"  Model trained on: {len(model_ftr.feature_names_in_)} features")
        print(f"  Match: {list(feature_lists['ftr']) == list(model_ftr.feature_names_in_)}")

    print("\nOT Model:")
    print(f"  Expected features: {len(feature_lists['ot'])}")
    if hasattr(model_ot, 'feature_names_in_'):
        print(f"  Model trained on: {len(model_ot.feature_names_in_)} features")
        print(f"  Match: {list(feature_lists['ot']) == list(model_ot.feature_names_in_)}")

    print()

def analyze_prediction_distributions(n_samples=100):
    """Compare prediction distributions between XGBoost and Rule-Based."""
    print("="*80)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*80)

    env = CallCenterEnv(data_path=DATA_PATH, test_indices_path=TEST_INDICES_PATH)
    policies = BaselinePolicies(data_path=DATA_PATH, test_indices_path=TEST_INDICES_PATH)

    xgboost_costs = []
    rule_based_costs = []
    xgboost_selections = []
    rule_based_selections = []

    print(f"\nAnalyzing {n_samples} random call scenarios...")

    for i in range(n_samples):
        obs, info = env.reset()
        call_data = policies._get_call_data_from_obs(obs)
        available_agents = policies._get_available_agents(obs)

        if len(available_agents) == 0:
            continue

        # Get XGBoost choice
        xgb_agent = policies.greedy_xgboost_policy(obs)
        xgb_agent_key = policies.agent_keys[xgb_agent]
        xgb_tmc, xgb_ftr, xgb_ot = policies._get_oracle_predictions(call_data, xgb_agent_key)
        xgb_cost = policies._calculate_cost(xgb_tmc, xgb_ftr, xgb_ot)

        # Get Rule-Based choice
        rb_agent = policies.rule_based_policy(obs)
        rb_agent_key = policies.agent_keys[rb_agent]
        rb_cost = policies._estimate_rule_based_cost(call_data, rb_agent_key)

        xgboost_costs.append(xgb_cost)
        rule_based_costs.append(rb_cost)
        xgboost_selections.append(xgb_agent)
        rule_based_selections.append(rb_agent)

    xgboost_costs = np.array(xgboost_costs)
    rule_based_costs = np.array(rule_based_costs)

    print(f"\nCost Predictions ({len(xgboost_costs)} samples):")
    print(f"  XGBoost:     mean={xgboost_costs.mean():.2f}, std={xgboost_costs.std():.2f}")
    print(f"  Rule-Based:  mean={rule_based_costs.mean():.2f}, std={rule_based_costs.std():.2f}")
    print(f"  Difference:  {abs(xgboost_costs.mean() - rule_based_costs.mean()):.2f} (€/call)")

    # Check how often they select the same agent
    same_agent = np.array(xgboost_selections) == np.array(rule_based_selections)
    agreement_pct = same_agent.mean() * 100
    print(f"\nAgent Selection Agreement: {agreement_pct:.1f}%")
    print(f"  Same agent: {same_agent.sum()}/{len(same_agent)} cases")

    return xgboost_costs, rule_based_costs, agreement_pct

def analyze_component_predictions(n_samples=50):
    """Analyze individual components (TMC, FTR, OT) predictions."""
    print("="*80)
    print("COMPONENT PREDICTION ANALYSIS")
    print("="*80)

    env = CallCenterEnv(data_path=DATA_PATH, test_indices_path=TEST_INDICES_PATH)
    policies = BaselinePolicies(data_path=DATA_PATH, test_indices_path=TEST_INDICES_PATH)

    xgb_tmcs = []
    xgb_ftrs = []
    xgb_ots = []
    rb_tmcs = []
    rb_ftrs = []
    rb_ots = []

    print(f"\nAnalyzing {n_samples} predictions...")

    for i in range(n_samples):
        obs, info = env.reset()
        call_data = policies._get_call_data_from_obs(obs)
        available_agents = policies._get_available_agents(obs)

        if len(available_agents) == 0:
            continue

        # Pick a random available agent
        agent_idx = np.random.choice(available_agents)
        agent_key = policies.agent_keys[agent_idx]

        # XGBoost predictions
        xgb_tmc, xgb_ftr, xgb_ot = policies._get_oracle_predictions(call_data, agent_key)
        xgb_tmcs.append(xgb_tmc)
        xgb_ftrs.append(xgb_ftr)
        xgb_ots.append(xgb_ot)

        # Rule-Based estimates
        topic_suffix = policies._determine_topic_suffix(call_data)
        topic_key = f"topic_{topic_suffix}" if topic_suffix else None
        topic_avg = policies.rule_based_averages.get(topic_key, policies.default_rule_average)
        agent_features = policies._get_agent_features(agent_key)

        # Extract rule-based values (simplified version of _estimate_rule_based_cost)
        rb_tmc = topic_avg.get('tmc', policies.default_rule_average['tmc'])
        rb_ftr = topic_avg.get('ftr_prob', policies.default_rule_average['ftr_prob'])
        rb_ot = topic_avg.get('ot_prob', policies.default_rule_average['ot_prob'])

        rb_tmcs.append(rb_tmc)
        rb_ftrs.append(rb_ftr)
        rb_ots.append(rb_ot)

    print(f"\nTMC Predictions (seconds):")
    print(f"  XGBoost:     {np.mean(xgb_tmcs):.1f} ± {np.std(xgb_tmcs):.1f}")
    print(f"  Rule-Based:  {np.mean(rb_tmcs):.1f} ± {np.std(rb_tmcs):.1f}")
    print(f"  Correlation: {np.corrcoef(xgb_tmcs, rb_tmcs)[0,1]:.3f}")

    print(f"\nFTR Probability:")
    print(f"  XGBoost:     {np.mean(xgb_ftrs):.3f} ± {np.std(xgb_ftrs):.3f}")
    print(f"  Rule-Based:  {np.mean(rb_ftrs):.3f} ± {np.std(rb_ftrs):.3f}")
    print(f"  Correlation: {np.corrcoef(xgb_ftrs, rb_ftrs)[0,1]:.3f}")

    print(f"\nOT Probability:")
    print(f"  XGBoost:     {np.mean(xgb_ots):.3f} ± {np.std(xgb_ots):.3f}")
    print(f"  Rule-Based:  {np.mean(rb_ots):.3f} ± {np.std(rb_ots):.3f}")
    print(f"  Correlation: {np.corrcoef(xgb_ots, rb_ots)[0,1]:.3f}")

    print()

def check_xgboost_r2():
    """Check the R² performance of XGBoost models on test set."""
    print("="*80)
    print("XGBOOST MODEL PERFORMANCE (R²)")
    print("="*80)

    from sklearn.metrics import r2_score, mean_absolute_error

    # Load test data
    full_data = pd.read_csv(DATA_PATH)
    test_indices = np.load(TEST_INDICES_PATH)
    test_data = full_data.loc[full_data.index.isin(test_indices)]

    # Load models and features
    model_tmc = joblib.load('models/model_tmc.joblib')
    model_ftr = joblib.load('models/model_ftr.joblib')
    model_ot = joblib.load('models/model_ot.joblib')

    scaler_tmc = joblib.load('models/scaler_tmc.joblib')
    scaler_ftr = joblib.load('models/scaler_ftr.joblib')
    scaler_ot = joblib.load('models/scaler_ot.joblib')

    with open('models/feature_lists.json', 'r') as f:
        feature_lists = json.load(f)

    # Prepare features
    X_test_tmc = test_data[feature_lists['tmc']].fillna(0)
    X_test_ftr = test_data[feature_lists['ftr']].fillna(0)
    X_test_ot = test_data[feature_lists['ot']].fillna(0)

    # Scale
    scaled_cols_tmc = list(getattr(scaler_tmc, 'feature_names_in_', []))
    scaled_cols_ftr = list(getattr(scaler_ftr, 'feature_names_in_', []))
    scaled_cols_ot = list(getattr(scaler_ot, 'feature_names_in_', []))

    if scaled_cols_tmc:
        X_test_tmc[scaled_cols_tmc] = scaler_tmc.transform(X_test_tmc[scaled_cols_tmc])
    if scaled_cols_ftr:
        X_test_ftr[scaled_cols_ftr] = scaler_ftr.transform(X_test_ftr[scaled_cols_ftr])
    if scaled_cols_ot:
        X_test_ot[scaled_cols_ot] = scaler_ot.transform(X_test_ot[scaled_cols_ot])

    # Get targets
    y_test_tmc = test_data['call_LEG_DURATION_SEC_QTY']
    y_test_ftr = test_data['call_FTR_1_SUM']  # or appropriate FTR column
    y_test_ot = test_data['call_FLAG_OT']

    # Predict
    pred_tmc = model_tmc.predict(X_test_tmc)
    pred_ftr = model_ftr.predict_proba(X_test_ftr)[:, 1]
    pred_ot = model_ot.predict_proba(X_test_ot)[:, 1] if model_ot.predict_proba(X_test_ot).shape[1] == 2 else model_ot.predict_proba(X_test_ot)[:, 0]

    # Calculate R²
    r2_tmc = r2_score(y_test_tmc, pred_tmc)
    mae_tmc = mean_absolute_error(y_test_tmc, pred_tmc)

    print(f"\nTMC Model (Regression):")
    print(f"  R² Score: {r2_tmc:.4f}")
    print(f"  MAE: {mae_tmc:.2f} seconds")
    print(f"  Mean actual: {y_test_tmc.mean():.1f} seconds")
    print(f"  Mean predicted: {pred_tmc.mean():.1f} seconds")

    # For classification models, use different metrics
    from sklearn.metrics import roc_auc_score

    try:
        auc_ftr = roc_auc_score(y_test_ftr, pred_ftr)
        print(f"\nFTR Model (Classification):")
        print(f"  ROC AUC: {auc_ftr:.4f}")
        print(f"  Mean actual: {y_test_ftr.mean():.3f}")
        print(f"  Mean predicted: {pred_ftr.mean():.3f}")
    except:
        print(f"\nFTR Model: Unable to calculate AUC")

    try:
        auc_ot = roc_auc_score(y_test_ot, pred_ot)
        print(f"\nOT Model (Classification):")
        print(f"  ROC AUC: {auc_ot:.4f}")
        print(f"  Mean actual: {y_test_ot.mean():.3f}")
        print(f"  Mean predicted: {pred_ot.mean():.3f}")
    except:
        print(f"\nOT Model: Unable to calculate AUC")

    print()
    return r2_tmc

if __name__ == '__main__':
    print("\n" + "="*80)
    print("DIAGNOSING GREEDY XGBOOST vs RULE-BASED PERFORMANCE GAP")
    print("="*80 + "\n")

    # 1. Feature consistency
    check_feature_consistency()

    # 2. XGBoost performance
    r2_tmc = check_xgboost_r2()

    # 3. Component predictions
    analyze_component_predictions(n_samples=100)

    # 4. Overall predictions
    xgb_costs, rb_costs, agreement = analyze_prediction_distributions(n_samples=200)

    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    print(f"\n1. XGBoost TMC R²: {r2_tmc:.4f}")
    print(f"   - Low R² suggests XGBoost has limited predictive power")
    print(f"   - This explains why it's close to simple rule-based approach")

    print(f"\n2. Agent Selection Agreement: {agreement:.1f}%")
    if agreement > 80:
        print("   - ⚠️  Models select the same agent >80% of the time")
        print("   - This explains the minimal performance difference")

    print(f"\n3. Cost Difference: {abs(xgb_costs.mean() - rb_costs.mean()):.2f} €/call")

    print("\n" + "="*80)
    print()
