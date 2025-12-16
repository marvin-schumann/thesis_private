#!/usr/bin/env python3
"""
Diagnose Small Sample Bias in Rule-Based Policy

Checks if low-sample agents are being selected and performing unrealistically well.
"""

import numpy as np
import pandas as pd
from baseline_policies import BaselinePolicies
from call_center_env import CallCenterEnv

# Configuration - Update DATA_PATH to your local data location
DATA_PATH = os.environ.get('NOS_DATA_PATH', 'data/full_merged_df.csv')
TEST_INDICES_PATH = 'models/test_indices.npy'

def main():
    print("="*80)
    print("SMALL SAMPLE BIAS DIAGNOSTIC")
    print("="*80)
    print()

    # Load data
    full_data = pd.read_csv(DATA_PATH)
    test_indices = np.load(TEST_INDICES_PATH)
    test_data = full_data.loc[full_data.index.isin(test_indices)]

    # Analyze agent distribution
    print("1. AGENT CALL DISTRIBUTION IN TEST SET")
    print("-" * 80)

    agent_col = 'RESOURCE_KEY'
    if agent_col not in test_data.columns:
        print(f"Error: {agent_col} not found in data")
        return

    # Convert to string to match agent_keys format in BaselinePolicies
    test_data[agent_col] = test_data[agent_col].astype(str)
    agent_counts = test_data[agent_col].value_counts()

    print(f"Total agents: {len(agent_counts)}")
    print(f"Total calls: {len(test_data)}")
    print(f"Mean calls/agent: {agent_counts.mean():.1f}")
    print(f"Median calls/agent: {agent_counts.median():.1f}")
    print(f"Min calls/agent: {agent_counts.min()}")
    print(f"Max calls/agent: {agent_counts.max()}")
    print()

    # Check low-sample agents
    print("Agents by sample size:")
    print(f"  < 10 calls: {(agent_counts < 10).sum()} agents ({(agent_counts < 10).sum() / len(agent_counts) * 100:.1f}%)")
    print(f"  < 30 calls: {(agent_counts < 30).sum()} agents ({(agent_counts < 30).sum() / len(agent_counts) * 100:.1f}%)")
    print(f"  < 100 calls: {(agent_counts < 100).sum()} agents ({(agent_counts < 100).sum() / len(agent_counts) * 100:.1f}%)")
    print(f"  ≥ 100 calls: {(agent_counts >= 100).sum()} agents ({(agent_counts >= 100).sum() / len(agent_counts) * 100:.1f}%)")
    print()

    # Check if rule-based thresholds protect against small samples
    print("2. RULE-BASED POLICY THRESHOLDS")
    print("-" * 80)

    policies = BaselinePolicies(DATA_PATH, 'models', TEST_INDICES_PATH)

    print(f"Minimum calls for topic-specific stats: {policies.rule_min_calls_topic}")
    print(f"Minimum calls for overall stats: {policies.rule_min_calls_overall}")
    print()

    # Simulate some calls and track agent selections
    print("3. AGENT SELECTION ANALYSIS (n=200 calls)")
    print("-" * 80)

    env = CallCenterEnv(data_path=DATA_PATH, test_indices_path=TEST_INDICES_PATH)

    rule_based_selections = []
    xgboost_selections = []
    agent_sample_sizes = []

    for i in range(200):
        obs, info = env.reset()
        available_agents = policies._get_available_agents(obs)

        if len(available_agents) == 0:
            continue

        # Get rule-based selection
        rb_agent = policies.rule_based_policy(obs)
        rb_agent_key = policies.agent_keys[rb_agent]

        # Get XGBoost selection
        xgb_agent = policies.greedy_xgboost_policy(obs)
        xgb_agent_key = policies.agent_keys[xgb_agent]

        # Get sample size for selected agents
        rb_sample_size = agent_counts.get(rb_agent_key, 0)
        xgb_sample_size = agent_counts.get(xgb_agent_key, 0)

        rule_based_selections.append(rb_agent_key)
        xgboost_selections.append(xgb_agent_key)
        agent_sample_sizes.append({
            'rb_agent': rb_agent_key,
            'rb_sample_size': rb_sample_size,
            'xgb_agent': xgb_agent_key,
            'xgb_sample_size': xgb_sample_size,
            'same_agent': rb_agent_key == xgb_agent_key
        })

    df = pd.DataFrame(agent_sample_sizes)

    print(f"Calls analyzed: {len(df)}")
    print()

    print("Rule-Based Agent Selection:")
    print(f"  Mean sample size: {df['rb_sample_size'].mean():.1f} calls")
    print(f"  Median sample size: {df['rb_sample_size'].median():.1f} calls")
    print(f"  Min sample size: {df['rb_sample_size'].min()} calls")
    print(f"  Selected agents with < 30 calls: {(df['rb_sample_size'] < 30).sum()} ({(df['rb_sample_size'] < 30).sum() / len(df) * 100:.1f}%)")
    print(f"  Selected agents with < 100 calls: {(df['rb_sample_size'] < 100).sum()} ({(df['rb_sample_size'] < 100).sum() / len(df) * 100:.1f}%)")
    print()

    print("XGBoost Agent Selection:")
    print(f"  Mean sample size: {df['xgb_sample_size'].mean():.1f} calls")
    print(f"  Median sample size: {df['xgb_sample_size'].median():.1f} calls")
    print(f"  Min sample size: {df['xgb_sample_size'].min()} calls")
    print(f"  Selected agents with < 30 calls: {(df['xgb_sample_size'] < 30).sum()} ({(df['xgb_sample_size'] < 30).sum() / len(df) * 100:.1f}%)")
    print(f"  Selected agents with < 100 calls: {(df['xgb_sample_size'] < 100).sum()} ({(df['xgb_sample_size'] < 100).sum() / len(df) * 100:.1f}%)")
    print()

    print("Agreement:")
    print(f"  Same agent selected: {df['same_agent'].sum()} / {len(df)} ({df['same_agent'].mean() * 100:.1f}%)")
    print()

    # Check if low-sample agents have suspiciously good performance
    print("4. LOW-SAMPLE AGENT PERFORMANCE")
    print("-" * 80)

    # Get agents with < 30 calls
    low_sample_agents = agent_counts[agent_counts < 30].index.tolist()
    high_sample_agents = agent_counts[agent_counts >= 100].index.tolist()

    if len(low_sample_agents) > 0:
        # Calculate mean TMC for each group
        low_sample_data = test_data[test_data[agent_col].isin(low_sample_agents)]
        high_sample_data = test_data[test_data[agent_col].isin(high_sample_agents)]

        tmc_col = 'call_LEG_DURATION_SEC_QTY'
        if tmc_col in test_data.columns:
            print(f"Average TMC by agent sample size:")
            print(f"  Agents with < 30 calls: {low_sample_data[tmc_col].mean():.1f} seconds (n={len(low_sample_agents)} agents)")
            print(f"  Agents with ≥ 100 calls: {high_sample_data[tmc_col].mean():.1f} seconds (n={len(high_sample_agents)} agents)")
            print(f"  Difference: {abs(low_sample_data[tmc_col].mean() - high_sample_data[tmc_col].mean()):.1f} seconds")
            print()

    print("5. DIAGNOSIS SUMMARY")
    print("-" * 80)

    # Check for issues
    issues = []

    if (df['rb_sample_size'] < 30).sum() > 0:
        issues.append(f"⚠️  Rule-Based selects agents with < 30 calls in {(df['rb_sample_size'] < 30).sum()} cases")

    if df['rb_sample_size'].min() < 10:
        issues.append(f"⚠️  Rule-Based selects agents with < 10 calls (min: {df['rb_sample_size'].min()})")

    if df['rb_sample_size'].mean() < df['xgb_sample_size'].mean() - 50:
        issues.append(f"⚠️  Rule-Based systematically selects lower-sample agents than XGBoost")

    if len(issues) > 0:
        print("POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        print()
        print("RECOMMENDATION:")
        print("  Consider increasing rule_min_calls_overall from 30 to 100")
        print("  This would prevent selection of low-sample agents")
    else:
        print("✓ No major small-sample bias detected")
        print("  Rule-Based policy appears to have adequate safeguards")

    print()
    print("="*80)

if __name__ == '__main__':
    main()
