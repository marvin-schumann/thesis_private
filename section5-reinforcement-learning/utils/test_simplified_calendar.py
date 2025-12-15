#!/usr/bin/env python3
"""
Quick test to verify simplified calendar day mode processes all calls.
"""

import sys
from call_center_env import CallCenterEnv
from baseline_policies import BaselinePolicies

DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'

print("=" * 70)
print("TESTING SIMPLIFIED CALENDAR DAY MODE")
print("=" * 70)
print()

# Initialize environment for calendar day 2024-01-17 (124 calls expected)
print("Initializing environment for 2024-01-17 (expected: 124 calls)...")
env = CallCenterEnv(
    data_path=DATA_PATH,
    assets_dir='models',
    test_indices_path='models/test_indices.npy',
    episode_mode='calendar_day',
    calendar_date='2024-01-17'
)

# Initialize baseline policies
print("Initializing baseline policies...")
baselines = BaselinePolicies(
    data_path=DATA_PATH,
    test_indices_path='models/test_indices.npy',
    assets_dir='models'
)

# Test with Greedy XGBoost policy
print("\nRunning Greedy XGBoost on calendar day 2024-01-17...")
obs, info = env.reset(seed=42)
done = False
calls_processed = 0
total_cost = 0
steps = 0

while not done:
    action = baselines.greedy_xgboost_policy(obs)
    obs, reward, done, truncated, info = env.step(action)
    steps += 1

    if info.get('status') in ['success', 'success_calendar_day_complete']:
        calls_processed += 1
        total_cost += info.get('cost', 0)

    if truncated:
        print(f"  ⚠ Episode truncated at step {steps}")
        break

print(f"\n✓ Episode complete:")
print(f"  Calls processed: {calls_processed}/124")
print(f"  Total steps: {steps}")
print(f"  Total cost: €{total_cost:.2f}")
print(f"  Cost per call: €{total_cost/calls_processed:.2f}")
print()

if calls_processed == 124:
    print("=" * 70)
    print("✓ SUCCESS: All 124 calls processed!")
    print("=" * 70)
    sys.exit(0)
else:
    print("=" * 70)
    print(f"✗ FAILED: Only {calls_processed}/124 calls processed")
    print("=" * 70)
    sys.exit(1)
