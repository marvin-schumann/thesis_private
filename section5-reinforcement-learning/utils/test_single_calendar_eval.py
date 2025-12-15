#!/usr/bin/env python3
"""Test calendar day evaluation for a single day"""
import json
from call_center_env import CallCenterEnv
from baseline_policies import BaselinePolicies

DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'

print("Initializing baseline policies...")
baselines = BaselinePolicies(
    data_path=DATA_PATH,
    test_indices_path='models/test_indices.npy',
    assets_dir='models'
)

date = '2024-01-17'
print(f"\nTesting calendar day: {date}")

# Initialize environment
env = CallCenterEnv(
    data_path=DATA_PATH,
    assets_dir='models',
    test_indices_path='models/test_indices.npy',
    episode_mode='calendar_day',
    calendar_date=date
)

# Run Random policy
print("  Running Random policy...")
obs, info = env.reset(seed=42)
done = False
calls = 0
while not done:
    action = baselines.random_policy(obs)
    obs, reward, done, truncated, info = env.step(action)
    if info.get('status') in ['success', 'success_calendar_day_complete']:
        calls += 1
print(f"    ✓ Random: {calls} calls processed")

# Run Greedy XGBoost policy
print("  Running Greedy XGBoost policy...")
obs, info = env.reset(seed=42)
done = False
calls = 0
while not done:
    action = baselines.greedy_xgboost_policy(obs)
    obs, reward, done, truncated, info = env.step(action)
    if info.get('status') in ['success', 'success_calendar_day_complete']:
        calls += 1
print(f"    ✓ Greedy XGBoost: {calls} calls processed")

print("\n✓ Test complete")
