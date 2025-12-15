#!/usr/bin/env python3
"""
Quick test script to validate calendar day mode functionality.
Tests both random mode (backwards compatibility) and calendar day mode.
"""

import sys
import logging
from call_center_env import CallCenterEnv

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'
TEST_INDICES = 'models/test_indices.npy'

print("="*70)
print("TESTING CALL CENTER ENVIRONMENT - DUAL MODE")
print("="*70)

# Test 1: Random mode (backwards compatibility)
print("\n[TEST 1] Random Mode (Backwards Compatibility)")
print("-" * 70)
try:
    env_random = CallCenterEnv(
        data_path=DATA_PATH,
        test_indices_path=TEST_INDICES,
        episode_mode='random'
    )
    obs, info = env_random.reset(seed=42)
    print(f"✓ Random mode initialized")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env_random.action_space.n} agents")

    # Take a few steps
    for i in range(5):
        action = env_random.action_space.sample()
        obs, reward, done, truncated, info = env_random.step(action)
        print(f"  Step {i+1}: reward={reward:.2f}, status={info['status']}")
        if done:
            break

    print("✓ Random mode test PASSED")
except Exception as e:
    print(f"✗ Random mode test FAILED: {e}")
    sys.exit(1)

# Test 2: Calendar day mode
print("\n[TEST 2] Calendar Day Mode")
print("-" * 70)
try:
    test_date = "2024-01-17"  # First date from selected_calendar_days.json

    env_calendar = CallCenterEnv(
        data_path=DATA_PATH,
        test_indices_path=TEST_INDICES,
        episode_mode='calendar_day',
        calendar_date=test_date
    )
    print(f"✓ Calendar day mode initialized for {test_date}")

    obs, info = env_calendar.reset(seed=42)
    print(f"✓ Episode reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Max steps: {env_calendar.max_steps_per_episode}")

    # Run through a few calls
    step_count = 0
    done = False
    while not done and step_count < 10:
        action = env_calendar.action_space.sample()
        obs, reward, done, truncated, info = env_calendar.step(action)
        step_count += 1
        print(f"  Step {step_count}: reward={reward:.2f}, status={info['status']}")

        if done:
            print(f"✓ Episode completed: {info['status']}")
            break

    print(f"✓ Calendar day mode test PASSED")
except Exception as e:
    print(f"✗ Calendar day mode test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Validation errors
print("\n[TEST 3] Validation Tests")
print("-" * 70)

# Test invalid mode
try:
    env_invalid = CallCenterEnv(
        data_path=DATA_PATH,
        test_indices_path=TEST_INDICES,
        episode_mode='invalid_mode'
    )
    print("✗ Should have raised ValueError for invalid mode")
    sys.exit(1)
except ValueError as e:
    print(f"✓ Correctly rejected invalid mode: {e}")

# Test calendar_day mode without date
try:
    env_no_date = CallCenterEnv(
        data_path=DATA_PATH,
        test_indices_path=TEST_INDICES,
        episode_mode='calendar_day',
        calendar_date=None
    )
    print("✗ Should have raised ValueError for missing calendar_date")
    sys.exit(1)
except ValueError as e:
    print(f"✓ Correctly rejected missing calendar_date: {e}")

print("\n" + "="*70)
print("✓ ALL TESTS PASSED")
print("="*70)
print("\nCalendar day mode is ready for evaluation!")
