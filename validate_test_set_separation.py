#!/usr/bin/env python3
"""
Validate train/test set separation.

Verifies:
1. Train/test indices exist and have correct properties
2. Zero overlap between train and test sets
3. Environment correctly filters to test set
4. Episodes only sample from test data
"""

import os
import numpy as np
import pandas as pd
from call_center_env import CallCenterEnv

# Paths
ASSETS_DIR = 'models'
DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'

def main():
    print("=" * 60)
    print("TRAIN/TEST SEPARATION VALIDATION")
    print("=" * 60)

    # Check 1: Files exist
    print("\n1. Checking index files exist...")
    train_path = os.path.join(ASSETS_DIR, 'train_indices.npy')
    test_path = os.path.join(ASSETS_DIR, 'test_indices.npy')

    if not os.path.exists(train_path):
        print(f"❌ FAILED: {train_path} not found")
        print("   Run modelling.py to generate indices")
        return False
    if not os.path.exists(test_path):
        print(f"❌ FAILED: {test_path} not found")
        print("   Run modelling.py to generate indices")
        return False
    print("✓ Both index files exist")

    # Check 2: Load indices
    print("\n2. Loading indices...")
    train_indices = np.load(train_path)
    test_indices = np.load(test_path)
    print(f"✓ Train set: {len(train_indices):,} samples")
    print(f"✓ Test set: {len(test_indices):,} samples")

    # Check 3: Verify split ratio
    print("\n3. Verifying split ratio...")
    total = len(train_indices) + len(test_indices)
    test_ratio = len(test_indices) / total
    print(f"✓ Split ratio: {test_ratio:.1%} test ({len(test_indices):,}/{total:,})")

    if abs(test_ratio - 0.20) > 0.01:
        print(f"⚠️  WARNING: Expected ~20% test split, got {test_ratio:.1%}")

    # Check 4: Verify zero overlap
    print("\n4. Verifying zero overlap...")
    overlap = np.intersect1d(train_indices, test_indices)
    print(f"✓ Overlap: {len(overlap)} samples")

    if len(overlap) > 0:
        print(f"❌ FAILED: Found {len(overlap)} overlapping indices!")
        print(f"   First few: {overlap[:10]}")
        return False
    print("✓ PASSED: 0% overlap between train and test")

    # Check 5: Verify environment filtering
    print("\n5. Verifying environment filtering...")
    env = CallCenterEnv(
        data_path=DATA_PATH,
        assets_dir=ASSETS_DIR,
        test_indices_path=test_path
    )

    print(f"✓ Environment loaded with {len(env.call_indices):,} calls")

    # Check 6: Verify all calls are from test set
    print("\n6. Verifying all environment calls are from test set...")
    env_indices_set = set(env.call_indices)
    test_indices_set = set(test_indices)

    not_in_test = env_indices_set - test_indices_set
    if len(not_in_test) > 0:
        print(f"❌ FAILED: Found {len(not_in_test)} calls NOT in test set")
        return False

    print(f"✓ PASSED: All {len(env.call_indices):,} calls are in test set")

    # Check 7: Sample episodes
    print("\n7. Running sample episodes...")
    for episode_num in range(3):
        obs, info = env.reset(seed=42 + episode_num)
        step = 0
        while step < 10:  # Just check first 10 calls
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            if done:
                break
        print(f"✓ Episode {episode_num + 1}: {step} steps completed")

    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Train indices: {len(train_indices):,} samples")
    print(f"✅ Test indices: {len(test_indices):,} samples")
    print(f"✅ Split ratio: {test_ratio:.1%} test")
    print(f"✅ Overlap: 0 samples (0.0%)")
    print(f"✅ Environment filtered correctly")
    print(f"✅ All calls in test set")
    print("\n🎉 PASSED: Train/test separation is working correctly!")
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
