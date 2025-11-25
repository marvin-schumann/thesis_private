#!/usr/bin/env python3
"""
Debug script to verify action mask dimensions are correct.
"""

import numpy as np
from call_center_env_masked import CallCenterEnvMasked
from sb3_contrib.common.wrappers import ActionMasker

DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'

def mask_fn(env):
    """Function that returns action mask for the current environment state."""
    return env.action_masks()

print("=" * 60)
print("DEBUGGING ACTION MASK DIMENSIONS")
print("=" * 60)

# Test base environment
print("\n1. Testing CallCenterEnvMasked (base)...")
base_env = CallCenterEnvMasked(data_path=DATA_PATH, assets_dir=ASSETS_DIR)

print(f"   Observation space: {base_env.observation_space.shape}")
print(f"   Action space: {base_env.action_space.n}")
print(f"   Number of agents: {base_env.num_agents}")

# Reset and get initial state
obs, info = base_env.reset(seed=42)
print(f"   Observation shape: {obs.shape}")

# Get action mask
mask = base_env.action_masks()
print(f"   Action mask shape: {mask.shape}")
print(f"   Action mask dtype: {mask.dtype}")
print(f"   Action mask sum: {np.sum(mask)}")
print(f"   Action mask min/max: {np.min(mask)}, {np.max(mask)}")

# Test wrapped environment
print("\n2. Testing ActionMasker wrapper...")
env = ActionMasker(base_env, mask_fn)

print(f"   Wrapped observation space: {env.observation_space.shape}")
print(f"   Wrapped action space: {env.action_space.n}")

# Reset wrapped environment
obs, info = env.reset(seed=42)
print(f"   Wrapped observation shape: {obs.shape}")

# Check if mask is in info
if 'action_mask' in info:
    print(f"   Info contains action_mask: {info['action_mask'].shape}")
else:
    print(f"   WARNING: Info does not contain action_mask")

# Test a few steps
print("\n3. Testing 5 steps...")
for i in range(5):
    mask = base_env.action_masks()
    valid_actions = np.where(mask == 1)[0]

    if len(valid_actions) == 0:
        print(f"   Step {i+1}: ERROR - No valid actions!")
        break

    action = np.random.choice(valid_actions)
    obs, reward, done, truncated, info = env.step(action)

    print(f"   Step {i+1}: obs.shape={obs.shape}, mask.shape={mask.shape}, "
          f"valid_actions={len(valid_actions)}, done={done}")

    if done or truncated:
        break

# Test the "all agents busy" case
print("\n4. Testing edge case: all agents unavailable...")
# Manually set all agents as busy
for agent_key in base_env.agent_keys:
    base_env.agent_available_at[agent_key] = base_env.current_time + 10000.0

mask = base_env.action_masks()
print(f"   Action mask shape: {mask.shape}")
print(f"   Action mask sum (should be >0 due to safety check): {np.sum(mask)}")

if np.sum(mask) == 0:
    print("   ERROR: Action mask is all zeros! Safety check failed!")
else:
    print("   ✓ Safety check working - mask has valid actions")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
