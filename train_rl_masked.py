#!/usr/bin/env python3
"""
Train RL agents with ACTION MASKING to fix the invalid action problem.

This script uses MaskablePPO from sb3-contrib, which respects the action masks
and only selects from available agents.
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

# Action masking support
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_masked_policy

# Import our masked environment
from call_center_env_masked import CallCenterEnvMasked

# --- Configuration ---
DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'
os.makedirs(ASSETS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO with action masking for call routing."
    )
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps. Defaults to 500000 (25000 if --quick-smoke).")
    parser.add_argument("--quick-smoke", action="store_true",
                        help="Quick test run (~25k steps).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--checkpoint-freq", type=int, default=50000,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Computation device ('cpu', 'cuda', 'auto').")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging.")
    return parser.parse_args()


def mask_fn(env):
    """
    Function that returns action mask for the current environment state.

    This is required by ActionMasker wrapper.
    """
    return env.action_masks()


def main():
    args = parse_args()

    # Determine training budget
    if args.quick_smoke:
        timesteps = args.timesteps if args.timesteps is not None else 25_000
    else:
        timesteps = args.timesteps if args.timesteps is not None else 500_000

    set_random_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("TRAINING MASKABLE PPO WITH ACTION MASKING")
    print("=" * 60)
    print(f"Training steps: {timesteps:,}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {args.device}")
    print()

    # Initialize masked environment
    print("Initializing Action-Masked Call Center Environment...")
    base_env = CallCenterEnvMasked(data_path=DATA_PATH, assets_dir=ASSETS_DIR)

    # Wrap with ActionMasker (required for MaskablePPO)
    env = ActionMasker(base_env, mask_fn)

    env.reset(seed=args.seed)
    print(f"✓ Environment initialized: {base_env.num_agents} agents")

    # Setup TensorBoard
    tensorboard_log = "./rl_tensorboard_logs/maskable_ppo" if args.tensorboard else None
    if tensorboard_log:
        os.makedirs(tensorboard_log, exist_ok=True)

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(ASSETS_DIR, 'checkpoints_masked_ppo'),
        name_prefix='masked_ppo'
    )

    print("\n" + "=" * 60)
    print("TRAINING MASKABLE PPO")
    print("=" * 60)

    # Create MaskablePPO model
    # Discount factor γ = 0.99 (same justification as before)
    # With action masking, the agent can ONLY select valid actions,
    # so it will handle all calls properly
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=args.device,
        # gamma=0.99,  # Using default
    )

    print("Model configuration:")
    print(f"  Policy: MlpPolicy")
    print(f"  Action masking: ENABLED")
    print(f"  Gamma: 0.99")
    print(f"  Device: {args.device}")
    print()

    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    elapsed = (time.time() - start_time) / 60.0

    # Save final model
    save_path = os.path.join(ASSETS_DIR, 'rl_model_masked_ppo')
    model.save(save_path)

    print(f"\n✓ Training complete in {elapsed:.2f} minutes")
    print(f"✓ Model saved to: {save_path}.zip")

    # Quick evaluation
    print("\n" + "=" * 60)
    print("QUICK EVALUATION (1 episode)")
    print("=" * 60)

    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    calls_handled = 0

    while not done and step < 10000:
        action, _states = model.predict(obs, deterministic=True, action_masks=env.env.action_masks())
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if 'status' in info and info['status'] == 'success':
            calls_handled += 1

    print(f"Episode stats:")
    print(f"  Steps: {step}")
    print(f"  Calls handled: {calls_handled}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final time: {env.env.current_time:.0f}s / {env.env.simulation_day_length:.0f}s")

    expected_calls = 590
    efficiency = calls_handled / expected_calls * 100
    print(f"\nEfficiency: {efficiency:.1f}% ({calls_handled}/{expected_calls} expected calls)")

    if efficiency > 90:
        print("\n🎉 SUCCESS: Agent is handling >90% of calls!")
    elif efficiency > 50:
        print("\n⚠️  PARTIAL: Agent is handling >50% but not all calls")
    else:
        print("\n❌ ISSUE: Agent still not handling enough calls")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("To evaluate this masked policy:")
    print("  python evaluate_policies_masked.py --episodes 10 --tag 'masked_ppo'")


if __name__ == '__main__':
    main()
