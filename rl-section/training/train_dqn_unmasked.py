#!/usr/bin/env python3
"""
Train DQN WITHOUT action masking to demonstrate the invalid action problem.

This is Experiment 1: Unmasked DQN
Purpose: Show that the action masking problem is algorithm-independent
Expected: Agent will fail to handle calls due to invalid actions
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

# Import base environment WITHOUT action masking
from call_center_env import CallCenterEnv

# --- Configuration ---
DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'
os.makedirs(ASSETS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DQN WITHOUT action masking (Experiment 1)."
    )
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps (default: 50000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--checkpoint-freq", type=int, default=10000,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Computation device ('cpu', 'cuda', 'auto').")
    return parser.parse_args()


def main():
    args = parse_args()

    set_random_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("EXPERIMENT 1: TRAINING UNMASKED DQN")
    print("=" * 60)
    print("Purpose: Demonstrate action masking problem is algorithm-independent")
    print(f"Training steps: {args.timesteps:,}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Action masking: DISABLED (expecting invalid action failures)")
    print()

    # Initialize base environment WITHOUT action masking
    print("Initializing Call Center Environment (NO action masking)...")
    env = CallCenterEnv(data_path=DATA_PATH, assets_dir=ASSETS_DIR)
    env.reset(seed=args.seed)
    print(f"✓ Environment initialized: {env.num_agents} agents")
    print(f"✓ Action space: Discrete({env.action_space.n})")

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(ASSETS_DIR, 'checkpoints_dqn_unmasked'),
        name_prefix='dqn_unmasked'
    )

    print("\n" + "=" * 60)
    print("TRAINING DQN (NO ACTION MASKING)")
    print("=" * 60)

    # Create DQN model with specified hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1,
        device=args.device,
        seed=args.seed
    )

    print("Model configuration:")
    print(f"  Policy: MlpPolicy")
    print(f"  Action masking: DISABLED")
    print(f"  Learning rate: 0.0003")
    print(f"  Buffer size: 100,000")
    print(f"  Batch size: 256")
    print(f"  Gamma: 0.99")
    print(f"  Exploration fraction: 0.1")
    print(f"  Exploration final eps: 0.05")
    print(f"  Device: {args.device}")
    print()

    # Train
    print("Starting training...")
    start_time = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=False
    )
    elapsed = (time.time() - start_time) / 60.0

    # Save final model
    save_path = os.path.join(ASSETS_DIR, 'rl_model_dqn_unmasked')
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
    invalid_actions = 0

    while not done and step < 10000:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Track successful calls
        if 'status' in info and info['status'] == 'success':
            calls_handled += 1
        # Track invalid actions (agent not available)
        elif 'status' in info and info['status'] == 'failed':
            invalid_actions += 1

    print(f"Episode stats:")
    print(f"  Steps: {step}")
    print(f"  Calls handled: {calls_handled}")
    print(f"  Invalid actions: {invalid_actions}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final time: {env.current_time:.0f}s / {env.simulation_day_length:.0f}s")

    expected_calls = 590
    efficiency = calls_handled / expected_calls * 100
    invalid_rate = invalid_actions / (calls_handled + invalid_actions) * 100 if (calls_handled + invalid_actions) > 0 else 0

    print(f"\nEfficiency: {efficiency:.1f}% ({calls_handled}/{expected_calls} expected calls)")
    print(f"Invalid action rate: {invalid_rate:.1f}%")

    if efficiency < 10:
        print("\n✓ EXPECTED FAILURE: Agent is barely handling any calls due to invalid actions")
        print("   This confirms the action masking problem affects DQN as well.")
    else:
        print("\n⚠️  UNEXPECTED: Agent is handling more calls than expected without masking")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("To evaluate this unmasked DQN policy:")
    print("  python evaluate_dqn_unmasked.py --episodes 10")


if __name__ == '__main__':
    main()
