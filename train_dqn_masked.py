#!/usr/bin/env python3
"""
Train DQN WITH action masking to demonstrate simulator noise problem.

This is Experiment 2: Masked DQN
Purpose: Show that simulator noise affects all RL algorithms, not just PPO
Expected: Agent will handle calls but underperform baseline policies due to simulator
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

# Action masking support - Note: DQN doesn't have native masking in SB3
# We'll use a custom wrapper approach
from sb3_contrib.common.wrappers import ActionMasker

# Import masked environment
from call_center_env_masked import CallCenterEnvMasked

# --- Configuration ---
DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'
os.makedirs(ASSETS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DQN WITH action masking (Experiment 2)."
    )
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total training timesteps (default: 200000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--checkpoint-freq", type=int, default=50000,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Computation device ('cpu', 'cuda', 'auto').")
    return parser.parse_args()


def mask_fn(env):
    """Action mask function for ActionMasker wrapper."""
    return env.action_masks()


def main():
    args = parse_args()

    set_random_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("EXPERIMENT 2: TRAINING MASKED DQN")
    print("=" * 60)
    print("Purpose: Demonstrate simulator noise affects all RL algorithms")
    print(f"Training steps: {args.timesteps:,}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Action masking: ENABLED")
    print()

    # Initialize masked environment
    print("Initializing Action-Masked Call Center Environment...")
    base_env = CallCenterEnvMasked(data_path=DATA_PATH, assets_dir=ASSETS_DIR)

    # Wrap with ActionMasker
    env = ActionMasker(base_env, mask_fn)
    env.reset(seed=args.seed)

    print(f"✓ Environment initialized: {base_env.num_agents} agents")
    print(f"✓ Action space: Discrete({base_env.action_space.n})")
    print(f"✓ Action masking: Enabled")

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(ASSETS_DIR, 'checkpoints_dqn_masked'),
        name_prefix='dqn_masked'
    )

    print("\n" + "=" * 60)
    print("TRAINING MASKED DQN")
    print("=" * 60)

    # Create DQN model with specified hyperparameters
    # Note: Standard DQN from SB3 doesn't natively support action masking
    # The ActionMasker wrapper will modify the Q-values of invalid actions
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
    print(f"  Action masking: ENABLED (via ActionMasker wrapper)")
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
    save_path = os.path.join(ASSETS_DIR, 'rl_model_dqn_masked')
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
    total_cost = 0
    calls_handled = 0

    while not done and step < 10000:
        # Get action mask and predict
        action_mask = env.env.action_masks()
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Track successful calls
        if 'status' in info and info['status'] == 'success':
            calls_handled += 1
            total_cost += info.get('cost', 0.0)

    avg_cost_per_call = total_cost / calls_handled if calls_handled > 0 else 0

    print(f"Episode stats:")
    print(f"  Steps: {step}")
    print(f"  Calls handled: {calls_handled}")
    print(f"  Total cost: €{total_cost:.2f}")
    print(f"  Avg cost per call: €{avg_cost_per_call:.2f}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final time: {env.env.current_time:.0f}s / {env.env.simulation_day_length:.0f}s")

    expected_calls = 590
    efficiency = calls_handled / expected_calls * 100
    print(f"\nEfficiency: {efficiency:.1f}% ({calls_handled}/{expected_calls} expected calls)")

    if efficiency > 95:
        print("\n✓ SUCCESS: Masked DQN handles >95% of calls (action masking working)")
    elif efficiency > 80:
        print("\n⚠️  GOOD: Handles most calls but some gaps remain")
    else:
        print("\n❌ ISSUE: Still significant call handling problems")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("To evaluate this masked DQN policy:")
    print("  python evaluate_dqn_masked.py --episodes 10")


if __name__ == '__main__':
    main()
