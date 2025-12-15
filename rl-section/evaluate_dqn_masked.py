#!/usr/bin/env python3
"""
Evaluate DQN WITH action masking.

This evaluates Experiment 2: Masked DQN
Tracks cost per call, calls handled, and overall performance.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from sb3_contrib.common.wrappers import ActionMasker

from call_center_env_masked import CallCenterEnvMasked

DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Masked DQN policy")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(ASSETS_DIR, 'rl_model_dqn_masked.zip'),
                        help="Path to trained model")
    parser.add_argument("--output", type=str,
                        default=os.path.join(ASSETS_DIR, 'dqn_masked_results.csv'),
                        help="Output CSV file")
    return parser.parse_args()


def mask_fn(env):
    """Action mask function for ActionMasker wrapper."""
    return env.action_masks()


def evaluate_episode(env, model, episode_num, base_seed):
    """Evaluate a single episode."""
    seed = base_seed + episode_num if base_seed is not None else None

    obs, info = env.reset(seed=seed)
    done = False
    step = 0

    episode_reward = 0.0
    episode_cost = 0.0
    calls_handled = 0

    while not done and step < 10000:
        # Predict action (ActionMasker wrapper will handle masking)
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        step += 1

        # Track successful calls
        if 'status' in info and info['status'] == 'success':
            episode_reward += reward
            episode_cost += info.get('cost', 0.0)
            calls_handled += 1

    return {
        'episode': episode_num + 1,
        'total_reward': episode_reward,
        'total_cost': episode_cost,
        'calls_handled': calls_handled,
        'steps': step,
        'final_time': env.env.current_time
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("EVALUATING MASKED DQN POLICY (EXPERIMENT 2)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print()

    # Load environment with action masking
    print("Loading masked environment...")
    base_env = CallCenterEnvMasked(
        data_path=DATA_PATH,
        assets_dir=ASSETS_DIR,
        test_indices_path=os.path.join(ASSETS_DIR, 'test_indices.npy')
    )
    env = ActionMasker(base_env, mask_fn)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = DQN.load(args.model_path)
    model.set_env(env)
    print("✓ Model loaded")

    # Evaluate
    print(f"\nEvaluating for {args.episodes} episodes...")
    results = []

    for ep in range(args.episodes):
        print(f"\nEpisode {ep+1}/{args.episodes}...")
        start = time.time()

        result = evaluate_episode(env, model, ep, args.seed)
        elapsed = time.time() - start

        results.append(result)

        print(f"  Calls: {result['calls_handled']}, Cost: €{result['total_cost']:.2f}, "
              f"Steps: {result['steps']}, Time: {elapsed:.1f}s")

    # Aggregate results
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    avg_calls = results_df['calls_handled'].mean()
    avg_cost = results_df['total_cost'].mean()
    avg_cost_per_call = avg_cost / avg_calls if avg_calls > 0 else 0
    std_cost_per_call = results_df['total_cost'].std() / avg_calls if avg_calls > 0 else 0

    print(f"Average calls per day: {avg_calls:.1f}")
    print(f"Average cost per day: €{avg_cost:.2f}")
    print(f"Average cost per call: €{avg_cost_per_call:.2f}")
    print(f"Std dev of cost: €{std_cost_per_call:.2f}")

    expected = 590
    efficiency = avg_calls / expected * 100
    print(f"\nEfficiency: {efficiency:.1f}% ({avg_calls:.0f}/{expected} expected calls)")

    # Save results
    summary = pd.DataFrame([{
        'policy_name': 'DQN (Masked)',
        'cost_per_call': avg_cost_per_call,
        'calls_per_day': avg_calls,
        'efficiency_pct': efficiency,
        'invalid_actions_pct': None,  # Not applicable with masking
        'std_dev': std_cost_per_call,
        'training_steps': 200000,  # As specified in experiment
        'seed': args.seed
    }])

    summary.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")

    # Compare with baselines (hardcoded from thesis)
    greedy_cost = 15.06  # From fresh validation
    masked_ppo_cost = 16.21  # From fresh validation

    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINES")
    print("=" * 60)
    print(f"Greedy XGBoost:  €{greedy_cost:.2f}/call (baseline)")
    print(f"Masked PPO:      €{masked_ppo_cost:.2f}/call")
    print(f"Masked DQN:      €{avg_cost_per_call:.2f}/call")

    if avg_cost_per_call > greedy_cost:
        diff = avg_cost_per_call - greedy_cost
        pct = (diff / greedy_cost) * 100
        print(f"\n✓ EXPECTED: Masked DQN underperforms Greedy by €{diff:.2f}/call ({pct:.1f}%)")
        print("   This confirms simulator noise affects DQN as well, not just PPO.")
    else:
        print(f"\n⚠️  UNEXPECTED: Masked DQN beats Greedy baseline")

    if efficiency > 95:
        print("\n🎉 SUCCESS: Action masking solved the call handling problem!")
    elif efficiency > 80:
        print("\n⚠️  GOOD: Much better but still some calls missing")
    else:
        print("\n❌ ISSUE: Still significant problems")


if __name__ == '__main__':
    main()
