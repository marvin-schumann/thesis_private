#!/usr/bin/env python3
"""
Evaluate DQN WITHOUT action masking.

This evaluates Experiment 1: Unmasked DQN
Tracks calls handled, efficiency, and invalid action rate.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from call_center_env import CallCenterEnv

DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Unmasked DQN policy")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(ASSETS_DIR, 'rl_model_dqn_unmasked.zip'),
                        help="Path to trained model")
    parser.add_argument("--output", type=str,
                        default=os.path.join(ASSETS_DIR, 'dqn_unmasked_results.csv'),
                        help="Output CSV file")
    return parser.parse_args()


def evaluate_episode(env, model, episode_num, base_seed):
    """Evaluate a single episode."""
    seed = base_seed + episode_num if base_seed is not None else None

    obs, info = env.reset(seed=seed)
    done = False
    step = 0

    episode_reward = 0.0
    episode_cost = 0.0
    calls_handled = 0
    invalid_actions = 0

    while not done and step < 10000:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

        # Track successful calls
        if 'status' in info and info['status'] == 'success':
            episode_reward += reward
            episode_cost += info.get('cost', 0.0)
            calls_handled += 1
        # Track invalid actions (agent not available)
        elif 'status' in info and info['status'] == 'invalid_action_agent_busy_or_off_shift':
            invalid_actions += 1

    return {
        'episode': episode_num + 1,
        'total_reward': episode_reward,
        'total_cost': episode_cost,
        'calls_handled': calls_handled,
        'invalid_actions': invalid_actions,
        'steps': step,
        'final_time': env.current_time
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("EVALUATING UNMASKED DQN POLICY (EXPERIMENT 1)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print()

    # Load environment WITHOUT action masking
    print("Loading environment (no action masking)...")
    env = CallCenterEnv(data_path=DATA_PATH, assets_dir=ASSETS_DIR)

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

        print(f"  Calls: {result['calls_handled']}, Invalid: {result['invalid_actions']}, "
              f"Steps: {result['steps']}, Time: {elapsed:.1f}s")

    # Aggregate results
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    avg_calls = results_df['calls_handled'].mean()
    avg_cost = results_df['total_cost'].mean()
    avg_invalid = results_df['invalid_actions'].mean()
    total_attempts = avg_calls + avg_invalid

    expected = 590
    efficiency = avg_calls / expected * 100 if expected > 0 else 0
    invalid_rate = avg_invalid / total_attempts * 100 if total_attempts > 0 else 0
    avg_cost_per_call = avg_cost / avg_calls if avg_calls > 0 else 0

    print(f"Average calls handled per day: {avg_calls:.1f}")
    print(f"Average cost per day: €{avg_cost:.2f}")
    print(f"Average cost per call: €{avg_cost_per_call:.2f}")
    print(f"Average invalid actions per day: {avg_invalid:.1f}")
    print(f"Total action attempts: {total_attempts:.1f}")
    print(f"\nEfficiency: {efficiency:.1f}% ({avg_calls:.0f}/{expected} expected calls)")
    print(f"Invalid action rate: {invalid_rate:.1f}%")

    # Save results
    summary = pd.DataFrame([{
        'policy_name': 'DQN (Unmasked)',
        'cost_per_call': avg_cost_per_call,
        'calls_per_day': avg_calls,
        'efficiency_pct': efficiency,
        'invalid_actions_pct': invalid_rate,
        'std_dev': results_df['total_cost'].std() / avg_calls if avg_calls > 0 else 0,
        'training_steps': 50000,  # As specified in experiment
        'seed': args.seed
    }])

    summary.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")

    # Interpretation
    if efficiency < 10:
        print("\n✓ EXPECTED FAILURE: DQN barely handles any calls without action masking")
        print("   This confirms the action masking problem is algorithm-independent.")
    elif efficiency < 50:
        print("\n⚠️  PARTIAL: DQN handles some calls but with high failure rate")
    else:
        print("\n❌ UNEXPECTED: DQN is handling more calls than expected without masking")


if __name__ == '__main__':
    main()
