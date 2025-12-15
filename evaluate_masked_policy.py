#!/usr/bin/env python3
"""
Evaluate the MaskablePPO policy trained with action masking.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from call_center_env_masked import CallCenterEnvMasked

DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MaskablePPO policy")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(ASSETS_DIR, 'rl_model_masked_ppo.zip'),
                        help="Path to trained model")
    parser.add_argument("--output", type=str,
                        default=os.path.join(ASSETS_DIR, 'masked_ppo_results.csv'),
                        help="Output CSV file")
    parser.add_argument("--tag", type=str, default="masked_ppo",
                        help="Tag for this evaluation run")
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
        # Get action mask and predict
        action_mask = env.env.action_masks()
        action, _states = model.predict(obs, deterministic=True, action_masks=action_mask)

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
    print("EVALUATING MASKABLE PPO POLICY")
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
    model = MaskablePPO.load(args.model_path)
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

    print(f"Average calls per day: {avg_calls:.1f}")
    print(f"Average cost per day: €{avg_cost:.2f}")
    print(f"Average cost per call: €{avg_cost_per_call:.2f}")

    expected = 590
    efficiency = avg_calls / expected * 100
    print(f"\nEfficiency: {efficiency:.1f}% ({avg_calls:.0f}/{expected} expected calls)")

    # Save results
    summary = pd.DataFrame([{
        'policy': 'MaskablePPO (Action Masked)',
        'run_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'avg_total_cost_per_day': avg_cost,
        'avg_cost_per_call': avg_cost_per_call,
        'avg_reward_per_day': results_df['total_reward'].mean(),
        'avg_calls_per_day': avg_calls,
        'tag': args.tag
    }])

    summary.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")

    if efficiency > 95:
        print("\n🎉 SUCCESS: Action masking solved the problem!")
    elif efficiency > 80:
        print("\n⚠️  GOOD: Much better but still some calls missing")
    else:
        print("\n❌ ISSUE: Still significant problems")


if __name__ == '__main__':
    main()
