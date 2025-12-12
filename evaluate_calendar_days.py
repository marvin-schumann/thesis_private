#!/usr/bin/env python3
"""
Calendar-Day-Based Policy Evaluation

This script evaluates routing policies on both:
1. Random-sampled episodes (baseline methodology)
2. Complete calendar days (calendar-day methodology)

This validates that policy performance rankings hold under realistic
operational conditions (complete calendar days vs synthetic episodes).

Evaluates 4 functional policies:
- Random
- Rule-Based
- Greedy XGBoost
- Masked PPO

Usage:
    python evaluate_calendar_days.py --mode both --episodes 10 --seed 42
    python evaluate_calendar_days.py --mode calendar_day --calendar-days-file models/selected_calendar_days.json
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from baseline_policies import BaselinePolicies
from call_center_env import CallCenterEnv
from call_center_env_masked import CallCenterEnvMasked

# Configuration
DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'

print("[DEBUG] Module loaded, setting up logging...")
logging.basicConfig(level=logging.WARNING)  # Reduce noise during evaluation
print("[DEBUG] Logging configured")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate policies on random episodes and/or calendar days"
    )
    parser.add_argument("--mode", type=str, choices=['random', 'calendar_day', 'both'],
                        default='both',
                        help="Evaluation mode: random episodes, calendar days, or both")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of random episodes (for 'random' or 'both' mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--calendar-days-file", type=str,
                        default=os.path.join(ASSETS_DIR, 'selected_calendar_days.json'),
                        help="JSON file with selected calendar days")
    parser.add_argument("--output-dir", type=str, default=ASSETS_DIR,
                        help="Directory for output CSV files")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip Random policy")
    parser.add_argument("--skip-topic-only", action="store_true",
                        help="Skip Topic-Only baseline policy")
    parser.add_argument("--skip-rule", action="store_true",
                        help="Skip Rule-Based policy")
    parser.add_argument("--skip-rule-conservative", action="store_true",
                        help="Skip Rule-Based Conservative policy")
    parser.add_argument("--skip-greedy", action="store_true",
                        help="Skip Greedy XGBoost policy")
    parser.add_argument("--skip-masked-ppo", action="store_true",
                        help="Skip Masked PPO policy")
    return parser.parse_args()


def mask_fn(env):
    """Action mask function for MaskablePPO."""
    return env.action_masks()


def evaluate_policy_random_mode(env, policy, policy_name: str, n_episodes: int,
                                 base_seed: Optional[int] = None) -> Dict:
    """
    Evaluate a policy on random-sampled episodes.

    Returns:
        Dict with episode-level results
    """
    is_sb3_model = hasattr(policy, 'predict')
    is_maskable = 'MaskablePPO' in policy_name or 'Masked' in policy_name

    results = []

    print(f"  > Running {n_episodes} random episodes for {policy_name}...")

    for episode in range(n_episodes):
        current_seed = base_seed + episode if base_seed is not None else None

        obs, info = env.reset(seed=current_seed)
        done = False
        step = 0

        episode_reward = 0.0
        episode_cost = 0.0
        calls_handled = 0
        invalid_actions = 0

        episode_start = time.time()

        while not done and step < 10000:
            if is_sb3_model:
                if is_maskable:
                    # Get action mask for MaskablePPO
                    action_mask = env.env.action_masks() if hasattr(env, 'env') else env.action_masks()
                    action, _states = policy.predict(obs, deterministic=True, action_masks=action_mask)
                else:
                    action, _states = policy.predict(obs, deterministic=True)
            else:
                action = policy(obs)

            obs, reward, done, truncated, info = env.step(action)
            step += 1

            if info.get('status') == 'success':
                episode_reward += reward
                episode_cost += info.get('cost', 0.0)
                calls_handled += 1
            elif 'invalid' in info.get('status', ''):
                invalid_actions += 1

            if done or truncated:
                break

        episode_time = time.time() - episode_start

        cost_per_call = episode_cost / calls_handled if calls_handled > 0 else 0.0

        results.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'total_cost': episode_cost,
            'calls_handled': calls_handled,
            'cost_per_call': cost_per_call,
            'steps': step,
            'invalid_actions': invalid_actions,
            'time_seconds': episode_time
        })

        print(f"    Ep {episode+1}/{n_episodes}: calls={calls_handled}, "
              f"cost={episode_cost:.2f}, cost/call=€{cost_per_call:.2f}, "
              f"steps={step}, time={episode_time:.1f}s")

    return {
        'mode': 'random',
        'policy': policy_name,
        'episodes': results
    }


def evaluate_policy_calendar_day_mode(env, policy, policy_name: str,
                                       calendar_dates: List[str],
                                       base_seed: Optional[int] = None) -> Dict:
    """
    Evaluate a policy on complete calendar days.

    Returns:
        Dict with day-level results
    """
    is_sb3_model = hasattr(policy, 'predict')
    is_maskable = 'MaskablePPO' in policy_name or 'Masked' in policy_name

    results = []

    print(f"  > Running {len(calendar_dates)} calendar days for {policy_name}...")

    for day_idx, calendar_date in enumerate(calendar_dates):
        # Create new env instance for this calendar day
        if is_maskable:
            base_env = CallCenterEnvMasked(
                data_path=DATA_PATH,
                assets_dir=ASSETS_DIR,
                test_indices_path=os.path.join(ASSETS_DIR, 'test_indices.npy'),
                episode_mode='calendar_day',
                calendar_date=calendar_date
            )
            day_env = ActionMasker(base_env, mask_fn)
        else:
            day_env = CallCenterEnv(
                data_path=DATA_PATH,
                assets_dir=ASSETS_DIR,
                test_indices_path=os.path.join(ASSETS_DIR, 'test_indices.npy'),
                episode_mode='calendar_day',
                calendar_date=calendar_date
            )

        current_seed = base_seed + day_idx if base_seed is not None else None
        obs, info = day_env.reset(seed=current_seed)
        done = False
        step = 0

        episode_reward = 0.0
        episode_cost = 0.0
        calls_handled = 0
        invalid_actions = 0

        day_start = time.time()

        while not done and step < 10000:
            if is_sb3_model:
                if is_maskable:
                    # Get action mask
                    action_mask = day_env.env.action_masks() if hasattr(day_env, 'env') else day_env.action_masks()
                    action, _states = policy.predict(obs, deterministic=True, action_masks=action_mask)
                else:
                    action, _states = policy.predict(obs, deterministic=True)
            else:
                action = policy(obs)

            obs, reward, done, truncated, info = day_env.step(action)
            step += 1

            status = info.get('status', '')
            if status in ['success', 'success_calendar_day_complete']:
                episode_reward += reward
                episode_cost += info.get('cost', 0.0)
                calls_handled += 1
            elif 'invalid' in status:
                invalid_actions += 1

            if done or truncated:
                break

        day_time = time.time() - day_start
        cost_per_call = episode_cost / calls_handled if calls_handled > 0 else 0.0

        results.append({
            'calendar_date': calendar_date,
            'day_index': day_idx + 1,
            'total_reward': episode_reward,
            'total_cost': episode_cost,
            'calls_handled': calls_handled,
            'cost_per_call': cost_per_call,
            'steps': step,
            'invalid_actions': invalid_actions,
            'time_seconds': day_time
        })

        print(f"    Day {day_idx+1}/{len(calendar_dates)} ({calendar_date}): "
              f"calls={calls_handled}, cost={episode_cost:.2f}, "
              f"cost/call=€{cost_per_call:.2f}, steps={step}, time={day_time:.1f}s")

    return {
        'mode': 'calendar_day',
        'policy': policy_name,
        'days': results
    }


def load_calendar_days(filepath: str) -> List[str]:
    """Load selected calendar days from JSON file."""
    with open(filepath, 'r') as f:
        dates = json.load(f)
    print(f"Loaded {len(dates)} calendar days from {filepath}")
    return dates


def main():
    args = parse_args()

    print("=" * 70)
    print("CALENDAR DAY EVALUATION")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Random episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load calendar days if needed
    calendar_dates = []
    if args.mode in ['calendar_day', 'both']:
        calendar_dates = load_calendar_days(args.calendar_days_file)
        print(f"Calendar days to evaluate: {len(calendar_dates)}")
        for i, date in enumerate(calendar_dates, 1):
            print(f"  {i}. {date}")
        print()

    # Initialize environments and policies for random mode
    random_env = None
    random_masked_env = None
    baselines = None

    if args.mode in ['random', 'both']:
        print("Initializing environment for random mode...")
        print("  [DEBUG] Creating CallCenterEnv...")
        random_env = CallCenterEnv(
            data_path=DATA_PATH,
            assets_dir=ASSETS_DIR,
            test_indices_path=os.path.join(ASSETS_DIR, 'test_indices.npy'),
            episode_mode='random'
        )
        print("  [DEBUG] CallCenterEnv created")

        # Masked environment for MaskablePPO
        print("  [DEBUG] Creating CallCenterEnvMasked...")
        base_masked_env = CallCenterEnvMasked(
            data_path=DATA_PATH,
            assets_dir=ASSETS_DIR,
            test_indices_path=os.path.join(ASSETS_DIR, 'test_indices.npy'),
            episode_mode='random'
        )
        print("  [DEBUG] CallCenterEnvMasked created")
        print("  [DEBUG] Creating ActionMasker...")
        random_masked_env = ActionMasker(base_masked_env, mask_fn)
        print("  [DEBUG] ActionMasker created")

        print("Initializing baseline policies...")
        baselines = BaselinePolicies(
            data_path=DATA_PATH,
            assets_dir=ASSETS_DIR,
            test_indices_path=os.path.join(ASSETS_DIR, 'test_indices.npy')
        )
        print("✓ Random mode setup complete\n")

    # Build policy list
    policies_to_evaluate = []

    if not args.skip_random:
        policies_to_evaluate.append(('Random', 'baseline', random_env))
    if not args.skip_topic_only:
        policies_to_evaluate.append(('Topic-Only', 'baseline', random_env))
    if not args.skip_rule:
        policies_to_evaluate.append(('Rule-Based', 'baseline', random_env))
    if not args.skip_rule_conservative:
        policies_to_evaluate.append(('Rule-Based Conservative', 'baseline', random_env))
    if not args.skip_greedy:
        policies_to_evaluate.append(('Greedy XGBoost', 'baseline', random_env))
    if not args.skip_masked_ppo:
        policies_to_evaluate.append(('Masked PPO', 'masked', random_masked_env))

    if not policies_to_evaluate:
        print("No policies selected. Exiting.")
        return

    # Initialize baselines if we have any baseline policies (regardless of mode)
    has_baseline_policies = any(p[1] == 'baseline' for p in policies_to_evaluate)
    if has_baseline_policies and baselines is None:
        print("Initializing baseline policies for evaluation...")
        baselines = BaselinePolicies(
            data_path=DATA_PATH,
            assets_dir=ASSETS_DIR,
            test_indices_path=os.path.join(ASSETS_DIR, 'test_indices.npy')
        )
        print("✓ Baseline policies initialized\n")

    print(f"Policies to evaluate: {', '.join([p[0] for p in policies_to_evaluate])}\n")

    # Evaluation results storage
    all_results = []

    # --- RANDOM MODE EVALUATION ---
    if args.mode in ['random', 'both']:
        print("=" * 70)
        print("RANDOM MODE EVALUATION")
        print("=" * 70)

        for policy_name, policy_type, env in policies_to_evaluate:
            print(f"\n--- Evaluating {policy_name} (Random Mode) ---")
            start_time = time.time()

            # Get policy function/model
            if policy_type == 'baseline':
                if policy_name == 'Random':
                    policy = baselines.random_policy
                elif policy_name == 'Topic-Only':
                    policy = baselines.naive_topic_only_policy
                elif policy_name == 'Rule-Based':
                    policy = baselines.rule_based_policy
                elif policy_name == 'Rule-Based Conservative':
                    policy = baselines.rule_based_conservative_policy
                elif policy_name == 'Greedy XGBoost':
                    policy = baselines.greedy_xgboost_policy
                else:
                    raise ValueError(f"Unknown baseline policy: {policy_name}")
            elif policy_type == 'masked':
                model_path = os.path.join(ASSETS_DIR, 'rl_model_masked_ppo.zip')
                if not os.path.exists(model_path):
                    print(f"  ✗ Skipping {policy_name}: model not found at {model_path}")
                    continue
                policy = MaskablePPO.load(model_path)
                policy.set_env(env)
                print(f"  ✓ Loaded model from {model_path}")
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")

            result = evaluate_policy_random_mode(env, policy, policy_name,
                                                  args.episodes, args.seed)
            elapsed = time.time() - start_time
            print(f"  ✓ Evaluation complete in {elapsed:.2f}s")
            all_results.append(result)

    # --- CALENDAR DAY MODE EVALUATION ---
    if args.mode in ['calendar_day', 'both']:
        print("\n" + "=" * 70)
        print("CALENDAR DAY MODE EVALUATION")
        print("=" * 70)

        for policy_name, policy_type, _ in policies_to_evaluate:
            print(f"\n--- Evaluating {policy_name} (Calendar Day Mode) ---")
            start_time = time.time()

            # Get policy function/model
            # Note: For calendar day mode, we create fresh env instances inside the function
            if policy_type == 'baseline':
                if policy_name == 'Random':
                    policy = baselines.random_policy
                elif policy_name == 'Topic-Only':
                    policy = baselines.naive_topic_only_policy
                elif policy_name == 'Rule-Based':
                    policy = baselines.rule_based_policy
                elif policy_name == 'Rule-Based Conservative':
                    policy = baselines.rule_based_conservative_policy
                elif policy_name == 'Greedy XGBoost':
                    policy = baselines.greedy_xgboost_policy
                else:
                    raise ValueError(f"Unknown baseline policy: {policy_name}")
            elif policy_type == 'masked':
                model_path = os.path.join(ASSETS_DIR, 'rl_model_masked_ppo.zip')
                if not os.path.exists(model_path):
                    print(f"  ✗ Skipping {policy_name}: model not found at {model_path}")
                    continue
                policy = MaskablePPO.load(model_path)
                print(f"  ✓ Loaded model from {model_path}")
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")

            # Pass None for env - will be created per calendar day
            result = evaluate_policy_calendar_day_mode(None, policy, policy_name,
                                                        calendar_dates, args.seed)
            elapsed = time.time() - start_time
            print(f"  ✓ Evaluation complete in {elapsed:.2f}s")
            all_results.append(result)

    # --- SAVE RESULTS ---
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for result in all_results:
        mode = result['mode']
        policy = result['policy']

        # Create DataFrame
        if mode == 'random':
            df = pd.DataFrame(result['episodes'])
            filename = f"calendar_eval_{mode}_{policy.replace(' ', '_')}_{timestamp}.csv"
        else:  # calendar_day
            df = pd.DataFrame(result['days'])
            filename = f"calendar_eval_{mode}_{policy.replace(' ', '_')}_{timestamp}.csv"

        output_path = os.path.join(args.output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved {mode} mode results for {policy}: {output_path}")

    # Create summary report
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    summary_rows = []
    for result in all_results:
        mode = result['mode']
        policy = result['policy']

        if mode == 'random':
            episodes_df = pd.DataFrame(result['episodes'])
            avg_cost_per_call = episodes_df['cost_per_call'].mean()
            std_cost_per_call = episodes_df['cost_per_call'].std()
            avg_calls = episodes_df['calls_handled'].mean()
        else:
            days_df = pd.DataFrame(result['days'])
            avg_cost_per_call = days_df['cost_per_call'].mean()
            std_cost_per_call = days_df['cost_per_call'].std()
            avg_calls = days_df['calls_handled'].mean()

        summary_rows.append({
            'mode': mode,
            'policy': policy,
            'avg_cost_per_call': avg_cost_per_call,
            'std_cost_per_call': std_cost_per_call,
            'avg_calls_handled': avg_calls,
            'n_episodes': len(result.get('episodes', result.get('days', [])))
        })

        print(f"\n{policy} ({mode} mode):")
        print(f"  Average cost per call: €{avg_cost_per_call:.2f} ± {std_cost_per_call:.2f}")
        print(f"  Average calls handled: {avg_calls:.1f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, f"calendar_eval_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("✓ CALENDAR DAY EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nNext step: Run analyze_calendar_day_results.py to compare results")
    print()


if __name__ == '__main__':
    main()
