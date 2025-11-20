import argparse
import logging
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

# Import our custom environment and baseline policies
from call_center_env import CallCenterEnv
from baseline_policies import BaselinePolicies

# Import masked environment and algorithms (if available)
try:
    from call_center_env_masked import CallCenterEnvMasked
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False

# --- 1. Configuration ---
DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and RL policies on the NOS call-center environment."
    )
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of simulated days per policy.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed to synchronise call sequences.")
    parser.add_argument("--include-rl", action="store_true",
                        help="Include trained RL models (DQN/PPO) if their .zip files exist.")
    parser.add_argument("--include-rl-masked", action="store_true",
                        help="Include trained RL models with action masking (MaskablePPO).")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip the random baseline.")
    parser.add_argument("--skip-rule", action="store_true",
                        help="Skip the rule-based policy.")
    parser.add_argument("--skip-greedy", action="store_true",
                        help="Skip the greedy XGBoost policy.")
    parser.add_argument("--output", type=str, default=os.path.join(ASSETS_DIR, 'final_evaluation_results.csv'),
                        help="CSV file to append evaluation results.")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag/notes column to add to the results.")
    return parser.parse_args()


def evaluate_policy(env, policy, n_episodes: int, base_seed: Optional[int] = None) -> Dict[str, float]:
    """
    Runs a given policy in the environment for n_episodes and returns metrics.

    'policy' can be a stable-baselines model (with a .predict() method)
    or a callable that takes (obs) and returns (action).
    """

    is_sb3_model = hasattr(policy, 'predict')
    # Check if this is a maskable model (requires action masks)
    is_maskable = SB3_CONTRIB_AVAILABLE and isinstance(policy, MaskablePPO)
    # Check if environment supports action masking
    env_has_masking = hasattr(env, 'action_masks')

    total_rewards = []
    total_costs = []
    total_calls_handled = []

    print(f"  > Running evaluation for {n_episodes} episodes...")
    if is_maskable and env_has_masking:
        print(f"  > Using action masking during evaluation")

    for episode in range(n_episodes):
        current_seed = None
        if base_seed is not None:
            current_seed = base_seed + episode
            random.seed(current_seed)
            np.random.seed(current_seed)

        obs, info = env.reset(seed=current_seed)
        print(f"    Episode {episode + 1}/{n_episodes}: starting...", flush=True)

        episode_start_time = time.perf_counter()

        done = False

        episode_reward = 0.0
        episode_cost = 0.0
        episode_calls = 0
        step_count = 0

        while not done:
            if is_sb3_model:
                if is_maskable and env_has_masking:
                    # Get action masks from environment
                    action_masks = env.action_masks()
                    action, _states = policy.predict(obs, action_masks=action_masks, deterministic=True)
                else:
                    action, _states = policy.predict(obs, deterministic=True)
            else:
                action = policy(obs)

            obs, reward, done, truncated, info = env.step(action)
            step_count += 1

            if info.get('status') == 'success':
                episode_reward += reward
                episode_cost += info.get('cost', 0.0)
                episode_calls += 1

            if step_count % 50 == 0:
                elapsed = time.perf_counter() - episode_start_time
                print(
                    f"      ▸ steps={step_count:<5} calls={episode_calls:<5} elapsed={elapsed:6.1f}s",
                    flush=True
                )

            if done or truncated:
                break

        elapsed_total = time.perf_counter() - episode_start_time
        print(
            f"    Episode {episode + 1}/{n_episodes} complete: calls={episode_calls}, "
            f"cost={episode_cost:.2f}, steps={step_count}, time={elapsed_total:.1f}s",
            flush=True
        )
        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        total_calls_handled.append(episode_calls)

    avg_reward = float(np.mean(total_rewards))
    avg_cost = float(np.mean(total_costs))
    avg_calls = float(np.mean(total_calls_handled))
    avg_cost_per_call = avg_cost / avg_calls if avg_calls > 0 else 0.0

    return {
        'avg_reward_per_day': avg_reward,
        'avg_total_cost_per_day': avg_cost,
        'avg_calls_per_day': avg_calls,
        'avg_cost_per_call': avg_cost_per_call
    }


def load_rl_models(include: bool, env) -> Dict[str, object]:
    rl_policies: Dict[str, object] = {}
    if not include:
        return rl_policies

    rl_paths = [
        ("4. DQN (RL Agent)", os.path.join(ASSETS_DIR, 'rl_model_dqn.zip'), DQN),
        ("5. PPO (RL Agent)", os.path.join(ASSETS_DIR, 'rl_model_ppo.zip'), PPO),
    ]

    for name, path, cls in rl_paths:
        if not os.path.exists(path):
            print(f"  ▸ Skipping {name}: model file not found at {path}")
            continue
        try:
            model = cls.load(path, device="auto")
            model.set_env(env)
            rl_policies[name] = model
            print(f"  ▸ Loaded {name} from {path}")
        except Exception as exc:
            print(f"  ▸ Failed to load {name} ({exc}). Skipping.")
    return rl_policies


def load_masked_rl_models(include: bool, env) -> Dict[str, object]:
    """Load RL models trained with action masking."""
    rl_policies: Dict[str, object] = {}
    if not include or not SB3_CONTRIB_AVAILABLE:
        if include and not SB3_CONTRIB_AVAILABLE:
            print("  ▸ Skipping masked models: sb3-contrib not installed")
        return rl_policies

    rl_paths = [
        ("6. PPO-Masked (RL Agent)", os.path.join(ASSETS_DIR, 'rl_model_ppo_masked.zip'), MaskablePPO),
    ]

    for name, path, cls in rl_paths:
        if not os.path.exists(path):
            print(f"  ▸ Skipping {name}: model file not found at {path}")
            continue
        try:
            model = cls.load(path, device="auto")
            # No need to set_env - evaluate_policy will handle action masking
            rl_policies[name] = model
            print(f"  ▸ Loaded {name} from {path} (action masking will be used during evaluation)")
        except Exception as exc:
            print(f"  ▸ Failed to load {name} ({exc}). Skipping.")
    return rl_policies


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("--- Starting Policy Evaluation ---")

    # Determine which environment to use
    if args.include_rl_masked:
        print("Initializing Masked Environment...")
        if SB3_CONTRIB_AVAILABLE:
            env = CallCenterEnvMasked(data_path=DATA_PATH, assets_dir=ASSETS_DIR)
            print("Masked environment initialized.")
        else:
            print("WARNING: sb3-contrib not installed. Using standard environment.")
            print("Install with: pip install sb3-contrib")
            env = CallCenterEnv(data_path=DATA_PATH, assets_dir=ASSETS_DIR)
    else:
        print("Initializing Environment...")
        env = CallCenterEnv(data_path=DATA_PATH, assets_dir=ASSETS_DIR)
        print("Environment initialized.")

    print("Initializing Baseline Policies...")
    baselines = BaselinePolicies(data_path=DATA_PATH, assets_dir=ASSETS_DIR)

    policies_to_evaluate: Dict[str, object] = {}
    if not args.skip_random:
        policies_to_evaluate["1. Random"] = baselines.random_policy
    if not args.skip_rule:
        policies_to_evaluate["2. Rule-Based"] = baselines.rule_based_policy
    if not args.skip_greedy:
        policies_to_evaluate["3. Greedy XGBoost"] = baselines.greedy_xgboost_policy

    policies_to_evaluate.update(load_rl_models(args.include_rl, env))
    policies_to_evaluate.update(load_masked_rl_models(args.include_rl_masked, env))

    if not policies_to_evaluate:
        print("No policies selected for evaluation. Exiting.")
        return

    print(f"Running evaluation for {args.episodes} simulated days per policy...")
    results = []
    for policy_name, policy in policies_to_evaluate.items():
        print(f"\n--- Evaluating: {policy_name} ---")
        start_time = time.perf_counter()
        metrics = evaluate_policy(env, policy, n_episodes=args.episodes, base_seed=args.seed)
        metrics['policy'] = policy_name
        metrics['tag'] = args.tag
        elapsed = time.perf_counter() - start_time
        print(f"  > Evaluation finished in {elapsed:.2f} seconds.")
        results.append(metrics)

    print(f"\n--- FINAL RESULTS (Averaged over {args.episodes} days) ---")
    results_df = pd.DataFrame(results).set_index('policy')
    columns_order = [
        'avg_total_cost_per_day',
        'avg_cost_per_call',
        'avg_reward_per_day',
        'avg_calls_per_day',
        'tag',
    ]
    results_df = results_df[columns_order]
    run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    results_df.insert(0, 'run_timestamp', run_timestamp)
    results_df = results_df.sort_values(by='avg_total_cost_per_day', ascending=True)
    print(results_df)

    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_header = not os.path.exists(output_path)
    results_df.to_csv(output_path, mode='a', header=write_header)
    print(f"\nResults appended to {output_path}")


if __name__ == '__main__':
    main()
