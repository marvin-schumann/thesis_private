#!/usr/bin/env python3
"""
Diagnostic script to understand why RL agents handle so few calls.
"""

import numpy as np
from stable_baselines3 import DQN, PPO
from call_center_env import CallCenterEnv

DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'

def diagnose_policy(model_path, model_class, name):
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {name}")
    print('='*60)

    env = CallCenterEnv(data_path=DATA_PATH)
    model = model_class.load(model_path)
    model.set_env(env)

    obs, info = env.reset(seed=42)
    done = False
    step = 0

    success_count = 0
    invalid_count = 0
    abandon_count = 0
    truncated_count = 0

    status_history = []

    while not done and step < 10000:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

        status = info.get('status', 'unknown')
        status_history.append(status)

        if status == 'success':
            success_count += 1
        elif status == 'invalid_action_agent_busy_or_off_shift':
            invalid_count += 1
        elif status == 'call_abandoned':
            abandon_count += 1
        elif status == 'truncated_max_steps_reached':
            truncated_count += 1

        if step % 1000 == 0:
            print(f'Step {step}: success={success_count}, invalid={invalid_count}, '
                  f'abandon={abandon_count}, time={env.current_time:.0f}s')

    print(f'\n=== FINAL STATS ===')
    print(f'Episode ended after {step} steps')
    print(f'Successful calls: {success_count}')
    print(f'Invalid actions: {invalid_count}')
    print(f'Abandoned calls: {abandon_count}')
    print(f'Truncated: {truncated_count}')
    print(f'Final time: {env.current_time:.0f}s / {env.simulation_day_length:.0f}s')
    print(f'\nReason episode ended: {"TIME LIMIT" if env.current_time >= env.simulation_day_length else "MAX STEPS" if step >= 10000 else "OTHER"}')

    # Show status breakdown
    print(f'\n=== STATUS BREAKDOWN ===')
    from collections import Counter
    status_counts = Counter(status_history)
    for status, count in status_counts.most_common():
        print(f'{status}: {count} ({100*count/len(status_history):.1f}%)')

    # Calculate expected vs actual
    expected_calls = 590  # Approximate
    efficiency = success_count / expected_calls * 100 if expected_calls > 0 else 0
    print(f'\n=== EFFICIENCY ===')
    print(f'Expected calls: ~{expected_calls}')
    print(f'Actual calls: {success_count}')
    print(f'Efficiency: {efficiency:.1f}%')

    return success_count, invalid_count, abandon_count

if __name__ == '__main__':
    print("Diagnosing RL Agent Behavior")
    print("=" * 60)

    try:
        dqn_success, dqn_invalid, dqn_abandon = diagnose_policy('models/rl_model_dqn.zip', DQN, 'DQN')
    except Exception as e:
        print(f"Error with DQN: {e}")

    try:
        ppo_success, ppo_invalid, ppo_abandon = diagnose_policy('models/rl_model_ppo.zip', PPO, 'PPO')
    except Exception as e:
        print(f"Error with PPO: {e}")
