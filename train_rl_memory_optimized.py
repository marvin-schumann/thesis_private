import argparse
import os
import time
from typing import Optional

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# Import our custom environment
from call_center_env import CallCenterEnv

# --- 1. Configuration ---
DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'
ASSETS_DIR = 'models'
os.makedirs(ASSETS_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DQN and PPO agents for the NOS call-center environment (memory-optimized)."
    )
    parser.add_argument("--timesteps-dqn", type=int, default=None,
                        help="Total training timesteps for DQN. Defaults to 200000 (25000 if --quick-smoke).")
    parser.add_argument("--timesteps-ppo", type=int, default=None,
                        help="Total training timesteps for PPO. Defaults to 200000 (25000 if --quick-smoke).")
    parser.add_argument("--quick-smoke", action="store_true",
                        help="Shortcut for a fast end-to-end run (~25k steps per agent).")
    parser.add_argument("--skip-dqn", action="store_true", help="Skip DQN training.")
    parser.add_argument("--skip-ppo", action="store_true", help="Skip PPO training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint-freq", type=int, default=None,
                        help="Save checkpoint every N steps. Defaults to 50000 (5000 if --quick-smoke).")
    parser.add_argument("--device", type=str, default="auto",
                        help="Computation device for SB3 models (e.g., 'cpu', 'cuda', 'auto').")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging (./rl_tensorboard_logs).")
    return parser.parse_args()


def resolve_training_budget(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.quick_smoke:
        timesteps_dqn = args.timesteps_dqn if args.timesteps_dqn is not None else 25_000
        timesteps_ppo = args.timesteps_ppo if args.timesteps_ppo is not None else 25_000
        checkpoint_freq = args.checkpoint_freq if args.checkpoint_freq is not None else 5_000
    else:
        timesteps_dqn = args.timesteps_dqn if args.timesteps_dqn is not None else 500_000
        timesteps_ppo = args.timesteps_ppo if args.timesteps_ppo is not None else 500_000
        checkpoint_freq = args.checkpoint_freq if args.checkpoint_freq is not None else 50_000
    return timesteps_dqn, timesteps_ppo, checkpoint_freq


def make_callback(save_path: str, checkpoint_freq: int, prefix: str) -> Optional[CheckpointCallback]:
    if checkpoint_freq <= 0:
        return None
    os.makedirs(save_path, exist_ok=True)
    return CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix=prefix
    )


def train_dqn_memory_optimized(env, timesteps: int, callback: Optional[CheckpointCallback],
                               save_path: str, tensorboard_path: Optional[str],
                               verbose: int = 1, device: str = "auto") -> float:
    """
    Train DQN with reduced buffer size to avoid memory issues.

    Original buffer size can use 6.89GB, this reduces it to ~2GB.
    """
    start_time = time.time()

    # Memory-optimized DQN configuration
    model = DQN(
        "MlpPolicy",
        env,
        verbose=verbose,
        tensorboard_log=tensorboard_path,
        device=device,
        # Reduce buffer size from default 1M to 200k (saves ~80% memory)
        buffer_size=200_000,
        # Reduce batch size from 32 to 64 for more stable learning with smaller buffer
        batch_size=64,
        # Start learning earlier since buffer is smaller
        learning_starts=1000,
        # More frequent target network updates for smaller buffer
        target_update_interval=500,
        # Keep other defaults
        learning_rate=1e-4,
        # Discount factor γ = 0.99
        # With ~600 calls/day, this gives effective horizon of 100 steps (1/(1-γ))
        # Balances immediate costs with medium-term agent availability planning
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )

    print(f"DQN Training Config (Memory-Optimized):")
    print(f"  Buffer size: 200,000 (vs default 1,000,000)")
    print(f"  Batch size: 64")
    print(f"  Estimated memory: ~2GB (vs ~7GB)")

    learn_kwargs = {
        "total_timesteps": timesteps,
        "progress_bar": True,
    }
    if callback is not None:
        learn_kwargs["callback"] = callback

    model.learn(**learn_kwargs)
    model.save(save_path)
    elapsed_minutes = (time.time() - start_time) / 60.0
    return elapsed_minutes


def train_ppo(env, timesteps: int, callback: Optional[CheckpointCallback],
              save_path: str, tensorboard_path: Optional[str],
              verbose: int = 1, device: str = "auto") -> float:
    """
    Train PPO (no memory issues since it doesn't use replay buffer).
    """
    start_time = time.time()

    # Discount factor γ = 0.99 (Stable-Baselines3 default for PPO)
    # Same justification as DQN: effective horizon of ~100 steps balances
    # immediate routing decisions with medium-term agent availability
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        tensorboard_log=tensorboard_path,
        device=device,
        # gamma=0.99,  # Using default
    )

    learn_kwargs = {
        "total_timesteps": timesteps,
        "progress_bar": True,
    }
    if callback is not None:
        learn_kwargs["callback"] = callback

    model.learn(**learn_kwargs)
    model.save(save_path)
    elapsed_minutes = (time.time() - start_time) / 60.0
    return elapsed_minutes


def main():
    args = parse_args()
    timesteps_dqn, timesteps_ppo, checkpoint_freq = resolve_training_budget(args)
    set_random_seed(args.seed)
    np.random.seed(args.seed)

    tensorboard_root = "./rl_tensorboard_logs" if args.tensorboard else None
    if tensorboard_root:
        os.makedirs(tensorboard_root, exist_ok=True)

    print("Initializing the Call Center Environment...")
    env = CallCenterEnv(data_path=DATA_PATH, assets_dir=ASSETS_DIR)
    env.reset(seed=args.seed)
    print(f"Environment initialized: {env.num_agents} agents, {len(env.call_indices)} call samples.")

    # --- DQN Training (Memory-Optimized) ---
    if not args.skip_dqn:
        print("\n" + "=" * 60)
        print("TRAINING DQN (Memory-Optimized)")
        print("=" * 60)
        dqn_callback = make_callback(
            save_path=os.path.join(ASSETS_DIR, "checkpoints_dqn"),
            checkpoint_freq=checkpoint_freq,
            prefix="dqn_checkpoint"
        )
        dqn_save_path = os.path.join(ASSETS_DIR, "rl_model_dqn")
        elapsed_dqn = train_dqn_memory_optimized(
            env=env,
            timesteps=timesteps_dqn,
            callback=dqn_callback,
            save_path=dqn_save_path,
            tensorboard_path=tensorboard_root,
            verbose=1,
            device=args.device
        )
        print(f"DQN training completed in {elapsed_dqn:.2f} minutes.")
        print(f"DQN model saved to: {dqn_save_path}.zip")

    # --- PPO Training ---
    if not args.skip_ppo:
        print("\n" + "=" * 60)
        print("TRAINING PPO")
        print("=" * 60)
        ppo_callback = make_callback(
            save_path=os.path.join(ASSETS_DIR, "checkpoints_ppo"),
            checkpoint_freq=checkpoint_freq,
            prefix="ppo_checkpoint"
        )
        ppo_save_path = os.path.join(ASSETS_DIR, "rl_model_ppo")
        elapsed_ppo = train_ppo(
            env=env,
            timesteps=timesteps_ppo,
            callback=ppo_callback,
            save_path=ppo_save_path,
            tensorboard_path=tensorboard_root,
            verbose=1,
            device=args.device
        )
        print(f"PPO training completed in {elapsed_ppo:.2f} minutes.")
        print(f"PPO model saved to: {ppo_save_path}.zip")

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    if not args.skip_dqn and not args.skip_ppo:
        total_time = elapsed_dqn + elapsed_ppo if not args.skip_dqn else elapsed_ppo
        print(f"Total training time: {total_time:.2f} minutes.")
    print("Next steps:")
    print("  1. Run evaluation: python evaluate_policies.py --episodes 3 --include-rl --tag 'after_bugfix'")
    print("  2. Check TensorBoard (if enabled): tensorboard --logdir=./rl_tensorboard_logs")


if __name__ == '__main__':
    main()
