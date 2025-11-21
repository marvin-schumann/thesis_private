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
        description="Train DQN and PPO agents for the NOS call-center environment."
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


def train_agent(model_cls, env, timesteps: int, callback: Optional[CheckpointCallback], save_path: str,
                tensorboard_path: Optional[str], verbose: int = 1, device: str = "auto") -> float:
    start_time = time.time()

    # Discount factor γ = 0.99 (Stable-Baselines3 default)
    # Justification: With ~600 calls per 8-hour day, rewards 100 steps away are
    # weighted at 0.99^100 ≈ 0.37. This reflects operational reality: immediate
    # routing decisions matter more than distant future states, but we still
    # consider medium-term consequences. Effective horizon: 1/(1-γ) ≈ 100 steps.
    model = model_cls(
        "MlpPolicy",
        env,
        verbose=verbose,
        tensorboard_log=tensorboard_path,
        device=device,
        # gamma=0.99,  # Using default (explicitly: gamma is 0.99 for both DQN and PPO)
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
    print("Environment initialized.")

    if not args.skip_dqn:
        print("\n--- Training DQN Agent ---")
        dqn_checkpoint_callback = make_callback(
            save_path=os.path.join(ASSETS_DIR, 'checkpoints_dqn'),
            checkpoint_freq=checkpoint_freq,
            prefix='dqn_model'
        )
        elapsed = train_agent(
            DQN,
            env,
            timesteps_dqn,
            dqn_checkpoint_callback,
            save_path=os.path.join(ASSETS_DIR, 'rl_model_dqn.zip'),
            tensorboard_path=os.path.join(tensorboard_root, "dqn") if tensorboard_root else None,
            device=args.device,
        )
        print(f"DQN training finished in {elapsed:.2f} minutes.")
    else:
        print("\n--- Skipping DQN training (--skip-dqn) ---")

    if not args.skip_ppo:
        print("\n--- Training PPO Agent ---")
        ppo_checkpoint_callback = make_callback(
            save_path=os.path.join(ASSETS_DIR, 'checkpoints_ppo'),
            checkpoint_freq=checkpoint_freq,
            prefix='ppo_model'
        )
        elapsed = train_agent(
            PPO,
            env,
            timesteps_ppo,
            ppo_checkpoint_callback,
            save_path=os.path.join(ASSETS_DIR, 'rl_model_ppo.zip'),
            tensorboard_path=os.path.join(tensorboard_root, "ppo") if tensorboard_root else None,
            device=args.device,
        )
        print(f"PPO training finished in {elapsed:.2f} minutes.")
    else:
        print("\n--- Skipping PPO training (--skip-ppo) ---")

    print("\n--- All Training Complete ---")
    if not args.skip_dqn:
        print(f"DQN model saved to: {os.path.join(ASSETS_DIR, 'rl_model_dqn.zip')}")
    if not args.skip_ppo:
        print(f"PPO model saved to: {os.path.join(ASSETS_DIR, 'rl_model_ppo.zip')}")


if __name__ == "__main__":
    main()
