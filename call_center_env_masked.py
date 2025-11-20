"""
Action-Masked Call Center Environment for Reinforcement Learning

This module extends the base CallCenterEnv with action masking support,
following best practices for constrained RL environments.

Action masking ensures RL agents only select from valid (available) agents,
improving sample efficiency and ensuring fair comparison with baseline policies.

References:
- Huang & Ontañón (2020): "A Closer Look at Invalid Action Masking"
- sb3-contrib documentation: https://sb3-contrib.readthedocs.io/
"""

import numpy as np
from call_center_env import CallCenterEnv


class CallCenterEnvMasked(CallCenterEnv):
    """
    Call Center Environment with Action Masking support.

    Extends the base CallCenterEnv to provide action masks that indicate
    which agents are currently available for selection. This prevents RL
    agents from wasting exploration on invalid actions.

    Usage with sb3-contrib:
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib import MaskablePPO

        def mask_fn(env):
            return env.action_masks()

        env = CallCenterEnvMasked(data_path=DATA_PATH)
        env = ActionMasker(env, mask_fn)
        model = MaskablePPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100000)
    """

    def __init__(self, data_path, assets_dir='models'):
        """Initialize the masked environment."""
        super().__init__(data_path, assets_dir)

        # Track masking statistics for analysis
        self.masking_stats = {
            'total_steps': 0,
            'avg_available_agents': 0.0,
            'min_available_agents': float('inf'),
            'max_available_agents': 0
        }

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask indicating which actions (agent selections) are valid.

        Returns:
            np.ndarray: Boolean array of shape (num_agents,) where True means the
                       agent is available (on shift and not busy), False otherwise.

        Example:
            masks = env.action_masks()
            # masks = [True, False, True, False, ...]
            # Only agents at index 0, 2, ... can be selected
        """
        availability_vector = self._get_agent_availability()

        # Convert to boolean mask (1.0 -> True, 0.0 -> False)
        mask = availability_vector == 1.0

        # Update statistics
        num_available = np.sum(mask)
        self.masking_stats['total_steps'] += 1
        self.masking_stats['avg_available_agents'] = (
            (self.masking_stats['avg_available_agents'] * (self.masking_stats['total_steps'] - 1) + num_available)
            / self.masking_stats['total_steps']
        )
        self.masking_stats['min_available_agents'] = min(
            self.masking_stats['min_available_agents'], num_available
        )
        self.masking_stats['max_available_agents'] = max(
            self.masking_stats['max_available_agents'], num_available
        )

        # Ensure at least one action is available (safety check)
        if not np.any(mask):
            # This should never happen in a well-designed environment,
            # but if all agents are busy, we need to handle it gracefully
            # In this case, we allow all actions and let the environment
            # handle the invalid action with penalties
            import warnings
            warnings.warn(
                "No valid agents available! This indicates a problem with "
                "shift scheduling or too many concurrent calls. Allowing all actions."
            )
            mask = np.ones(self.num_agents, dtype=bool)

        return mask

    def reset(self, seed=None, options=None):
        """Reset environment and masking statistics."""
        # Reset masking stats for new episode
        self.masking_stats = {
            'total_steps': 0,
            'avg_available_agents': 0.0,
            'min_available_agents': float('inf'),
            'max_available_agents': 0
        }

        return super().reset(seed=seed, options=options)

    def get_masking_stats(self):
        """
        Get statistics about action masking during the episode.

        Returns:
            dict: Statistics including average, min, and max available agents
        """
        return self.masking_stats.copy()


if __name__ == '__main__':
    # Test the masked environment
    print("Testing CallCenterEnvMasked...")

    DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'

    env = CallCenterEnvMasked(data_path=DATA_PATH)

    print(f"\nEnvironment initialized with {env.num_agents} agents.")

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")

    # Test action masking
    masks = env.action_masks()
    print(f"\nAction masks shape: {masks.shape}")
    print(f"Number of available agents: {np.sum(masks)}/{env.num_agents}")
    print(f"Percentage available: {100 * np.sum(masks) / env.num_agents:.1f}%")

    # Test a few steps with masked actions
    print("\nTesting masked action selection:")
    for step in range(5):
        masks = env.action_masks()
        available_indices = np.where(masks)[0]

        if len(available_indices) > 0:
            # Select random available agent
            action = np.random.choice(available_indices)
            obs, reward, done, truncated, info = env.step(action)

            print(f"  Step {step + 1}: Selected agent {action} (available), "
                  f"Reward: {reward:.2f}, Status: {info.get('status', 'unknown')}")
        else:
            print(f"  Step {step + 1}: No agents available!")
            break

        if done or truncated:
            break

    # Print masking statistics
    stats = env.get_masking_stats()
    print(f"\nMasking Statistics:")
    print(f"  Average available agents: {stats['avg_available_agents']:.1f}")
    print(f"  Min available agents: {stats['min_available_agents']}")
    print(f"  Max available agents: {stats['max_available_agents']}")

    print("\n✅ CallCenterEnvMasked test complete!")
