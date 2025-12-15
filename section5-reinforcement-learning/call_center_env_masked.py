"""
Action-Masked version of CallCenterEnv

This version implements action_masks() method to restrict RL agents
to only select from available agents, preventing invalid action loops.

Based on: call_center_env.py with action masking added
"""

import gymnasium
from gymnasium import spaces
import numpy as np

# Import the base environment
from call_center_env import CallCenterEnv


class CallCenterEnvMasked(CallCenterEnv):
    """
    Action-masked version of CallCenterEnv.

    Adds action_masks() method that returns a binary mask indicating
    which agents are currently available (1) or unavailable (0).

    This is compatible with sb3-contrib's MaskablePPO.
    """

    def action_masks(self):
        """
        Return a binary mask indicating valid actions.

        Returns:
            np.ndarray: Binary mask where 1 = valid action, 0 = invalid action
                        Shape: (num_agents,)

        Note: MaskablePPO requires at least one valid action. If all agents
        are unavailable (busy or off-shift), we unmask all agents to allow
        the RL agent to make a choice (the environment will handle the
        unavailability by making the agent wait).
        """
        mask = self._get_agent_availability()

        # Safety check: ensure at least one action is valid
        if np.sum(mask) == 0:
            # All agents unavailable - unmask all to prevent distribution error
            mask = np.ones_like(mask)

        return mask

    def reset(self, seed=None, options=None):
        """Reset and return initial observation + info with action mask."""
        obs, info = super().reset(seed=seed, options=options)
        # Add action mask to info (some implementations expect this)
        info['action_mask'] = self.action_masks()
        return obs, info

    def step(self, action):
        """Step function - identical to base but adds mask to info."""
        obs, reward, done, truncated, info = super().step(action)
        # Add current action mask to info
        info['action_mask'] = self.action_masks()
        return obs, reward, done, truncated, info


if __name__ == '__main__':
    # Test the masked environment
    print("Testing Action-Masked Call Center Environment...")

    DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'

    env = CallCenterEnvMasked(data_path=DATA_PATH)

    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Check action mask
    mask = env.action_masks()
    print(f"\nAction mask shape: {mask.shape}")
    print(f"Available agents: {np.sum(mask)} / {len(mask)}")
    print(f"Percentage available: {100*np.sum(mask)/len(mask):.1f}%")

    # Test a few steps with random VALID actions
    print("\nTesting 10 steps with masked random actions...")
    for i in range(10):
        mask = env.action_masks()
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            print(f"Step {i+1}: No valid actions available!")
            break

        # Select random valid action
        action = np.random.choice(valid_actions)
        obs, reward, done, truncated, info = env.step(action)

        status = info.get('status', 'unknown')
        print(f"Step {i+1}: action={action}, status={status}, reward={reward:.2f}, "
              f"available={np.sum(mask)}")

        if done or truncated:
            print(f"Episode ended: done={done}, truncated={truncated}")
            break

    print("\n✓ Action masking test complete")
