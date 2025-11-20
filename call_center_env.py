import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib
import json
import os
import random
import logging

from residual_adjustments import ResidualAdjuster

logger = logging.getLogger(__name__)


class CallCenterEnv(gymnasium.Env):
    """
    A custom Gymnasium environment for simulating the NOS call center routing problem.

    **State (Observation):**
    A 1D vector containing:
    1.  Call Features: A flattened vector of the current call's features.
    2.  Agent Availability: A binary vector (1=free, 0=busy) for all agents.
    3.  Queue/Time Features: `[queue_length, current_hour_of_day]`

    **Action:**
    A discrete integer representing the *index* of the chosen agent from `self.agent_keys`.
    """
    
    def __init__(self, data_path, assets_dir='models'):
        super().__init__()

        # --- 1. Load All Assets (Oracles & Data) ---
        self.assets_dir = assets_dir
        
        # Load models
        self.model_tmc = joblib.load(os.path.join(self.assets_dir, 'model_tmc.joblib'))
        self.model_ftr = joblib.load(os.path.join(self.assets_dir, 'model_ftr.joblib'))
        self.model_ot = joblib.load(os.path.join(self.assets_dir, 'model_ot.joblib'))

        # Load scalers
        self.scaler_tmc = joblib.load(os.path.join(self.assets_dir, 'scaler_tmc.joblib'))
        self.scaler_ftr = joblib.load(os.path.join(self.assets_dir, 'scaler_ftr.joblib'))
        self.scaler_ot = joblib.load(os.path.join(self.assets_dir, 'scaler_ot.joblib'))
        self.scaled_cols_tmc = list(getattr(self.scaler_tmc, 'feature_names_in_', []))
        self.scaled_cols_ftr = list(getattr(self.scaler_ftr, 'feature_names_in_', []))
        self.scaled_cols_ot = list(getattr(self.scaler_ot, 'feature_names_in_', []))

        # Load residual adjustment statistics
        residuals_path = os.path.join(self.assets_dir, 'gc_residuals_summary.csv')
        self.residual_adjuster = None
        if os.path.exists(residuals_path):
            try:
                self.residual_adjuster = ResidualAdjuster(residuals_path)
                logger.info(f"Loaded residual adjustments from {residuals_path}.")
            except Exception as exc:
                logger.warning(f"Failed to load residual adjustments from {residuals_path}: {exc}")
        else:
            logger.warning(f"No residual adjustment file found at {residuals_path}. Proceeding without bias corrections.")
        
        topic_stats_path = os.path.join(self.assets_dir, 'topic_stats.csv')
        self.topic_stats = None
        if os.path.exists(topic_stats_path):
            try:
                self.topic_stats = pd.read_csv(topic_stats_path)
                logger.info(f"Loaded topic statistics from {topic_stats_path}.")
            except Exception as exc:
                logger.warning(f"Failed to load topic statistics from {topic_stats_path}: {exc}")

        with open(os.path.join(self.assets_dir, 'feature_lists.json'), 'r') as f:
            self.feature_lists = json.load(f)
            
        self.features_tmc = self.feature_lists['tmc']
        self.features_ftr = self.feature_lists['ftr']
        self.features_ot = self.feature_lists['ot']

        # Load the raw data for sampling calls and agents
        logger.info(f"Loading merged dataset from {data_path}...")
        self.full_data = pd.read_csv(data_path)
        logger.info(f"Full dataset loaded with shape {self.full_data.shape}.")
        data_dir = os.path.dirname(data_path)
        # Initialize RNG for reproducibility
        self._rng = np.random.default_rng()

        # Attempt to load GCS features from gcs_unique.csv in the same directory
        self.gcs_data = None
        gcs_path = os.path.join(data_dir, "gcs_unique.csv")
        if os.path.exists(gcs_path):
            try:
                gcs_df = pd.read_csv(gcs_path)
                if "RESOURCE_KEY" in gcs_df.columns:
                    self.gcs_data = gcs_df.set_index("RESOURCE_KEY")
                    logger.info(f"Loaded GCS features from {gcs_path} with shape {self.gcs_data.shape}.")
                else:
                    logger.warning(f"Found gcs_unique.csv at {gcs_path} but no 'RESOURCE_KEY' column. Using dummy agents.")
            except Exception as exc:
                logger.warning(f"Failed to load GCS data from {gcs_path}: {exc}. Using dummy agents.")
        else:
            logger.warning(f"No gcs_unique.csv found alongside {data_path}. Using dummy agents.")
        
        # --- 2. Prepare Simulation Data ---
        
        # Create a clean list of all agent keys and their features
        if self.gcs_data is not None and not self.gcs_data.empty:
            self.agent_data_df = self.gcs_data.copy()
            self.agent_data_df.index = self.agent_data_df.index.astype(str)
            self.agent_keys = self.agent_data_df.index.tolist()
            logger.info(f"Using {len(self.agent_keys)} agents from gcs_unique.csv.")
        elif 'RESOURCE_KEY' in self.full_data.columns:
            unique_resources = self.full_data['RESOURCE_KEY'].dropna().unique()
            if len(unique_resources) > 0:
                limited = min(50, len(unique_resources))
                self.agent_keys = [str(res) for res in unique_resources[:limited]]
                self.agent_data_df = pd.DataFrame(index=self.agent_keys)
                logger.warning(f"No GCS data available. Created {len(self.agent_keys)} dummy agents based on RESOURCE_KEY values.")
            else:
                self.agent_keys = [f"agent_{i}" for i in range(10)]
                self.agent_data_df = pd.DataFrame(index=self.agent_keys)
                logger.warning("No RESOURCE_KEY found. Created 10 dummy agents for testing.")
        else:
            self.agent_keys = [f"agent_{i}" for i in range(10)]
            self.agent_data_df = pd.DataFrame(index=self.agent_keys)
            logger.warning("No agent data found. Created 10 dummy agents for testing.")

        self.num_agents = len(self.agent_keys)
        logger.info(f"Total agents configured: {self.num_agents}.")
        
        # Prepare call/client feature dataframe for sampling without materializing all rows as dicts
        call_cols = [col for col in self.full_data.columns if col.startswith('call_') or col.startswith('client_')]
        self.call_feature_df = self.full_data[call_cols].copy()
        topic_feature_cols = self._augment_call_features_with_topic_stats()
        if topic_feature_cols:
            call_cols = sorted(list(set(call_cols + topic_feature_cols)))
        logger.info(f"Prepared call feature dataframe with shape {self.call_feature_df.shape}.")
        self.call_indices = self.call_feature_df.index.to_numpy()
        self.all_call_features = sorted(list(set(call_cols)))

        # Shift definitions (start_hour, end_hour, target_count)
        self.shifts = [
            (0, 8, 250),   # single 8-hour shift covering the active window
        ]
        
        # --- 3. Define Environment Spaces ---
        
        # Action: Choose one agent (by index)
        self.action_space = spaces.Discrete(self.num_agents)
        
        # Observation Space: [call_features, agent_availability, queue_features]
        call_features_len = len(self.all_call_features)
        agent_status_len = self.num_agents
        queue_features_len = 2 # [queue_length, current_hour_of_day]
        
        obs_space_len = call_features_len + agent_status_len + queue_features_len
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_len,), dtype=np.float32
        )
        
        # --- 4. Simulation Parameters ---
        # Simulation time is in seconds. Here we simulate an 8-hour day.
        self.current_time = 0.0 
        self.simulation_day_length = 8 * 60 * 60 
        
        # Call arrival rates (calls per hour) for our Time-Dependent Poisson Process
        self.hourly_arrival_rates = {
            0: 50, 1: 80, 2: 100, 3: 120, 4: 100, 5: 80, 6: 60, 7: 40
        }

        # Waiting behaviour and abandonment controls
        self.wait_penalty_per_second = 0.02       # € cost per second a call waits
        self.idle_penalty_per_second = 0.05       # € cost per second spent idling/invalid actions
        self.abandonment_threshold = 600.0        # seconds (10 minutes)
        self.abandonment_penalty = 500.0          # € penalty when a call abandons
        self.invalid_action_penalty = 100.0       # base penalty for picking a busy/off agent
        self.invalid_action_wait_seconds = 60.0   # seconds to advance time on invalid action
        # self.call_completion_bonus = 200.0      # REMOVED: Reward is now pure cost minimization (-cost)
        self.abandoned_calls = 0

    def _augment_call_features_with_topic_stats(self):
        additional_cols = []
        if self.topic_stats is None:
            return additional_cols

        topic_col = 'call_TOPIC_CLASSIFIC_ENTRY_AT_FT'
        if topic_col not in self.call_feature_df.columns:
            logger.warning("Topic statistics loaded but topic column not present in call features.")
            return additional_cols

        stats_map = self.topic_stats.set_index(topic_col)
        if '__GLOBAL__' in stats_map.index:
            global_stats = stats_map.loc['__GLOBAL__']
        else:
            global_stats = stats_map.mean(numeric_only=True)

        for col in ['call_TOPIC_AVG_TMC', 'call_TOPIC_AVG_FTR', 'call_TOPIC_AVG_OT', 'call_TOPIC_COUNT']:
            if col not in stats_map.columns:
                continue
            values = self.call_feature_df[topic_col].map(stats_map[col])
            fill_value = float(global_stats[col]) if col in global_stats else 0.0
            self.call_feature_df[col] = values.fillna(fill_value).astype(float)
            additional_cols.append(col)

        return additional_cols

    def _get_next_call_arrival(self):
        """
        Samples the time until the next call arrival using a Time-Dependent Poisson Process.
        """
        current_hour = int((self.current_time % self.simulation_day_length) / 3600)
        rate_per_hour = self.hourly_arrival_rates.get(current_hour, 10) # 10 as default
        rate_per_second = rate_per_hour / 3600

        # Sample from an exponential distribution (inter-arrival time of a Poisson process)
        # Use self._rng for consistent random number generation
        rng = getattr(self, "_rng", np.random.default_rng())
        time_until_next_call = rng.exponential(1.0 / rate_per_second)
        return self.current_time + time_until_next_call

    def _sample_call(self):
        """
        Samples the next call from our historical data following the prepared order.
        """
        if not hasattr(self, "_call_order") or len(self._call_order) == 0:
            raise ValueError("Call order not initialized. Did you forget to reset the environment?")

        if self._call_pointer >= len(self._call_order):
            # Reshuffle for subsequent cycles to avoid repetition bias
            self._call_order = self._rng.permutation(self.call_indices)
            self._call_pointer = 0

        idx = self._call_order[self._call_pointer]
        self._call_pointer += 1
        return self.call_feature_df.loc[idx].to_dict()
        
    def _assign_shifts(self):
        """
        Assigns all agents to one of the defined shifts for the day.
        Returns a dict of {agent_key: (start_time, end_time)}
        """
        agent_shifts = {}
        agents_array = np.array(self.agent_keys, dtype=object)
        self._rng.shuffle(agents_array)
        agents_to_assign = list(agents_array)

        day_start_time = int(self.current_time / self.simulation_day_length) * self.simulation_day_length

        assigned_count = 0
        for shift_idx, (start_hour, end_hour, target_count) in enumerate(self.shifts):
            count_for_shift = min(target_count, len(agents_to_assign) - assigned_count)
            for _ in range(count_for_shift):
                if assigned_count >= len(agents_to_assign):
                    break
                agent_key = agents_to_assign[assigned_count]
                agent_shifts[agent_key] = (
                    day_start_time + start_hour * 3600,
                    day_start_time + end_hour * 3600
                )
                assigned_count += 1

        # Any remaining agents (if total counts < num_agents) are assigned to the last shift
        while assigned_count < len(agents_to_assign):
            agent_key = agents_to_assign[assigned_count]
            start_hour, end_hour, _ = self.shifts[-1]
            agent_shifts[agent_key] = (
                day_start_time + start_hour * 3600,
                day_start_time + end_hour * 3600
            )
            assigned_count += 1

        return agent_shifts

    def _get_agent_availability(self):
        """
        Gets the current availability of all agents as a binary vector.
        1 = free, 0 = busy or off-shift.
        """
        availability_vector = np.zeros(self.num_agents, dtype=np.float32)
        for i, agent_key in enumerate(self.agent_keys):
            # Is agent on shift?
            shift_start, shift_end = self.agent_shifts[agent_key]
            is_on_shift = (self.current_time >= shift_start) and (self.current_time < shift_end)
            
            # Is agent not busy with another call?
            is_not_busy = (self.agent_available_at[agent_key] <= self.current_time)
            
            if is_on_shift and is_not_busy:
                availability_vector[i] = 1.0
        return availability_vector

    def _get_observation(self):
        """
        Constructs the full observation vector for the current state.
        """
        # 1. Call Features
        call_features_vector = np.zeros(len(self.all_call_features), dtype=np.float32)
        for i, feature_name in enumerate(self.all_call_features):
            value = self.current_call.get(feature_name, 0.0)
            # Convert to float, handling strings and other types
            try:
                if isinstance(value, (int, float, np.number)):
                    call_features_vector[i] = float(value)
                elif isinstance(value, str):
                    # Try to convert string to float, if fails use 0.0
                    try:
                        call_features_vector[i] = float(value)
                    except (ValueError, TypeError):
                        call_features_vector[i] = 0.0
                else:
                    call_features_vector[i] = 0.0
            except (ValueError, TypeError):
                call_features_vector[i] = 0.0
        
        # 2. Agent Availability
        agent_availability_vector = self._get_agent_availability()
        
        # 3. Queue/Time Features
        # For simplicity, we assume a queue of 1 (the current call)
        # and no other calls are waiting. This can be expanded later.
        queue_length = 1.0 
        current_hour = (self.current_time % self.simulation_day_length) / 3600.0
        queue_features_vector = np.array([queue_length, current_hour], dtype=np.float32)
        
        # Concatenate all parts into one flat vector
        observation = np.concatenate([
            call_features_vector,
            agent_availability_vector,
            queue_features_vector
        ])
        
        # Replace any NaN or Inf values with 0.0
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        
        return observation.astype(np.float32)

    def _build_feature_vector(self, call_data, agent_key, feature_list):
        """
        Helper to construct the 1D feature vector for a model.
        """
        # Handle empty agent_data_df (dummy agents)
        if len(self.agent_data_df.columns) == 0:
            agent_features = {}
        else:
            agent_features = self.agent_data_df.loc[agent_key].to_dict()
        
        # Combine data sources
        combined_data = {**call_data, **agent_features}
        
        # Build the vector aligned with the feature list
        combined_series = pd.Series(combined_data)
        feature_values = combined_series.reindex(feature_list, fill_value=0)
        feature_values = pd.to_numeric(feature_values, errors='coerce').fillna(0.0)
        
        # Return as single-row DataFrame aligned with feature_list
        return feature_values.to_frame().T.astype(np.float32)

    def _get_oracle_predictions(self, call_data, agent_key):
        """
        Uses the saved XGBoost models to predict outcomes.
        """
        # TMC
        vec_tmc_df = self._build_feature_vector(call_data, agent_key, self.features_tmc)
        vec_tmc_scaled_df = vec_tmc_df.copy()
        if self.scaled_cols_tmc:
            vec_tmc_scaled_df[self.scaled_cols_tmc] = self.scaler_tmc.transform(vec_tmc_df[self.scaled_cols_tmc])
        pred_tmc = self.model_tmc.predict(vec_tmc_scaled_df)[0]
        # Ensure TMC is not negative and has a minimum (e.g., 30 seconds)
        pred_tmc = max(30.0, float(pred_tmc)) 

        # FTR
        vec_ftr_df = self._build_feature_vector(call_data, agent_key, self.features_ftr)
        vec_ftr_scaled_df = vec_ftr_df.copy()
        if self.scaled_cols_ftr:
            vec_ftr_scaled_df[self.scaled_cols_ftr] = self.scaler_ftr.transform(vec_ftr_df[self.scaled_cols_ftr])
        pred_ftr_prob = self.model_ftr.predict_proba(vec_ftr_scaled_df)[0][1] # Prob of class 1

        # OT
        vec_ot_df = self._build_feature_vector(call_data, agent_key, self.features_ot)
        vec_ot_scaled_df = vec_ot_df.copy()
        if self.scaled_cols_ot:
            vec_ot_scaled_df[self.scaled_cols_ot] = self.scaler_ot.transform(vec_ot_df[self.scaled_cols_ot])
        ot_proba = self.model_ot.predict_proba(vec_ot_scaled_df)[0]
        # Handle case where model only has one class (DummyClassifier)
        if len(ot_proba) == 2:
            pred_ot_prob = ot_proba[1]  # Prob of class 1
        else:
            pred_ot_prob = float(ot_proba[0])  # Use the single class probability

        return self._apply_residual_adjustments(pred_tmc, pred_ftr_prob, pred_ot_prob, call_data, agent_key)

    def _apply_residual_adjustments(self, pred_tmc, pred_ftr_prob, pred_ot_prob, call_data, agent_key):
        """
        Applies residual bias and stochastic noise based on historical residual statistics.
        """
        if self.residual_adjuster is None:
            return pred_tmc, pred_ftr_prob, pred_ot_prob

        topic_value = call_data.get('call_TOPIC_CLASSIFIC_ENTRY_AT_FT')
        adjustments = self.residual_adjuster.get_adjustments(agent_key, topic_value)

        # Bias adjustments
        bias_tmc = adjustments['tmc'].bias
        bias_ftr = adjustments['ftr'].bias
        bias_ot = adjustments['ot'].bias

        adjusted_tmc = pred_tmc + bias_tmc
        adjusted_ftr = pred_ftr_prob + bias_ftr
        adjusted_ot = pred_ot_prob + bias_ot

        # Inject stochastic residual noise
        rng = getattr(self, "_rng", None)
        noise = self.residual_adjuster.sample_noise(agent_key, topic_value, rng=rng)
        adjusted_tmc += noise['tmc']
        adjusted_ftr += noise['ftr']
        adjusted_ot += noise['ot']

        # Enforce sensible bounds
        adjusted_tmc = max(30.0, float(adjusted_tmc))
        adjusted_ftr = float(np.clip(adjusted_ftr, 0.0, 1.0))
        adjusted_ot = float(np.clip(adjusted_ot, 0.0, 1.0))

        return adjusted_tmc, adjusted_ftr, adjusted_ot

    def _calculate_cost(self, tmc, ftr_prob, ot_prob):
        """
        Calculates the cost of a call based on the group's cost function.
        """
        cost_per_minute = 0.35
        cost_per_ot = 22.0
        
        # For TMC_pop, we use the predicted TMC as a proxy (as discussed)
        tmc_pop = tmc 
        
        cost_duration = (tmc / 60.0) * cost_per_minute
        cost_repeat = (1.0 - ftr_prob) * (tmc_pop / 60.0) * cost_per_minute
        cost_ot = ot_prob * cost_per_ot
        
        total_cost = cost_duration + cost_repeat + cost_ot
        return total_cost

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)
            random.seed(seed)
            np.random.seed(seed)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()

        # Prepare call order for this episode
        if len(self.call_indices) == 0:
            raise ValueError("No call indices available for sampling.")
        self._call_order = self._rng.permutation(self.call_indices)
        self._call_pointer = 0

        # Reset simulation time (e.g., start of a new day)
        self.current_time = 0.0

        # Reset step counter for episode truncation safety
        self.episode_step_count = 0
        self.max_steps_per_episode = 10000  # Safety limit to prevent infinite loops

        # Assign agents to shifts for this new day
        self.agent_shifts = self._assign_shifts()

        # Reset agent availability (all agents are free at time 0, but subject to shifts)
        self.agent_available_at = {agent_key: 0.0 for agent_key in self.agent_keys}
        self.abandoned_calls = 0

        # Set the first call
        self.current_time = self._get_next_call_arrival()
        self.current_call = self._sample_call()
        self.current_call_arrival_time = self.current_time
        self.current_call_wait_time = 0.0

        # Get first observation
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):

        self.episode_step_count += 1

        # Safety truncation
        if self.episode_step_count >= self.max_steps_per_episode:
            observation = self._get_observation()
            reward = -1000.0
            info = {'status': 'truncated_max_steps_reached', 'steps': self.episode_step_count}
            return observation, reward, True, True, info

        # Abandonment check before action
        if getattr(self, "current_call_wait_time", 0.0) >= self.abandonment_threshold:
            return self._handle_call_abandonment()

        chosen_agent_index = int(action)
        chosen_agent_key = self.agent_keys[chosen_agent_index]
        is_available = self._get_agent_availability()[chosen_agent_index] == 1.0

        if not is_available:
            # Invalid action: advance time and accumulate wait time
            # This simulates the call waiting in queue while agent tries again
            self.current_time += self.invalid_action_wait_seconds
            self.current_call_wait_time += self.invalid_action_wait_seconds

            # Calculate penalties for invalid action + idle time
            idle_penalty = self.invalid_action_wait_seconds * self.idle_penalty_per_second
            wait_penalty = self.invalid_action_wait_seconds * self.wait_penalty_per_second
            total_penalty = self.invalid_action_penalty + idle_penalty + wait_penalty
            reward = -total_penalty

            # Check if call should be abandoned after waiting
            if self.current_call_wait_time >= self.abandonment_threshold:
                return self._handle_call_abandonment()

            # Check if simulation day has ended
            done = self.current_time >= self.simulation_day_length

            observation = self._get_observation()
            info = {
                'status': 'invalid_action_agent_busy_or_off_shift',
                'wait_time': self.current_call_wait_time,
                'abandoned_calls': self.abandoned_calls,
                'penalty': total_penalty
            }
            return observation, reward, done, False, info

        # Valid action: handle the call
        pred_tmc, pred_ftr, pred_ot = self._get_oracle_predictions(self.current_call, chosen_agent_key)
        cost = self._calculate_cost(pred_tmc, pred_ftr, pred_ot)
        reward = -cost  # Pure cost minimization (no bonus)
        wait_time_served = getattr(self, "current_call_wait_time", 0.0)

        self.agent_available_at[chosen_agent_key] = self.current_time + pred_tmc

        self.current_time = self._get_next_call_arrival()
        self.current_call = self._sample_call()
        self.current_call_arrival_time = self.current_time
        self.current_call_wait_time = 0.0

        observation = self._get_observation()
        done = self.current_time >= self.simulation_day_length
        info = {
            'status': 'success',
            'cost': cost,
            'reward': reward,
            'pred_tmc': pred_tmc,
            'pred_ftr': pred_ftr,
            'pred_ot': pred_ot,
            'steps': self.episode_step_count,
            'wait_time': wait_time_served,
            'abandoned_calls': self.abandoned_calls
        }

        return observation, reward, done, False, info

    def _handle_call_abandonment(self):
        wait_time = getattr(self, "current_call_wait_time", 0.0)
        wait_penalty = wait_time * self.wait_penalty_per_second
        total_penalty = self.abandonment_penalty + wait_penalty
        reward = -total_penalty
        self.abandoned_calls += 1

        self.current_time = self._get_next_call_arrival()
        self.current_call = self._sample_call()
        self.current_call_arrival_time = self.current_time
        self.current_call_wait_time = 0.0

        observation = self._get_observation()
        done = self.current_time >= self.simulation_day_length
        info = {
            'status': 'call_abandoned',
            'penalty': total_penalty,
            'wait_time': wait_time,
            'abandoned_calls': self.abandoned_calls
        }
        return observation, reward, done, False, info


if __name__ == '__main__':
    # This block is for testing the environment
    print("Testing the custom Call Center Environment...")
    
    DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'
    
    env = CallCenterEnv(data_path=DATA_PATH)
    
    # Test the reset function
    print("Testing reset...")
    obs, info = env.reset()
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Initial observation (first 10 features): {obs[:10]}")
    
    # Test the step function with a random action
    print("\nTesting step...")
    random_action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(random_action)
    
    print(f"Action taken: {random_action}")
    print(f"Reward: {reward}")
    print(f"Info: {info}")
    print(f"New observation (first 10 features): {obs[:10]}")
    
    # Test a full episode
    print("\nTesting a full episode...")
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    while not done:
        action = env.action_space.sample() # Use a random agent for testing
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if done or truncated:
            break
            
    print(f"Episode finished after {step_count} steps.")
    print(f"Total reward: {total_reward}")

