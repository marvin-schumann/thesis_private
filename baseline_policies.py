import numpy as np
import pandas as pd
import joblib
import json
import os
import random
import warnings

from residual_adjustments import ResidualAdjuster


class BaselinePolicies:
    """
    Holds the logic for the baseline policies (Rule-Based and Greedy XGBoost)
    that the RL agent will be compared against.
    """
    
    def __init__(self, data_path, assets_dir='models', test_indices_path=None):
        print("Loading baseline policy assets...")
        self.assets_dir = assets_dir
        
        # --- 1. Load All Assets (Oracles & Data) ---
        self.model_tmc = joblib.load(os.path.join(self.assets_dir, 'model_tmc.joblib'))
        self.model_ftr = joblib.load(os.path.join(self.assets_dir, 'model_ftr.joblib'))
        self.model_ot = joblib.load(os.path.join(self.assets_dir, 'model_ot.joblib'))

        self.scaler_tmc = joblib.load(os.path.join(self.assets_dir, 'scaler_tmc.joblib'))
        self.scaler_ftr = joblib.load(os.path.join(self.assets_dir, 'scaler_ftr.joblib'))
        self.scaler_ot = joblib.load(os.path.join(self.assets_dir, 'scaler_ot.joblib'))
        self.scaled_cols_tmc = list(getattr(self.scaler_tmc, 'feature_names_in_', []))
        self.scaled_cols_ftr = list(getattr(self.scaler_ftr, 'feature_names_in_', []))
        self.scaled_cols_ot = list(getattr(self.scaler_ot, 'feature_names_in_', []))

        residuals_path = os.path.join(self.assets_dir, 'gc_residuals_summary.csv')
        self.residual_adjuster = None
        if os.path.exists(residuals_path):
            try:
                self.residual_adjuster = ResidualAdjuster(residuals_path)
                print(f"Loaded residual adjustments from {residuals_path}.")
            except Exception as exc:
                print(f"Warning: Failed to load residual adjustments ({exc}). Continuing without biases.")
        else:
            print(f"Warning: Residual adjustment file not found at {residuals_path}.")
        
        with open(os.path.join(self.assets_dir, 'feature_lists.json'), 'r') as f:
            self.feature_lists = json.load(f)
            
        self.features_tmc = self.feature_lists['tmc']
        self.features_ftr = self.feature_lists['ftr']
        self.features_ot = self.feature_lists['ot']

        self.full_data = pd.read_csv(data_path)

        # Filter to test set if indices provided
        if test_indices_path is not None:
            if os.path.exists(test_indices_path):
                test_indices = np.load(test_indices_path)
                self.full_data = self.full_data.loc[self.full_data.index.isin(test_indices)]
                print(f"✓ Baseline policies filtered to test set: {len(self.full_data):,} samples")
            else:
                raise FileNotFoundError(f"Test indices file not found: {test_indices_path}")
        else:
            print("⚠️  WARNING: Baseline policies using FULL dataset")

        data_dir = os.path.dirname(data_path)
        print(f"Loaded merged dataset for policies from {data_path} with shape {self.full_data.shape}.")

        self.topic_stats = None
        topic_stats_path = os.path.join(self.assets_dir, 'topic_stats.csv')
        if os.path.exists(topic_stats_path):
            try:
                self.topic_stats = pd.read_csv(topic_stats_path)
                print(f"Loaded topic statistics from {topic_stats_path}.")
            except Exception as exc:
                print(f"Warning: failed to load topic statistics ({exc}). Continuing without them.")

        # Attempt to load GCS features from gcs_unique.csv in the same directory
        self.gcs_data = None
        gcs_path = os.path.join(data_dir, "gcs_unique.csv")
        if os.path.exists(gcs_path):
            try:
                gcs_df = pd.read_csv(gcs_path)
                if "RESOURCE_KEY" in gcs_df.columns:
                    self.gcs_data = gcs_df.set_index("RESOURCE_KEY")
                    print(f"Loaded GCS features for policies from {gcs_path} with shape {self.gcs_data.shape}.")
                else:
                    print(f"gcs_unique.csv at {gcs_path} missing 'RESOURCE_KEY'. Using fallback agents.")
            except Exception as exc:
                print(f"Failed to load GCS data from {gcs_path}: {exc}. Using fallback agents.")
        else:
            print(f"No gcs_unique.csv found alongside {data_path}. Using fallback agents.")
        
        # Create a clean list of all agent keys and their features
        if self.gcs_data is not None and not self.gcs_data.empty:
            self.agent_data_df = self.gcs_data.copy()
            self.agent_data_df.index = self.agent_data_df.index.astype(str)
            self.agent_keys = self.agent_data_df.index.tolist()
            print(f"Using {len(self.agent_keys)} agents from gcs_unique.csv.")
        elif 'RESOURCE_KEY' in self.full_data.columns:
            unique_resources = self.full_data['RESOURCE_KEY'].dropna().unique()
            if len(unique_resources) > 0:
                self.agent_keys = [str(res) for res in unique_resources[:50]]  # Limit to 50 agents
                self.agent_data_df = pd.DataFrame(index=self.agent_keys)
                print(f"No GCS data available. Created {len(self.agent_keys)} dummy agents based on RESOURCE_KEY values.")
            else:
                self.agent_keys = [f"agent_{i}" for i in range(10)]
                self.agent_data_df = pd.DataFrame(index=self.agent_keys)
                print("No RESOURCE_KEY found. Created 10 dummy agents for testing.")
        else:
            self.agent_keys = [f"agent_{i}" for i in range(10)]
            self.agent_data_df = pd.DataFrame(index=self.agent_keys)
            print("No agent data found. Created 10 dummy agents for testing.")
        
        self.num_agents = len(self.agent_keys)

        # Create a clean list of all call/client features
        call_cols = [col for col in self.full_data.columns if col.startswith('call_') or col.startswith('client_')]
        self.all_call_features = sorted(list(set(call_cols)))
        self.call_feature_df = self.full_data[call_cols].copy()
        topic_feature_cols = self._augment_call_features_with_topic_stats()
        if topic_feature_cols:
            self.all_call_features = sorted(list(set(self.all_call_features + topic_feature_cols)))
        print(f"Prepared policy call feature dataframe with shape {self.call_feature_df.shape}.")
        
        # --- 2. Load Rule-Based Model Averages ---
        # (This is a simplified version of the logic from the PBL report)
        print("Calculating rule-based averages...")
        self.rule_based_averages = {}
        # Assuming 'call_TOPIC_CLASSIFIC_ENTRY_AT_FT' is the key categorical feature
        # We also need the target variables to calculate these averages
        # Try different possible column names for targets
        possible_targets = {
            'tmc': ['call_LEG_DURATION_SEC_QTY'],
            'ftr': ['call_FTR_depen', 'call_FTR_CALCULATED', 'call_FTR_1_SUM'],
            'ot': ['call_FLAG_OT']
        }
        
        # Find which target columns actually exist
        available_targets = {}
        for target_type, col_names in possible_targets.items():
            for col_name in col_names:
                if col_name in self.full_data.columns:
                    available_targets[target_type] = col_name
                    break
        
        if 'call_TOPIC_CLASSIFIC_ENTRY_AT_FT' in self.full_data.columns and len(available_targets) > 0:
            # One-hot encode the topic to get separate columns
            topic_dummies = pd.get_dummies(self.full_data['call_TOPIC_CLASSIFIC_ENTRY_AT_FT'], prefix='topic')
            target_cols = [v for v in available_targets.values()]
            temp_df = pd.concat([self.full_data[target_cols], topic_dummies], axis=1)
            
            for topic_col in topic_dummies.columns:
                topic_calls = temp_df[temp_df[topic_col] == 1]
                if not topic_calls.empty:
                    avg_dict = {}
                    if 'tmc' in available_targets:
                        avg_dict['tmc'] = topic_calls[available_targets['tmc']].mean()
                    if 'ftr' in available_targets:
                        avg_dict['ftr_prob'] = topic_calls[available_targets['ftr']].mean()
                    if 'ot' in available_targets:
                        avg_dict['ot_prob'] = topic_calls[available_targets['ot']].mean()
                    if avg_dict:
                        self.rule_based_averages[topic_col] = avg_dict
            print("Rule-based averages calculated.")
        else:
            print("Warning: Required columns for rule-based averages not found. Rule-based model will be random.")
        
        # Global defaults used if topic-specific averages unavailable
        self.default_rule_average = {
            'tmc': None,
            'ftr_prob': None,
            'ot_prob': None
        }
        # Minimum calls required to trust agent-level averages
        self.rule_min_calls_overall = 20
        self.rule_min_calls_topic = 5
        if 'tmc' in available_targets:
            self.default_rule_average['tmc'] = float(self.full_data[available_targets['tmc']].mean())
        if 'ftr' in available_targets:
            self.default_rule_average['ftr_prob'] = float(self.full_data[available_targets['ftr']].mean())
        if 'ot' in available_targets:
            self.default_rule_average['ot_prob'] = float(self.full_data[available_targets['ot']].mean())
        # Fallback constants if still None
        if self.default_rule_average['tmc'] is None:
            self.default_rule_average['tmc'] = 600.0
        if self.default_rule_average['ftr_prob'] is None:
            self.default_rule_average['ftr_prob'] = 0.6
        if self.default_rule_average['ot_prob'] is None:
            self.default_rule_average['ot_prob'] = 0.1

        print("Baseline policies initialized.")

    def _augment_call_features_with_topic_stats(self):
        additional_cols = []
        if self.topic_stats is None:
            return additional_cols

        topic_col = 'call_TOPIC_CLASSIFIC_ENTRY_AT_FT'
        if topic_col not in self.call_feature_df.columns:
            print("Warning: topic statistics loaded but topic column missing in call features.")
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

    def _build_feature_vector(self, call_data, agent_key, feature_list):
        """ Helper to construct the 1D feature vector for a model. """
        # Handle empty agent_data_df (dummy agents)
        if len(self.agent_data_df.columns) == 0:
            agent_features = {}
        else:
            agent_features = self.agent_data_df.loc[agent_key].to_dict()
        
        combined_data = {**call_data, **agent_features}
        combined_series = pd.Series(combined_data)
        feature_values = combined_series.reindex(feature_list, fill_value=0)
        feature_values = pd.to_numeric(feature_values, errors='coerce').fillna(0.0)
        return feature_values.to_frame().T.astype(np.float32)

    def _determine_topic_suffix(self, call_data):
        topic_cols = [col for col, val in call_data.items()
                      if col.startswith('call_TOPIC_CLASSIFIC_ENTRY_AT_FT_') and val == 1]
        if topic_cols:
            suffix = topic_cols[0].split('call_TOPIC_CLASSIFIC_ENTRY_AT_FT_')[1]
            return suffix
        return None

    def _get_agent_features(self, agent_key):
        if len(self.agent_data_df.columns) == 0:
            return {}
        return self.agent_data_df.loc[agent_key].to_dict()

    def _estimate_rule_based_cost(self, call_data, agent_key):
        topic_suffix = self._determine_topic_suffix(call_data)
        topic_key = f"topic_{topic_suffix}" if topic_suffix else None
        topic_avg = self.rule_based_averages.get(topic_key, self.default_rule_average)

        agent_features = self._get_agent_features(agent_key)

        overall_call_count = float(agent_features.get('gc_COUNT_CALLS', 0)) if agent_features else 0.0
        topic_call_count = 0.0
        if topic_suffix and agent_features:
            topic_call_count = float(agent_features.get(f'gc_COUNT_CALLS_{topic_suffix}', 0))

        use_topic_stats = topic_suffix and topic_call_count >= self.rule_min_calls_topic
        use_overall_stats = overall_call_count >= self.rule_min_calls_overall

        def _get_value(agent_dict, base_name, default_topic, default_global):
            # topic-specific metric
            if use_topic_stats and agent_dict:
                topic_column = f"{base_name}_{topic_suffix}"
                if topic_column in agent_dict:
                    val = agent_dict[topic_column]
                    if val is not None and not pd.isna(val) and float(val) > 0:
                        return float(val)
            # overall metric
            if use_overall_stats and agent_dict and base_name in agent_dict:
                val = agent_dict[base_name]
                if val is not None and not pd.isna(val) and float(val) > 0:
                    return float(val)
            # fall back to topic average first, then global default
            if default_topic is not None:
                return float(default_topic)
            return float(default_global)

        mean_tmc = _get_value(
            agent_features,
            'gc_MEAN_TMC',
            topic_avg.get('tmc'),
            self.default_rule_average['tmc']
        )
        mean_ftr = _get_value(
            agent_features,
            'gc_MEAN_FTR',
            topic_avg.get('ftr_prob'),
            self.default_rule_average['ftr_prob']
        )
        ot_prob = _get_value(
            agent_features,
            'gc_OTS_BY_CALL',
            topic_avg.get('ot_prob'),
            self.default_rule_average['ot_prob']
        )

        # Ensure sensible bounds
        mean_tmc = max(30.0, float(mean_tmc))
        mean_ftr = float(np.clip(mean_ftr, 0.0, 1.0))
        ot_prob = float(np.clip(ot_prob, 0.0, 1.0))

        cost_duration = (mean_tmc / 60.0) * 0.35
        cost_repeat = (1.0 - mean_ftr) * (mean_tmc / 60.0) * 0.35
        cost_ot = ot_prob * 22.0
        total_cost = cost_duration + cost_repeat + cost_ot
        return float(total_cost)

    def _get_oracle_predictions(self, call_data, agent_key):
        """ Uses the saved XGBoost models to predict outcomes. """
        vec_tmc_df = self._build_feature_vector(call_data, agent_key, self.features_tmc)
        vec_tmc_scaled_df = vec_tmc_df.copy()
        if self.scaled_cols_tmc:
            vec_tmc_scaled_df[self.scaled_cols_tmc] = self.scaler_tmc.transform(vec_tmc_df[self.scaled_cols_tmc])
        pred_tmc = max(30.0, float(self.model_tmc.predict(vec_tmc_scaled_df)[0]))

        vec_ftr_df = self._build_feature_vector(call_data, agent_key, self.features_ftr)
        vec_ftr_scaled_df = vec_ftr_df.copy()
        if self.scaled_cols_ftr:
            vec_ftr_scaled_df[self.scaled_cols_ftr] = self.scaler_ftr.transform(vec_ftr_df[self.scaled_cols_ftr])
        pred_ftr_prob = self.model_ftr.predict_proba(vec_ftr_scaled_df)[0][1]

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

        return self._apply_residual_biases(pred_tmc, pred_ftr_prob, pred_ot_prob, call_data, agent_key)

    def _apply_residual_biases(self, pred_tmc, pred_ftr_prob, pred_ot_prob, call_data, agent_key):
        if self.residual_adjuster is None:
            return pred_tmc, pred_ftr_prob, pred_ot_prob

        topic_value = call_data.get('call_TOPIC_CLASSIFIC_ENTRY_AT_FT')
        adjustments = self.residual_adjuster.get_adjustments(agent_key, topic_value)

        adjusted_tmc = max(30.0, float(pred_tmc + adjustments['tmc'].bias))
        adjusted_ftr = float(np.clip(pred_ftr_prob + adjustments['ftr'].bias, 0.0, 1.0))
        adjusted_ot = float(np.clip(pred_ot_prob + adjustments['ot'].bias, 0.0, 1.0))
        return adjusted_tmc, adjusted_ftr, adjusted_ot

    def _calculate_cost(self, tmc, ftr_prob, ot_prob):
        """ Calculates the cost of a call. """
        cost_per_minute = 0.35
        cost_per_ot = 22.0
        tmc_pop = tmc 
        
        cost_duration = (tmc / 60.0) * cost_per_minute
        cost_repeat = (1.0 - ftr_prob) * (tmc_pop / 60.0) * cost_per_minute
        cost_ot = ot_prob * cost_per_ot
        
        total_cost = cost_duration + cost_repeat + cost_ot
        return total_cost

    def _get_call_data_from_obs(self, observation):
        """ Extracts the call_data dict from the flat observation vector. """
        call_features_len = len(self.all_call_features)
        call_features_vector = observation[:call_features_len]
        
        # Reconstruct the call_data dict
        call_data = {self.all_call_features[i]: call_features_vector[i] for i in range(call_features_len)}
        return call_data

    def _get_available_agents(self, observation):
        """ Extracts the indices of available agents from the observation. """
        call_features_len = len(self.all_call_features)
        agent_status_len = self.num_agents
        
        agent_status_vector = observation[call_features_len : call_features_len + agent_status_len]
        
        available_agent_indices = np.where(agent_status_vector == 1.0)[0]
        return available_agent_indices

    # --- PUBLIC POLICY FUNCTIONS ---

    def greedy_xgboost_policy(self, observation):
        """
        The Greedy XGBoost Policy.
        Selects the available agent with the minimum predicted cost from the XGBoost models.
        """
        call_data = self._get_call_data_from_obs(observation)
        available_agent_indices = self._get_available_agents(observation)
 
        if len(available_agent_indices) == 0:
            # No agents available, this shouldn't happen if env logic is correct
            # but as a fallback, just pick a random agent (will be penalized by env)
            return random.randint(0, self.num_agents - 1)
 
        try:
            best_agent_index = self._greedy_xgboost_vectorized(call_data, available_agent_indices)
        except Exception as exc:
            warnings.warn(f"Vectorized greedy evaluation failed ({exc}); falling back to loop.")
            best_agent_index = self._greedy_xgboost_loop(call_data, available_agent_indices)

        return best_agent_index

    def _greedy_xgboost_loop(self, call_data, available_agent_indices):
        best_agent_index = -1
        min_cost = float('inf')

        for agent_index in available_agent_indices:
            agent_key = self.agent_keys[agent_index]

            pred_tmc, pred_ftr, pred_ot = self._get_oracle_predictions(call_data, agent_key)
            cost = self._calculate_cost(pred_tmc, pred_ftr, pred_ot)

            if cost < min_cost:
                min_cost = cost
                best_agent_index = agent_index

        return best_agent_index

    def _greedy_xgboost_vectorized(self, call_data, available_agent_indices):
        if len(available_agent_indices) == 1:
            return available_agent_indices[0]

        agent_keys = [self.agent_keys[idx] for idx in available_agent_indices]
        agent_features_df = self.agent_data_df.loc[agent_keys].copy() if len(self.agent_data_df.columns) else pd.DataFrame(index=agent_keys)

        call_series = pd.Series(call_data)
        call_repeated_df = pd.DataFrame([call_series.values] * len(agent_keys), columns=call_series.index, index=agent_keys)

        combined_df = pd.concat([call_repeated_df, agent_features_df], axis=1)

        vec_tmc_df = combined_df.reindex(columns=self.features_tmc, fill_value=0.0).astype(np.float32)
        vec_tmc_scaled_df = vec_tmc_df.copy()
        if self.scaled_cols_tmc:
            vec_tmc_scaled_df[self.scaled_cols_tmc] = self.scaler_tmc.transform(vec_tmc_df[self.scaled_cols_tmc])
        tmc_preds = self.model_tmc.predict(vec_tmc_scaled_df)

        vec_ftr_df = combined_df.reindex(columns=self.features_ftr, fill_value=0.0).astype(np.float32)
        vec_ftr_scaled_df = vec_ftr_df.copy()
        if self.scaled_cols_ftr:
            vec_ftr_scaled_df[self.scaled_cols_ftr] = self.scaler_ftr.transform(vec_ftr_df[self.scaled_cols_ftr])
        ftr_probs = self.model_ftr.predict_proba(vec_ftr_scaled_df)[:, 1]

        vec_ot_df = combined_df.reindex(columns=self.features_ot, fill_value=0.0).astype(np.float32)
        vec_ot_scaled_df = vec_ot_df.copy()
        if self.scaled_cols_ot:
            vec_ot_scaled_df[self.scaled_cols_ot] = self.scaler_ot.transform(vec_ot_df[self.scaled_cols_ot])
        ot_proba_matrix = self.model_ot.predict_proba(vec_ot_scaled_df)
        ot_probs = ot_proba_matrix[:, 1] if ot_proba_matrix.shape[1] == 2 else ot_proba_matrix[:, 0]

        tmc_preds = np.maximum(tmc_preds, 30.0).astype(float)
        ftr_probs = np.clip(ftr_probs, 0.0, 1.0)
        ot_probs = np.clip(ot_probs, 0.0, 1.0)

        costs = []
        for tmc, ftr_prob, ot_prob in zip(tmc_preds, ftr_probs, ot_probs):
            cost_duration = (tmc / 60.0) * 0.35
            cost_repeat = (1.0 - ftr_prob) * (tmc / 60.0) * 0.35
            cost_ot = ot_prob * 22.0
            total_cost = cost_duration + cost_repeat + cost_ot
            costs.append(total_cost)

        min_idx = int(np.argmin(costs))
        return available_agent_indices[min_idx]

    def rule_based_policy(self, observation):
        """
        The Rule-Based Policy.
        Estimates the cost for each available agent using historical averages
        and selects the agent with the minimum estimated cost.
        """
        call_data = self._get_call_data_from_obs(observation)
        available_agent_indices = self._get_available_agents(observation)

        if len(available_agent_indices) == 0:
            return random.randint(0, self.num_agents - 1)
        
        best_agent_index = None
        best_estimated_cost = float('inf')
        for agent_index in available_agent_indices:
            agent_key = self.agent_keys[agent_index]
            estimated_cost = self._estimate_rule_based_cost(call_data, agent_key)
            if estimated_cost < best_estimated_cost:
                best_estimated_cost = estimated_cost
                best_agent_index = agent_index

        if best_agent_index is None:
            return random.choice(available_agent_indices)

        return best_agent_index

    def naive_topic_only_policy(self, observation):
        """
        Naive Topic-Only Baseline (matching group work modelling).

        Uses ONLY call topic averages, completely ignoring agent characteristics.
        This replicates the baseline from the group work phase that achieved:
        - 41% pairwise accuracy
        - R² = -0.0885 for TMC prediction

        Since all agents get the same cost estimate for a given topic,
        we simply select a random available agent (no differentiation possible).

        This is fundamentally different from rule_based_policy which uses
        agent-specific historical performance.
        """
        call_data = self._get_call_data_from_obs(observation)
        available_agent_indices = self._get_available_agents(observation)

        if len(available_agent_indices) == 0:
            return random.randint(0, self.num_agents - 1)

        # Get topic-only averages (same for all agents)
        topic_suffix = self._determine_topic_suffix(call_data)
        topic_key = f"topic_{topic_suffix}" if topic_suffix else None
        topic_avg = self.rule_based_averages.get(topic_key, self.default_rule_average)

        # Calculate cost using ONLY topic statistics (no agent differentiation)
        mean_tmc = max(30.0, float(topic_avg.get('tmc', self.default_rule_average['tmc'])))
        mean_ftr = float(np.clip(topic_avg.get('ftr_prob', self.default_rule_average['ftr_prob']), 0.0, 1.0))
        ot_prob = float(np.clip(topic_avg.get('ot_prob', self.default_rule_average['ot_prob']), 0.0, 1.0))

        # Since all agents have the same estimated cost for this topic,
        # we cannot differentiate between them → select randomly from available
        # (This mimics the group work approach where agent selection was poor)
        return random.choice(available_agent_indices)

    def rule_based_conservative_policy(self, observation):
        """
        Conservative Rule-Based Policy with higher minimum call thresholds.

        Uses the same hierarchical lookup logic as rule_based_policy, but with
        stricter thresholds to avoid small sample bias:
        - rule_min_calls_topic: 50 (vs 5 in standard)
        - rule_min_calls_overall: 100 (vs 20 in standard)

        This prevents selection of agents with unreliable statistics based on
        only a few calls, addressing the small sample bias identified in diagnostics.
        """
        call_data = self._get_call_data_from_obs(observation)
        available_agent_indices = self._get_available_agents(observation)

        if len(available_agent_indices) == 0:
            return random.randint(0, self.num_agents - 1)

        # Temporarily override thresholds for conservative evaluation
        original_topic_threshold = self.rule_min_calls_topic
        original_overall_threshold = self.rule_min_calls_overall

        self.rule_min_calls_topic = 50
        self.rule_min_calls_overall = 100

        best_agent_index = None
        best_estimated_cost = float('inf')
        for agent_index in available_agent_indices:
            agent_key = self.agent_keys[agent_index]
            estimated_cost = self._estimate_rule_based_cost(call_data, agent_key)
            if estimated_cost < best_estimated_cost:
                best_estimated_cost = estimated_cost
                best_agent_index = agent_index

        # Restore original thresholds
        self.rule_min_calls_topic = original_topic_threshold
        self.rule_min_calls_overall = original_overall_threshold

        if best_agent_index is None:
            return random.choice(available_agent_indices)

        return best_agent_index

    def random_policy(self, observation):
        """
        The pure Random Policy.
        Selects a random agent from all *available* agents.
        """
        available_agent_indices = self._get_available_agents(observation)
        
        if len(available_agent_indices) == 0:
            return random.randint(0, self.num_agents - 1)
            
        return random.choice(available_agent_indices)


if __name__ == '__main__':
    # This block is for testing the baseline policies
    print("Testing the Baseline Policies...")
    
    DATA_PATH = '/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/full_merged_df.csv'
    
    # We need an environment to get a sample observation
    from call_center_env import CallCenterEnv
    env = CallCenterEnv(data_path=DATA_PATH)
    obs, info = env.reset()
    
    # Initialize the policies
    policies = BaselinePolicies(data_path=DATA_PATH)
    
    # Test each policy
    greedy_action = policies.greedy_xgboost_policy(obs)
    print(f"\nGreedy XGBoost Policy selected agent index: {greedy_action}")
    
    rule_action = policies.rule_based_policy(obs)
    print(f"Rule-Based Policy selected agent index: {rule_action}")

    random_action = policies.random_policy(obs)
    print(f"Random Policy selected agent index: {random_action}")

    # Check that the actions are valid (i.e., the agent is available)
    available_agents = policies._get_available_agents(obs)
    print(f"\nAvailable agent indices: {available_agents}")
    print(f"Is Greedy action valid? {greedy_action in available_agents}")
    print(f"Is Rule-Based action valid? {rule_action in available_agents}")
    print(f"Is Random action valid? {random_action in available_agents}")

