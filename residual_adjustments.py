#!/usr/bin/env python3
"""Utilities for loading and applying residual bias and noise adjustments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

METRICS = ("tmc", "ftr", "ot")


@dataclass(frozen=True)
class ResidualStat:
    bias: float
    std: float
    count: int


class ResidualAdjuster:
    """Encapsulates residual statistics and provides fallbacks for missing data."""

    def __init__(
        self,
        csv_path: Path,
        agent_min_count: int = 100,
        topic_min_count: int = 400,
        agent_bias_shrinkage: float = 200.0,
        topic_bias_shrinkage: float = 500.0,
        global_bias_shrinkage: float = 0.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Residual summary CSV not found at {self.csv_path}")

        self.agent_min_count = agent_min_count
        self.topic_min_count = topic_min_count
        self.bias_shrinkage = {
            "agent": max(0.0, agent_bias_shrinkage),
            "topic": max(0.0, topic_bias_shrinkage),
            "global": max(0.0, global_bias_shrinkage),
        }

        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise ValueError(f"Residual summary CSV at {self.csv_path} is empty")

        self.agent_df = df[df["level"] == "agent"].set_index("key")
        self.topic_df = df[df["level"] == "topic"].set_index("key")

        global_df = df[df["level"] == "global"].set_index("key")
        if global_df.empty:
            raise ValueError("Residual summary missing global baseline row")
        self.global_row = global_df.iloc[0]

    def _extract_stats_from_row(self, row: pd.Series, level: str) -> Dict[str, ResidualStat]:
        stats: Dict[str, ResidualStat] = {}
        for metric in METRICS:
            bias = float(row.get(f"{metric}_bias_mean", 0.0))
            std = float(row.get(f"{metric}_residual_std", 0.0))
            count = int(row.get(f"{metric}_count", 0))
            shrinkage = self.bias_shrinkage.get(level, 0.0)
            if shrinkage > 0.0:
                shrink_factor = min(1.0, count / (count + shrinkage))
            else:
                shrink_factor = 1.0
            bias *= shrink_factor
            std *= math.sqrt(shrink_factor) if shrink_factor < 1.0 else 1.0
            stats[metric] = ResidualStat(bias=bias, std=max(0.0, std), count=count)
        return stats

    def _select_row(self, agent_key: Optional[str], topic: Optional[str]) -> Tuple[pd.Series, str]:
        # Prefer agent-level stats if count threshold met
        if agent_key and agent_key in self.agent_df.index:
            agent_row = self.agent_df.loc[agent_key]
            if agent_row.get("tmc_count", 0) >= self.agent_min_count:
                return agent_row, "agent"
        else:
            agent_row = None

        # Fallback to topic-level if available and sufficient data
        if topic and topic in self.topic_df.index:
            topic_row = self.topic_df.loc[topic]
            if topic_row.get("tmc_count", 0) >= self.topic_min_count:
                return topic_row, "topic"

        # If agent exists but did not meet threshold and topic missing, we still retain agent row as next best option
        if agent_row is not None:
            return agent_row, "agent"

        return self.global_row, "global"

    def get_adjustments(self, agent_key: Optional[str], topic: Optional[str]) -> Dict[str, ResidualStat]:
        row, level = self._select_row(agent_key, topic)
        return self._extract_stats_from_row(row, level)

    def apply_bias(self, predictions: Dict[str, float], agent_key: Optional[str], topic: Optional[str]) -> Dict[str, float]:
        stats = self.get_adjustments(agent_key, topic)
        adjusted: Dict[str, float] = {}
        for metric, value in predictions.items():
            metric_stats = stats.get(metric)
            bias = metric_stats.bias if metric_stats else 0.0
            adjusted[metric] = value + bias
        return adjusted

    def sample_noise(self, agent_key: Optional[str], topic: Optional[str], rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        stats = self.get_adjustments(agent_key, topic)
        generator = rng if rng is not None else np.random.default_rng()
        noise: Dict[str, float] = {}
        for metric, metric_stats in stats.items():
            if metric_stats.std > 0.0:
                noise[metric] = float(generator.normal(0.0, metric_stats.std))
            else:
                noise[metric] = 0.0
        return noise
