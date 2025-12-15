#!/usr/bin/env python3
"""Compute agent- and topic-level residual statistics for the call-center models.

This script loads the historical merged dataset together with the trained XGBoost
models and scalers, reproduces the oracle predictions for the historical calls,
and measures the residuals (actual - predicted) for each agent and topic. The
aggregated residual bias (mean) and dispersion (standard deviation) are written
out so the simulator can inject agent-specific adjustments and stochasticity.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_PATH = (
    "/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/"
    "PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/"
    "full_merged_df.csv"
)
DEFAULT_GCS_PATH = (
    "/Users/marvinschumann/Library/CloudStorage/OneDrive-SharedLibraries-NovaSBE/"
    "PBL - NOS (Consultants) - General/03 Data/01 Full Datasets/Cleaned Data/07052025/"
    "gcs_unique.csv"
)
DEFAULT_ASSETS_DIR = "models"
DEFAULT_OUTPUT_PATH = Path(DEFAULT_ASSETS_DIR) / "gc_residuals_summary.csv"

RESOURCE_KEY_COL = "RESOURCE_KEY"
TOPIC_COL = "call_TOPIC_CLASSIFIC_ENTRY_AT_FT"
METRICS = ("tmc", "ftr", "ot")


def configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_assets(assets_dir: Path) -> Tuple[object, object, object, object, object, object, Dict[str, List[str]]]:
    LOGGER.info("Loading models and scalers from %s", assets_dir)
    model_tmc = joblib.load(assets_dir / "model_tmc.joblib")
    model_ftr = joblib.load(assets_dir / "model_ftr.joblib")
    model_ot = joblib.load(assets_dir / "model_ot.joblib")

    scaler_tmc = joblib.load(assets_dir / "scaler_tmc.joblib")
    scaler_ftr = joblib.load(assets_dir / "scaler_ftr.joblib")
    scaler_ot = joblib.load(assets_dir / "scaler_ot.joblib")

    with open(assets_dir / "feature_lists.json", "r", encoding="utf-8") as f:
        feature_lists = json.load(f)

    return (
        model_tmc,
        model_ftr,
        model_ot,
        scaler_tmc,
        scaler_ftr,
        scaler_ot,
        feature_lists,
    )


def infer_target_columns(columns: Iterable[str]) -> Tuple[str, str, str]:
    columns_set = set(columns)

    tmc_col = "call_LEG_DURATION_SEC_QTY"
    if tmc_col not in columns_set:
        raise ValueError(f"Required TMC column '{tmc_col}' not found in dataset")

    ftr_candidates = [
        "call_FTR_CALCULATED",
        "call_FTR_depen",
        "call_FTR_1_SUM",
    ]
    ftr_col = next((col for col in ftr_candidates if col in columns_set), None)
    if ftr_col is None:
        raise ValueError(f"None of the FTR columns {ftr_candidates} found in dataset")

    ot_candidates = ["call_FLAG_OT", "call_OT_FLAG", "call_FLAG_OT_SUM"]
    ot_col = next((col for col in ot_candidates if col in columns_set), None)
    if ot_col is None:
        raise ValueError(f"None of the OT columns {ot_candidates} found in dataset")

    return tmc_col, ftr_col, ot_col


def prepare_feature_frame(
    raw_chunk: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Return a numeric DataFrame aligned with the provided feature list."""
    frame = raw_chunk.reindex(columns=feature_names, fill_value=0.0)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame.astype(np.float32)


def apply_scaler(df: pd.DataFrame, scaler: object, scaled_cols: Sequence[str]) -> pd.DataFrame:
    if not scaled_cols:
        return df
    scaled_df = df.copy()
    scaled_df.loc[:, scaled_cols] = scaler.transform(df[scaled_cols])
    return scaled_df


def extract_actuals(
    chunk: pd.DataFrame,
    tmc_col: str,
    ftr_col: str,
    ot_col: str,
    max_tmc: Optional[float],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    chunk = chunk.loc[chunk[RESOURCE_KEY_COL].notna()].copy()
    if chunk.empty:
        return chunk, chunk.index.to_series(), chunk.index.to_series(), chunk.index.to_series()

    chunk[RESOURCE_KEY_COL] = chunk[RESOURCE_KEY_COL].astype(str)

    # TMC
    actual_tmc = pd.to_numeric(chunk[tmc_col], errors="coerce")
    if max_tmc is not None:
        actual_tmc = actual_tmc.where(actual_tmc <= max_tmc)

    # Filter invalid rows early so subsequent series align
    valid_mask = actual_tmc.notna()
    chunk = chunk.loc[valid_mask].copy()
    actual_tmc = actual_tmc.loc[valid_mask]

    # FTR
    actual_ftr = pd.to_numeric(chunk[ftr_col], errors="coerce").fillna(0.0)
    if ftr_col == "call_FTR_CALCULATED":
        actual_ftr = (actual_ftr > 0).astype(float)
    else:
        actual_ftr = actual_ftr.astype(float)

    # OT
    actual_ot = pd.to_numeric(chunk[ot_col], errors="coerce").fillna(0.0)
    actual_ot = actual_ot.clip(lower=0.0, upper=1.0).astype(float)

    return chunk.reset_index(drop=True), actual_tmc.reset_index(drop=True), actual_ftr.reset_index(drop=True), actual_ot.reset_index(drop=True)


def predict_models(
    chunk_features: pd.DataFrame,
    model_tmc,
    model_ftr,
    model_ot,
    scaler_tmc,
    scaler_ftr,
    scaler_ot,
    features_tmc: Sequence[str],
    features_ftr: Sequence[str],
    features_ot: Sequence[str],
    scaled_cols_tmc: Sequence[str],
    scaled_cols_ftr: Sequence[str],
    scaled_cols_ot: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TMC predictions
    tmc_features = prepare_feature_frame(chunk_features, features_tmc)
    tmc_scaled = apply_scaler(tmc_features, scaler_tmc, scaled_cols_tmc)
    pred_tmc = model_tmc.predict(tmc_scaled)

    # FTR predictions (probability of class 1)
    ftr_features = prepare_feature_frame(chunk_features, features_ftr)
    ftr_scaled = apply_scaler(ftr_features, scaler_ftr, scaled_cols_ftr)
    ftr_prob_matrix = model_ftr.predict_proba(ftr_scaled)
    if ftr_prob_matrix.ndim == 2 and ftr_prob_matrix.shape[1] >= 2:
        pred_ftr = ftr_prob_matrix[:, 1]
    else:
        pred_ftr = ftr_prob_matrix.ravel()

    # OT predictions (probability of class 1)
    ot_features = prepare_feature_frame(chunk_features, features_ot)
    ot_scaled = apply_scaler(ot_features, scaler_ot, scaled_cols_ot)
    ot_prob_matrix = model_ot.predict_proba(ot_scaled)
    if ot_prob_matrix.ndim == 2 and ot_prob_matrix.shape[1] >= 2:
        pred_ot = ot_prob_matrix[:, 1]
    else:
        pred_ot = ot_prob_matrix.ravel()

    return pred_tmc.astype(float), pred_ftr.astype(float), pred_ot.astype(float)


def accumulate_statistics(
    residual_df: pd.DataFrame,
    agent_sum: pd.DataFrame,
    agent_sumsq: pd.DataFrame,
    agent_count: pd.Series,
    topic_sum: pd.DataFrame,
    topic_sumsq: pd.DataFrame,
    topic_count: pd.Series,
    global_sum: pd.Series,
    global_sumsq: pd.Series,
    global_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, int]:
    if residual_df.empty:
        return (
            agent_sum,
            agent_sumsq,
            agent_count,
            topic_sum,
            topic_sumsq,
            topic_count,
            global_sum,
            global_sumsq,
            global_count,
        )

    metric_cols = ["tmc_residual", "ftr_residual", "ot_residual"]
    metric_sq_cols = [f"{col}_sq" for col in metric_cols]

    # Agents
    agent_group = residual_df.groupby(RESOURCE_KEY_COL, observed=True)
    agent_sum_chunk = agent_group[metric_cols].sum().rename(columns={
        "tmc_residual": "tmc",
        "ftr_residual": "ftr",
        "ot_residual": "ot",
    })
    agent_sumsq_chunk = agent_group[metric_sq_cols].sum().rename(columns={
        "tmc_residual_sq": "tmc",
        "ftr_residual_sq": "ftr",
        "ot_residual_sq": "ot",
    })
    agent_count_chunk = agent_group.size()

    agent_sum = agent_sum.add(agent_sum_chunk, fill_value=0.0)
    agent_sumsq = agent_sumsq.add(agent_sumsq_chunk, fill_value=0.0)
    agent_count = agent_count.add(agent_count_chunk, fill_value=0).astype(int)

    # Topics
    topic_group = residual_df.groupby("topic", observed=True)
    topic_sum_chunk = topic_group[metric_cols].sum().rename(columns={
        "tmc_residual": "tmc",
        "ftr_residual": "ftr",
        "ot_residual": "ot",
    })
    topic_sumsq_chunk = topic_group[metric_sq_cols].sum().rename(columns={
        "tmc_residual_sq": "tmc",
        "ftr_residual_sq": "ftr",
        "ot_residual_sq": "ot",
    })
    topic_count_chunk = topic_group.size()

    topic_sum = topic_sum.add(topic_sum_chunk, fill_value=0.0)
    topic_sumsq = topic_sumsq.add(topic_sumsq_chunk, fill_value=0.0)
    topic_count = topic_count.add(topic_count_chunk, fill_value=0).astype(int)

    # Global aggregates
    global_sum = global_sum.add(residual_df[metric_cols].sum(), fill_value=0.0)
    global_sumsq = global_sumsq.add(residual_df[metric_sq_cols].sum(), fill_value=0.0)
    global_count += len(residual_df)

    return (
        agent_sum,
        agent_sumsq,
        agent_count,
        topic_sum,
        topic_sumsq,
        topic_count,
        global_sum,
        global_sumsq,
        global_count,
    )


def build_summary_rows(
    level: str,
    key_iterable: Iterable,
    sum_df: pd.DataFrame,
    sumsq_df: pd.DataFrame,
    count_series: pd.Series,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key in key_iterable:
        row: Dict[str, object] = {"level": level, "key": key}
        total_count = int(count_series.get(key, 0))

        for metric in METRICS:
            metric_sum = float(sum_df.loc[key, metric]) if key in sum_df.index else 0.0
            metric_sumsq = float(sumsq_df.loc[key, metric]) if key in sumsq_df.index else 0.0

            if total_count > 0:
                mean = metric_sum / total_count
                variance = max((metric_sumsq / total_count) - (mean ** 2), 0.0)
                std_dev = math.sqrt(variance)
            else:
                mean = 0.0
                std_dev = 0.0

            row[f"{metric}_bias_mean"] = float(mean)
            row[f"{metric}_residual_std"] = float(std_dev)
            row[f"{metric}_count"] = total_count

        rows.append(row)

    return rows


def finalise_statistics(
    agent_sum: pd.DataFrame,
    agent_sumsq: pd.DataFrame,
    agent_count: pd.Series,
    topic_sum: pd.DataFrame,
    topic_sumsq: pd.DataFrame,
    topic_count: pd.Series,
    global_sum: pd.Series,
    global_sumsq: pd.Series,
    global_count: int,
) -> pd.DataFrame:
    agent_keys = agent_count.index.tolist()
    topic_keys = topic_count.index.tolist()

    rows: List[Dict[str, object]] = []
    rows.extend(build_summary_rows("agent", agent_keys, agent_sum, agent_sumsq, agent_count))
    rows.extend(build_summary_rows("topic", topic_keys, topic_sum, topic_sumsq, topic_count))

    # Global row
    global_row: Dict[str, object] = {"level": "global", "key": "ALL"}
    for metric, metric_col, metric_sq_col in zip(
        METRICS,
        ["tmc_residual", "ftr_residual", "ot_residual"],
        ["tmc_residual_sq", "ftr_residual_sq", "ot_residual_sq"],
    ):
        metric_sum = float(global_sum.get(metric_col, 0.0))
        metric_sumsq = float(global_sumsq.get(metric_sq_col, 0.0))
        if global_count > 0:
            mean = metric_sum / global_count
            variance = max((metric_sumsq / global_count) - (mean ** 2), 0.0)
            std_dev = math.sqrt(variance)
        else:
            mean = 0.0
            std_dev = 0.0
        global_row[f"{metric}_bias_mean"] = float(mean)
        global_row[f"{metric}_residual_std"] = float(std_dev)
        global_row[f"{metric}_count"] = int(global_count)

    rows.append(global_row)

    summary_df = pd.DataFrame(rows)
    summary_df.sort_values(by=["level", "key"], inplace=True)
    return summary_df.reset_index(drop=True)


def compute_residual_summary(
    data_path: Path,
    gcs_path: Optional[Path],
    assets_dir: Path,
    output_path: Path,
    chunksize: int,
    max_tmc: Optional[float],
) -> Path:
    (
        model_tmc,
        model_ftr,
        model_ot,
        scaler_tmc,
        scaler_ftr,
        scaler_ot,
        feature_lists,
    ) = load_assets(assets_dir)

    features_tmc = feature_lists["tmc"]
    features_ftr = feature_lists["ftr"]
    features_ot = feature_lists["ot"]

    scaled_cols_tmc = list(getattr(scaler_tmc, "feature_names_in_", []))
    scaled_cols_ftr = list(getattr(scaler_ftr, "feature_names_in_", []))
    scaled_cols_ot = list(getattr(scaler_ot, "feature_names_in_", []))

    LOGGER.info("Reading merged dataset from %s", data_path)

    agent_sum = pd.DataFrame(columns=METRICS, dtype=float)
    agent_sumsq = pd.DataFrame(columns=METRICS, dtype=float)
    agent_count = pd.Series(dtype=int)

    topic_sum = pd.DataFrame(columns=METRICS, dtype=float)
    topic_sumsq = pd.DataFrame(columns=METRICS, dtype=float)
    topic_count = pd.Series(dtype=int)

    global_sum = pd.Series(dtype=float)
    global_sumsq = pd.Series(dtype=float)
    global_count = 0

    tmc_col = ftr_col = ot_col = None

    reader = pd.read_csv(data_path, chunksize=chunksize)
    total_rows = 0

    for chunk_idx, chunk in enumerate(reader, start=1):
        if chunk.empty:
            continue

        if tmc_col is None:
            tmc_col, ftr_col, ot_col = infer_target_columns(chunk.columns)
            LOGGER.info(
                "Using target columns: TMC=%s, FTR=%s, OT=%s", tmc_col, ftr_col, ot_col
            )

        chunk, actual_tmc, actual_ftr, actual_ot = extract_actuals(
            chunk, tmc_col, ftr_col, ot_col, max_tmc
        )
        if chunk.empty:
            continue

        chunk_features = chunk.copy()  # Preserve non-feature columns for grouping later

        pred_tmc, pred_ftr, pred_ot = predict_models(
            chunk_features,
            model_tmc,
            model_ftr,
            model_ot,
            scaler_tmc,
            scaler_ftr,
            scaler_ot,
            features_tmc,
            features_ftr,
            features_ot,
            scaled_cols_tmc,
            scaled_cols_ftr,
            scaled_cols_ot,
        )

        pred_tmc = np.maximum(pred_tmc, 30.0)
        pred_ftr = np.clip(pred_ftr, 0.0, 1.0)
        pred_ot = np.clip(pred_ot, 0.0, 1.0)

        tmc_residual = actual_tmc.to_numpy(dtype=float) - pred_tmc
        ftr_residual = actual_ftr.to_numpy(dtype=float) - pred_ftr
        ot_residual = actual_ot.to_numpy(dtype=float) - pred_ot

        topics = (
            chunk[TOPIC_COL].fillna("UNKNOWN").astype(str)
            if TOPIC_COL in chunk.columns
            else pd.Series(["UNKNOWN"] * len(chunk), index=chunk.index)
        )

        residual_df = pd.DataFrame(
            {
                RESOURCE_KEY_COL: chunk[RESOURCE_KEY_COL].astype(str),
                "topic": topics.to_numpy(dtype=str),
                "tmc_residual": tmc_residual,
                "ftr_residual": ftr_residual,
                "ot_residual": ot_residual,
                "tmc_residual_sq": np.square(tmc_residual),
                "ftr_residual_sq": np.square(ftr_residual),
                "ot_residual_sq": np.square(ot_residual),
            }
        )

        (
            agent_sum,
            agent_sumsq,
            agent_count,
            topic_sum,
            topic_sumsq,
            topic_count,
            global_sum,
            global_sumsq,
            global_count,
        ) = accumulate_statistics(
            residual_df,
            agent_sum,
            agent_sumsq,
            agent_count,
            topic_sum,
            topic_sumsq,
            topic_count,
            global_sum,
            global_sumsq,
            global_count,
        )

        total_rows += len(chunk)
        LOGGER.info(
            "Processed chunk %d: rows=%d, cumulative_rows=%d",
            chunk_idx,
            len(chunk),
            total_rows,
        )

    LOGGER.info("Finished processing %d rows.", total_rows)

    summary_df = finalise_statistics(
        agent_sum,
        agent_sumsq,
        agent_count,
        topic_sum,
        topic_sumsq,
        topic_count,
        global_sum,
        global_sumsq,
        global_count,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    LOGGER.info("Residual summary written to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute GC residual bias and noise statistics for the simulator."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(DEFAULT_DATA_PATH),
        help="Path to full_merged_df.csv (merged historical dataset).",
    )
    parser.add_argument(
        "--gcs-path",
        type=Path,
        default=Path(DEFAULT_GCS_PATH),
        help="Optional path to gcs_unique.csv (not directly used but kept for reference).",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(DEFAULT_ASSETS_DIR),
        help="Directory containing the trained models, scalers, and feature lists.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help="Destination CSV for the residual summary.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Number of rows per chunk when streaming the dataset.",
    )
    parser.add_argument(
        "--max-tmc",
        type=float,
        default=10800.0,
        help="Maximum TMC (seconds) to keep, matching the modelling outlier filter.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging for progress reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    compute_residual_summary(
        data_path=args.data_path,
        gcs_path=args.gcs_path,
        assets_dir=args.assets_dir,
        output_path=args.output,
        chunksize=args.chunksize,
        max_tmc=args.max_tmc,
    )


if __name__ == "__main__":
    main()
