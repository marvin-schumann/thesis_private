# Calendar-Day Evaluation - Quick Reference Guide

## Overview

Calendar-day evaluation validates that routing policy performance rankings hold under realistic operational conditions (complete calendar days) vs. the baseline methodology (synthetic episodes with random sampling).

## Implementation Summary

### Phase 1: Calendar Day Selection ✓
**Script**: `select_calendar_days.py`
**Output**: `models/selected_calendar_days.json`, `models/selected_calendar_days.csv`

Selected 10 representative calendar days from test set:
- Call range: 110-135 calls/day (±1 std dev from mean)
- Temporal diversity: 2 days/month × 5 months
- Mix of weekdays (7) and weekends (3)

**Selected dates**: 2024-01-17, 2024-01-29, 2024-02-10, 2024-02-19, 2024-03-21, 2024-03-23, 2024-04-13, 2024-04-15, 2024-05-09, 2024-05-13

### Phase 2: Environment Modification ✓
**File**: `call_center_env.py`

Added dual-mode support:
- `episode_mode='random'` - Original synthetic episodes (backwards compatible)
- `episode_mode='calendar_day'` - Chronological replay of actual calendar days

**Key changes**:
- New `__init__` parameters: `episode_mode`, `calendar_date`
- New method: `_setup_calendar_day_episode()`
- Modified: `reset()`, `_sample_call()`, `_get_next_call_arrival()`, `step()`
- StopIteration handling for calendar day completion

**Validation**: `test_calendar_day_mode.py` - All tests passed ✓

### Phase 3: Unified Evaluation Script ✓
**Script**: `evaluate_calendar_days.py`

Evaluates 4 functional policies on both modes:
- Random (baseline)
- Rule-Based
- Greedy XGBoost
- Masked PPO

**Features**:
- Dual-mode evaluation (random and/or calendar_day)
- Per-episode/day detailed results
- Aggregate summary statistics
- CSV output for analysis

### Phase 4: Statistical Analysis Script ✓
**Script**: `analyze_calendar_day_results.py`

**Analyses**:
- Policy ranking comparison (Spearman correlation)
- Pairwise cost differences (absolute and relative)
- Statistical validation (within ±20% threshold)
- Markdown report generation

---

## Running the Evaluation

### Quick Start (Full Pipeline)

```bash
# 1. Day selection (already done)
python select_calendar_days.py

# 2. Run evaluation (both modes)
python evaluate_calendar_days.py --mode both --episodes 10 --seed 42

# 3. Analyze results
python analyze_calendar_day_results.py --results-dir models

# Estimated time: 60-90 minutes
```

### Individual Mode Evaluation

**Random Mode Only** (10 episodes):
```bash
python evaluate_calendar_days.py --mode random --episodes 10 --seed 42
```

**Calendar Day Mode Only** (10 days):
```bash
python evaluate_calendar_days.py --mode calendar_day --seed 42
```

### Selective Policy Evaluation

Skip specific policies to save time:
```bash
# Skip Random and Masked PPO (faster)
python evaluate_calendar_days.py --mode both --skip-random --skip-masked-ppo
```

Options:
- `--skip-random` - Skip Random policy
- `--skip-rule` - Skip Rule-Based policy
- `--skip-greedy` - Skip Greedy XGBoost policy
- `--skip-masked-ppo` - Skip Masked PPO policy

---

## Output Files

### Evaluation Results
Located in `models/` directory:

**Per-Policy Results** (timestamped):
- `calendar_eval_random_Random_YYYYMMDD_HHMMSS.csv`
- `calendar_eval_random_Rule_Based_YYYYMMDD_HHMMSS.csv`
- `calendar_eval_random_Greedy_Xgboost_YYYYMMDD_HHMMSS.csv`
- `calendar_eval_random_Masked_Ppo_YYYYMMDD_HHMMSS.csv`
- `calendar_eval_calendar_day_Random_YYYYMMDD_HHMMSS.csv`
- (etc. for all policies in calendar_day mode)

**Summary**:
- `calendar_eval_summary_YYYYMMDD_HHMMSS.csv`

### Analysis Report
- `models/CALENDAR_DAY_ANALYSIS_YYYYMMDD_HHMMSS.md`

Contains:
- Executive summary with validation verdict
- Policy performance comparison table
- Rankings comparison (both modes)
- Statistical tests results
- Interpretation and recommendations

---

## Expected Results

### Validation Criteria

**Success**: ✓ VALIDATION SUCCESSFUL
- Policy rankings preserved (Spearman ρ > 0.9, p < 0.05)
- Absolute costs within ±20% for all policies

**Partial**: ⚠ PARTIAL VALIDATION
- Rankings preserved but some costs exceed ±20%
- Further investigation recommended

**Failure**: ✗ VALIDATION FAILED
- Rankings not preserved
- Methodology may not be robust

### Typical Performance (from original request)

Expected absolute costs within **10-20%** between methodologies:
- Baseline: ~€15/call (random mode)
- Calendar: ~€14-18/call (expected range)

Rankings should remain:
1. Greedy XGBoost (best)
2. Rule-Based
3. Masked PPO
4. Random (worst)

---

## Troubleshooting

### Common Issues

**1. "No calls found for calendar date"**
- Ensure `models/selected_calendar_days.json` exists
- Check test set coverage (may need to regenerate calendar days)

**2. "Model file not found" for Masked PPO**
- Ensure `models/rl_model_masked_ppo.zip` exists
- Skip with `--skip-masked-ppo` if not trained

**3. Memory issues during evaluation**
- Run modes separately (`--mode random`, then `--mode calendar_day`)
- Reduce number of policies evaluated

**4. Analysis script can't find results**
- Specify timestamp: `--timestamp 20251211_120000`
- Check `models/` directory for CSV files

---

## File Reference

### Scripts (in order of execution)
1. `select_calendar_days.py` - Day selection
2. `evaluate_calendar_days.py` - Evaluation
3. `analyze_calendar_day_results.py` - Analysis

### Supporting Files
- `test_calendar_day_mode.py` - Environment validation
- `call_center_env.py` - Modified environment (both modes)
- `call_center_env_masked.py` - Inherits dual-mode support
- `baseline_policies.py` - Policy implementations

### Data Files
- `models/selected_calendar_days.json` - Selected dates
- `models/selected_calendar_days.csv` - Day statistics
- `models/test_indices.npy` - Test set indices

---

## Advanced Usage

### Custom Calendar Day Selection

Edit criteria in `select_calendar_days.py`:
```python
CALL_RANGE = (110, 135)  # Calls per day range
DAYS_TO_SELECT = 10      # Number of days
DAYS_PER_MONTH = 2       # Temporal diversity
```

### Programmatic Access

```python
from call_center_env import CallCenterEnv

# Random mode (original)
env = CallCenterEnv(
    data_path=DATA_PATH,
    test_indices_path='models/test_indices.npy',
    episode_mode='random'
)

# Calendar day mode
env = CallCenterEnv(
    data_path=DATA_PATH,
    test_indices_path='models/test_indices.npy',
    episode_mode='calendar_day',
    calendar_date='2024-01-17'
)

obs, info = env.reset(seed=42)
# ... standard gym loop
```

---

## Timeline

- **Phase 1-4**: Implementation (completed)
- **Phase 5**: Execution (60-90 minutes)
  - Random mode: ~30-45 min (10 episodes × 4 policies)
  - Calendar day mode: ~30-45 min (10 days × 4 policies)
  - Analysis: ~1 min

**Total estimated time**: 60-90 minutes for full evaluation

---

## Questions?

- Script help: `python <script>.py --help`
- Environment testing: `python test_calendar_day_mode.py`
- Logs: Check console output for INFO/WARNING messages

---

**Status**: Implementation complete (Phases 1-4) ✓
**Next**: Execute Phase 5 (full evaluation)
**Last Updated**: 2025-12-11
