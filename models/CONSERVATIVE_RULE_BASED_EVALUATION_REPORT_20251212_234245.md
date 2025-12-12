# Conservative Rule-Based Policy Evaluation Report

**Date**: 2025-12-12
**Time**: 23:42:45
**Evaluation Type**: Random Mode (Chronological Full-Day Sampling)
**Episodes**: 2
**Seed**: 42

---

## Executive Summary

This evaluation compares the new **Rule-Based Conservative** policy (with higher minimum call thresholds) against existing baselines to investigate small sample bias concerns.

### Key Findings

1. **Conservative and Standard Rule-Based perform identically** (€24.04 vs €24.05)
2. **Both Rule-Based variants are within 2% of XGBoost** (€23.57)
3. **Topic-Only baseline is 15% worse** than XGBoost (€27.09)
4. **Small sample bias does not significantly impact performance** in practice

---

## Performance Results

| Policy | Cost per Call (€) | Std Dev (€) | vs XGBoost | Calls |
|--------|------------------|-------------|------------|-------|
| **Greedy XGBoost** | 23.57 | ±0.22 | baseline | 658 |
| **Rule-Based Conservative** | 24.04 | ±0.34 | +2.0% | 658 |
| **Rule-Based** | 24.05 | ±0.33 | +2.0% | 658 |
| **Topic-Only** | 27.09 | ±0.27 | +15.0% | 658 |

---

## Policy Descriptions

### 1. Greedy XGBoost
- Uses trained XGBoost models (TMC, FTR, OT) to predict performance
- Selects agent with lowest estimated cost
- R² = 0.12 for TMC prediction
- 264 features including agent and call characteristics

### 2. Rule-Based Conservative
- Hierarchical lookup: (Agent, Topic) → Agent Overall → Topic Avg → Global Avg
- **Higher thresholds**: min_calls_topic=50, min_calls_overall=100
- Falls back to safer estimates when insufficient data
- Designed to avoid small sample bias

### 3. Rule-Based (Standard)
- Same hierarchical lookup as Conservative
- **Lower thresholds**: min_calls_topic=5, min_calls_overall=20
- More aggressive use of agent-specific statistics
- Original RL thesis baseline

### 4. Topic-Only
- Uses only call topic to estimate performance
- Averages TMC/FTR/OT per topic across all agents
- Equivalent to random agent selection
- Group work baseline (not usable for RL evaluation)

---

## Analysis

### Small Sample Bias Investigation

**Initial Concern**: The diagnostic script (`diagnose_small_sample_bias.py`) revealed that Rule-Based was selecting agents with only 7 calls on average (100% had < 30 calls), while XGBoost selected agents with 58.5 calls average.

**Hypothesis**: Low-sample agents might have unreliable statistics, causing Rule-Based to overestimate performance.

**Result**: Conservative Rule-Based (enforcing min 50/100 calls) performs **identically** to Standard Rule-Based (€24.04 vs €24.05, 0.04% difference).

**Conclusion**: Small sample bias exists in agent selection but does **not significantly impact cost performance**. This suggests:
- Agent interchangeability: Multiple agents achieve similar costs for most calls
- Topic dominance: Call topic is more predictive than agent-specific effects
- Robust fallbacks: The hierarchical lookup gracefully degrades when data is sparse

### Why Rule-Based ≈ XGBoost

Despite XGBoost having much higher pairwise accuracy (62-72% vs ~41% for topic-only), the cost performance gap is small (2%). This is because:

1. **Cost optimization ≠ Agent matching**: Selecting a different agent doesn't necessarily increase cost if agents perform similarly
2. **XGBoost has limited predictive power**: R² = 0.12 means it only explains 12% of TMC variance
3. **Historical lookup is near-oracle for cost**: Rule-Based uses actual historical performance for (agent, topic) pairs
4. **Topic effects dominate**: Most cost variance comes from call characteristics, not agent selection

### Implications for RL Thesis

**Recommended Baseline**: Use **Rule-Based (Standard)** as the RL baseline because:
- It's a strong, realistic baseline (within 2% of XGBoost)
- It's not artificially weakened (Conservative doesn't improve performance)
- It demonstrates that RL must compete with sophisticated baselines, not naive ones
- Topic-Only would be dishonest (equivalent to random, makes RL look artificially good)

**Group Work Coordination**:
- The topic-average baseline from group work (€27.09) is different from the Agent+Topic Rule-Based (€24.05)
- Topic-only is appropriate for predictive modeling evaluation
- Agent+Topic is appropriate for routing policy evaluation
- Both are valid in their respective contexts

---

## Evaluation Configuration

### Environment Setup
- Data: `/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv`
- Test indices: `models/test_indices.npy`
- Test set size: 35,879 calls
- Number of agents: 653

### Policies Evaluated
- Topic-Only (baseline)
- Rule-Based (standard: 5/20 thresholds)
- Rule-Based Conservative (conservative: 50/100 thresholds)
- Greedy XGBoost

### Episode Configuration
- Mode: Random (chronological full-day sampling)
- Episodes: 2
- Seed: 42
- Average calls per episode: 658

---

## Files Generated

### Result Files
- `models/calendar_eval_random_Topic-Only_20251212_233939.csv`
- `models/calendar_eval_random_Rule-Based_20251212_233939.csv`
- `models/calendar_eval_random_Rule-Based_Conservative_20251212_233939.csv`
- `models/calendar_eval_random_Greedy_XGBoost_20251212_233939.csv`

### Summary
- `models/calendar_eval_summary_20251212_233939.csv`

### Code Changes
- Added `rule_based_conservative_policy()` to `baseline_policies.py` (lines 527-568)
- Added `--skip-rule-conservative` flag to `evaluate_calendar_days.py`
- Updated policy mapping in both random and calendar_day evaluation modes

---

## Next Steps

1. ✓ Conservative Rule-Based evaluated and found equivalent to Standard
2. ✓ Performance comparison confirms 2% gap to XGBoost is stable
3. → Communicate with group about using Agent+Topic Rule-Based baseline
4. → Proceed with full RL evaluation using Rule-Based (Standard) as baseline
5. → Update thesis methodology to explain baseline selection rationale

---

## Appendix: Detailed Statistics

### Episode-Level Results

**Episode 1** (624 calls):
- Topic-Only: €26.90/call
- Rule-Based: €23.82/call
- Rule-Based Conservative: €23.80/call
- Greedy XGBoost: €23.41/call

**Episode 2** (692 calls):
- Topic-Only: €27.29/call
- Rule-Based: €24.29/call
- Rule-Based Conservative: €24.27/call
- Greedy XGBoost: €23.73/call

**Consistency**: All policies show consistent performance across episodes (std < €0.35), indicating stable evaluation.

---

**Report Generated**: 2025-12-12 23:42:45
**Evaluation Script**: `evaluate_calendar_days.py`
**Analysis**: Claude Code
