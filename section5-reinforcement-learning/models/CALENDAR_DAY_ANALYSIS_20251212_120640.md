# Calendar Day Evaluation Analysis Report

**Generated**: 20251212_120640

## Executive Summary

This analysis compares routing policy performance between two evaluation methodologies:
1. **Random Mode**: Synthetic 8-hour episodes with random call sampling
2. **Calendar Day Mode**: Complete historical calendar days in chronological order

### Key Findings

- **Policy Rankings Preserved**: ✗ NO (ρ = 1.000, p = nan)
- **Costs Within ±20%**: ✓ YES (100% of policies)
- **Mean Absolute Difference**: €3.21 per call
- **Mean Relative Difference**: 12.8%

### Overall Conclusion: ✗ VALIDATION FAILED

Policy rankings are not preserved between methodologies. Results may not be robust.

---

## 1. Policy Performance Comparison

| Policy | Random Mode (€/call) | Calendar Day Mode (€/call) | Absolute Diff (€) | Relative Diff (%) |
|--------|---------------------|---------------------------|------------------|------------------|
| Greedy Xgboost | 23.57 ± 0.22 | 20.27 ± 1.64 | -3.30 | -14.0% ✓ |
| Random | 27.09 ± 0.27 | 23.98 ± 1.14 | -3.12 | -11.5% ✓ |

## 2. Policy Rankings

### Random Mode Rankings
1. **Greedy Xgboost**: €23.57/call
2. **Random**: €27.09/call

### Calendar Day Mode Rankings
1. **Greedy Xgboost**: €20.27/call
2. **Random**: €23.98/call

**Spearman Rank Correlation**: ρ = 1.000, p = nan
**Interpretation**: Strong positive correlation - rankings preserved

## 3. Statistical Analysis

- **Mean Absolute Difference**: €3.21 per call
- **Mean Relative Difference**: 12.8%
- **Policies Within ±20%**: 100% (2/2)

## 4. Interpretation

### Validation Failed - Rankings Not Preserved ✗

**Critical Issue**: Policy rankings differ between random and calendar-day methodologies.

**Possible Causes:**
1. Simulator noise overwhelming policy differences in shorter episodes
2. Temporal patterns in calendar days affecting policy performance
3. Overfitting to random-sampling methodology

**Recommendation**: Further investigation required before using results.

---

**Analysis Complete**: 20251212_120640
