# Statistical Power Analysis Report

**Generated**: 20251212_170037

## Executive Summary

This report calculates the required sample size for statistically valid policy comparisons using two-sample t-test power analysis.

### Current Sample Sizes
- **Random mode**: n = 2 episodes
- **Calendar day mode**: n = 10 days

### Observed Standard Deviations
- **Random mode**: σ = €0.25
- **Calendar day mode**: σ = €1.03

### Current Policy Gaps (Random vs Greedy XGBoost)
- **Random mode**: €3.52 per call
- **Calendar day mode**: €3.05 per call

---

## Required Sample Sizes

### For 80% Power (α = 0.05)

| Minimum Detectable Effect | Random Mode | Calendar Day Mode |
|---------------------------|-------------|-------------------|
| €0.50 per call            | 4 episodes  | 67 days |
| €1.00 per call            | 1 episodes  | 17 days |
| €1.50 per call            | 1 episodes  | 8 days |
| €2.00 per call            | 1 episodes  | 5 days |

### For 90% Power (α = 0.05)

| Minimum Detectable Effect | Random Mode | Calendar Day Mode |
|---------------------------|-------------|-------------------|
| €0.50 per call            | 6 episodes  | 90 days |
| €1.00 per call            | 2 episodes  | 23 days |
| €1.50 per call            | 1 episodes  | 10 days |
| €2.00 per call            | 1 episodes  | 6 days |

---

## Analysis of Current Sample Size

### What can we detect with n = 10 calendar days?

**With 80% power (α = 0.05)**:
- Minimum detectable effect: €1.29 per call

**With 90% power (α = 0.05)**:
- Minimum detectable effect: €1.49 per call

**Current observed gap**: €3.05 per call

### What can we detect with n = 2 random episodes?

**With 80% power (α = 0.05)**:
- Minimum detectable effect: €0.70 per call

**With 90% power (α = 0.05)**:
- Minimum detectable effect: €0.80 per call

**Current observed gap**: €3.52 per call

---

## Recommendations

### Scenario 1: Detect €1.00 Differences (Conservative) ✓ RECOMMENDED

**Rationale**: €1.00 per call is a meaningful operational difference. Over 1000 calls per day, this represents €1000/day = €365,000/year.

**Required Sample Sizes (80% power)**:
- Random mode: 1 episodes (currently have 2)
- Calendar day mode: 17 days (currently have 10)

**Status**:
- Random mode: ✓ SUFFICIENT
- Calendar day mode: ✗ INSUFFICIENT - Need 7 more days

### Scenario 2: Detect €2.00 Differences (Moderate)

**Rationale**: Given current gap is ~€3, detecting €2 differences is reasonable threshold.

**Required Sample Sizes (80% power)**:
- Random mode: 1 episodes (currently have 2)
- Calendar day mode: 5 days (currently have 10)

**Status**:
- Random mode: ✓ SUFFICIENT
- Calendar day mode: ✓ SUFFICIENT

### Scenario 3: Current Sample Size (n = 10)

**Calendar Day Mode Assessment**:
- Can detect €1.29 differences with 80% power
- Can detect €1.49 differences with 90% power
- Current gap is €3.05

**Conclusion**: ✓ SUFFICIENT - Current gap (€3.05) exceeds minimum detectable effect (€1.29)

---

## Final Recommendation

### For Calendar Day Mode (Most Realistic)

**Current Status**: n = 10 days

**To detect €1.00 differences with 80% power**:
- Required: 17 days
- Action: Run 7 more calendar days

**To detect €2.00 differences with 80% power**:
- Required: 5 days
- Action: ✓ Proceed with current data

---

## Technical Notes

### Assumptions
- Two-sample independent t-test (comparing different policies)
- Normal distribution of cost-per-call measurements
- Equal variance across policies (conservative estimate using pooled std)
- Two-tailed test (α = 0.05)

### Formulas Used

**Sample Size Calculation**:
```
n = 2 × ((z_α/2 + z_β)² × σ²) / δ²
```

Where:
- z_α/2 = 1.96 (for α = 0.05, two-tailed)
- z_β = 0.84 (for 80% power) or 1.28 (for 90% power)
- σ = standard deviation (€1.03 for calendar mode)
- δ = minimum detectable effect (e.g., €1.00)

**Minimum Detectable Effect**:
```
δ = √(2 × ((z_α/2 + z_β)² × σ²) / n)
```

### References
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.)
- Faul, F., Erdfelder, E., Lang, A.-G., & Buchner, A. (2007). G*Power 3

---

**Report Generated**: 20251212_170037
