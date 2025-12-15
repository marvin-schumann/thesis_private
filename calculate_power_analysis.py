#!/usr/bin/env python3
"""
Statistical Power Analysis for Sample Size Calculation

This script calculates the required number of episodes/days needed for
statistically valid policy comparisons using power analysis.

Author: Generated for Thesis
Date: 2025-12-12
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

def calculate_sample_size(std_dev, min_effect, alpha=0.05, power=0.80):
    """
    Calculate required sample size for two-sample t-test.

    Parameters:
    - std_dev: Standard deviation of measurements (€ per call)
    - min_effect: Minimum detectable effect size (€ per call)
    - alpha: Significance level (default 0.05, two-tailed)
    - power: Statistical power (default 0.80)

    Returns:
    - Required sample size per group (n)

    Formula: n = 2 × ((z_α/2 + z_β)² × σ²) / δ²
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)  # 1.96 for α=0.05
    z_beta = stats.norm.ppf(power)          # 0.84 for 80%, 1.28 for 90%

    n = 2 * ((z_alpha + z_beta) ** 2) * (std_dev ** 2) / (min_effect ** 2)
    return np.ceil(n)

def calculate_min_detectable_effect(std_dev, n, alpha=0.05, power=0.80):
    """
    Calculate minimum detectable effect given sample size.

    Parameters:
    - std_dev: Standard deviation of measurements (€ per call)
    - n: Sample size per group
    - alpha: Significance level (default 0.05)
    - power: Statistical power (default 0.80)

    Returns:
    - Minimum detectable effect (€ per call)

    Formula: δ = √(2 × ((z_α/2 + z_β)² × σ²) / n)
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    delta = np.sqrt(2 * ((z_alpha + z_beta) ** 2) * (std_dev ** 2) / n)
    return delta

def generate_report(results_df, summary_df, timestamp):
    """
    Generate markdown report with power analysis results.

    Parameters:
    - results_df: DataFrame with power analysis results
    - summary_df: DataFrame with evaluation summary
    - timestamp: Timestamp string for report
    """

    # Extract data from summary
    random_mode = summary_df[summary_df['mode'] == 'random']
    calendar_mode = summary_df[summary_df['mode'] == 'calendar_day']

    pooled_std_random = random_mode['std_cost_per_call'].mean()
    pooled_std_calendar = calendar_mode['std_cost_per_call'].mean()

    n_random = int(random_mode['n_episodes'].iloc[0])
    n_calendar = int(calendar_mode['n_episodes'].iloc[0])

    # Calculate minimum detectable effects for current sample sizes
    mde_random_80 = calculate_min_detectable_effect(pooled_std_random, n_random, power=0.80)
    mde_random_90 = calculate_min_detectable_effect(pooled_std_random, n_random, power=0.90)
    mde_calendar_80 = calculate_min_detectable_effect(pooled_std_calendar, n_calendar, power=0.80)
    mde_calendar_90 = calculate_min_detectable_effect(pooled_std_calendar, n_calendar, power=0.90)

    # Calculate current observed gap
    greedy_random = random_mode[random_mode['policy'] == 'Greedy XGBoost']['avg_cost_per_call'].iloc[0]
    random_random = random_mode[random_mode['policy'] == 'Random']['avg_cost_per_call'].iloc[0]
    gap_random = abs(random_random - greedy_random)

    greedy_calendar = calendar_mode[calendar_mode['policy'] == 'Greedy XGBoost']['avg_cost_per_call'].iloc[0]
    random_calendar = calendar_mode[calendar_mode['policy'] == 'Random']['avg_cost_per_call'].iloc[0]
    gap_calendar = abs(random_calendar - greedy_calendar)

    # Generate report
    report = f"""# Statistical Power Analysis Report

**Generated**: {timestamp}

## Executive Summary

This report calculates the required sample size for statistically valid policy comparisons using two-sample t-test power analysis.

### Current Sample Sizes
- **Random mode**: n = {n_random} episodes
- **Calendar day mode**: n = {n_calendar} days

### Observed Standard Deviations
- **Random mode**: σ = €{pooled_std_random:.2f}
- **Calendar day mode**: σ = €{pooled_std_calendar:.2f}

### Current Policy Gaps (Random vs Greedy XGBoost)
- **Random mode**: €{gap_random:.2f} per call
- **Calendar day mode**: €{gap_calendar:.2f} per call

---

## Required Sample Sizes

### For 80% Power (α = 0.05)

| Minimum Detectable Effect | Random Mode | Calendar Day Mode |
|---------------------------|-------------|-------------------|
| €0.50 per call            | {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==0.5) & (results_df['power']==0.80)]['required_n'].iloc[0])} episodes  | {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==0.5) & (results_df['power']==0.80)]['required_n'].iloc[0])} days |
| €1.00 per call            | {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} episodes  | {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} days |
| €1.50 per call            | {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==1.5) & (results_df['power']==0.80)]['required_n'].iloc[0])} episodes  | {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.5) & (results_df['power']==0.80)]['required_n'].iloc[0])} days |
| €2.00 per call            | {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} episodes  | {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} days |

### For 90% Power (α = 0.05)

| Minimum Detectable Effect | Random Mode | Calendar Day Mode |
|---------------------------|-------------|-------------------|
| €0.50 per call            | {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==0.5) & (results_df['power']==0.90)]['required_n'].iloc[0])} episodes  | {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==0.5) & (results_df['power']==0.90)]['required_n'].iloc[0])} days |
| €1.00 per call            | {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==1.0) & (results_df['power']==0.90)]['required_n'].iloc[0])} episodes  | {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.0) & (results_df['power']==0.90)]['required_n'].iloc[0])} days |
| €1.50 per call            | {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==1.5) & (results_df['power']==0.90)]['required_n'].iloc[0])} episodes  | {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.5) & (results_df['power']==0.90)]['required_n'].iloc[0])} days |
| €2.00 per call            | {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==2.0) & (results_df['power']==0.90)]['required_n'].iloc[0])} episodes  | {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==2.0) & (results_df['power']==0.90)]['required_n'].iloc[0])} days |

---

## Analysis of Current Sample Size

### What can we detect with n = {n_calendar} calendar days?

**With 80% power (α = 0.05)**:
- Minimum detectable effect: €{mde_calendar_80:.2f} per call

**With 90% power (α = 0.05)**:
- Minimum detectable effect: €{mde_calendar_90:.2f} per call

**Current observed gap**: €{gap_calendar:.2f} per call

### What can we detect with n = {n_random} random episodes?

**With 80% power (α = 0.05)**:
- Minimum detectable effect: €{mde_random_80:.2f} per call

**With 90% power (α = 0.05)**:
- Minimum detectable effect: €{mde_random_90:.2f} per call

**Current observed gap**: €{gap_random:.2f} per call

---

## Recommendations

### Scenario 1: Detect €1.00 Differences (Conservative) ✓ RECOMMENDED

**Rationale**: €1.00 per call is a meaningful operational difference. Over 1000 calls per day, this represents €1000/day = €365,000/year.

**Required Sample Sizes (80% power)**:
- Random mode: {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} episodes (currently have {n_random})
- Calendar day mode: {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} days (currently have {n_calendar})

**Status**:
- Random mode: {"✓ SUFFICIENT" if n_random >= results_df[(results_df['mode']=='random') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0] else f"✗ INSUFFICIENT - Need {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0]) - n_random} more episodes"}
- Calendar day mode: {"✓ SUFFICIENT" if n_calendar >= results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0] else f"✗ INSUFFICIENT - Need {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0]) - n_calendar} more days"}

### Scenario 2: Detect €2.00 Differences (Moderate)

**Rationale**: Given current gap is ~€3, detecting €2 differences is reasonable threshold.

**Required Sample Sizes (80% power)**:
- Random mode: {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} episodes (currently have {n_random})
- Calendar day mode: {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} days (currently have {n_calendar})

**Status**:
- Random mode: {"✓ SUFFICIENT" if n_random >= results_df[(results_df['mode']=='random') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0] else f"✗ INSUFFICIENT - Need {int(results_df[(results_df['mode']=='random') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0]) - n_random} more episodes"}
- Calendar day mode: {"✓ SUFFICIENT" if n_calendar >= results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0] else f"✗ INSUFFICIENT - Need {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0]) - n_calendar} more days"}

### Scenario 3: Current Sample Size (n = {n_calendar})

**Calendar Day Mode Assessment**:
- Can detect €{mde_calendar_80:.2f} differences with 80% power
- Can detect €{mde_calendar_90:.2f} differences with 90% power
- Current gap is €{gap_calendar:.2f}

**Conclusion**: {"✓ SUFFICIENT - Current gap (€{:.2f}) exceeds minimum detectable effect (€{:.2f})".format(gap_calendar, mde_calendar_80) if gap_calendar > mde_calendar_80 else "✗ INSUFFICIENT - Current gap may not be reliably detected"}

---

## Final Recommendation

### For Calendar Day Mode (Most Realistic)

**Current Status**: n = {n_calendar} days

**To detect €1.00 differences with 80% power**:
- Required: {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} days
- Action: {"✓ Proceed with current data" if n_calendar >= results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0] else f"Run {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==1.0) & (results_df['power']==0.80)]['required_n'].iloc[0]) - n_calendar} more calendar days"}

**To detect €2.00 differences with 80% power**:
- Required: {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0])} days
- Action: {"✓ Proceed with current data" if n_calendar >= results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0] else f"Run {int(results_df[(results_df['mode']=='calendar') & (results_df['min_effect']==2.0) & (results_df['power']==0.80)]['required_n'].iloc[0]) - n_calendar} more calendar days"}

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
- σ = standard deviation (€{pooled_std_calendar:.2f} for calendar mode)
- δ = minimum detectable effect (e.g., €1.00)

**Minimum Detectable Effect**:
```
δ = √(2 × ((z_α/2 + z_β)² × σ²) / n)
```

### References
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.)
- Faul, F., Erdfelder, E., Lang, A.-G., & Buchner, A. (2007). G*Power 3

---

**Report Generated**: {timestamp}
"""

    return report

def main():
    """Main execution function."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*70)
    print("Statistical Power Analysis for Sample Size Calculation")
    print("="*70)
    print()

    # Load existing results
    print("Step 1: Loading evaluation data...")
    summary_file = 'models/calendar_eval_summary_20251212_161439.csv'
    summary = pd.read_csv(summary_file)
    print(f"✓ Loaded: {summary_file}")
    print()

    # Extract standard deviations
    print("Step 2: Extracting standard deviations...")
    random_mode_data = summary[summary['mode'] == 'random']
    calendar_mode_data = summary[summary['mode'] == 'calendar_day']

    pooled_std_random = random_mode_data['std_cost_per_call'].mean()
    pooled_std_calendar = calendar_mode_data['std_cost_per_call'].mean()

    print(f"  Random mode pooled std:       σ = €{pooled_std_random:.2f}")
    print(f"  Calendar day mode pooled std: σ = €{pooled_std_calendar:.2f}")
    print()

    # Calculate for different scenarios
    print("Step 3: Calculating required sample sizes...")
    effect_sizes = [0.5, 1.0, 1.5, 2.0]  # €0.50, €1.00, €1.50, €2.00
    powers = [0.80, 0.90]

    results = []
    for mode, std in [('random', pooled_std_random), ('calendar', pooled_std_calendar)]:
        for delta in effect_sizes:
            for pwr in powers:
                n = calculate_sample_size(std, delta, power=pwr)
                results.append({
                    'mode': mode,
                    'std_dev': std,
                    'min_effect': delta,
                    'power': pwr,
                    'required_n': int(n)
                })

    results_df = pd.DataFrame(results)

    # Display key results
    print("\n--- Quick Summary ---")
    print(f"\nTo detect €1.00 differences with 80% power:")
    n_calendar_1euro = int(results_df[(results_df['mode']=='calendar') &
                                      (results_df['min_effect']==1.0) &
                                      (results_df['power']==0.80)]['required_n'].iloc[0])
    print(f"  Calendar day mode: {n_calendar_1euro} days")

    n_random_1euro = int(results_df[(results_df['mode']=='random') &
                                    (results_df['min_effect']==1.0) &
                                    (results_df['power']==0.80)]['required_n'].iloc[0])
    print(f"  Random mode:       {n_random_1euro} episodes")

    print(f"\nTo detect €2.00 differences with 80% power:")
    n_calendar_2euro = int(results_df[(results_df['mode']=='calendar') &
                                      (results_df['min_effect']==2.0) &
                                      (results_df['power']==0.80)]['required_n'].iloc[0])
    print(f"  Calendar day mode: {n_calendar_2euro} days")

    n_random_2euro = int(results_df[(results_df['mode']=='random') &
                                    (results_df['min_effect']==2.0) &
                                    (results_df['power']==0.80)]['required_n'].iloc[0])
    print(f"  Random mode:       {n_random_2euro} episodes")
    print()

    # Save results
    print("Step 4: Saving results...")
    results_csv = f'models/power_analysis_results_{timestamp}.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Saved: {results_csv}")

    # Generate report
    print("Step 5: Generating markdown report...")
    report = generate_report(results_df, summary, timestamp)
    report_file = f'models/POWER_ANALYSIS_REPORT_{timestamp}.md'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"✓ Saved: {report_file}")
    print()

    print("="*70)
    print("Power Analysis Complete!")
    print("="*70)
    print(f"\nView full report: {report_file}")
    print()

    # Display current status
    n_calendar = int(calendar_mode_data['n_episodes'].iloc[0])
    n_random = int(random_mode_data['n_episodes'].iloc[0])

    print("CURRENT STATUS:")
    print(f"  Calendar days: {n_calendar} (need {n_calendar_1euro} for €1.00 detection)")
    print(f"  Random episodes: {n_random} (need {n_random_1euro} for €1.00 detection)")

    if n_calendar >= n_calendar_1euro:
        print("\n✓ SUFFICIENT: Current calendar day sample size is adequate!")
    else:
        print(f"\n✗ INSUFFICIENT: Need {n_calendar_1euro - n_calendar} more calendar days")

    print()

if __name__ == "__main__":
    main()
