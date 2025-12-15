#!/usr/bin/env python3
"""
Select 30 Consecutive Calendar Days for Final Thesis Evaluation

This script selects January 1-30, 2024 (one business month) from the test set
for the final thesis evaluation with n=30 calendar days.

Author: Generated for Thesis
Date: 2025-12-12
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_PATH = '/Users/marvin.schumann/Library/CloudStorage/OneDrive-Personal/Documents/UNI/Nova SBE/04 Thesis/Data/07052025/full_merged_df.csv'
TEST_INDICES_PATH = 'models/test_indices.npy'
OUTPUT_JSON = 'models/selected_calendar_days_30.json'
OUTPUT_CSV = 'models/selected_calendar_days_30_stats.csv'

def main():
    """Select 30 consecutive calendar days and generate statistics."""

    print("="*70)
    print("FINAL THESIS EVALUATION - Calendar Day Selection")
    print("="*70)
    print()

    # Load test set
    print("Step 1: Loading test set data...")
    test_indices = np.load(TEST_INDICES_PATH)
    print(f"  ✓ Loaded {len(test_indices):,} test indices")

    full_data = pd.read_csv(DATA_PATH)
    print(f"  ✓ Loaded full dataset: {len(full_data):,} calls")

    test_data = full_data.loc[full_data.index.isin(test_indices)].copy()
    print(f"  ✓ Test set: {len(test_data):,} calls")
    print()

    # Parse dates
    print("Step 2: Parsing call dates...")
    test_data['date'] = pd.to_datetime(test_data['call_LEG_START_TIME_DAT']).dt.date

    # Get date range
    min_date = test_data['date'].min()
    max_date = test_data['date'].max()
    print(f"  ✓ Test set date range: {min_date} to {max_date}")
    print()

    # Select January 1-30, 2024 (30 consecutive days = 1 business month)
    print("Step 3: Selecting January 1-30, 2024 (30 consecutive days)...")
    selected_dates = [f"2024-01-{day:02d}" for day in range(1, 31)]
    print(f"  ✓ Selected period: {selected_dates[0]} to {selected_dates[-1]}")
    print(f"  ✓ Total days: {len(selected_dates)}")
    print()

    # Calculate daily statistics
    print("Step 4: Calculating daily statistics...")
    daily_stats = test_data.groupby('date').size()

    selected_stats = []
    total_calls = 0

    for date_str in selected_dates:
        date_obj = pd.to_datetime(date_str).date()
        call_count = daily_stats.get(date_obj, 0)
        total_calls += call_count

        selected_stats.append({
            'date': date_str,
            'call_count': int(call_count),
            'day_of_week': int(pd.to_datetime(date_str).dayofweek),
            'day_name': pd.to_datetime(date_str).day_name()
        })

    stats_df = pd.DataFrame(selected_stats)

    # Print summary statistics
    print("  Summary Statistics:")
    print(f"    Total calls: {total_calls:,}")
    print(f"    Mean calls/day: {stats_df['call_count'].mean():.1f}")
    print(f"    Std calls/day: {stats_df['call_count'].std():.1f}")
    print(f"    Min calls/day: {stats_df['call_count'].min()}")
    print(f"    Max calls/day: {stats_df['call_count'].max()}")
    print()

    # Check for days with zero calls
    zero_days = stats_df[stats_df['call_count'] == 0]
    if len(zero_days) > 0:
        print(f"  ⚠ Warning: {len(zero_days)} days have zero calls:")
        for _, row in zero_days.iterrows():
            print(f"    - {row['date']} ({row['day_name']})")
        print()

    # Save outputs
    print("Step 5: Saving outputs...")

    # Save JSON (for evaluation script)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(selected_dates, f, indent=2)
    print(f"  ✓ Saved JSON: {OUTPUT_JSON}")

    # Save CSV (for analysis)
    stats_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Saved CSV: {OUTPUT_CSV}")
    print()

    # Display day-of-week distribution
    print("Day-of-Week Distribution:")
    dow_summary = stats_df.groupby('day_name')['call_count'].agg(['count', 'mean', 'sum'])
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_summary = dow_summary.reindex([d for d in dow_order if d in dow_summary.index])

    for day_name, row in dow_summary.iterrows():
        print(f"  {day_name:9s}: {int(row['count'])} days, "
              f"mean={row['mean']:.1f} calls/day, total={int(row['sum'])} calls")
    print()

    # Final validation
    print("="*70)
    print("VALIDATION")
    print("="*70)

    checks = []
    checks.append(("Total days selected", len(selected_dates) == 30, len(selected_dates)))
    checks.append(("All dates in test set", all(pd.to_datetime(d).date() >= min_date and
                                                  pd.to_datetime(d).date() <= max_date
                                                  for d in selected_dates), "Yes"))
    checks.append(("Consecutive dates", all(selected_dates[i] < selected_dates[i+1]
                                             for i in range(len(selected_dates)-1)), "Yes"))
    checks.append(("Output files created", Path(OUTPUT_JSON).exists() and
                                            Path(OUTPUT_CSV).exists(), "Yes"))

    all_passed = True
    for check_name, passed, value in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name:30s}: {status} ({value})")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready for evaluation")
    else:
        print("✗ SOME CHECKS FAILED - Please review")

    print("="*70)
    print()

    print("Next Steps:")
    print(f"  1. Review {OUTPUT_CSV} to verify selected days")
    print(f"  2. Run evaluation: python evaluate_calendar_days.py \\")
    print(f"       --mode both --episodes 30 --seed 42 \\")
    print(f"       --calendar-days-file {OUTPUT_JSON}")
    print()

if __name__ == '__main__':
    main()
