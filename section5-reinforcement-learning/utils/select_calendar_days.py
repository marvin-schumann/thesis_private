#!/usr/bin/env python3
"""
Select Representative Calendar Days for Evaluation

This script identifies 10 typical calendar days from the test set for calendar-day
based evaluation. Selection criteria:
- Call count: 110-135 (within ±1 std dev of mean 121.6)
- Temporal diversity: 2 days per month for 5 months
- Mix of weekdays and weekends

Outputs:
- models/selected_calendar_days.csv: DataFrame with daily statistics
- models/selected_calendar_days.json: List of date strings for easy loading

Author: Thesis Calendar-Day Evaluation
Date: 2025-12-11
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

# Configuration - Update DATA_PATH to your local data location
DATA_PATH = os.environ.get('NOS_DATA_PATH', 'data/full_merged_df.csv')
TEST_INDICES_PATH = 'models/test_indices.npy'
OUTPUT_CSV = 'models/selected_calendar_days.csv'
OUTPUT_JSON = 'models/selected_calendar_days.json'
SEED = 42

# Selection criteria
CALL_RANGE = (110, 135)  # ±1 std dev from mean 121.6
DAYS_TO_SELECT = 10
DAYS_PER_MONTH = 2


def main():
    # Use global constants, but allow adjustment if needed
    days_to_select = DAYS_TO_SELECT

    print("="*70)
    print("CALENDAR DAY SELECTION FOR EVALUATION")
    print("="*70)
    print(f"\nCriteria:")
    print(f"  Call range: {CALL_RANGE[0]}-{CALL_RANGE[1]} calls/day")
    print(f"  Days to select: {days_to_select}")
    print(f"  Temporal diversity: {DAYS_PER_MONTH} days per month")
    print(f"  Random seed: {SEED}\n")

    # Load test set
    print("Loading data...")
    test_indices = np.load(TEST_INDICES_PATH)
    print(f"  Test set size: {len(test_indices):,} calls")

    # Load full data
    print(f"  Reading: {DATA_PATH}")
    full_data = pd.read_csv(DATA_PATH)
    test_data = full_data.loc[full_data.index.isin(test_indices)].copy()
    print(f"  Test data loaded: {len(test_data):,} rows")

    # Parse dates
    print("\nParsing dates...")
    test_data['date'] = pd.to_datetime(test_data['call_LEG_START_TIME_DAT']).dt.date
    test_data['timestamp'] = pd.to_datetime(test_data['call_LEG_START_TIME_DAT'])
    test_data['month'] = test_data['timestamp'].dt.month
    test_data['day_of_week'] = test_data['timestamp'].dt.dayofweek
    test_data['day_name'] = test_data['timestamp'].dt.day_name()

    # Calculate daily statistics
    print("  Grouping by calendar day...")
    daily_stats = test_data.groupby('date').agg({
        'call_LEG_IF_ID': 'count',
        'month': 'first',
        'day_of_week': 'first',
        'day_name': 'first'
    }).rename(columns={'call_LEG_IF_ID': 'call_count'})

    print(f"\nTest set statistics:")
    print(f"  Total calendar days: {len(daily_stats)}")
    print(f"  Date range: {daily_stats.index.min()} to {daily_stats.index.max()}")
    print(f"  Calls per day (mean): {daily_stats['call_count'].mean():.1f}")
    print(f"  Calls per day (median): {daily_stats['call_count'].median():.1f}")
    print(f"  Calls per day (std): {daily_stats['call_count'].std():.1f}")
    print(f"  Calls per day (min): {daily_stats['call_count'].min()}")
    print(f"  Calls per day (max): {daily_stats['call_count'].max()}")

    # Filter to typical days
    print(f"\nFiltering to typical days ({CALL_RANGE[0]}-{CALL_RANGE[1]} calls)...")
    typical_days = daily_stats[
        (daily_stats['call_count'] >= CALL_RANGE[0]) &
        (daily_stats['call_count'] <= CALL_RANGE[1])
    ]
    print(f"  Typical days available: {len(typical_days)}")

    if len(typical_days) < days_to_select:
        print(f"\n⚠️  WARNING: Only {len(typical_days)} typical days available, need {days_to_select}")
        print("  Adjusting selection to available days...")
        days_to_select = len(typical_days)

    # Sample with diversity (2 days per month for first 5 months)
    print(f"\nSampling {days_to_select} days with temporal diversity...")
    np.random.seed(SEED)

    selected_dates = []
    months_available = typical_days['month'].value_counts().sort_index()

    print(f"  Available months: {months_available.index.tolist()}")

    # Try to get 2 days from each of the first 5 months
    months_to_sample = months_available.index[:5] if len(months_available) >= 5 else months_available.index

    for month in months_to_sample:
        month_days = typical_days[typical_days['month'] == month]
        sample_size = min(DAYS_PER_MONTH, len(month_days), days_to_select - len(selected_dates))

        if sample_size > 0:
            sampled = month_days.sample(n=sample_size, random_state=SEED+month)
            selected_dates.extend(sampled.index.tolist())
            print(f"    Month {month}: sampled {sample_size} days")

        if len(selected_dates) >= days_to_select:
            break

    # Fill to target if needed
    if len(selected_dates) < days_to_select:
        print(f"\n  Need {days_to_select - len(selected_dates)} more days...")
        remaining = typical_days[~typical_days.index.isin(selected_dates)]
        if len(remaining) > 0:
            additional = remaining.sample(
                n=min(days_to_select - len(selected_dates), len(remaining)),
                random_state=999
            )
            selected_dates.extend(additional.index.tolist())
            print(f"    Added {len(additional)} additional days")

    # Sort chronologically
    selected_dates = sorted(selected_dates[:days_to_select])

    # Generate output DataFrame
    output_df = daily_stats.loc[selected_dates].copy()
    output_df.index.name = 'date'
    output_df = output_df.reset_index()

    # Save outputs
    print(f"\nSaving results...")
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV saved: {OUTPUT_CSV}")

    date_strings = [str(d) for d in selected_dates]
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(date_strings, f, indent=2)
    print(f"  JSON saved: {OUTPUT_JSON}")

    # Display selections
    print("\n" + "="*70)
    print("SELECTED CALENDAR DAYS")
    print("="*70)
    print(f"\n{len(selected_dates)} days selected:\n")

    for i, row in output_df.iterrows():
        date_obj = pd.to_datetime(row['date'])
        print(f"  {i+1:2d}. {row['date']} ({row['day_name']:3s}) - {row['call_count']:3d} calls - Month {row['month']}")

    # Summary statistics
    print(f"\nSelection summary:")
    print(f"  Total days: {len(output_df)}")
    print(f"  Weekdays: {sum(output_df['day_of_week'] < 5)}")
    print(f"  Weekends: {sum(output_df['day_of_week'] >= 5)}")
    print(f"  Months covered: {output_df['month'].nunique()}")
    print(f"  Call count range: {output_df['call_count'].min()}-{output_df['call_count'].max()}")
    print(f"  Call count mean: {output_df['call_count'].mean():.1f}")

    print("\n" + "="*70)
    print("✓ CALENDAR DAY SELECTION COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_JSON}")
    print(f"\nNext: Modify call_center_env.py to add calendar-day mode")
    print()


if __name__ == '__main__':
    main()
