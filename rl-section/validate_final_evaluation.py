#!/usr/bin/env python3
"""
Data Integrity Validation for Final Thesis Evaluation

Validates the 240 evaluation runs (4 policies × 2 modes × 30 samples)
before statistical analysis.

Usage: python validate_final_evaluation.py <timestamp>
"""

import sys
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def validate_evaluation(timestamp):
    """Run comprehensive validation checks on evaluation data."""

    print("="*70)
    print("FINAL EVALUATION VALIDATION")
    print("="*70)
    print(f"Timestamp: {timestamp}")
    print()

    checks = {
        'files_exist': False,
        'sample_sizes': False,
        'no_missing': False,
        'no_extreme_outliers': False,
        'zero_invalid_actions': False,
        'calendar_coverage': False
    }

    # Check 1: All 8 CSV files exist (excluding summary)
    print("Check 1: Verifying all policy-mode CSV files exist...")
    random_pattern = f'models/calendar_eval_random_*_{timestamp}.csv'
    calendar_pattern = f'models/calendar_eval_calendar_day_*_{timestamp}.csv'

    random_files = glob.glob(random_pattern)
    calendar_files = glob.glob(calendar_pattern)

    # Exclude summary file
    random_files = [f for f in random_files if 'summary' not in f]
    calendar_files = [f for f in calendar_files if 'summary' not in f]

    all_files = random_files + calendar_files

    print(f"  Random mode files: {len(random_files)}")
    print(f"  Calendar day files: {len(calendar_files)}")
    print(f"  Total: {len(all_files)}")

    checks['files_exist'] = len(all_files) == 8

    if len(all_files) != 8:
        print(f"  ✗ Expected 8 files, found {len(all_files)}")
        print("  Files found:")
        for f in sorted(all_files):
            print(f"    - {Path(f).name}")
    else:
        print("  ✓ All 8 files found")
    print()

    # Check 2: Each file has exactly 30 rows
    print("Check 2: Verifying sample sizes (n=30)...")
    sample_sizes_ok = True
    for f in all_files:
        df = pd.read_csv(f)
        if len(df) != 30:
            print(f"  ✗ {Path(f).name}: {len(df)} rows (expected 30)")
            sample_sizes_ok = False

    if sample_sizes_ok:
        print("  ✓ All files have exactly 30 rows")
    checks['sample_sizes'] = sample_sizes_ok
    print()

    # Check 3: No missing cost_per_call values
    print("Check 3: Checking for missing values...")
    all_data = pd.concat([pd.read_csv(f) for f in all_files])
    missing_count = all_data['cost_per_call'].isna().sum()

    if missing_count > 0:
        print(f"  ✗ Found {missing_count} missing cost_per_call values")
        checks['no_missing'] = False
    else:
        print("  ✓ No missing values")
        checks['no_missing'] = True
    print()

    # Check 4: Outlier detection (>3 IQR from median)
    print("Check 4: Detecting extreme outliers...")
    outliers_found = False

    for mode in ['random', 'calendar_day']:
        mode_pattern = f'models/calendar_eval_{mode}_*_{timestamp}.csv'
        mode_files = [f for f in glob.glob(mode_pattern) if 'summary' not in f]
        mode_data = pd.concat([pd.read_csv(f) for f in mode_files])

        median = mode_data['cost_per_call'].median()
        q1 = mode_data['cost_per_call'].quantile(0.25)
        q3 = mode_data['cost_per_call'].quantile(0.75)
        iqr = q3 - q1

        outliers = ((mode_data['cost_per_call'] < median - 3*iqr) |
                   (mode_data['cost_per_call'] > median + 3*iqr))

        if outliers.any():
            print(f"  ⚠ {mode}: {outliers.sum()} extreme outliers (>3 IQR)")
            outliers_found = True
        else:
            print(f"  ✓ {mode}: No extreme outliers")

    checks['no_extreme_outliers'] = not outliers_found
    print()

    # Check 5: Zero invalid actions
    print("Check 5: Checking invalid action counts...")
    if 'invalid_actions' in all_data.columns:
        total_invalid = all_data['invalid_actions'].sum()
        if total_invalid > 0:
            print(f"  ✗ Found {total_invalid} invalid actions")
            checks['zero_invalid_actions'] = False
        else:
            print("  ✓ Zero invalid actions")
            checks['zero_invalid_actions'] = True
    else:
        print("  ⚠ Column 'invalid_actions' not found, skipping check")
        checks['zero_invalid_actions'] = True
    print()

    # Check 6: All 30 calendar days evaluated
    print("Check 6: Verifying calendar day coverage...")
    all_dates = set()
    for f in calendar_files:
        df = pd.read_csv(f)
        if 'calendar_date' in df.columns:
            all_dates.update(df['calendar_date'].unique())

    if len(all_dates) == 30:
        print(f"  ✓ All 30 unique calendar days evaluated")
        checks['calendar_coverage'] = True
    else:
        print(f"  ✗ Only {len(all_dates)} unique calendar days (expected 30)")
        checks['calendar_coverage'] = False
    print()

    # Summary CSV check
    print("Bonus Check: Summary file...")
    summary_file = f'models/calendar_eval_summary_{timestamp}.csv'
    if Path(summary_file).exists():
        summary = pd.read_csv(summary_file)
        print(f"  ✓ Summary file exists with {len(summary)} rows")
        print(f"  Policies: {', '.join(summary['policy'].unique())}")
    else:
        print(f"  ✗ Summary file not found")
    print()

    # Print final results
    print("="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        check_name = check.replace('_', ' ').title()
        print(f"  {check_name:30s}: {status}")

    all_passed = all(checks.values())
    print()
    print("="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Data is ready for analysis")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"✗ {len(failed)} CHECKS FAILED - Review before analysis")
        print(f"  Failed: {', '.join(failed)}")
    print("="*70)
    print()

    return checks, all_passed

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_final_evaluation.py <timestamp>")
        print("Example: python validate_final_evaluation.py 20251212_180000")
        sys.exit(1)

    timestamp = sys.argv[1]
    checks, passed = validate_evaluation(timestamp)

    sys.exit(0 if passed else 1)
