#!/usr/bin/env python3
"""
Generate train/test indices for reproducible train/test separation.

This script loads the full merged dataset, performs the same 80/20 stratified split
that was used for XGBoost training, and saves the train and test indices.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths - Update DATA_PATH to your local data location
DATA_PATH = os.environ.get('NOS_DATA_PATH', 'data/full_merged_df.csv')
MODELS_DIR = 'models'

print("=" * 60)
print("GENERATING TRAIN/TEST INDICES")
print("=" * 60)

# Load data
print(f"\nLoading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df):,} rows")

# Remove NaN rows (same as modelling.py does before split)
print("\nRemoving NaN rows...")
original_len = len(df)
df = df.dropna()
print(f"✓ Removed {original_len - len(df):,} rows with NaN")
print(f"✓ Final dataset: {len(df):,} rows")

# Check for OT column (used for stratification)
ot_col = 'call_OT_Flag'  # Standard column name from feature engineering
if ot_col not in df.columns:
    # Try alternative names
    possible_ot_cols = [col for col in df.columns if 'OT' in col.upper() or 'overtime' in col.lower()]
    if possible_ot_cols:
        ot_col = possible_ot_cols[0]
        print(f"Using OT column: {ot_col}")
    else:
        print("WARNING: No OT column found for stratification, using random split")
        ot_col = None

# Perform stratified train/test split (same as modelling.py)
print(f"\nPerforming 80/20 stratified split (random_state=42)...")
if ot_col and ot_col in df.columns:
    stratify_on = df[ot_col]
    print(f"Stratifying on: {ot_col}")
else:
    stratify_on = None
    print("No stratification (random split)")

# Create X and y (we don't need them, just the indices)
X = df.drop(columns=[ot_col] if ot_col and ot_col in df.columns else [])
y = df[[ot_col]] if ot_col and ot_col in df.columns else df.iloc[:, 0:1]  # Dummy y if no OT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_on
)

# Extract indices
train_indices = X_train.index.values
test_indices = X_test.index.values

print(f"\n✓ Train set: {len(train_indices):,} samples ({len(train_indices)/len(df)*100:.1f}%)")
print(f"✓ Test set: {len(test_indices):,} samples ({len(test_indices)/len(df)*100:.1f}%)")

# Verify no overlap
overlap = np.intersect1d(train_indices, test_indices)
print(f"\nVerifying separation...")
print(f"✓ Overlap: {len(overlap)} samples (0.0%)")

if len(overlap) > 0:
    print("❌ ERROR: Overlap detected!")
    exit(1)

# Save indices
os.makedirs(MODELS_DIR, exist_ok=True)
train_path = os.path.join(MODELS_DIR, 'train_indices.npy')
test_path = os.path.join(MODELS_DIR, 'test_indices.npy')

np.save(train_path, train_indices)
np.save(test_path, test_indices)

print(f"\n✓ Saved train indices to: {train_path}")
print(f"✓ Saved test indices to: {test_path}")

# Verify files
train_verify = np.load(train_path)
test_verify = np.load(test_path)
print(f"\nVerification:")
print(f"✓ Train indices file: {len(train_verify):,} samples")
print(f"✓ Test indices file: {len(test_verify):,} samples")

print("\n" + "=" * 60)
print("✅ SUCCESS: Train/test indices generated successfully!")
print("=" * 60)
print(f"\nFiles created:")
print(f"  - {train_path} ({os.path.getsize(train_path):,} bytes)")
print(f"  - {test_path} ({os.path.getsize(test_path):,} bytes)")
print(f"\nNext step: Run validate_test_set_separation.py")
