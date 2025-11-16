#!/usr/bin/env python3
import json
import os
import joblib
import pandas as pd

ASSETS_DIR = 'models'

def main():
    feature_list_path = os.path.join(ASSETS_DIR, 'feature_lists.json')
    if not os.path.exists(feature_list_path):
        raise FileNotFoundError(f"Feature list file not found: {feature_list_path}")

    with open(feature_list_path, 'r') as f:
        feature_lists = json.load(f)

    models_info = [
        ('tmc', 'model_tmc.joblib'),
        ('ftr', 'model_ftr.joblib'),
        ('ot', 'model_ot.joblib'),
    ]

    for key, filename in models_info:
        model_path = os.path.join(ASSETS_DIR, filename)
        if not os.path.exists(model_path):
            print(f"\nModel file missing for {key.upper()}: {model_path}")
            continue

        model = joblib.load(model_path)
        features = feature_lists.get(key)
        if features is None:
            print(f"\nNo feature list found for {key.upper()} in feature_lists.json")
            continue

        importances = getattr(model, 'feature_importances_', None)
        if importances is None:
            print(f"\n{key.upper()} model ({type(model).__name__}) does not expose feature_importances_. Skipping.")
            continue

        if len(importances) != len(features):
            print(f"\n{key.upper()} model feature importance length mismatch "
                  f"(importances={len(importances)}, features={len(features)}). Skipping.")
            continue

        df = pd.DataFrame({'feature': features, 'importance': importances})
        df = df.sort_values('importance', ascending=False)

        print(f"\nTop 20 features for {key.upper()} model ({filename}):")
        print(df.head(20).to_string(index=False))

if __name__ == '__main__':
    main()

