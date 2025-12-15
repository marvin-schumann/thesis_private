# Section 3: End-to-End vs Fragmented Cost Prediction

**Author:** Maria Teresa Daffan

## Research Question

Does predicting call cost directly (end-to-end) outperform the fragmented approach of predicting individual components (TMC, FTR, OT) and combining them?

## Methodology

This section compares two approaches:

1. **Fragmented Approach** (Section 1): Train separate models for TMC, FTR, and OT, then combine predictions using the cost formula
2. **End-to-End Approach**: Train a single model to predict total call cost directly

### Models Tested
- Linear Regression
- XGBoost
- Neural Network

### Cost Formula
```
Cost = (TMC/60) × 0.35 + (1-FTR) × (TMC/60) × 0.35 + OT × 22
```

## Key Results

| Approach | Model | R² | Spearman Á |
|----------|-------|-----|------------|
| E2E | XGBoost | 0.108 | 0.324 |
| Fragmented | XGBoost | ~0.12 (TMC) | - |

**Key Finding**: End-to-end XGBoost achieves slightly better accuracy, but fragmented approach offers better interpretability for understanding cost drivers.

## Dependencies

This section uses the **same data methodology as Section 1** but trains models independently.

## Files

| File | Description |
|------|-------------|
| `train.py` | Training script for E2E models |
| `predict.py` | Inference script for cost prediction |
| `model_e2e_xgb.joblib` | Trained E2E XGBoost model |
| `Chapter3_E2E_models.ipynb` | Main analysis notebook |
| `Chapter3_basic_simulation.ipynb` | Simulation experiments |
| `requirements.txt` | Python dependencies |

## How to Run

### Training
```bash
cd section3-e2e-models/model_e2e_xgb
python train.py
```

### Prediction
```bash
python predict.py
```

### Notebooks
Open the Jupyter notebooks for detailed analysis:
```bash
jupyter notebook Chapter3_E2E_models.ipynb
```

## Requirements

See `requirements.txt` for dependencies. Main packages:
- scikit-learn
- xgboost
- pandas
- numpy

## Notes

- Data is not included (NOS confidential)
- Model file (`model_e2e_xgb.joblib`) contains trained E2E XGBoost
- Notebooks contain visualizations and detailed comparisons
