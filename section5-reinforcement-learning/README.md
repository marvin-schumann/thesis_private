# Section 5: Reinforcement Learning for Call Routing

**Author:** Marvin Schumann

## Research Question

Can reinforcement learning with action masking outperform heuristic baselines for routing ~1,200 daily calls to 653 agents, optimizing for a composite cost metric?

## Methodology

- **Environment**: Custom Gymnasium environment simulating NOS call center operations
- **Algorithm**: PPO with action masking (MaskablePPO from SB3-Contrib)
- **Architecture**: MLP [256, 256], learning rate 3e-4
- **Training**: 200,000 timesteps (~8 hours on M3 Pro)
- **Action Space**: 653 agents (discrete)
- **State Space**: 859-dimensional observation (call features + agent availability)

## Key Results

| Policy | Avg Cost/Call | vs Greedy |
|--------|---------------|-----------|
| Greedy XGBoost | €20.30 | baseline |
| Masked PPO | €21.62 | +6.5% |
| Rule-Based | €26.91 | +32.6% |
| Random | €26.88 | +32.4% |

**Key Finding**: Masked PPO underperforms Greedy XGBoost by 6.5%, primarily due to low simulator fidelity (cost correlation r=0.08, Spearman ρ=0.24).

**Methodological Contribution**: Action masking is critical—improves valid action rate from ~6% to ~100%.

## Dependencies

This section **depends on Section 1's trained XGBoost models** for:
- TMC (call duration) prediction
- FTR (first time resolution) prediction
- OT (technician visit) prediction

The simulator uses these models to predict call outcomes for any agent assignment.

## Directory Structure

```
section5-reinforcement-learning/
├── call_center_env.py           # Core RL environment
├── call_center_env_masked.py    # Environment with action masking
├── baseline_policies.py         # Baseline policies (Rule-Based, Random, Greedy)
├── residual_adjustments.py      # Residual noise adjustments
├── training/                    # RL training scripts
├── evaluation/                  # Policy evaluation scripts
├── validation/                  # Validation scripts
├── analysis/                    # Analysis and diagnostics
├── utils/                       # Utility scripts
├── models/                      # Trained models and results
├── figures/                     # Thesis figures
├── documentation/               # Current documentation
├── archive/                     # Historical documentation
└── docs/                        # Technical documentation
```

## Running Scripts

All scripts must be run from the `section5-reinforcement-learning/` directory with `PYTHONPATH=.` set:

```bash
cd section5-reinforcement-learning
PYTHONPATH=. python3 evaluation/evaluate_calendar_days.py --help
PYTHONPATH=. python3 training/train_rl_masked.py --help
```

### Quick Commands

**Evaluate all policies** (30 episodes, both modes):
```bash
cd section5-reinforcement-learning
PYTHONPATH=. python3 evaluation/evaluate_calendar_days.py \
  --mode both \
  --episodes 30 \
  --seed 42 \
  --calendar-days-file models/selected_calendar_days_30.json
```

**Train Masked PPO**:
```bash
cd section5-reinforcement-learning
PYTHONPATH=. python3 training/train_rl_masked.py \
  --timesteps 500000 \
  --seed 42
```

**Run validation**:
```bash
cd section5-reinforcement-learning
PYTHONPATH=. python3 validation/validate_simulator.py
```

**Generate analysis**:
```bash
cd section5-reinforcement-learning
PYTHONPATH=. python3 analysis/analyze_final_evaluation.py \
  --timestamp 20251212_215243
```

## Key Files

### Core Components
- `call_center_env.py` - Gym environment for call center routing
- `call_center_env_masked.py` - Version with action masking support
- `baseline_policies.py` - Rule-Based, Random, and Greedy XGBoost policies
- `residual_adjustments.py` - Adds residual noise to break circularity

### Main Scripts
- `evaluation/evaluate_calendar_days.py` - Main evaluation script (30 episodes × 2 modes)
- `training/train_rl_masked.py` - Train Masked PPO agent
- `validation/validate_simulator.py` - Validate simulator fidelity
- `analysis/analyze_final_evaluation.py` - Statistical analysis

### Results
- `models/` - XGBoost models, PPO checkpoints, evaluation CSVs, reports
- `figures/` - All 7 thesis figures (PNG format)

## Documentation

- **For thesis writing**: See `documentation/CLAUDE_CODE_HANDOFF_THESIS_QUESTIONS.md`
- **Technical details**: See `docs/COMPREHENSIVE_AUDIT_REPORT_DEC2025.md`
- **Final results**: See `docs/section5_final_results_summary.md`

## Requirements

All dependencies are listed in the root `requirements.txt`. Install with:

```bash
pip install -r ../requirements.txt
```

## Notes

- All scripts use the test set at: `models/test_indices.npy` (35,879 calls, 653 agents)
- Calendar days: `models/selected_calendar_days_30.json` (January 1-30, 2024)
- Data path is hardcoded in scripts (OneDrive location)
- Models directory: `section5-reinforcement-learning/models/`
