# Repository Structure

**Last Updated**: 2025-12-15

This repository contains the complete thesis project, organized into three main sections:

## Directory Overview

```
thesis_private/
├── rl-section/              # Chapter 5: Reinforcement Learning (Individual Work)
├── group-baseline/          # Group Project: NOS Call Center Pipeline
├── section3-e2e-models/     # Chapter 3: Basic Simulation Models
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
├── README.md                # Project overview
├── CITATION.bib             # Citation information
├── LICENSE                  # License file
└── Thesis_full_v01_251214_v01.pdf  # Complete thesis document
```

---

## 1. RL Section (Chapter 5) - Individual Work

**Path**: `rl-section/`

Organized into 12 subdirectories:

### Core Components
- **environment/** - RL environment implementations
  - `call_center_env.py` - Base environment
  - `call_center_env_masked.py` - Environment with action masking

- **policies/** - Policy implementations
  - `baseline_policies.py` - Rule-Based, Random, Greedy XGBoost policies

### Experiment Execution
- **training/** - RL training scripts
  - `train_rl_masked.py` - Masked PPO training
  - `train_dqn_*.py` - DQN experiments (masked/unmasked)

- **evaluation/** - Evaluation scripts (5 scripts)
  - `evaluate_calendar_days.py` - Main evaluation script
  - `evaluate_policies.py` - Baseline policy evaluation
  - `evaluate_masked_policy.py` - PPO evaluation
  - `evaluate_dqn_*.py` - DQN evaluation

- **validation/** - Validation scripts (3 scripts)
  - `validate_simulator.py` - Simulator fidelity checks
  - `validate_test_set_separation.py` - Train/test split validation
  - `validate_final_evaluation.py` - 240-run validation

### Analysis & Utilities
- **analysis/** - Analysis and diagnostic scripts
  - `analyze_final_evaluation.py` - Statistical analysis
  - `analyze_calendar_day_results.py` - Calendar day analysis
  - `diagnose_*.py` - Diagnostic scripts
  - `calculate_power_analysis.py` - Statistical power calculation

- **utils/** - Utility scripts
  - `generate_thesis_update_guide.py` - Thesis table generator
  - `select_30_consecutive_days.py` - Calendar day selector
  - `generate_calendar_day_json.py` - JSON generator
  - `test_*.py` - Testing utilities

### Results & Documentation
- **models/** - Trained models and results
  - XGBoost models (TMC, FTR, OT predictors)
  - PPO/DQN checkpoints
  - CSV evaluation results
  - Markdown reports

- **figures/** - Thesis figures (7 PNG files)
  - `figure5_1_simulator_validation.png`
  - `figure5_2_action_masking_impact.png`
  - `figure5_3_policy_performance.png`
  - `figure5_4_learning_curve.png`
  - `figure5_5_cost_distribution.png`
  - `figure5_6_simulator_fidelity.png`
  - `figure5_7_action_masking_mechanism.png`

- **documentation/** - Current documentation (6 files)
  - `CLAUDE_CODE_HANDOFF_THESIS_QUESTIONS.md` - Q&A for thesis writing
  - `FIGURE_CAPTIONS.txt` - Figure captions for Chapter 5
  - `SYSTEM_VALIDATION_SUMMARY.md` - Validation summary
  - `DQN_EXPERIMENTS_REPORT.md` - DQN results
  - `CALENDAR_DAY_EVALUATION_GUIDE.md` - Evaluation guide
  - `FIGURE_PLACEMENT_GUIDE.md` - Figure placement guide

- **archive/** - Historical documentation (8 files)
  - Older reports and summaries
  - Investigation reports
  - Legacy analysis documents

- **docs/** - Technical documentation (5 files)
  - `COMPREHENSIVE_AUDIT_REPORT_DEC2025.md` - Full audit
  - `rl_mathematical_formulation.md` - Mathematical notation
  - `mdp_vs_bandit_justification.md` - MDP justification
  - `rl_validation_findings.md` - Validation findings
  - `section5_final_results_summary.md` - Final results

---

## 2. Group Baseline (Group Project)

**Path**: `group-baseline/`

Contains the original group deliverable:

- **libPBL2425NovaNOS/** - NOS pipeline library
  - `modelling/` - XGBoost training pipeline
  - `preprocessing/` - Data preprocessing
  - `config/` - Configuration files

- **config/** - Group configuration files
- **notebooks/** - Jupyter notebooks from group work
- **scripts/** - Group analysis scripts

---

## 3. Section 3 E2E Models (Chapter 3)

**Path**: `section3-e2e-models/`

Contains basic simulation models from Chapter 3:

- **model_e2e_xgb/** - End-to-end XGBoost models
  - TMC, FTR, OT predictors for Chapter 3

---

## 4. Root Files

### Standard Repository Files
- **README.md** - Project overview and setup instructions
- **LICENSE** - MIT License
- **CITATION.bib** - BibTeX citation for thesis
- **requirements.txt** - Python package dependencies

### Tests
- **tests/** - Unit tests and test utilities

### Thesis Document
- **Thesis_full_v01_251214_v01.pdf** - Complete thesis PDF

---

## Key File Locations

### For Running Experiments
- RL environment: `rl-section/environment/call_center_env.py`
- Main evaluation: `rl-section/evaluation/evaluate_calendar_days.py`
- XGBoost training: `group-baseline/libPBL2425NovaNOS/modelling/modelling.py`

### For Analysis
- Statistical analysis: `rl-section/analysis/analyze_final_evaluation.py`
- Simulator validation: `rl-section/validation/validate_simulator.py`
- Power analysis: `rl-section/analysis/calculate_power_analysis.py`

### For Thesis Writing
- Handoff Q&A: `rl-section/documentation/CLAUDE_CODE_HANDOFF_THESIS_QUESTIONS.md`
- Figure captions: `rl-section/documentation/FIGURE_CAPTIONS.txt`
- Final results: `rl-section/docs/section5_final_results_summary.md`
- All figures: `rl-section/figures/`

---

## Branch Structure

- **main** - Original group deliverable (frozen)
- **marvin-thesis** - Primary thesis development branch
- **organize-rl-and-group** - Current branch with clean organization
- **backup-20251215** - Backup before reorganization

---

## Notes

1. **RL Section**: All individual thesis work for Chapter 5
2. **Group Baseline**: Original group project (shared with 3 co-authors)
3. **Section 3**: Chapter 3 models (already uploaded separately)
4. **Clean Separation**: No file overlap between sections

**Organization Date**: December 15, 2025
