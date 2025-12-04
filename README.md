# Reinforcement Learning for Call Center Routing: Master's Thesis Repository

**Author:** Marvin Schumann
**Institution:** Nova School of Business and Economics
**Year:** 2025

This repository contains the complete codebase for a master's thesis on applying Reinforcement Learning to optimize call center agent routing, building upon the NOS Project for telecom customer service optimization.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [RL Experiments](#rl-experiments)
6. [NOS Project (Base System)](#nos-project-base-system)
7. [Documentation](#documentation)
8. [Setup Instructions](#setup-instructions)
9. [License](#license)

---

## Project Overview

This thesis investigates the application of **Reinforcement Learning (RL)** to optimize call center operations, specifically focusing on **agent routing** - the problem of assigning incoming calls to available agents to minimize operational costs while maintaining service quality.

### Two-Part System

**Part 1: NOS Project (Base System)**
- **Original Group Project**: Predictive modeling pipeline for telecom call center metrics
- **Purpose**: Train XGBoost models to predict call duration (TMC), first contact resolution (FTR), and technician dispatches (OT)
- **Output**: Predictive models used as cost estimators in the RL environment

**Part 2: RL Thesis Work (Individual Contribution)**
- **Custom Gym Environment**: `CallCenterEnv` simulating agent routing decisions
- **Action Masking**: Critical constraint enforcement for agent availability
- **Baseline Policies**: Greedy XGBoost, Rule-Based, Random assignment
- **RL Algorithms**: Masked PPO (main), DQN (comparative analysis)
- **Simulator Validation**: Statistical tests confirming environment fidelity

---

## Key Results

### Main Findings

1. **Action Masking is Critical (Algorithm-Independent)**
   - Unmasked PPO: ~98% invalid actions, <2% efficiency
   - Unmasked DQN: 89.2% invalid actions, 8% efficiency
   - Masked PPO: 0% invalid actions, 104% efficiency ✅

2. **MaskablePPO is the Only Viable RL Solution**
   - Masked PPO: 614.5 calls/day (€16.21/call)
   - Masked DQN + ActionMasker: 30.6 calls/day (€16.92/call)
   - Generic wrappers fail with value-based methods

3. **Simulator Noise Affects Learning Quality**
   - Greedy XGBoost (no learning): €15.06/call
   - Masked PPO (learned): €16.21/call (+7.6%)
   - Prediction errors in simulator limit RL performance

### Performance Comparison

| Policy | Calls/Day | Cost/Call | Efficiency | Invalid Actions |
|--------|-----------|-----------|------------|-----------------|
| **Baselines** | | | | |
| Greedy XGBoost | 614.5 | €15.06 | 104% | 0% |
| Rule-Based | 614.5 | €15.54 | 104% | 0% |
| Random | 614.5 | €18.18 | 104% | 0% |
| **Masked RL** | | | | |
| Masked PPO | 614.5 | €16.21 | 104% | 0% |
| Masked DQN | 30.6 | €16.92 | 5% | ~95% |
| **Unmasked RL** | | | | |
| Unmasked PPO | ~1.3 | N/A | <2% | ~98% |
| Unmasked DQN | 47.0 | €15.35 | 8% | 89.2% |

---

## Repository Structure

```
thesis_private/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── docs/                              # Comprehensive documentation
│   ├── SYSTEM_VALIDATION_SUMMARY.md   # Final validation report
│   ├── DQN_EXPERIMENTS_REPORT.md      # DQN analysis
│   ├── section5_final_results_summary.md
│   ├── rl_mathematical_formulation.md
│   ├── rl_validation_findings.md
│   └── mdp_vs_bandit_justification.md
│
├── figures/                           # Publication-quality figures
│   ├── figure5_1_simulator_validation.png
│   ├── figure5_2_action_masking_impact.png
│   ├── figure5_3_policy_performance.png
│   ├── figure5_4_learning_curve.png
│   ├── figure5_5_cost_distribution.png
│   ├── figure5_6_simulator_fidelity.png
│   └── figure5_7_action_masking_mechanism.png
│
├── models/                            # Trained models and results
│   ├── rl_model_masked_ppo.zip        # Final Masked PPO model
│   ├── rl_model_dqn_masked.zip        # Masked DQN model
│   ├── rl_model_dqn_unmasked.zip      # Unmasked DQN model
│   ├── model_ftr.joblib               # XGBoost FTR predictor
│   ├── model_ot.joblib                # XGBoost OT predictor
│   ├── model_tmc.joblib               # XGBoost TMC predictor
│   ├── final_evaluation_results.csv   # Baseline results
│   ├── masked_ppo_results.csv         # Masked PPO results
│   ├── dqn_unmasked_results.csv       # DQN unmasked results
│   ├── dqn_masked_results.csv         # DQN masked results
│   └── checkpoints_*/                 # Training checkpoints
│
├── config/                            # Configuration
│   ├── settings.py                    # Pipeline settings
│   └── .env                           # Database credentials (not in git)
│
├── RL Environment & Policies/
├── call_center_env.py                 # Core RL environment (unmasked)
├── call_center_env_masked.py          # Action-masked environment
├── baseline_policies.py               # Greedy, Rule-Based, Random
│
├── RL Training Scripts/
├── train_rl_masked.py                 # Train Masked PPO
├── train_dqn_unmasked.py              # Train Unmasked DQN
├── train_dqn_masked.py                # Train Masked DQN
│
├── RL Evaluation Scripts/
├── evaluate_policies.py               # Evaluate all baselines
├── evaluate_masked_policy.py          # Evaluate Masked PPO
├── evaluate_dqn_unmasked.py           # Evaluate Unmasked DQN
├── evaluate_dqn_masked.py             # Evaluate Masked DQN
│
├── Validation & Visualization/
├── validate_simulator.py              # Statistical simulator validation
├── generate_thesis_figures.py         # Generate publication figures
│
├── NOS Project (Original)/
├── libPBL2425NovaNOS/                 # Core NOS pipeline library
│   ├── data_access/                   # Database loaders
│   ├── data_cleaning/                 # Data cleaning modules
│   └── modelling/                     # XGBoost training
├── check_feature_importance.py
├── compute_gc_residuals.py
├── residual_adjustments.py
├── run_pipeline.py                    # Run NOS pipeline
│
├── notebooks/                         # Jupyter notebooks (exploration)
└── tests/                             # Unit tests
```

---

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository_url>
cd thesis_private

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Baseline Evaluation

```bash
# Evaluate Greedy XGBoost, Rule-Based, and Random policies
python evaluate_policies.py --episodes 10 --seed 42
```

### 3. Train Masked PPO

```bash
# Train for 200k timesteps (recommended)
python train_rl_masked.py --timesteps 200000 --seed 42 --device cpu
```

### 4. Evaluate Masked PPO

```bash
# Evaluate trained Masked PPO model
python evaluate_masked_policy.py --episodes 10 --seed 42
```

### 5. Generate Thesis Figures

```bash
# Generate all publication-quality figures
python generate_thesis_figures.py
```

---

## RL Experiments

### Environment: `CallCenterEnv`

**State Space:**
- Call features: topic, wait time, priority
- Agent features: availability, skill level, performance history
- Temporal features: time of day, day of week

**Action Space:**
- Discrete: Assign call to one of N available agents
- Constraint: Only available agents (not on call, within shift)

**Reward Function:**
```python
reward = -(call_handling_cost + technician_cost + repeat_call_cost + wait_time_penalty)
```

**Episode Termination:**
- Simulation day length reached (default: 28,800s = 8 hours)
- No more calls in queue

### Training Commands

```bash
# Masked PPO (main algorithm)
python train_rl_masked.py \
    --timesteps 200000 \
    --seed 42 \
    --device cpu \
    --checkpoint-freq 50000

# Unmasked DQN (comparison)
python train_dqn_unmasked.py \
    --timesteps 50000 \
    --seed 42 \
    --device cpu \
    --checkpoint-freq 10000

# Masked DQN (comparison)
python train_dqn_masked.py \
    --timesteps 200000 \
    --seed 42 \
    --device cpu \
    --checkpoint-freq 50000
```

### Evaluation Commands

```bash
# Evaluate all baselines
python evaluate_policies.py --episodes 10 --seed 42

# Evaluate Masked PPO
python evaluate_masked_policy.py --episodes 10 --seed 42

# Evaluate Unmasked DQN
python evaluate_dqn_unmasked.py --episodes 10 --seed 42

# Evaluate Masked DQN
python evaluate_dqn_masked.py --episodes 10 --seed 42
```

---

## NOS Project (Base System)

The original NOS project provides the predictive models used for cost estimation in the RL environment.

### Pipeline Stages

1. **Data Access** (`libPBL2425NovaNOS/data_access/`)
   - Connect to PostgreSQL database
   - Fetch call, client, and agent data

2. **Data Cleaning** (`libPBL2425NovaNOS/data_cleaning/`)
   - Clean call data
   - Clean client data
   - Clean agent (GC) data

3. **Data Merging** (`libPBL2425NovaNOS/data_cleaning/merger.py`)
   - Merge cleaned datasets
   - Create final modeling dataset

4. **Modeling** (`libPBL2425NovaNOS/modelling/modelling.py`)
   - Train XGBoost models for TMC, FTR, OT
   - Evaluate model performance
   - Simulate GC assignment

### Running NOS Pipeline

```bash
# Run complete NOS pipeline
python run_pipeline.py
```

**Note:** Requires database credentials in `config/.env`:
```bash
DB_HOST="your_db_host"
DB_PORT="your_db_port"
DB_NAME="your_db_name"
DB_USER="your_db_user"
DB_PASSWORD="your_db_password"
```

---

## Documentation

### Comprehensive Reports

- **`SYSTEM_VALIDATION_SUMMARY.md`**: Final validation of all thesis components
  - Environment validation
  - Baseline policy validation
  - Masked PPO validation
  - DQN experiments validation
  - Bug fixes and corrections
  - Thesis claims validation

- **`DQN_EXPERIMENTS_REPORT.md`**: Complete DQN analysis
  - Unmasked DQN results (89.2% invalid actions)
  - Masked DQN results (5.2% efficiency)
  - Comparative analysis with PPO
  - Bug discovery and correction

### Additional Documentation

- `docs/section5_final_results_summary.md`: Results chapter summary
- `docs/rl_mathematical_formulation.md`: MDP formulation
- `docs/rl_validation_findings.md`: Validation findings
- `docs/mdp_vs_bandit_justification.md`: MDP vs bandit justification

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL (for NOS pipeline only)
- 8GB+ RAM (for RL training)

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository_url>
   cd thesis_private
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure database** (NOS pipeline only):
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your credentials
   ```

3. **Verify installation**:
   ```bash
   python -c "import gymnasium; import stable_baselines3; print('RL dependencies OK')"
   python -c "import xgboost; import pandas; print('NOS dependencies OK')"
   ```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code or reference this work, please cite:

```bibtex
@mastersthesis{schumann2025rl_call_center,
  author  = {Schumann, Marvin},
  title   = {Reinforcement Learning for Call Center Agent Routing:
             An Action Masking Approach},
  school  = {Nova School of Business and Economics},
  year    = {2025},
  type    = {Master's Thesis},
  address = {Lisbon, Portugal}
}
```

---

## Contact

**Marvin Schumann**
Nova School of Business and Economics
Email: [your.email@novaims.unl.pt]

---

## Acknowledgments

- Nova School of Business and Economics
- NOS Portugal (Data partner)
- Original NOS Project Team (Base system development)
- Thesis Supervisors

---

**Last Updated:** December 2025
**Status:** ✅ Validated and ready for thesis submission
