# Optimizing the Match Between Contact Managers and Clients for a Portuguese Telco Company

**Master's Thesis in Business Analytics**
Nova School of Business and Economics, 2024

## Authors

- Maria Teresa Daffan
- Runa Maria Kleppek
- Marvin Schumann
- Raquel Santos

**Company Partner:** NOS (Portuguese telecommunications)

## Project Overview

NOS routes approximately 300,000 calls per month to ~653 contact managers (GCs). This thesis develops machine learning approaches to optimize call routing by predicting and minimizing call handling costs.

The cost of each call is defined as:

```
Cost = (TMC/60) × 0.35 + (1-FTR) × (TMC/60) × 0.35 + OT × 22
```

Where:
- **TMC** (Tempo Médio de Chamada): Average call duration in seconds
- **FTR** (First Time Resolution): Whether the issue was resolved on first contact (0 or 1)
- **OT** (Ordem de Trabalho): Whether a technician visit was required (0 or 1)

## Repository Structure

```
├── section1-group-baseline/      # XGBoost models for TMC, FTR, OT prediction
├── section2-llm-ivr/             # LLM-powered IVR prototype
├── section3-e2e-models/          # End-to-end vs fragmented cost prediction
├── section4-recommender/         # Agent training recommender system
├── section5-reinforcement-learning/  # RL-based call routing optimization
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── CITATION.bib                  # Citation information
└── LICENSE                       # MIT License
```

## Section Summaries

### Section 1: Group Baseline (XGBoost Models)
**Authors:** All

Establishes the baseline prediction models using XGBoost:
- **TMC Model**: Regression (R² = 0.12)
- **FTR Model**: Classification (AUC = 0.72)
- **OT Model**: Classification (AUC = 0.65)

These models are used by Sections 3 and 5.

### Section 2: LLM-Powered IVR
**Author:** Runa Maria Kleppek

Prototype of an AI-powered Interactive Voice Response system using GPT-4o, demonstrating natural language conversation and structured handover package generation.

### Section 3: End-to-End vs Fragmented Models
**Author:** Maria Teresa Daffan

Compares predicting call cost directly (E2E) versus combining individual component predictions. E2E XGBoost achieves R² = 0.108, Spearman ρ = 0.324.

### Section 4: Agent Recommender System
**Author:** Raquel Santos

Collaborative filtering system for personalized agent training recommendations based on a 652 GCs × 102 topics performance matrix.

### Section 5: Reinforcement Learning
**Author:** Marvin Schumann

PPO with action masking for call routing optimization. Key finding: RL underperforms Greedy XGBoost by 6.5% due to simulator fidelity limitations, but demonstrates the critical importance of action masking.

## Section Dependencies

```
Section 1 (Group Baseline)
    │
    ├──── Section 3 (uses same data methodology)
    │
    └──── Section 5 (DEPENDS ON trained XGBoost models)
                └── Builds simulator using Section 1's predictions

Section 2 (LLM IVR) ─── Standalone

Section 4 (Recommender) ─── Standalone (uses derived GC-topic matrix)
```

## Quick Start

### Prerequisites
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`

### Running Each Section

**Section 1 (Group Baseline):**
```bash
cd section1-group-baseline
python run_pipeline.py
```

**Section 2 (LLM IVR):**
```bash
cd section2-llm-ivr
python nos_voice_ivr_prototype.py
```

**Section 3 (E2E Models):**
```bash
cd section3-e2e-models/model_e2e_xgb
python train.py
```

**Section 4 (Recommender):**
```bash
cd section4-recommender
jupyter notebook Thesis.Section4.ipynb
```

**Section 5 (Reinforcement Learning):**
```bash
cd section5-reinforcement-learning
PYTHONPATH=. python3 evaluation/evaluate_calendar_days.py --help
```

## Data Availability

The data used in this thesis is proprietary to NOS and cannot be shared publicly. The code is provided for reference and reproducibility purposes.

## Key Abbreviations

| Abbreviation | Portuguese | English |
|--------------|------------|---------|
| GC | Gestor de Cliente | Contact Manager/Agent |
| TMC | Tempo Médio de Chamada | Average Call Time |
| FTR | - | First Time Resolution |
| OT | Ordem de Trabalho | Work Order (Technician Visit) |

## Citation

See `CITATION.bib` for citation information.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
