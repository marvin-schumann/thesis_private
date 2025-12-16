# Section 4: Agent Training Prioritization System

**Author:** Raquel Santos

## Research Question

Can collaborative filtering and matrix factorization techniques be used to systematically identify agent skill gaps and generate business-relevant training priorities across call topics, based on relative performance patterns?

## Scope of This Notebook

This notebook implements a training prioritization system, not a strength-based recommender or an operational decision engine. Its objective is to identify performance weaknesses that merit targeted training intervention, while aligning recommendations with business-relevant topic structures.

The notebook includes:
- Data preparation and performance aggregation
- Rating construction methodology
- Model comparison using a unified evaluation pipeline
- Final training prioritization using matrix factorization with topic reaggregation

## Methodology

### Performance Matrix Construction
- Matrix dimensions: 652 GCs × 102 call topics
- Entries reflect relative agent performance by topic
- Sparsity: ~72%, as agents are not exposed to all topics
- Only information available at call time is used to avoid leakage

### Rating Transformation
Agent–topic performance is converted to an ordinal 1–5 rating scale:
- Ratings are based on median-centered standard deviation bands
- The scale captures relative underperformance or overperformance within each topic
- Lower ratings indicate training-relevant weaknesses, not absolute inability

This transformation enables collaborative filtering while preserving the diagnostic nature of the signal.

### Models Tested
The following approaches are evaluated under a consistent train/test pipeline:

**User-based Collaborative Filtering**
Estimates expected agent performance by referencing peers with similar overall performance profiles.

**Item-based Collaborative Filtering**
Leverages similarity across call topics based on shared agent performance patterns.

**Matrix Factorization (MF)**
Learns latent agent–topic factors to infer missing or weak performance signals in sparse settings.

Model comparison focuses on predictive accuracy and robustness under sparsity, rather than deployment or real-time use.

### Final Training Prioritization System
The final system combines matrix factorization with business-driven topic reaggregation to generate actionable training priorities.

For each agent, the system:
1. Identifies topics with consistently low predicted performance
2. Reaggregates granular call topics into business-relevant training areas
3. Produces a ranked list of training priorities, emphasizing:
   - Severity of relative underperformance
   - Consistency across related topics
   - Practical relevance for training design

The output is intended to support training allocation and curriculum planning, not performance ranking or call routing.

## Business Relevance

This system is designed as a complement to efficiency-driven call routing.

While routing optimization tends to:
- Concentrate expertise
- Reduce exposure for weaker agents

This training prioritization approach:
- Surfaces persistent or hidden skill gaps
- Enables targeted, needs-based training
- Helps mitigate skill polarization in outsourced, high-turnover call center environments

## Dependencies

This notebook is analytically standalone but relies on GC–topic performance metrics derived from the same underlying call-level dataset used elsewhere in the thesis.

### External Mapping File
The final training prioritization step requires an external Excel file:

| File | Description |
|------|-------------|
| `topics_agg.xlsx` | Mapping table used to reaggregate granular call topics into business-relevant training topic groups |

- The file is loaded during the final recommender stage
- It defines the correspondence between original topic hierarchies and aggregated training domains
- The file is not included due to data confidentiality but is required to run the notebook end-to-end

## Files

| File | Description |
|------|-------------|
| `Thesis.Section4.ipynb` | End-to-end implementation of the agent training prioritization system |

## How to Run

```bash
cd section4-recommender
# Ensure topics_agg.xlsx is present in the working directory
jupyter notebook Thesis.Section4.ipynb
```

## Requirements

- Python 3.8+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- scipy (sparse matrix operations)

## Notes

- Underlying data is confidential (NOS)
- Outputs are agent-specific and developmental
- Results should be interpreted as decision support for training, not as automated performance judgments
