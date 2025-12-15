# Section 4: Agent Training Recommender System

**Author:** Raquel Santos

## Research Question

Can collaborative filtering techniques identify personalized training recommendations for call center agents based on their performance patterns across different call topics?

## Methodology

- **Matrix Construction**: 652 GCs × 102 topics performance matrix
- **Sparsity**: 72% (not all agents handle all topics)
- **Rating Scale**: 1-5 based on median-centered σ-bands (standard deviation bands)

### Methods Implemented
1. **User-based Collaborative Filtering**: Find similar agents, recommend based on their strengths
2. **Item-based Collaborative Filtering**: Find similar topics, recommend based on topic patterns
3. **Matrix Factorization**: Latent factor decomposition for pattern discovery

## Key Results

The recommender system identifies:
- Topics where each agent underperforms relative to peers
- Similar agents who excel at those topics
- Personalized training priorities per agent

## Dependencies

This section is **standalone** but uses GC-topic performance metrics derived from the same base call data.

## Files

| File | Description |
|------|-------------|
| `Thesis.Section4.ipynb` | Main Jupyter notebook with full implementation and analysis |

## How to Run

```bash
cd section4-recommender
jupyter notebook Thesis.Section4.ipynb
```

## Requirements

- Python 3.8+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- scipy (for sparse matrices)

## Methodology Details

### Rating Calculation
Agent performance on each topic is converted to a 1-5 rating:
- Rating based on how many standard deviations from median performance
- Higher rating = better performance relative to peers

### Recommendation Generation
For each agent:
1. Identify topics with low ratings (underperformance)
2. Find similar agents with high ratings on those topics
3. Generate ranked training recommendations

## Notes

- Data is not included (NOS confidential)
- The notebook contains visualizations of the performance matrix
- Recommendations are personalized per agent
