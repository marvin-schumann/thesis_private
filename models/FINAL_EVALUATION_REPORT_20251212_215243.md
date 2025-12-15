# Final Thesis Evaluation Report

**Generated**: 2025-12-12 22:17:47
**Timestamp**: 20251212_215243
**Sample Size**: n=30 per mode (exceeds statistical requirement by 76%)

---

## Executive Summary

This report presents publication-ready results from the final thesis evaluation with **n=30 samples per mode**, providing **>95% statistical power** for detecting €1.00 cost differences.

**Key Findings**:
- All 4 policies evaluated: Random, Rule-Based, Greedy XGBoost, Masked PPO
- Statistical rigor: 95% CIs (t-distribution), Bonferroni correction (α'=0.0083), Cohen's d effect sizes
- Methodology validation: Spearman ρ = 1.000 (p=0.0000)

---

## 1. Random Mode Results (n=30)

| Policy | Mean (€/call) | Std | 95% CI | Median | IQR |
|--------|---------------|-----|--------|--------|-----|
| Greedy XGBoost | 23.51 | 0.30 | [23.40, 23.63] | 23.49 | 0.44 |
| Rule-Based | 23.97 | 0.35 | [23.84, 24.10] | 23.96 | 0.58 |
| Masked PPO | 24.75 | 0.33 | [24.62, 24.87] | 24.76 | 0.57 |
| Random | 26.97 | 0.38 | [26.83, 27.12] | 27.10 | 0.69 |


**Rankings (Random Mode)**:
1. **Greedy XGBoost**: €23.51/call
2. **Rule-Based**: €23.97/call
3. **Masked PPO**: €24.75/call
4. **Random**: €26.97/call


---

## 2. Calendar Day Mode Results (n=30)

| Policy | Mean (€/call) | Std | 95% CI | Median | IQR |
|--------|---------------|-----|--------|--------|-----|
| Greedy XGBoost | 20.30 | 1.03 | [19.91, 20.68] | 20.26 | 1.31 |
| Rule-Based | 20.69 | 0.94 | [20.34, 21.05] | 20.73 | 1.22 |
| Masked PPO | 21.62 | 0.96 | [21.26, 21.97] | 21.61 | 1.19 |
| Random | 23.35 | 1.12 | [22.93, 23.77] | 23.43 | 1.36 |


**Rankings (Calendar Day Mode)**:
1. **Greedy XGBoost**: €20.30/call
2. **Rule-Based**: €20.69/call
3. **Masked PPO**: €21.62/call
4. **Random**: €23.35/call


---

## 3. Pairwise Comparisons (Bonferroni-corrected α' = 0.0083)

### Random Mode

| Comparison | t-statistic | p-value | Significant? | Cohen's d | Effect |
|------------|-------------|---------|--------------|-----------|--------|
| Greedy XGBoost vs Masked PPO | t(58)=-15.21 | 0.0000 | ✓ | -3.93 | Large |
| Greedy XGBoost vs Random | t(58)=-39.13 | 0.0000 | ✓ | -10.10 | Large |
| Greedy XGBoost vs Rule-Based | t(58)=-5.46 | 0.0000 | ✓ | -1.41 | Large |
| Masked PPO vs Random | t(58)=-24.27 | 0.0000 | ✓ | -6.27 | Large |
| Masked PPO vs Rule-Based | t(58)=8.90 | 0.0000 | ✓ | 2.30 | Large |
| Random vs Rule-Based | t(58)=31.92 | 0.0000 | ✓ | 8.24 | Large |


### Calendar Day Mode

| Comparison | t-statistic | p-value | Significant? | Cohen's d | Effect |
|------------|-------------|---------|--------------|-----------|--------|
| Greedy XGBoost vs Masked PPO | t(58)=-5.13 | 0.0000 | ✓ | -1.33 | Large |
| Greedy XGBoost vs Random | t(58)=-11.00 | 0.0000 | ✓ | -2.84 | Large |
| Greedy XGBoost vs Rule-Based | t(58)=-1.56 | 0.1246 | ✗ | -0.40 | Medium |
| Masked PPO vs Random | t(58)=-6.45 | 0.0000 | ✓ | -1.67 | Large |
| Masked PPO vs Rule-Based | t(58)=3.76 | 0.0004 | ✓ | 0.97 | Large |
| Random vs Rule-Based | t(58)=9.95 | 0.0000 | ✓ | 2.57 | Large |


---

## 4. Methodology Validation

**Spearman Rank Correlation (Random vs Calendar Day)**:
- ρ = 1.000
- p-value = 0.0000
- Rankings preserved: ✓ YES

| Policy | Random Rank | Calendar Rank | Random Cost | Calendar Cost |
|--------|-------------|---------------|-------------|---------------|
| Random | - | - | €26.97 | €23.35 |
| Rule-Based | - | - | €23.97 | €20.69 |
| Greedy XGBoost | - | - | €23.51 | €20.30 |
| Masked PPO | - | - | €24.75 | €21.62 |


**Interpretation**: Policy rankings ARE preserved across methodologies, validating random sampling approach.

---

## 5. Statistical Rigor

**Sample Size**:
- Required (80% power): n=17
- Final sample: n=30
- Over-sampling: 76%
- Achieved power: >95% for €1.00 detection

**Statistical Methods**:
- Confidence intervals: 95%, t-distribution (df=29, t_crit=2.045)
- Pairwise tests: Independent samples t-test
- Multiple testing: Bonferroni correction (α' = 0.05/6 = 0.0083)
- Effect sizes: Cohen's d (small: <0.2, medium: 0.2-0.8, large: >0.8)

---

**Report Complete**: 2025-12-12 22:17:47
