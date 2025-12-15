# Investigation Plan: XGBoost vs Rule-Based Performance Gap Discrepancy

## Problem Statement

**Critical Discrepancy**: In final RL evaluation, Greedy XGBoost (€20.30/call) and Rule-Based (€20.69/call) show only 1.92% difference. However, in group work:
- Rule-Based had 40% pairwise accuracy (worse than random 50%)
- Rule-Based had negative R²
- XGBoost significantly outperformed Rule-Based

This doesn't make sense. Need to verify:
1. Are the XGBoost models the same as group work?
2. Is the rule-based policy implementation the same?
3. Did code drift introduce bugs?
4. Are we measuring different things?

---

## Investigation Steps

### Phase 1: Find Group Work Artifacts (CRITICAL)

**Goal**: Locate original group work notebooks, evaluation scripts, and documented results

**Actions**:
1. Search `libPBL2425NovaNOS/` for Jupyter notebooks with:
   - Pairwise accuracy calculations
   - R² scores (TMC, FTR, OT)
   - Model evaluation/comparison
   - Rule-based baseline evaluation

2. Search for Python scripts with:
   - `pairwise_accuracy` or `pairwise` in function names
   - Model comparison/evaluation code
   - Original rule-based implementation

3. Look for documentation:
   - README files with results
   - Performance reports
   - Model validation outputs

**Key Metrics to Find**:
- Exact pairwise accuracy values (mentioned: ~40% for Rule-Based)
- Exact R² values (mentioned: negative for Rule-Based, positive for XGBoost)
- How pairwise accuracy was calculated
- TMC R² from group work (compare to current 11%)

---

### Phase 2: Verify XGBoost Models

**Goal**: Confirm current models are identical to group work models

**Actions**:
1. Check model file metadata:
   - Creation dates of `models/model_tmc.joblib`, `model_ftr.joblib`, `model_ot.joblib`
   - Compare to git history of model training scripts

2. Verify training process:
   - Run `modelling.py` logs to see recorded R² scores
   - Compare feature engineering between group work and current code
   - Verify same dataset split (test_indices.npy)

3. Test model predictions:
   - Use diagnostic script to calculate R² on test set
   - Compare to group work R² values
   - Check if feature construction is identical

**Expected Outcome**: Models should be identical with ~11% R² for TMC

---

### Phase 3: Compare Rule-Based Implementations

**Goal**: Verify rule-based policy hasn't changed or been "accidentally improved"

**Actions**:
1. Find original rule-based implementation in group work
   - Search for baseline/oracle implementations
   - Check notebooks for rule-based logic

2. Compare implementations line-by-line:
   - Current: `baseline_policies.py::_estimate_rule_based_cost()` (lines 246-307)
   - Current: Uses topic+agent historical stats from `agent_topic_stats.csv`
   - Original: Find equivalent code in `libPBL2425NovaNOS/`

3. Check git history:
   - When was `baseline_policies.py` created?
   - Were there modifications to rule-based logic?
   - When was `agent_topic_stats.csv` generated?

**Red Flags to Look For**:
- Current rule-based uses sophisticated topic+agent conditioning
- If original was simpler (e.g., only topic-based), this could explain performance improvement
- Check if "rule-based" in group work vs RL thesis means different things

---

### Phase 4: Reconcile Evaluation Metrics

**Goal**: Understand if we're measuring different things

**Actions**:
1. **Pairwise Accuracy** (group work metric):
   - Find exact definition from group work code
   - Likely: "Did model select same agent as actual historical routing?"
   - This measures agent selection correctness, NOT cost optimization

2. **Cost per Call** (RL evaluation metric):
   - Current: Expected cost based on TMC, FTR, OT predictions
   - Different objective: Minimize cost vs match historical routing

3. **Key Insight**:
   - Low pairwise accuracy doesn't necessarily mean high cost
   - If multiple agents have similar costs, "wrong" agent selection could still achieve low cost
   - Rule-Based might pick different agents but with similar costs to XGBoost choices

**Test This Hypothesis**:
- Calculate pairwise accuracy for current evaluation
- Check agent selection agreement between XGBoost and Rule-Based
- See if they pick different agents with similar costs

---

### Phase 5: Run Diagnostic Analysis

**Goal**: Quantify the actual differences in model behavior

**Actions**:
1. Fix and run `diagnose_policy_gap.py`:
   - Fix missing engineered features error
   - Calculate R² on test set for all models
   - Compare component predictions (TMC, FTR, OT)
   - Analyze agent selection agreement

2. Calculate pairwise accuracy for current models:
   - Load test set with actual historical routing
   - Compare predictions to actual agent assignments
   - See if we reproduce ~40% for Rule-Based

3. Analyze cost distributions:
   - Are agent costs highly variable or clustered?
   - Do multiple agents achieve similar costs?
   - Is cost optimization easier than agent selection?

---

## Investigation Priorities

**PRIORITY 1** (HIGHEST): Find group work artifacts
- Without original code/notebooks, we can't compare implementations
- Need exact pairwise accuracy calculation method
- Need documented R² values

**PRIORITY 2**: Check git history
- When were baseline_policies.py and models created?
- Any modifications to rule-based logic?
- Timeline of development

**PRIORITY 3**: Compare implementations
- Line-by-line comparison of rule-based logic
- Verify XGBoost feature construction

**PRIORITY 4**: Run diagnostics
- Calculate missing metrics (pairwise accuracy, R²)
- Test hypothesis about cost vs accuracy

---

## Expected Findings

**Scenario A**: Implementations Differ
- Rule-based in group work was simpler (naive averaging)
- Current rule-based uses sophisticated topic+agent conditioning
- This explains performance convergence

**Scenario B**: Different Metrics
- Pairwise accuracy and cost are weakly correlated
- Multiple agents achieve similar costs
- XGBoost and Rule-Based select different agents with similar costs

**Scenario C**: Model Drift
- Models were retrained and improved
- Current models have different performance characteristics
- Need to verify model provenance

**Scenario D**: Data/Feature Bug
- Feature construction differs between group work and RL thesis
- Models are same but receiving different inputs
- Test set differs

---

## Success Criteria

Investigation complete when we can answer:
1. ✓ Are current XGBoost models the same as group work? (Yes/No + proof)
2. ✓ Is rule-based implementation the same? (Yes/No + line-by-line comparison)
3. ✓ What explains the performance gap discrepancy? (Root cause identified)
4. ✓ Should we trust the RL evaluation results? (Recommendation)

---

## Next Steps After Investigation

**If implementations differ**: Document changes, justify current approach
**If metrics differ**: Explain why cost optimization shows smaller gap than pairwise accuracy
**If models differ**: Decide whether to use original models or current models
**If bug found**: Fix and re-run evaluation

---

**READY FOR USER APPROVAL**
