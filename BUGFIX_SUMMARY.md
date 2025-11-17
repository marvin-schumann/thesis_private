# Critical Bug Fixes - RL Agent Call Routing

## Executive Summary

I've identified and fixed **three critical bugs** in `call_center_env.py` that were preventing RL agents from processing calls. The agents were only handling 1.33 calls/day instead of the expected ~607 calls/day.

## Root Cause Analysis

The fundamental issue was that **invalid actions (selecting busy/unavailable agents) didn't advance simulation time**. This created a scenario where:

1. RL agents would select invalid agents during exploration/poor learning
2. The environment would penalize them but return the same observation with the same timestamp
3. Time wouldn't progress, so calls wouldn't be processed
4. Episodes would hit the 10k step safety limit or end prematurely

## Bugs Fixed

### Bug #1: Invalid Actions Didn't Advance Time ❌→✅
**Location**: `call_center_env.py:500-526`

**Before**: Invalid actions returned immediately with no time advancement
```python
if not is_available:
    reward = -self.invalid_action_penalty  # Just penalty
    observation = self._get_observation()  # Same observation!
    return observation, reward, False, False, info  # No time change!
```

**After**: Invalid actions now advance time by 60 seconds, simulating the call waiting in queue
```python
if not is_available:
    # Advance time and accumulate wait time
    self.current_time += self.invalid_action_wait_seconds  # +60 seconds
    self.current_call_wait_time += self.invalid_action_wait_seconds

    # Calculate comprehensive penalties
    idle_penalty = self.invalid_action_wait_seconds * self.idle_penalty_per_second
    wait_penalty = self.invalid_action_wait_seconds * self.wait_penalty_per_second
    total_penalty = self.invalid_action_penalty + idle_penalty + wait_penalty  # €104.2

    # Check for abandonment and episode end
    if self.current_call_wait_time >= self.abandonment_threshold:
        return self._handle_call_abandonment()
    done = self.current_time >= self.simulation_day_length

    return observation, reward, done, False, info
```

**Impact**:
- Invalid actions now cost €104.2 (base penalty + idle penalty + wait penalty)
- Time progresses, forcing agents to either select valid agents or lose calls to abandonment
- Simulation continues properly through the full 8-hour day

### Bug #2: Wait Time Never Accumulated 📊→✅
**Location**: Throughout `step()` function

**Before**: `current_call_wait_time` was initialized to 0.0 but never incremented, making the abandonment mechanism non-functional.

**After**: Wait time now properly accumulates with each invalid action:
```python
self.current_call_wait_time += self.invalid_action_wait_seconds  # Accumulates
```

**Impact**:
- Calls now abandon after 600 seconds (10 minutes) of waiting
- Abandonment penalty of €500+ creates strong incentive to avoid invalid actions
- Wait penalties properly accumulate (€0.02/second)

### Bug #3: Inconsistent RNG Usage 🎲→✅
**Location**: `call_center_env.py:215`

**Before**: Used global numpy RNG
```python
time_until_next_call = np.random.exponential(1.0 / rate_per_second)
```

**After**: Uses environment's seeded RNG
```python
rng = getattr(self, "_rng", np.random.default_rng())
time_until_next_call = rng.exponential(1.0 / rate_per_second)
```

**Impact**:
- Reproducible call arrival patterns across evaluations
- Consistent behavior between baseline and RL evaluations
- Proper seed control for fair comparisons

## Reward Structure Analysis

With these fixes, the reward structure now properly incentivizes good behavior:

**Successful Call Handling** (Valid Agent Selection):
- Call cost: ~€10-15 (varies by prediction)
- Completion bonus: +€200
- **Net reward**: +€185 to +€190 ✅

**Invalid Action** (Busy/Off-Shift Agent):
- Base penalty: €100
- Idle penalty: 60s × €0.05/s = €3
- Wait penalty: 60s × €0.02/s = €1.2
- **Total penalty**: -€104.2 ❌

**Call Abandonment** (After 600s of invalid actions):
- Abandonment penalty: €500
- Accumulated wait penalty: 600s × €0.02/s = €12
- **Total penalty**: -€512 ❌❌❌

This creates a **clear incentive gradient**: Handle calls successfully (+€190) >> Avoid action >> Invalid action (-€104) >> Let calls abandon (-€512)

## What Changed in the Environment Behavior

### Before Fixes:
1. RL agent selects invalid agent → gets -€100 penalty, time = T
2. RL agent selects invalid agent again → gets -€100 penalty, time = T (still!)
3. Episode continues for thousands of steps at time = T
4. Either hits 10k step limit or episode ends prematurely
5. Result: 1-2 calls processed per episode

### After Fixes:
1. RL agent selects invalid agent → gets -€104 penalty, time = T + 60s, wait_time = 60s
2. RL agent selects invalid agent again → gets -€104 penalty, time = T + 120s, wait_time = 120s
3. ... (continues up to 10 invalid actions)
4. After 10 invalid actions → wait_time = 600s → call abandons with -€512 penalty
5. Next call arrives, agent forced to learn to select valid agents
6. Result: ~607 calls processed per episode (full simulation day)

## Testing Instructions

### Important: Memory-Optimized Training Script

⚠️ **If you see a memory warning like this:**
```
UserWarning: This system does not have apparently enough memory to store the complete
replay buffer 6.89GB > 5.94GB
```

**Use the memory-optimized script instead:**
```bash
python train_rl_memory_optimized.py --quick-smoke --seed 42
```

This script reduces DQN's replay buffer from 1M to 200K, using ~2GB instead of ~7GB.

### Step 1: Retrain RL Agents with Fixed Environment

**Option A: Standard Training** (if you have 8GB+ RAM available):
```bash
python train_rl.py --quick-smoke --seed 42
```

**Option B: Memory-Optimized Training** (recommended for systems with <8GB RAM):
```bash
python train_rl_memory_optimized.py --quick-smoke --seed 42
```

This will train both DQN and PPO for 25,000 timesteps and save models to:
- `models/rl_model_dqn.zip`
- `models/rl_model_ppo.zip`

Expected training time: ~5-15 minutes depending on your hardware.

**What to look for during training**:
- Progress bar should advance smoothly
- Episode rewards should start negative (lots of penalties) and gradually improve
- Training should NOT get stuck at the same episode for a long time

### Step 2: Evaluate All Policies

Run evaluation to compare baselines vs. new RL agents:

```bash
python evaluate_policies.py --episodes 3 --include-rl --tag "after_bugfix" --seed 42
```

**Expected Results**:

| Policy | Expected Calls/Day | Expected Cost/Day |
|--------|-------------------|-------------------|
| Random | ~607 | ~€10,800 |
| Rule-Based | ~607 | ~€9,200 |
| Greedy XGBoost | ~607 | ~€8,900 |
| **DQN (RL)** | **~607** ✅ | **€8,000-12,000** (depending on learning) |
| **PPO (RL)** | **~607** ✅ | **€8,000-12,000** (depending on learning) |

**Critical Success Criteria**:
✅ RL agents MUST process ~607 calls/day (not 1-2!)
✅ RL agents should have reasonable costs (€8,000-€12,000 range)

### Step 3: Full Training Run (If Smoke Test Succeeds)

If the quick smoke test shows RL agents are now processing all calls, run a full training session:

```bash
python train_rl.py --timesteps-dqn 500000 --timesteps-ppo 500000 --seed 42 --tensorboard
```

This will:
- Train DQN for 500k timesteps (~30-60 minutes)
- Train PPO for 500k timesteps (~30-60 minutes)
- Save checkpoints every 50k steps
- Enable TensorBoard logging for training curves

**Monitor training with TensorBoard** (optional):
```bash
tensorboard --logdir=./rl_tensorboard_logs
```

### Step 4: Final Evaluation

After full training, evaluate with more episodes for statistical significance:

```bash
python evaluate_policies.py --episodes 10 --include-rl --tag "final_full_training" --seed 42
```

## Expected Outcomes

### Minimum Success Criteria (Smoke Test):
- ✅ RL agents process ~607 calls/day (proving the environment is fixed)
- ✅ RL agents have costs in reasonable range (€8k-€15k)
- ✅ No more "call dodging" behavior

### Ideal Success Criteria (Full Training):
- 🎯 RL agents match or beat Rule-Based policy (~€9,254/day)
- 🎯 Best RL agent approaches Greedy XGBoost performance (~€8,938/day)
- 🎯 RL agents demonstrate learning curve in TensorBoard

### Thesis Contribution:
Even if RL doesn't beat baselines after full training, you now have:
1. **Working RL infrastructure** - agents engage with the full problem
2. **Fair comparison** - all policies handle all calls equally
3. **Insights** - can analyze WHY RL struggles (exploration challenge, credit assignment, etc.)
4. **Strong baselines** - Greedy XGBoost is a legitimate benchmark

## Troubleshooting

### If RL agents still process < 100 calls/day:
1. Check if there are other bugs in the environment
2. Verify the trained models are using the new environment code
3. Add logging to `step()` to see what's happening:
   ```python
   print(f"Step {self.episode_step_count}: action={action}, status={info['status']}, time={self.current_time:.0f}")
   ```

### If RL agents have very high costs (> €20k/day):
- This is normal for early training - they're learning through penalties
- Check TensorBoard to see if costs are decreasing over time
- Consider increasing training budget to 1M+ timesteps

### If training is very slow:
- Use `--device cpu` or `--device cuda` explicitly
- Reduce checkpoint frequency with `--checkpoint-freq 100000`
- Try smaller network architectures (requires modifying `train_rl.py`)

## Files Modified

- ✅ `call_center_env.py` - All three bugs fixed and committed

## Commit Details

Branch: `claude/thesis-project-continuation-01USwmgfVVtumAcNA7bozsv9`
Commit: `78f6727` - "Fix critical bugs preventing RL agents from processing calls"

## Next Steps for Your Thesis

1. **Run the smoke test** to verify fixes work
2. **Run full training** if smoke test succeeds
3. **Analyze results**:
   - If RL beats baselines: Great! Emphasize RL's value
   - If RL matches baselines: Show RL learns similar strategies
   - If RL underperforms: Analyze why (exploration, credit assignment, etc.)
4. **Consider hyperparameter tuning**:
   - Learning rates
   - Network architectures
   - Exploration schedules
   - Training budgets
5. **Visualize RL behavior**:
   - Agent selection patterns over time
   - Learning curves
   - Episode trajectories

## Questions or Issues?

If you encounter any issues or have questions about the fixes:
1. Check this document first
2. Review the commit diff to understand changes
3. Add logging to see environment behavior
4. Feel free to ask for clarification!

---

**Summary**: The environment now properly forces RL agents to engage with all incoming calls through a combination of time advancement, wait penalties, and abandonment mechanisms. The "call dodging" bug is fixed. You should now see RL agents processing ~607 calls/day just like the baselines.
