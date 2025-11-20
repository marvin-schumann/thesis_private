# Mathematical Formulation: Call Routing as a Markov Decision Process

**Document Purpose**: LaTeX-ready mathematical definitions for Master's Thesis Section 5

**Author**: Marvin Schumann
**Date**: November 2025

---

## 1. Problem Definition

We formulate the call center routing problem as a **Markov Decision Process (MDP)** where an agent must sequentially assign incoming calls to available contact managers (agents) to minimize expected cost over an operational period.

---

## 2. MDP Formulation

A Markov Decision Process is defined by the tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$:

### 2.1 State Space ($\mathcal{S}$)

The state $s_t \in \mathcal{S}$ at time $t$ is a vector:

$$
s_t = [\mathbf{x}_t, \mathbf{a}_t, \mathbf{h}_t] \in \mathbb{R}^{N_s}
$$

where:

1. **Call Features** $\mathbf{x}_t \in \mathbb{R}^{N_c}$: Features of the current incoming call
   - Client demographics (age, subscription type)
   - Call metadata (topic classification, time-of-day, day-of-week)
   - Client history (rolling window metrics: mean call duration, issue counts)
   - Topic-specific statistics (average TMC/FTR/OT for this topic)

   Dimension: $N_c \approx 50$ features

2. **Agent Availability** $\mathbf{a}_t \in \{0, 1\}^{N_a}$: Binary vector indicating which agents are available
   $$
   a_t^{(i)} = \begin{cases}
   1 & \text{if agent } i \text{ is free at time } t \\
   0 & \text{if agent } i \text{ is busy or off-shift}
   \end{cases}
   $$

   Dimension: $N_a = 250$ agents

3. **Temporal Context** $\mathbf{h}_t \in \mathbb{R}^2$: Time-related features
   $$
   \mathbf{h}_t = [\text{calls\_in\_system}, \text{hour\_of\_day}]
   $$
   - $\text{calls\_in\_system}$: Number of calls currently being processed (fixed at 1 in our implementation)
   - $\text{hour\_of\_day}$: Current hour within the operational day $\in [0, 8)$

**Total State Dimension**:
$$N_s = N_c + N_a + 2 \approx 50 + 250 + 2 = 302$$

### 2.2 Action Space ($\mathcal{A}$)

The action space is discrete:

$$
\mathcal{A} = \{0, 1, \ldots, N_a - 1\}
$$

An action $a_t \in \mathcal{A}$ represents the **index of the selected agent** from the available pool. With $N_a = 250$ agents, the action space has 250 possible actions.

**Invalid Actions**: If the selected agent is unavailable ($a_t^{(i)} = 0$), the action is considered **invalid** and incurs a penalty (see Section 2.4).

### 2.3 Transition Dynamics ($P$)

The transition function $P: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$ describes the probability distribution over next states:

$$
P(s_{t+1} | s_t, a_t)
$$

#### State Transitions:

1. **Call Features ($\mathbf{x}_t$)**: New call arrives independently
   $$
   \mathbf{x}_{t+1} \sim \text{Historical Distribution}
   $$
   Calls are sampled from the historical dataset, shuffled per episode.

2. **Agent Availability ($\mathbf{a}_t$)**: Updates based on selected action
   $$
   a_{t+1}^{(i)} = \begin{cases}
   0 & \text{if agent } i \text{ was selected at } t \text{ and is now busy until } t + \text{TMC} \\
   1 & \text{if agent } i \text{ was previously busy but call completed} \\
   0/1 & \text{based on shift schedule otherwise}
   \end{cases}
   $$

3. **Temporal Context ($\mathbf{h}_t$)**: Time advances
   $$
   \text{hour\_of\_day}_{t+1} = \text{hour\_of\_day}_t + \Delta t / 3600
   $$
   where $\Delta t$ is the inter-arrival time sampled from a time-dependent Poisson process:
   $$
   \Delta t \sim \text{Exp}(\lambda_h)
   $$
   with arrival rate $\lambda_h$ depending on the current hour $h$.

**Markov Property**: The next state $s_{t+1}$ depends only on $s_t$ and $a_t$, not on the full history. This holds because:
- Call features are sampled independently
- Agent availability is fully determined by current state and action
- Time advancement is deterministic

### 2.4 Reward Function ($R$)

The reward function $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ assigns immediate reward for taking action $a$ in state $s$:

#### Valid Actions:
For a valid action (agent $i$ is available):

$$
R(s_t, a_t) = -C(s_t, a_t)
$$

where $C(s_t, a_t)$ is the **cost** of routing the call to agent $i$:

$$
C(s_t, a_t) = C_{\text{duration}} + C_{\text{repeat}} + C_{\text{OT}}
$$

**Cost Components**:

1. **Duration Cost**: Cost of agent's time on the call
   $$
   C_{\text{duration}} = \frac{\text{TMC}(s_t, a_t)}{60} \times c_{\text{minute}}
   $$
   where:
   - $\text{TMC}(s_t, a_t)$: Predicted call duration in seconds for call $s_t$ with agent $a_t$
   - $c_{\text{minute}} = 0.35$ €/minute: Cost per minute of agent time

2. **Repeat Call Cost**: Expected cost if customer calls back (First Time Resolution failure)
   $$
   C_{\text{repeat}} = \left(1 - P(\text{FTR} | s_t, a_t)\right) \times \frac{\text{TMC}(s_t, a_t)}{60} \times c_{\text{minute}}
   $$
   where $P(\text{FTR} | s_t, a_t)$: Predicted probability of First Time Resolution

3. **Onsite Technician Cost**: Expected cost of dispatching a technician
   $$
   C_{\text{OT}} = P(\text{OT} | s_t, a_t) \times c_{\text{technician}}
   $$
   where:
   - $P(\text{OT} | s_t, a_t)$: Predicted probability of technician dispatch
   - $c_{\text{technician}} = 22.00$ €: Cost of a technician visit

**Total Cost**:
$$
\boxed{C(s_t, a_t) = \frac{\text{TMC}(s_t, a_t)}{60} \times c_{\text{minute}} \times \left(1 + \left(1 - P(\text{FTR} | s_t, a_t)\right)\right) + P(\text{OT} | s_t, a_t) \times c_{\text{technician}}}
$$

**Typical Values**:
- TMC = 600 seconds → Duration cost = €3.50
- FTR = 0.6 → Repeat cost = €1.40
- OT = 0.1 → OT cost = €2.20
- **Total ≈ €7.10** per call

#### Invalid Actions:
For invalid actions (agent is unavailable):

$$
R(s_t, a_t) = -\left(c_{\text{base}} + \Delta t \times c_{\text{wait}} + \Delta t \times c_{\text{idle}}\right)
$$

where:
- $c_{\text{base}} = 100$ €: Base penalty for selecting unavailable agent
- $\Delta t = 60$ seconds: Time penalty (call waits in queue)
- $c_{\text{wait}} = 0.02$ €/second: Cost per second customer waits
- $c_{\text{idle}} = 0.05$ €/second: Cost per second of system idle time

**Total Invalid Action Penalty**: $\approx -104.20$ €

#### Call Abandonment:
If a call waits too long (>600 seconds):

$$
R(s_t, a_t) = -\left(c_{\text{abandon}} + t_{\text{wait}} \times c_{\text{wait}}\right)
$$

where:
- $c_{\text{abandon}} = 500$ €: Abandonment penalty (lost customer goodwill)
- $t_{\text{wait}}$: Total wait time before abandonment

**Total Abandonment Penalty**: $\approx -512$ €

### 2.5 Discount Factor ($\gamma$)

$$
\gamma = 0.99
$$

**Justification**:
With approximately 600 calls per 8-hour operational period, the effective horizon is ~600 steps. With $\gamma = 0.99$, rewards 100 steps in the future are weighted at:

$$
0.99^{100} \approx 0.366
$$

This reflects operational reality: immediate routing decisions matter more than distant future states, but we still consider medium-term consequences.

**Effective Horizon**:
$$
\frac{1}{1 - \gamma} = \frac{1}{1 - 0.99} = 100 \text{ steps}
$$

---

## 3. Objective

The goal is to find an optimal policy $\pi^*: \mathcal{S} \rightarrow \mathcal{A}$ that minimizes the expected cumulative cost over an operational period:

$$
\pi^* = \arg\min_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t C(s_t, \pi(s_t)) \right]
$$

Equivalently, in the RL formulation (maximizing reward):

$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, \pi(s_t)) \right]
$$

where $T \approx 600$ is the number of calls in an 8-hour period.

---

## 4. Oracle Functions (Simulator)

The simulator uses trained supervised learning models as **oracle functions** to predict outcomes:

### 4.1 Call Duration (TMC)

$$
\text{TMC}(s_t, a_t) = f_{\text{TMC}}(\mathbf{x}_t, \mathbf{g}_{a_t}) + \beta_{a_t, \tau_t}^{\text{TMC}} + \epsilon_{\text{TMC}}
$$

where:
- $f_{\text{TMC}}$: XGBoost regression model (trained in Section 4)
- $\mathbf{x}_t$: Call features
- $\mathbf{g}_{a_t}$: Agent $a_t$'s feature vector (historical performance metrics)
- $\beta_{a_t, \tau_t}^{\text{TMC}}$: Agent-topic-specific bias correction
- $\epsilon_{\text{TMC}} \sim \mathcal{N}(0, \sigma_{a_t, \tau_t}^2)$: Stochastic noise (optional)

**Trained Model Performance**: Test $R^2 = 0.12$ (weak predictor)

### 4.2 First Time Resolution (FTR)

$$
P(\text{FTR} | s_t, a_t) = f_{\text{FTR}}(\mathbf{x}_t, \mathbf{g}_{a_t}) + \beta_{a_t, \tau_t}^{\text{FTR}}
$$

where:
- $f_{\text{FTR}}$: XGBoost binary classifier (trained in Section 4)
- Output: Probability $\in [0, 1]$

**Trained Model Performance**: Test ROC-AUC = 0.73 (acceptable)

### 4.3 Onsite Technician (OT)

$$
P(\text{OT} | s_t, a_t) = f_{\text{OT}}(\mathbf{x}_t, \mathbf{g}_{a_t}) + \beta_{a_t, \tau_t}^{\text{OT}}
$$

where:
- $f_{\text{OT}}$: XGBoost binary classifier (trained in Section 4)
- Output: Probability $\in [0, 1]$

**Trained Model Performance**: Test ROC-AUC = 0.65 (mediocre)

### 4.4 Residual Bias Corrections

To account for systematic prediction errors, we apply agent-topic-specific bias adjustments:

$$
\beta_{a, \tau} = \frac{1}{N_{a,\tau}} \sum_{i \in \mathcal{D}_{a,\tau}} (y_i - \hat{y}_i)
$$

where:
- $\mathcal{D}_{a,\tau}$: Set of historical calls handled by agent $a$ with topic $\tau$
- $N_{a,\tau}$: Number of such calls
- $y_i$: Actual outcome
- $\hat{y}_i$: Model prediction

**Shrinkage**: To prevent overfitting on agents/topics with few observations:

$$
\beta_{a,\tau}^{\text{adjusted}} = \beta_{a,\tau} \times \min\left(1, \frac{N_{a,\tau}}{N_{a,\tau} + \lambda}\right)
$$

where $\lambda = 200$ (shrinkage parameter).

---

## 5. MDP vs. Contextual Bandit Discussion

### 5.1 Why MDP?

We model this as an MDP (rather than a Contextual Bandit) because:

1. **Temporal Dependencies**: Agent availability creates state transitions
   - Routing a call to agent $i$ makes them unavailable for future calls
   - This creates coupling between sequential decisions

2. **Multi-Step Optimization**: We optimize over an 8-hour shift (~600 calls)
   - Discount factor $\gamma = 0.99$ balances immediate vs. future costs
   - Greedy policies (contextual bandits) may suboptimally concentrate load

3. **Resource Constraints**: Limited agent availability requires forward-looking decisions
   - Assigning too many calls to one agent creates bottlenecks

### 5.2 Caveat: Weak Temporal Coupling

However, we acknowledge that the temporal dependencies are **relatively weak**:

- **Call features** ($\mathbf{x}_t$) are sampled independently (no call-to-call correlation)
- **Agent availability** ($\mathbf{a}_t$) is the ONLY state component that changes due to actions
- This makes our problem a **"quasi-MDP"** or **"MDP with weak coupling"**

**Alternative View**: One could frame this as a **Contextual Bandit with resource constraints**, where:
- Context: $\mathbf{x}_t$ (call features)
- Arms: $\mathcal{A}$ (agents)
- Constraint: Some arms unavailable at each time step

This is a valid perspective and could be explored in future work (e.g., using Upper Confidence Bound methods adapted for resource constraints).

---

## 6. RL Algorithms Implemented

### 6.1 Deep Q-Network (DQN)

DQN learns an action-value function $Q(s, a; \theta)$ parameterized by a neural network:

$$
Q(s, a; \theta) \approx Q^*(s, a) = \mathbb{E}\left[ \sum_{t'=t}^{\infty} \gamma^{t'-t} R(s_{t'}, a_{t'}) \mid s_t = s, a_t = a, \pi^* \right]
$$

**Policy**: $\epsilon$-greedy
$$
\pi(s) = \begin{cases}
\arg\max_{a \in \mathcal{A}} Q(s, a; \theta) & \text{with probability } 1 - \epsilon \\
\text{random action} & \text{with probability } \epsilon
\end{cases}
$$

**Loss Function** (Temporal Difference):
$$
\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
$$

where:
- $\mathcal{D}$: Replay buffer
- $\theta^-$: Target network parameters (updated periodically)

**Hyperparameters**:
- Replay buffer size: 200,000 (memory-optimized) or 1,000,000 (standard)
- Batch size: 64
- Learning rate: $10^{-4}$
- Target network update frequency: 500 steps
- $\epsilon$ decay: 1.0 → 0.05 over first 10% of training

### 6.2 Proximal Policy Optimization (PPO)

PPO directly learns a stochastic policy $\pi(a|s; \theta)$ using a policy gradient method.

**Policy**: Softmax over actions
$$
\pi(a | s; \theta) = \frac{\exp(f_a(s; \theta))}{\sum_{a' \in \mathcal{A}} \exp(f_{a'}(s; \theta))}
$$

where $f_a(s; \theta)$ is the logit for action $a$ output by a neural network.

**Objective** (clipped surrogate):
$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where:
- $r_t(\theta) = \frac{\pi(a_t | s_t; \theta)}{\pi(a_t | s_t; \theta_{\text{old}})}$: Probability ratio
- $\hat{A}_t$: Advantage estimate (via Generalized Advantage Estimation)
- $\epsilon = 0.2$: Clipping parameter

**Hyperparameters**:
- Horizon: 2048 steps
- Batch size: 64
- Learning rate: $3 \times 10^{-4}$
- Entropy coefficient: 0.01 (encourages exploration)
- GAE $\lambda$: 0.95

---

## 7. Baseline Policies

For comparison, we implement three baseline policies:

### 7.1 Random Policy

$$
\pi_{\text{random}}(s) \sim \text{Uniform}(\{a \in \mathcal{A} : a_t^{(a)} = 1\})
$$

Selects uniformly at random from available agents.

### 7.2 Rule-Based Policy

Uses historical averages for each agent-topic pair:

$$
\pi_{\text{rule}}(s_t) = \arg\min_{a \in \mathcal{A}_{\text{avail}}} \left( \bar{C}_{a, \tau_t} \right)
$$

where:
- $\bar{C}_{a, \tau}$: Historical average cost for agent $a$ on topic $\tau$
- $\mathcal{A}_{\text{avail}} = \{a : a_t^{(a)} = 1\}$: Set of available agents

Fallback to global averages if agent-topic pair has insufficient data.

### 7.3 Greedy XGBoost Policy

Uses oracle models to predict cost for each available agent, selects minimum:

$$
\pi_{\text{greedy}}(s_t) = \arg\min_{a \in \mathcal{A}_{\text{avail}}} C(s_t, a)
$$

where $C(s_t, a)$ is computed using the XGBoost oracle functions.

This is the **strongest baseline** as it has full knowledge of the simulator's cost function.

---

## 8. Notation Summary

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| $\mathcal{S}$ | State space | $\mathbb{R}^{302}$ |
| $\mathcal{A}$ | Action space | $\{0, \ldots, 249\}$ |
| $s_t$ | State at time $t$ | Vector of length 302 |
| $a_t$ | Action at time $t$ | Integer in $\{0, \ldots, 249\}$ |
| $\mathbf{x}_t$ | Call features | $\mathbb{R}^{50}$ |
| $\mathbf{a}_t$ | Agent availability | $\{0,1\}^{250}$ |
| $\mathbf{h}_t$ | Temporal context | $\mathbb{R}^2$ |
| $R(s, a)$ | Reward function | $\mathbb{R}$ |
| $C(s, a)$ | Cost function | $\mathbb{R}_+$ |
| $\gamma$ | Discount factor | 0.99 |
| $\pi$ | Policy | $\mathcal{S} \to \mathcal{A}$ |
| $Q(s, a)$ | Action-value function | $\mathbb{R}$ |
| $\text{TMC}$ | Call duration (seconds) | $\mathbb{R}_+$ |
| $\text{FTR}$ | First Time Resolution | $\{0, 1\}$ or $[0,1]$ (prob) |
| $\text{OT}$ | Onsite Technician | $\{0, 1\}$ or $[0,1]$ (prob) |
| $N_a$ | Number of agents | 250 |
| $N_c$ | Number of call features | ~50 |
| $N_s$ | State dimension | 302 |
| $T$ | Episode length | ~600 calls |

---

## 9. Implementation Notes

1. **Reward Scaling**:
   - Rewards are in the range $[-512, -7]$ approximately
   - No normalization applied (tested and found unnecessary)

2. **State Normalization**:
   - Call features: No global normalization (already preprocessed in Section 4)
   - Agent availability: Binary (0/1)
   - Hour-of-day: Raw values in $[0, 8)$

3. **Action Masking**:
   - NOT implemented (agents learn to avoid unavailable actions via penalties)
   - Future work could mask invalid actions at the policy level

4. **Episode Termination**:
   - Episode ends when simulation time reaches 8 hours (28,800 seconds)
   - Or when 10,000 steps reached (safety truncation)

---

## 10. LaTeX Export

The formulations above are ready for direct inclusion in LaTeX. Key environments:

- Use `\mathbb{R}` for real spaces
- Use `\mathcal{S}, \mathcal{A}` for sets
- Use `\text{}` for operator names (e.g., `\text{TMC}`)
- Use `\boxed{}` for important equations

Example snippet for thesis:

```latex
\subsection{MDP Formulation}

We model the call routing problem as a Markov Decision Process (MDP)
$\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$, where:

\paragraph{State Space.} The state $s_t \in \mathcal{S}$ at time $t$ is:
$$
s_t = [\mathbf{x}_t, \mathbf{a}_t, \mathbf{h}_t] \in \mathbb{R}^{N_s}
$$
where $\mathbf{x}_t \in \mathbb{R}^{N_c}$ are call features,
$\mathbf{a}_t \in \{0,1\}^{N_a}$ is agent availability,
and $\mathbf{h}_t \in \mathbb{R}^2$ is temporal context.
```

---

**Document Status**: Complete and ready for thesis inclusion
**Next Steps**: Insert into Section 5.2 of thesis manuscript
