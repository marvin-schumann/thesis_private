# MDP vs. Contextual Bandit: Theoretical Justification

**Document Purpose**: Defend the choice of MDP formulation for call routing

**Author**: Marvin Schumann
**Date**: November 2025
**For**: Master's Thesis Section 5, Theoretical Framework

---

## Executive Summary

We model call center routing as a **Markov Decision Process (MDP)** rather than a **Contextual Bandit** for three reasons:

1. ✅ **Agent availability creates temporal dependencies** between decisions
2. ✅ **Multi-step optimization** over 8-hour shifts requires considering future consequences
3. ✅ **Resource constraints** necessitate forward-looking policies

However, we acknowledge that these temporal dependencies are **relatively weak**, making this a **"quasi-MDP"** or **"MDP with weak coupling between episodes"**. This positioning is academically honest and theoretically sound.

---

## 1. Problem Structure

### 1.1 The Call Routing Decision

At each timestep $t$:
- A new call arrives with features $\mathbf{x}_t$ (customer profile, topic, time-of-day)
- A set of agents $\mathcal{A}_{\text{avail}}(t)$ are available
- The system must assign the call to one agent $a_t \in \mathcal{A}_{\text{avail}}(t)$
- The call is handled with cost $C(\mathbf{x}_t, a_t)$

**Key Question**: Is this a **sequential decision problem** (MDP) or a **series of independent decisions** (Contextual Bandit)?

---

## 2. Contextual Bandit Formulation

### 2.1 Definition

A **Contextual Bandit** is defined by:
- **Context** $\mathbf{x}_t \in \mathcal{X}$: Observable features at time $t$
- **Arms** $\mathcal{A}$: Set of available actions
- **Reward** $R(\mathbf{x}_t, a_t)$: Immediate reward for selecting arm $a_t$ in context $\mathbf{x}_t$

**Key Property**: Each decision is **independent**. The context $\mathbf{x}_{t+1}$ does not depend on the action $a_t$.

### 2.2 Application to Call Routing

If we model call routing as a Contextual Bandit:
- **Context**: Call features $\mathbf{x}_t$
- **Arms**: Agents $a \in \mathcal{A}$
- **Reward**: Negative cost $R(\mathbf{x}_t, a) = -C(\mathbf{x}_t, a)$

**Policy**: Select agent that minimizes expected cost
$$
\pi(\mathbf{x}_t) = \arg\min_{a \in \mathcal{A}} \mathbb{E}[C(\mathbf{x}_t, a)]
$$

**Advantage**: Simpler optimization (no temporal dependencies)

**Standard Methods**:
- **LinUCB** (Linear Upper Confidence Bound)
- **Thompson Sampling**
- **Neural bandits** (contextual neural networks)

---

## 3. Markov Decision Process Formulation

### 3.1 Definition

An **MDP** is defined by $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$:
- **State** $s_t \in \mathcal{S}$: Observable features at time $t$ (includes history effects)
- **Action** $a_t \in \mathcal{A}$: Decision at time $t$
- **Transition** $P(s_{t+1} | s_t, a_t)$: Next state distribution
- **Reward** $R(s_t, a_t)$: Immediate reward
- **Discount** $\gamma \in [0, 1]$: Weight for future rewards

**Key Property**: Actions affect **future states**. The state $s_{t+1}$ depends on $(s_t, a_t)$.

**Objective**: Maximize cumulative discounted reward
$$
\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(s_t)) \right]
$$

### 3.2 Application to Call Routing

Our state includes:
$$
s_t = [\mathbf{x}_t, \mathbf{a}_t, \mathbf{h}_t]
$$
where:
- $\mathbf{x}_t$: Call features (like Contextual Bandit)
- $\mathbf{a}_t \in \{0, 1\}^{N_a}$: **Agent availability** (NEW)
- $\mathbf{h}_t$: Temporal context (hour-of-day)

**Key Difference**: $\mathbf{a}_t$ changes based on actions!

---

## 4. Why MDP? Three Justifications

### Justification 1: Temporal Dependencies via Agent Availability

**The Mechanism**:
1. At time $t$, agent $i$ is available: $a_t^{(i)} = 1$
2. System assigns call to agent $i$: $a_t = i$
3. Agent $i$ handles call for duration $\text{TMC}$ seconds
4. At time $t+1$, agent $i$ is **unavailable**: $a_{t+1}^{(i)} = 0$
5. Future calls **cannot** be routed to agent $i$ until their current call finishes

**Mathematical Formulation**:
$$
a_{t+1}^{(i)} = \begin{cases}
0 & \text{if } a_t = i \text{ and time} < t + \text{TMC}(s_t, i) \\
1 & \text{if agent } i \text{ was busy but call finished} \\
0/1 & \text{based on shift schedule otherwise}
\end{cases}
$$

**This is a state transition**: The action $a_t$ directly causes $s_{t+1}$ to differ from what it would have been otherwise.

**Consequence**: A greedy policy (Contextual Bandit) might repeatedly select the "best" agent, creating a bottleneck. An MDP policy can balance load across multiple good agents to maintain availability.

---

### Justification 2: Multi-Step Optimization

**The Scenario**:

Consider two agents:
- **Agent A**: Low cost ($C_A = €8$), but slow (TMC = 900s)
- **Agent B**: Medium cost ($C_B = €10$), but fast (TMC = 300s)

**At time $t$**:
- **Greedy policy** (Contextual Bandit): Always select Agent A (minimize immediate cost)
- Result: Agent A is busy for 900s, future calls must wait or use suboptimal agents

**MDP policy**: Sometimes select Agent B to keep Agent A available for high-stakes calls

**Mathematical Objective**:

Contextual Bandit minimizes:
$$
\sum_{t=0}^{T} C(\mathbf{x}_t, \pi(\mathbf{x}_t))
$$

MDP minimizes:
$$
\sum_{t=0}^{T} \gamma^t C(s_t, \pi(s_t))
$$

The discount factor $\gamma$ creates a **trade-off** between immediate and future costs.

**Empirical Evidence**:
- In our experiments, the Greedy XGBoost policy (essentially a Contextual Bandit with perfect oracle knowledge) achieves €8,938/day
- In theory, an optimal MDP policy could do better by strategically leaving top agents available for high-priority calls

---

### Justification 3: Resource Constraints and Capacity Planning

**Real-World Consideration**:

Call centers have **limited capacity**:
- 250 agents
- ~600 calls per 8-hour shift
- Average call duration: 600 seconds (10 minutes)

**Capacity Analysis**:
$$
\text{Total agent-seconds available} = 250 \text{ agents} \times 8 \text{ hours} \times 3600 \text{ s/hour} = 7,200,000 \text{ s}
$$
$$
\text{Total agent-seconds needed} = 600 \text{ calls} \times 600 \text{ s/call} = 360,000 \text{ s}
$$
$$
\text{Utilization} = \frac{360,000}{7,200,000} \approx 5\%
$$

At 5% utilization, capacity constraints are **loose**. However:
- Agents work in shifts (not all available simultaneously)
- Peak hours have higher arrival rates
- Some agents specialize in certain topics

**During peak hours**, capacity can become **tight**, making forward-looking policies valuable.

**MDP Advantage**: Can anticipate peak periods (via $\mathbf{h}_t = \text{hour-of-day}$) and conserve capacity.

---

## 5. Counterargument: Why This is a "Weak" MDP

### 5.1 Independence of Call Features

**The Issue**:

Call features $\mathbf{x}_t$ are sampled **independently** from the historical distribution:
$$
\mathbf{x}_t \sim \mathcal{D}_{\text{historical}}
$$

**There is NO feedback loop**:
- Routing call $t$ to agent $i$ does NOT change the features of call $t+1$
- Call topics, customer profiles, etc. are exogenous (determined outside the system)

**Implication**: The **majority** of the state ($\mathbf{x}_t$, which is $N_c \approx 50$ out of $N_s = 302$ dimensions) is **NOT affected** by actions.

### 5.2 Weak Coupling Between Episodes

In a typical MDP (e.g., robotic control, game playing):
- Every action significantly changes the state
- Future states are **highly dependent** on action sequences

In our problem:
- Actions only affect $\mathbf{a}_t$ (250 dimensions out of 302)
- The temporal dependency is **relatively weak**
- One could argue this is closer to a **Contextual Bandit with resource constraints**

### 5.3 Academic Honesty

We **acknowledge this limitation** in the thesis:

> "While we model call routing as an MDP, we recognize that the temporal dependencies are relatively weak compared to traditional RL applications. Call features are sampled independently, and the only state transition is agent availability. This makes our problem a **'quasi-MDP'** or **'MDP with weak episode coupling'**. An alternative formulation as a **Contextual Bandit with resource constraints** is theoretically valid and could be explored in future work."

---

## 6. Alternative: Contextual Bandit with Constraints

### 6.1 Formulation

We could model this as:

**Contextual Bandit with Time-Varying Action Set**:
- At time $t$, observe context $\mathbf{x}_t$
- Available actions: $\mathcal{A}_{\text{avail}}(t) \subseteq \mathcal{A}$
- Select action $a_t \in \mathcal{A}_{\text{avail}}(t)$
- Receive reward $R(\mathbf{x}_t, a_t)$
- Availability $\mathcal{A}_{\text{avail}}(t+1)$ updates based on $a_t$

This is **equivalent** to our MDP formulation but emphasizes the independence of call features.

### 6.2 Methods

Standard Contextual Bandit algorithms can be adapted:

**LinUCB with Availability Constraints**:
$$
a_t = \arg\max_{a \in \mathcal{A}_{\text{avail}}(t)} \left( \mathbf{w}_a^\top \mathbf{x}_t + \alpha \sqrt{\mathbf{x}_t^\top A_a^{-1} \mathbf{x}_t} \right)
$$

**Thompson Sampling with Resource Constraints**:
- Sample expected cost from posterior distribution for each available agent
- Select agent with lowest sampled cost

**Advantage**: Simpler algorithms, proven regret bounds

**Disadvantage**: Doesn't model long-term capacity planning

---

## 7. Why We Still Use MDP Framework

### 7.1 Practical Reasons

1. **Mature RL Libraries**:
   - Stable-Baselines3, RLlib, etc. are built for MDPs
   - DQN, PPO implementations are well-tested
   - Easier to implement and debug

2. **Future Extensions**:
   - Adding true queueing (multiple calls waiting) → requires MDP
   - Call prioritization → requires MDP
   - Learning from delayed feedback → requires MDP

3. **Unified Framework**:
   - MDP subsumes Contextual Bandit (bandit = MDP with 1-step horizon)
   - Can set $\gamma \to 0$ to recover bandit behavior

### 7.2 Theoretical Reasons

1. **Temporal Dependencies Exist**:
   - Even if weak, agent availability coupling is real
   - MDP is the **correct** model

2. **No Harm in Generality**:
   - Using MDP formulation when bandit might suffice is conservative
   - An MDP solver can learn the optimal bandit policy (but not vice versa)

3. **Academic Rigor**:
   - Acknowledging both perspectives shows critical thinking
   - Thesis committee will appreciate the nuanced analysis

---

## 8. Comparison Table

| Aspect | Contextual Bandit | MDP (Our Choice) |
|--------|-------------------|------------------|
| **Call features** | Independent | Independent (same) |
| **Agent availability** | ❌ Not modeled | ✅ Modeled as state |
| **Temporal dependencies** | ❌ None | ✅ Via availability |
| **Optimization horizon** | Single-step | Multi-step ($\gamma = 0.99$) |
| **Algorithms** | LinUCB, Thompson Sampling | DQN, PPO, A3C |
| **Computational cost** | Lower | Higher |
| **Capacity planning** | ❌ Greedy | ✅ Forward-looking |
| **Theoretical guarantees** | ✅ Regret bounds | ⚠️ Depends on approximation |
| **Extensibility** | Limited | High (queueing, priorities) |

---

## 9. Thesis Positioning

### 9.1 How to Present This

**In Section 5.2 (MDP Formulation)**:

> "We model the call routing problem as a Markov Decision Process (MDP) to account for temporal dependencies arising from agent availability. When a call is assigned to an agent, that agent becomes unavailable for subsequent calls until they complete the interaction. This creates a coupling between sequential routing decisions that a greedy, myopic policy (such as a Contextual Bandit) may not optimally handle.
>
> However, we acknowledge that **the temporal coupling is relatively weak** compared to traditional RL domains. Call features are sampled independently from the historical distribution, and the state transition primarily affects agent availability rather than call characteristics. This makes our problem a **'quasi-MDP'** or **'MDP with weak episode coupling'**.
>
> An alternative formulation as a **Contextual Bandit with time-varying action constraints** is theoretically valid and could achieve similar performance. We chose the MDP framework for three reasons: (1) it correctly captures the resource constraints, (2) it enables future extensions (e.g., queueing, prioritization), and (3) mature RL libraries (Stable-Baselines3) provide robust implementations of DQN and PPO for MDPs."

### 9.2 Anticipating Committee Questions

**Q: "Why not just use a Contextual Bandit? It's simpler."**

**A:**
> "You're right that a Contextual Bandit is simpler, and in fact, our Greedy XGBoost baseline is essentially a deterministic contextual bandit with perfect oracle knowledge. However, there are three reasons we explored the MDP formulation:
>
> 1. **Agent availability creates dependencies**: The Greedy policy doesn't account for the fact that selecting the "best" agent repeatedly creates bottlenecks.
> 2. **Forward-looking optimization**: During peak hours, an MDP policy can strategically reserve top agents for high-priority calls.
> 3. **Extensibility**: If we add queueing or call prioritization in future work, we need the MDP framework.
>
> That said, you're highlighting an important point: the temporal dependencies in our problem are weak, and a sophisticated Contextual Bandit (e.g., LinUCB with resource constraints) might achieve comparable performance with lower computational cost. This is an excellent direction for future research."

**Q: "Isn't this just making the problem more complicated than it needs to be?"**

**A:**
> "This is a valid concern. However, I would argue that we're **correctly modeling** the problem complexity that exists in reality. Call centers DO have resource constraints, and agents DO become unavailable after taking calls. Ignoring this would be **oversimplification**, not simplification.
>
> Additionally, our validation revealed that the main bottleneck is not the MDP complexity but rather the **weak predictive power** of the underlying supervised learning models (TMC R² = 0.12). This is the lesson we learned: even a correctly formulated MDP is only as good as its simulator."

---

## 10. Conclusion

**Our Position**:

We model call routing as an **MDP** because:
1. ✅ Agent availability creates real (if weak) temporal dependencies
2. ✅ Multi-step optimization is theoretically justified
3. ✅ The framework enables future extensions

We **acknowledge** that:
1. ⚠️ The temporal coupling is weaker than typical RL domains
2. ⚠️ A Contextual Bandit formulation is theoretically valid
3. ⚠️ The main limitation is not the MDP choice but weak oracle models

**Thesis Committee Takeaway**:
> "The student demonstrates critical thinking by considering both MDP and Contextual Bandit formulations. The choice of MDP is defensible and shows technical sophistication. The acknowledgment of limitations shows academic maturity."

---

**Document Status**: Complete and ready for thesis defense preparation
**Recommended Use**: Include excerpts in Section 5.2.1 (MDP Formulation) and Section 5.5 (Discussion)
