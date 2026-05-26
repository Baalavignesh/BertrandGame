# Bertrand Q-Learning Multi-Market Simulation

A research simulation framework for studying algorithmic pricing behavior in Bertrand competition. Two Q-learning agents (Firm A and Firm B) compete across two parallel markets with different discount factors. The central question is whether independent pricing algorithms tacitly collude to supra-competitive prices.

See **[RUNNING.md](RUNNING.md)** for installation and usage instructions.

---

## Table of Contents

- [Economic Model](#economic-model)
- [Q-Learning Implementation](#q-learning-implementation)
- [Multi-Market Architecture](#multi-market-architecture)
- [Convergence Mechanism](#convergence-mechanism)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [File Structure](#file-structure)

---

## Economic Model

### Bertrand Competition

Firms simultaneously set prices; the lowest-priced firm captures all market demand.

**Demand function:**
```
Q(p) = max(0, a - b×p)
```

**Profit function:**
```
π(p_i, p_j) = (p_i - c) × Q(p_i)      if p_i < p_j   (wins market)
              (p_i - c) × Q(p_i) / 2   if p_i = p_j   (splits market)
              0                          if p_i > p_j   (loses market)
```

Default parameters: `a = 100`, `b = 1`, `c = 0.1`.

**Nash equilibrium:** Standard Bertrand competition predicts `p* = c`. Q-learning agents frequently converge above this, exhibiting tacit collusion.

### Price Grid

Prices are discretized into `k` evenly-spaced levels on `[p_min, p_max]`. Default: `k = 4`, range `[0.0, 0.65]`, giving `{0.0, 0.22, 0.43, 0.65}`.

---

## Q-Learning Implementation

### Agent Architecture

Each agent maintains a Q-table over state–action pairs:

**State:** `(pA_m1, pA_m2, pB_m1, pB_m2)` — prices from the previous period in both markets.

**Action:** `(price_m1, price_m2)` — the agent picks prices for both markets simultaneously.

**Reward:** Combined profit `π_m1 + π_m2`.

**Next state:** `s' = a` (the actions become the next state, matching the MATLAB reference implementation).

### Update Rule

```
Q(s, a) ← (1 - α) × Q(s, a) + α × [r + γ̄ × max_{a'} Q(s', a')]
```

where `γ̄ = (γ_M1 + γ_M2) / 2` is the average discount factor across both markets.

- `α = 0.15` (learning rate)
- `γ_M1 = δ`, `γ_M2 = 0.7 × δ`

### Exploration

Epsilon-greedy with step-based exponential decay:
```
ε(t) = max(ε_min, exp(-β × t))
```
- `ε_start = 1.0`, `ε_min = 0.001`, `β = 2×10⁻⁵`

### Optimistic Initialization

Q-values are initialized to encourage early exploration of all actions:
```
Q(s, a) = avg_π(s) / (1 - γ̄)
```

---

## Multi-Market Architecture

Two markets run in parallel, sharing the same two agents:

| | Market 1 | Market 2 |
|---|---|---|
| Discount factor | δ | 0.7 × δ |
| Agents | Shared Agent A, Shared Agent B | Same agents |

```
┌────────────────────────────────────────────────────────┐
│                SHARED AGENT A (one Q-table)            │
└────────────────────┬───────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Market 1 (γ = δ)         Market 2 (γ = 0.7δ)
        │                         │
        └────────────┬────────────┘
                     │
┌────────────────────┴───────────────────────────────────┐
│                SHARED AGENT B (one Q-table)            │
└────────────────────────────────────────────────────────┘

Convergence: both markets must independently stabilize
Unified price = ((M1_A + M1_B)/2 + (M2_A + M2_B)/2) / 2
```

The "tug-of-war" between `γ_M1` (higher) and `γ_M2` (lower) produces Q-values that represent a weighted average of both markets' perspectives, modeling a firm deploying the same algorithm across multiple markets simultaneously.

---

## Convergence Mechanism

Convergence requires both markets to independently hold the same prices for `price_convergence_count` consecutive steps:

```python
if m1_stable_count >= N and m2_stable_count >= N:
    converged = True
```

Independent tracking prevents complementary fluctuations (M1 rising while M2 falls) from falsely signaling stability.

**Unified convergence price:**
```
unified_price = ((price_1A + price_1B)/2 + (price_2A + price_2B)/2) / 2
```

### Post-Convergence Undercut Experiment

After convergence, Firm B deviates by setting the highest valid price strictly below Firm A's converged price in a given market. Both firms then follow greedy policies (no Q-updates) for 15 steps, recording the price trajectory.

---

## Configuration

All defaults are in `config.py`. Modify there to change behaviour globally.

| Parameter | Default | Description |
|---|---|---|
| `MARGINAL_COST` | 0.1 | Production cost |
| `PRICE_MIN` | 0.0 | Minimum price |
| `PRICE_MAX` | 0.65 | Maximum price |
| `NUM_PRICE_LEVELS` | 4 | Discrete price levels |
| `DEMAND_INTERCEPT` | 100.0 | Demand intercept `a` |
| `DEMAND_SLOPE` | 1.0 | Demand slope `b` |
| `LEARNING_RATE` | 0.15 | Q-learning `α` |
| `GAMMA_MULTIPLIER` | 0.7 | Market 2 discount = γ × 0.7 |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_MIN` | 0.001 | Minimum exploration rate |
| `STEP_BETA` | 2×10⁻⁵ | Exploration decay rate `β` |
| `DEFAULT_MAX_STEPS` | 1,000,000 | Max training steps |
| `DEFAULT_PRICE_CONVERGENCE_COUNT` | 1000 | Consecutive stable steps required |

---

## Output Format

### CSV Columns

| Column | Description |
|---|---|
| `discount_factor_a` | Market 1 discount factor (δ) |
| `discount_factor_b` | Market 2 discount factor (0.7×δ) |
| `converged` | Whether simulation converged |
| `convergence_step` | Step at which convergence occurred |
| `converged_price` | Unified convergence price |
| `m1_stable_count` / `m2_stable_count` | Steps each market was stable |
| `price_1a`, `price_1b` | Market 1 converged prices |
| `price_2a`, `price_2b` | Market 2 converged prices |
| `converged_q_value_a` / `_b` | Shared agent Q-values at convergence |
| `m1_uc_pa_*`, `m1_uc_pb_*` | Market 1 undercut trajectory (15 steps) |
| `m2_uc_pa_*`, `m2_uc_pb_*` | Market 2 undercut trajectory (15 steps) |
| `time_to_converge` | Wall-clock time (seconds) |

---

## File Structure

```
Simulation/
├── core.py                      # Q-learning agents, environments, training loop
├── batch_simulator.py           # Sequential batch runner; programmatic API
├── parallel_batch_simulator.py  # Parallel batch runner (multiprocessing)
├── config.py                    # All tunable constants
├── web_app.py                   # Streamlit web interface
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Module dependency chain

```
config.py → core.py → batch_simulator.py → web_app.py
```

- **`core.py`**: `MarketConfig`, `BertrandMultiAgentEnvironment`, `MATLABQLearningAgent`, `train_dual_market_interleaved()`, `run_undercut_experiment()`
- **`batch_simulator.py`**: `run_single_simulation()`, `run_batch_simulations()`, CLI entry point
- **`parallel_batch_simulator.py`**: Parallel wrapper using `multiprocessing.Pool`; separate CSV per gamma multiplier; real-time progress bar with ETA

---

## Research Applications

This framework is designed to study:

1. **Algorithmic collusion** — Do Q-learning agents converge to supra-competitive prices?
2. **Discount factor effects** — How does patience (δ) affect collusive outcomes?
3. **Cross-market learning** — How do shared Q-tables affect convergence speed and stability?
4. **Deviation incentives** — How do agents respond when a rival undercuts?
