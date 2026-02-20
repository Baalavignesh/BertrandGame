# Bertrand Q-Learning Multi-Market Simulation

A research simulation framework for studying algorithmic pricing behavior in Bertrand competition using Q-learning agents. This project investigates how autonomous pricing algorithms learn to compete (or collude) across multiple parallel markets with different discount factors.

## Table of Contents

- [Overview](#overview)
- [Economic Model](#economic-model)
- [Q-Learning Implementation](#q-learning-implementation)
- [Multi-Market Architecture](#multi-market-architecture)
- [Shared Q-Tables](#shared-q-tables)
- [Convergence Mechanism](#convergence-mechanism)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [File Structure](#file-structure)

---

## Overview

This simulation studies **algorithmic collusion** — the phenomenon where independent pricing algorithms learn to coordinate on supra-competitive prices without explicit communication. The framework implements:

- **Two parallel markets** running in interleaved steps
- **Two shared Q-learning agents** (Firm A and Firm B operate in both markets)
- **Different discount factors** per market during Q-updates
- **Independent convergence tracking** per market
- **Unified convergence price** reported when both markets stabilize

```
┌─────────────────────────────────────────────────────────────────┐
│              Shared Q-Table Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              SHARED AGENT A (one Q-table)               │   │
│   │    Updates from M1 (γ) and M2 (0.7×γ) are averaged     │   │
│   └─────────────────────────────────────────────────────────┘   │
│         │                                       │               │
│         ▼                                       ▼               │
│   Market 1 (γ = δ)                    Market 2 (γ = 0.7×δ)     │
│   ┌───────────────┐                   ┌───────────────┐         │
│   │  Environment  │                   │  Environment  │         │
│   └───────────────┘                   └───────────────┘         │
│         │                                       │               │
│         ▼                                       ▼               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              SHARED AGENT B (one Q-table)               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│                    Convergence Check                             │
│              (Both markets must stabilize)                       │
│                           │                                      │
│                           ▼                                      │
│                    Unified Price                                 │
│          ((M1_A+M1_B)/2 + (M2_A+M2_B)/2) / 2                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Economic Model

### Bertrand Competition

The simulation models **Bertrand price competition** where firms simultaneously set prices and the lowest-priced firm captures the entire market demand.

**Demand Function:**
```
Q(p) = max(0, a - b×p)
```
- `a` = demand intercept (default: 100)
- `b` = demand slope (default: 1)

**Profit Function:**
```
π(p_i, p_j) = {
    (p_i - c) × Q(p_i)     if p_i < p_j   (wins market)
    (p_i - c) × Q(p_i)/2   if p_i = p_j   (splits market)
    0                       if p_i > p_j   (loses market)
}
```
- `c` = marginal cost (default: 0.1)

**Nash Equilibrium:** In standard Bertrand competition, the Nash equilibrium is `p* = c` (marginal cost pricing). However, Q-learning agents often converge to higher prices, exhibiting tacit collusion.

### Price Grid

Prices are discretized into `k` levels between `[p_min, p_max]`:
```
prices = {p_min, p_min + Δ, p_min + 2Δ, ..., p_max}
where Δ = (p_max - p_min) / (k - 1)
```

Default: 4 levels from 0.0 to 0.65 → {0.0, 0.22, 0.43, 0.65}

---

## Q-Learning Implementation

### Agent Architecture

Each agent maintains a Q-table mapping state-action pairs to expected cumulative rewards:
```
Q(s, a) → expected future profit
```

**State:** `(my_last_price, rival_last_price)` — a tuple of the prices from the previous period.

**Actions:** All valid prices ≥ marginal cost.

### Q-Learning Update Rule

The discount factor `γ` is passed explicitly on each update:
```
Q(s,a) ← (1-α)×Q(s,a) + α×[r + γ×max_a' Q(s',a')]
```

- `α` = learning rate (default: 0.15)
- `γ` = discount factor (passed per update, varies by market)
- `r` = immediate reward (profit)

### Exploration Strategy

**Epsilon-greedy with exponential decay:**
```
ε(t) = max(ε_min, ε_start × e^(-β×t))
```

- `ε_start` = 1.0 (100% exploration initially)
- `ε_min` = 0.001 (0.1% minimum exploration)
- `β` = 2×10⁻⁵ (decay rate)

### Optimistic Initialization

Q-values are initialized optimistically to encourage exploration:
```
Q(s,a) = π(s,a) / (1 - γ)
```
This assumes the agent will receive the same profit forever, encouraging trying all actions.

---

## Multi-Market Architecture

### Dual Market Setup

The simulation runs **two markets in parallel with shared agents**:

| Property | Market 1 | Market 2 |
|----------|----------|----------|
| Discount Factor | δ | 0.7 × δ |
| Agents | Shared Agent A, Shared Agent B | Same agents |
| Environment | Independent | Independent |

**Rationale:** Different discount factors model firms with different "patience" or planning horizons. Market 2's lower γ means agents value future rewards less.

### Interleaved Execution

Both markets execute **step-by-step together** using the same shared agents:

```python
for step in range(max_steps):
    # Both markets use the SAME agents
    action_1a = agent_a.select_action(state_1a)
    action_1b = agent_b.select_action(state_1b)
    action_2a = agent_a.select_action(state_2a)  # Same agent_a
    action_2b = agent_b.select_action(state_2b)  # Same agent_b
    
    # Environment steps
    env1.step(action_1a, action_1b)
    env2.step(action_2a, action_2b)
    
    # Q-updates with different discount factors
    agent_a.update(..., discount_factor=γ_m1)      # Market 1 update
    agent_a.update(..., discount_factor=γ_m2)      # Market 2 update
    ...
```

---

## Shared Q-Tables

### Why Shared Q-Tables?

The shared Q-table architecture models a scenario where:
- The **same pricing algorithm** is deployed across multiple markets
- The algorithm learns from experiences in **all markets simultaneously**
- It develops a **unified policy** that balances performance across markets

### How Different γ Values Affect Learning

Each market uses its own discount factor during Q-updates:

```python
# Market 1 update (higher γ, values future more)
agent_a.update(state, action, reward, next_state, done, discount_factor_m1)

# Market 2 update (lower γ, values future less)  
agent_a.update(state, action, reward, next_state, done, discount_factor_m2)
```

This creates a **"tug-of-war"** effect on the Q-values:
- Market 1 (higher γ) pushes Q-values **higher**
- Market 2 (lower γ) pushes Q-values **lower**
- The final Q-values represent a **weighted average** of both markets' perspectives

### Benefits

1. **Faster convergence** — Double the training samples per state-action pair
2. **Knowledge transfer** — Experiences in one market inform the other
3. **Realistic modeling** — Same algorithm deployed across multiple markets
4. **Simpler analysis** — 2 Q-values to track instead of 4

---

## Convergence Mechanism

### Independent Market Stability

Convergence requires **both markets to independently stabilize** their prices:

```python
# Track Market 1 stability
if current_prices_m1 == last_prices_m1:
    m1_stable_count += 1
else:
    m1_stable_count = 1

# Track Market 2 stability  
if current_prices_m2 == last_prices_m2:
    m2_stable_count += 1
else:
    m2_stable_count = 1

# Converge only when BOTH are stable
if m1_stable_count >= N and m2_stable_count >= N:
    converged = True
```

**Why independent tracking?** If we only tracked the unified price, complementary fluctuations could cancel out (e.g., M1 price rises while M2 falls), falsely indicating stability.

### Unified Convergence Price

Once both markets converge, the unified price is calculated:

```
unified_price = ((price_1A + price_1B)/2 + (price_2A + price_2B)/2) / 2
```

This represents the average collusive price across both markets.

---

## Installation

### Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib
- Streamlit (for web interface)

### Setup

```bash
# Clone the repository
cd Bertrand_Q-Learning/Simulation

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Command Line (Batch Simulations)

```bash
# Test mode: 10 simulations for γ ∈ {0.05, 0.5, 1.0}
python batch_simulator.py --mode test

# Full scale: 100,000 simulations for γ ∈ {0.05, 0.10, ..., 1.0}
python batch_simulator.py --mode full
```

### Parallel Batch Simulations

For large-scale runs, use `parallel_batch_simulator.py` which distributes simulations across multiple CPU cores using Python `multiprocessing`. This provides significant speedup on multi-core machines (e.g., ~8× faster on 8 cores).

**Features:**
- **Parallel execution** via `multiprocessing.Pool` with `imap_unordered`
- **Real-time progress bar** showing completed/total, elapsed time, ETA, and throughput (sims/s)
- **Periodic CSV checkpoints** (every 100 sims by default) to prevent data loss
- **Separate CSV per multiplier** — one output file for each Market 2 gamma multiplier
- **All parameters configurable** via CLI arguments

```bash
# Full run: 3 multipliers × 20 δ values × 500 sims = 30,000 simulations
python parallel_batch_simulator.py \
  --multipliers 0.5 0.75 1.0 \
  --n-sims 500 \
  --max-steps 1000000 \
  --convergence-count 1000 \
  --learning-rate 0.15 \
  --step-beta 2e-5 \
  --price-levels 4 \
  --price-min 0 --price-max 0.65 \
  --marginal-cost 0.10 \
  --workers 8

# Quick test
python parallel_batch_simulator.py \
  --multipliers 1.0 --n-sims 2 --max-steps 50000 --convergence-count 50
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--multipliers` | `0.5 0.75 1.0` | Market 2 gamma multiplier values |
| `--n-sims` | `500` | Simulations per discount factor |
| `--max-steps` | `1,000,000` | Max training steps per simulation |
| `--convergence-count` | `1000` | Consecutive stable steps for convergence |
| `--learning-rate` | `0.15` | Q-learning α |
| `--step-beta` | `2e-5` | Exploration decay rate β |
| `--price-levels` | `4` | Number of discrete price levels |
| `--price-min` | `0.0` | Minimum price |
| `--price-max` | `0.65` | Maximum price |
| `--marginal-cost` | `0.10` | Production cost |
| `--workers` | `cpu_count - 1` | Number of parallel worker processes |
| `--discount-factors` | `0.0, 0.05, ..., 0.95` | List of δ values to test |
| `--output-dir` | `.` | Directory for output CSVs |
| `--checkpoint-every` | `100` | Save partial results every N sims |

**Output files:** `results_mult_0.50.csv`, `results_mult_0.75.csv`, `results_mult_1.00.csv`

### Web Interface

```bash
streamlit run web_app.py
```

The web interface provides:
- Configurable simulation parameters
- Real-time progress tracking
- Results visualization
- CSV/log file downloads

### Programmatic Usage

```python
from batch_simulator import run_single_simulation

result = run_single_simulation(
    discount_factor=0.5,      # Market 1 uses 0.5, Market 2 uses 0.35
    max_steps=1_000_000,
    price_convergence_count=1000,
    verbose=True,
)

print(f"Converged: {result['converged']}")
print(f"Unified Price: {result['converged_price']}")
print(f"Market 1: {result['price_1a']}, {result['price_1b']}")
print(f"Market 2: {result['price_2a']}, {result['price_2b']}")
print(f"Shared Q-value A: {result['converged_q_value_a']}")
print(f"Shared Q-value B: {result['converged_q_value_b']}")
```

---

## Configuration

### config.py Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MARGINAL_COST` | 0.1 | Production cost for all firms |
| `PRICE_MIN` | 0.0 | Minimum price in grid |
| `PRICE_MAX` | 0.65 | Maximum price in grid |
| `NUM_PRICE_LEVELS` | 4 | Number of discrete price levels |
| `DEMAND_INTERCEPT` | 100.0 | Demand function intercept (a) |
| `DEMAND_SLOPE` | 1.0 | Demand function slope (b) |
| `LEARNING_RATE` | 0.15 | Q-learning α |
| `GAMMA_MULTIPLIER` | 0.7 | Market 2 discount = γ × 0.7 |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_MIN` | 0.001 | Minimum exploration rate |
| `STEP_BETA` | 2×10⁻⁵ | Exploration decay rate |

### Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 1,000,000 | Maximum training steps |
| `price_convergence_count` | 1000 | Steps each market must be stable |

---

## Output Format

### CSV Columns

| Column | Description |
|--------|-------------|
| `discount_factor_a` | Market 1's discount factor (δ) |
| `discount_factor_b` | Market 2's discount factor (0.7×δ) |
| `converged` | Whether simulation converged |
| `convergence_step` | Step at which convergence occurred |
| `converged_price` | Unified convergence price |
| `m1_stable_count` | Steps Market 1 was stable |
| `m2_stable_count` | Steps Market 2 was stable |
| `price_1a`, `price_1b` | Market 1 converged prices |
| `price_2a`, `price_2b` | Market 2 converged prices |
| `converged_q_value_a` | Shared Agent A's Q-value at convergence |
| `converged_q_value_b` | Shared Agent B's Q-value at convergence |
| `m1_uc_pa_*`, `m1_uc_pb_*` | Market 1 undercut trajectory |
| `m2_uc_pa_*`, `m2_uc_pb_*` | Market 2 undercut trajectory |
| `time_to_converge` | Wall-clock time (seconds) |

### Example Output

```
BOTH MARKETS CONVERGED at step 123,456
Market 1 stable for 1000 steps: A=0.43, B=0.43
Market 2 stable for 1024 steps: A=0.43, B=0.43
Unified price: 0.4300

Shared Q-values:
  Agent A: 0.1378
  Agent B: 0.1378
```

---

## File Structure

```
Simulation/
├── core.py                      # Q-learning agents, environments, training functions
├── batch_simulator.py           # Sequential batch simulation runner with CLI
├── parallel_batch_simulator.py  # Parallel batch simulation runner (multiprocessing)
├── config.py                    # Configuration constants
├── web_app.py                   # Streamlit web interface
├── requirements.txt             # Python dependencies
├── undercut.ipynb               # Jupyter notebook for analysis
├── simulation_results_*.csv     # Output CSV files (batch_simulator)
├── results_mult_*.csv           # Output CSV files (parallel_batch_simulator)
├── simulation_log_*.txt         # Detailed log files
└── README.md                    # This file
```

### Core Modules

**`core.py`** contains:
- `MarketConfig` — Market parameter configuration
- `BertrandMultiAgentEnvironment` — Bertrand competition environment
- `MATLABQLearningAgent` — Q-learning agent with explicit discount_factor in update()
- `train_dual_market_interleaved()` — Dual market training with shared agents
- `run_undercut_experiment()` — Post-convergence deviation analysis

**`batch_simulator.py`** contains:
- `run_single_simulation()` — Run one dual-market simulation with shared agents
- `run_batch_simulations()` — Run multiple simulations sequentially across discount factors
- CLI interface for test/full modes

**`parallel_batch_simulator.py`** contains:
- Parallel wrapper around `run_single_simulation()` using `multiprocessing.Pool`
- Real-time progress bar with ETA and throughput tracking
- Support for multiple Market 2 gamma multipliers in a single run
- CLI interface with full parameter control

---

## Research Applications

This framework is designed to study:

1. **Algorithmic Collusion:** Do Q-learning agents converge to supra-competitive prices?
2. **Discount Factor Effects:** How does patience (γ) affect collusive outcomes?
3. **Cross-Market Learning:** How do shared Q-tables affect convergence speed and stability?
4. **Deviation Incentives:** How do agents respond when a rival undercuts?

---

## License

Research use only. Please cite appropriately if used in academic work.
