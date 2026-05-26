# Code Walkthrough

A concise guide to how the simulation works, module by module.

---

## Overall Flow

```
config.py          → define all constants (prices, learning rate, etc.)
core.py            → market environment + Q-learning agents + training loop
batch_simulator.py → wire everything together; run one or many simulations
web_app.py         → Streamlit UI calling batch_simulator
```

---

## 1. Market Environment (`core.py`)

`BertrandMultiAgentEnvironment` handles the price grid and per-step profit calculation.
Two instances are created — one per market — but they share the same configuration.

```python
env1 = BertrandMultiAgentEnvironment(market_config)  # Market 1 (γ = δ)
env2 = BertrandMultiAgentEnvironment(market_config)  # Market 2 (γ = 0.7δ)
```

Profit follows the standard Bertrand rule — lowest price wins the market; a tie splits demand 50/50:

```python
def _profit(self, my_price, rival_price, my_cost):
    if my_price < rival_price:
        return (my_price - my_cost) * self._demand(my_price)
    if my_price == rival_price:
        return (my_price - my_cost) * self._demand(my_price) / 2
    return 0.0
```

---

## 2. Q-Learning Agent (`core.py`)

`MATLABQLearningAgent` is a tabular Q-learner. Its key design choices:

**State and action space**

Each agent sees all four prices from the previous period and picks prices for both markets in one joint action:

```python
State     = (pA_m1, pA_m2, pB_m1, pB_m2)   # 4-tuple
ActionPair = (price_m1, price_m2)             # 2-tuple
```

With `k = 4` price levels this gives `4⁴ = 256` states and `4² = 16` action pairs.

**Optimistic initialisation**

Q-values start high to encourage exploring all actions before settling:

```python
Q(s, a) = avg_profit(s) / (1 - γ̄)
```

**Epsilon-greedy exploration with step-based decay**

```python
ε(t) = max(ε_min, exp(-β × t))   # β = 2×10⁻⁵
```

Exploration is tied to the global step counter (not episodes), so both markets contribute to the same decay clock.

**Q-update**

```python
Q(s, a) ← (1 - α) × Q(s, a) + α × [r + γ̄ × max_{a'} Q(s', a')]
```

The next state is simply the actions just taken (`s' = a`), matching the MATLAB reference.

---

## 3. Dual-Market Training Loop (`core.py → train_dual_market_interleaved`)

Both markets step together on every iteration. The two agents are **shared** — the same Q-table is updated by experience from both markets.

```python
for step in range(max_steps):
    # Each agent picks a joint action (price_m1, price_m2)
    action_a = agent_a.select_action(state)
    action_b = agent_b.select_action(state)

    # Per-market profits
    pi1_a = env1._profit(action_a[0], action_b[0], cost_a) * scale
    pi2_a = env2._profit(action_a[1], action_b[1], cost_a) * scale

    reward_a = pi1_a + pi2_a          # combined reward

    next_state = (action_a[0], action_a[1], action_b[0], action_b[1])

    # Single Q-update using the average of both discount factors
    agent_a.update(state, action_a, reward_a, next_state, done=False, discount_factor=γ̄)
```

`γ̄ = (γ_M1 + γ_M2) / 2` creates a tug-of-war: Market 1 (higher γ) pushes Q-values up while Market 2 (lower γ) pulls them down, producing a weighted-average policy.

---

## 4. Convergence Detection

Each market tracks how many consecutive steps its prices have been unchanged. Convergence is declared only when **both** markets are independently stable:

```python
if current_prices_m1 == last_prices_m1:
    m1_stable_count += 1
else:
    m1_stable_count = 1

if m1_stable_count >= threshold and m2_stable_count >= threshold:
    converged = True
```

The reported convergence price is the average across both markets and both firms:

```
unified_price = ((p1A + p1B)/2 + (p2A + p2B)/2) / 2
```

---

## 5. Post-Convergence Undercut Experiment (`core.py → run_undercut_experiment`)

Once converged, Firm B deviates by undercutting Firm A in one market:

```python
# Firm B picks the highest valid price strictly below Firm A's converged price
prices_below_a = [p for p in valid_prices if p < conv_price_a]
undercut_price = max(prices_below_a)
```

Both firms then follow their greedy policies (no further Q-updates) for 15 steps, and the resulting price trajectory is recorded. This tests whether the equilibrium is self-enforcing — i.e., whether Firm A responds by punishing the deviation.

---

## 6. Batch Simulation (`batch_simulator.py`)

`run_single_simulation()` wraps the above into a single call, returns a flat dictionary of results (convergence status, prices, Q-values, undercut trajectories), and is the unit used by both the CLI runner and the web app.

`run_batch_simulations()` loops over a list of discount factors, runs `n` independent simulations for each, and aggregates results into a CSV.

```python
result = run_single_simulation(
    discount_factor=0.5,
    max_steps=1_000_000,
    price_convergence_count=1000,
    verbose=True,
)
# result["converged"], result["converged_price"], result["price_1a"], ...
```

For large-scale runs, `parallel_batch_simulator.py` distributes `run_single_simulation()` calls across CPU cores via `multiprocessing.Pool`, writing one output CSV per Market 2 gamma multiplier.
