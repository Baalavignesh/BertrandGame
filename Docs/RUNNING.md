# Running Instructions

## Prerequisites

Python 3.8 or later. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 1. Web Interface

The web app provides an interactive UI for running simulations and visualising results.

```bash
streamlit run web_app.py
```

A browser tab will open automatically (default: `http://localhost:8501`).

### Using the web app

1. **Configure parameters** in the left sidebar:
   - *Simulations per γ* — number of independent runs for each discount factor
   - *Max steps* — training budget per simulation (1M is the default)
   - *Price convergence threshold* — consecutive stable steps required
   - Learning rate, exploration decay, market settings

2. **Click "Run Batch Simulation"** to start. A progress bar tracks completion.

3. **View results** across four tabs:
   - *Results Table* — full per-simulation data
   - *Charts* — price trajectories and convergence steps by discount factor
   - *Summary Stats* — grouped statistics with undercut experiment breakdown
   - *Full Log* — raw simulation log

4. **Download results** using the CSV and Log buttons that appear after the run.

> The discount factors tested are fixed to `{0.05, 0.10, ..., 0.95}` as defined in `config.py`. All other parameters are adjustable from the sidebar.

---

## 2. Command Line

### Sequential batch simulator

```bash
# Test mode: 10 simulations for δ ∈ {0.05, 0.5, 1.0}
python batch_simulator.py --mode test

# Full mode: 100,000 simulations for δ ∈ {0.05, 0.10, ..., 0.95}
python batch_simulator.py --mode full
```

Output is saved to `simulation_results_test.csv` or `simulation_results_full.csv`, plus a timestamped log file.

### Parallel batch simulator

For large-scale runs, use the multiprocessing-based simulator. It distributes simulations across CPU cores and saves a separate CSV per Market 2 gamma multiplier.

```bash
# Default run: 3 multipliers × 19 δ values × 500 sims
python parallel_batch_simulator.py \
  --multipliers 0.5 0.75 1.0 \
  --n-sims 500 \
  --workers 8

# Quick test
python parallel_batch_simulator.py \
  --multipliers 1.0 --n-sims 2 --max-steps 50000 --convergence-count 50
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--multipliers` | `0.5 0.75 1.0` | Market 2 gamma multiplier values |
| `--n-sims` | `500` | Simulations per discount factor |
| `--max-steps` | `1,000,000` | Max training steps per simulation |
| `--convergence-count` | `1000` | Consecutive stable steps for convergence |
| `--learning-rate` | `0.15` | Q-learning α |
| `--step-beta` | `2e-5` | Exploration decay rate β |
| `--price-levels` | `4` | Number of discrete price levels |
| `--workers` | `cpu_count - 1` | Parallel worker processes |
| `--output-dir` | `.` | Directory for output CSVs |

Output files: `results_mult_0.50.csv`, `results_mult_0.75.csv`, `results_mult_1.00.csv`.

### Programmatic usage

```python
from batch_simulator import run_single_simulation

result = run_single_simulation(
    discount_factor=0.5,           # Market 1 uses δ, Market 2 uses 0.7×δ
    max_steps=1_000_000,
    price_convergence_count=1000,
    verbose=True,
)

print(f"Converged:      {result['converged']}")
print(f"Unified price:  {result['converged_price']:.4f}")
print(f"Market 1:       A={result['price_1a']:.2f}, B={result['price_1b']:.2f}")
print(f"Market 2:       A={result['price_2a']:.2f}, B={result['price_2b']:.2f}")
print(f"Q-value A:      {result['converged_q_value_a']:.4f}")
print(f"Q-value B:      {result['converged_q_value_b']:.4f}")
```
