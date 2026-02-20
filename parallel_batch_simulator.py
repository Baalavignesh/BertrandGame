"""
Parallel Batch Simulator for Multi-Agent Q-Learning

Runs simulations in parallel using Python multiprocessing for massive speedup.
Includes real-time progress tracking with ETA.

Usage:
    # Run all 3 multipliers (0.5, 0.75, 1.0) with 500 sims each
    python parallel_batch_simulator.py

    # Custom run
    python parallel_batch_simulator.py --multipliers 0.5 0.75 1.0 --n-sims 500 --workers 8

    # Quick test
    python parallel_batch_simulator.py --multipliers 1.0 --n-sims 2 --max-steps 50000 --convergence-count 50
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
from batch_simulator import run_single_simulation


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (runs in child process)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args: Tuple) -> Dict:
    """
    Worker function for multiprocessing. Runs a single simulation with the
    given parameters. Must be a top-level function for pickling.

    Args:
        args: Tuple of (discount_factor, sim_idx, params_dict)

    Returns:
        Dict with simulation results
    """
    discount_factor, sim_idx, params = args

    # Each worker gets its own unique seed based on time + identity
    seed = (int(time.time() * 1e6) + os.getpid() + sim_idx) % (2**31)

    result = run_single_simulation(
        discount_factor=discount_factor,
        max_steps=params["max_steps"],
        price_convergence_count=params["convergence_count"],
        learning_rate=params["learning_rate"],
        step_beta=params["step_beta"],
        num_price_levels=params["price_levels"],
        price_min=params["price_min"],
        price_max=params["price_max"],
        marginal_cost=params["marginal_cost"],
        gamma_multiplier=params["gamma_multiplier"],
        seed=seed,
        verbose=False,  # No logging in workers for speed
    )

    result["simulation_id"] = sim_idx
    result["seed"] = seed
    result["gamma_multiplier"] = params["gamma_multiplier"]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Progress display
# ─────────────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_progress(completed: int, total: int, start_time: float, 
                   multiplier: float, bar_width: int = 30):
    """Print a compact progress bar to stdout."""
    elapsed = time.time() - start_time
    pct = completed / total if total > 0 else 0
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Calculate ETA
    if completed > 0:
        rate = completed / elapsed
        remaining = (total - completed) / rate
        eta_str = format_time(remaining)
        rate_str = f"{rate:.1f}"
    else:
        eta_str = "..."
        rate_str = "..."

    line = (
        f"\r  [{bar}] {completed:,}/{total:,} "
        f"({pct*100:.1f}%) | "
        f"{format_time(elapsed)} elapsed | "
        f"ETA {eta_str} | "
        f"{rate_str} sims/s  "
    )
    sys.stdout.write(line)
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Main parallel runner
# ─────────────────────────────────────────────────────────────────────────────

def run_parallel_batch(
    discount_factors: List[float],
    n_simulations: int,
    gamma_multiplier: float,
    params: Dict,
    output_file: str,
    num_workers: int,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """
    Run batch simulations in parallel using multiprocessing.

    Args:
        discount_factors: List of delta values to test
        n_simulations: Simulations per delta
        gamma_multiplier: Market 2 multiplier
        params: Dict of simulation parameters
        output_file: CSV output path
        num_workers: Number of parallel workers
        checkpoint_every: Save intermediate results every N completions

    Returns:
        DataFrame with all results
    """
    # Build task list: (discount_factor, global_sim_id, params_with_multiplier)
    params_with_mult = {**params, "gamma_multiplier": gamma_multiplier}
    tasks = []
    sim_id = 0
    for delta in discount_factors:
        for _ in range(n_simulations):
            tasks.append((delta, sim_id, params_with_mult))
            sim_id += 1

    total = len(tasks)
    print(f"\n  Total simulations: {total:,}")
    print(f"  Workers: {num_workers}")
    print(f"  Checkpointing every {checkpoint_every} sims to: {output_file}")
    print()

    all_results = []
    start_time = time.time()

    # Use imap_unordered for best throughput + incremental results
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_worker, tasks, chunksize=4):
            all_results.append(result)
            completed = len(all_results)

            # Progress bar
            print_progress(completed, total, start_time, gamma_multiplier)

            # Periodic checkpoint
            if completed % checkpoint_every == 0:
                df_temp = pd.DataFrame(all_results)
                df_temp.to_csv(output_file, index=False)

    # Final save
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)

    elapsed = time.time() - start_time
    print(f"\n  ✅ Done! {total:,} sims in {format_time(elapsed)} "
          f"({total/elapsed:.1f} sims/s)")
    print(f"  📄 Saved to: {output_file}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel Batch Simulator for Q-Learning Simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Multiplier and simulation count
    parser.add_argument(
        "--multipliers", type=float, nargs="+", default=[0.5, 0.75, 1.0],
        help="Market 2 gamma multiplier values to test"
    )
    parser.add_argument(
        "--n-sims", type=int, default=500,
        help="Number of simulations per discount factor"
    )

    # Simulation parameters
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--convergence-count", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--step-beta", type=float, default=2e-5)
    parser.add_argument("--price-levels", type=int, default=4)
    parser.add_argument("--price-min", type=float, default=0.0)
    parser.add_argument("--price-max", type=float, default=0.65)
    parser.add_argument("--marginal-cost", type=float, default=0.10)

    # Parallelism
    parser.add_argument(
        "--workers", type=int, default=max(1, mp.cpu_count() - 1),
        help="Number of parallel workers (default: CPU count - 1)"
    )

    # Discount factors
    parser.add_argument(
        "--discount-factors", type=float, nargs="+",
        default=config.DISCOUNT_FACTORS,
        help="List of discount factors (delta values)"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save result CSVs"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="Save intermediate results every N completions"
    )

    args = parser.parse_args()

    # Build params dict for workers
    params = {
        "max_steps": args.max_steps,
        "convergence_count": args.convergence_count,
        "learning_rate": args.learning_rate,
        "step_beta": args.step_beta,
        "price_levels": args.price_levels,
        "price_min": args.price_min,
        "price_max": args.price_max,
        "marginal_cost": args.marginal_cost,
    }

    total_start = time.time()

    print("=" * 70)
    print("PARALLEL BATCH SIMULATOR")
    print("=" * 70)
    print(f"  Multipliers:     {args.multipliers}")
    print(f"  Discount factors: {args.discount_factors}")
    print(f"  Sims per δ:       {args.n_sims}")
    print(f"  Max steps:        {args.max_steps:,}")
    print(f"  Convergence:      {args.convergence_count}")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Step beta:        {args.step_beta}")
    print(f"  Price levels:     {args.price_levels}")
    print(f"  Price range:      [{args.price_min}, {args.price_max}]")
    print(f"  Marginal cost:    {args.marginal_cost}")
    print(f"  Workers:          {args.workers}")
    total_all = len(args.multipliers) * len(args.discount_factors) * args.n_sims
    print(f"  Total sims:       {total_all:,}")
    print("=" * 70)

    all_dfs = []

    for mult in args.multipliers:
        print(f"\n{'─' * 70}")
        print(f"  MULTIPLIER = {mult}")
        print(f"  Market 2 uses γ₂ = {mult} × δ")
        print(f"{'─' * 70}")

        output_file = os.path.join(
            args.output_dir, f"results_mult_{mult:.2f}.csv"
        )

        df = run_parallel_batch(
            discount_factors=args.discount_factors,
            n_simulations=args.n_sims,
            gamma_multiplier=mult,
            params=params,
            output_file=output_file,
            num_workers=args.workers,
            checkpoint_every=args.checkpoint_every,
        )
        all_dfs.append(df)

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print("ALL RUNS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time: {format_time(total_elapsed)}")
    print(f"  Total sims: {total_all:,}")
    print(f"  Throughput:  {total_all / total_elapsed:.1f} sims/s")
    print(f"\n  Output files:")
    for mult in args.multipliers:
        fname = os.path.join(args.output_dir, f"results_mult_{mult:.2f}.csv")
        if os.path.exists(fname):
            rows = len(pd.read_csv(fname))
            size_mb = os.path.getsize(fname) / (1024 * 1024)
            print(f"    {fname} ({rows:,} rows, {size_mb:.1f} MB)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
