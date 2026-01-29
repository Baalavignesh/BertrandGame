"""
Batch Simulator for Multi-Agent Q-Learning with Multiple Markets

This script runs multiple simulations across different discount factors and collects
convergence metrics for analysis. It supports running 2 independent markets, each
with 2 agents (firms A and B).

Configuration:
- Price range: [0, 1] (normalized)
- Production cost: 0 for both firms
- For each discount factor in {0.05, 0.10, ..., 0.95, 1.0}, run simulations
- Market 1 uses δ, Market 2 uses 0.7 × δ
- Collect: converged price, converged Q-value, time to converge, if converged or not

Extended for 2 markets:
- Market 1: Firm 1A and Firm 1B
- Market 2: Firm 2A and Firm 2B
- Each market runs independently with the same environment configuration
- Results include converged prices for all 4 agents (1A, 1B, 2A, 2B)

Usage:
    # Test mode: 10 simulations for discount factors [0.05, 0.5, 1.0]
    python batch_simulator.py --mode test

    # Full scale: 100,000 simulations for discount factors [0.05, 0.10, ..., 0.95, 1.0]
    python batch_simulator.py --mode full
"""

import argparse
import logging
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import configuration constants
import config

# Import from core module - no duplicate code!
from core import (
    MarketConfig,
    BertrandMultiAgentEnvironment,
    MATLABQLearningAgent,
    train_dual_market_interleaved,
    run_undercut_experiment,
)

# Global logger
logger = logging.getLogger("batch_simulator")


def setup_logging(log_file: Optional[str] = None, verbose: bool = True) -> str:
    """
    Set up logging to both console and file.
    
    Args:
        log_file: Path to log file. If None, generates a timestamped filename.
        verbose: If True, also log to console.
    
    Returns:
        Path to the log file.
    """
    # Generate log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"simulation_log_{timestamp}.txt"
    
    # Clear any existing handlers
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    
    # File handler - always write to file
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only if verbose
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return log_file


def log(message: str = "") -> None:
    """Log a message to both console and file."""
    logger.info(message)


def run_single_simulation(
    discount_factor: float,
    max_steps: int = config.DEFAULT_MAX_STEPS,
    price_convergence_count: int = config.DEFAULT_PRICE_CONVERGENCE_COUNT,
    learning_rate: float = config.LEARNING_RATE,
    step_beta: float = config.STEP_BETA,
    num_price_levels: int = config.NUM_PRICE_LEVELS,
    price_min: float = config.PRICE_MIN,
    price_max: float = config.PRICE_MAX,
    marginal_cost: float = config.MARGINAL_COST,
    gamma_multiplier: float = config.GAMMA_MULTIPLIER,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """
    Run a single simulation with 2 markets running in parallel (interleaved steps).
    
    Both markets take actions at each step together, and convergence is based on a
    unified price: ((M1_A + M1_B)/2 + (M2_A + M2_B)/2) / 2

    Args:
        discount_factor: The discount factor (delta) for Market 1
                        Market 2 uses: gamma_multiplier × delta
        max_steps: Maximum training steps
        price_convergence_count: Consecutive steps for unified price convergence
        learning_rate: Q-learning update rate (alpha)
        step_beta: Exploration decay parameter (beta)
        num_price_levels: Number of price levels (k)
        price_min: Minimum price
        price_max: Maximum price
        marginal_cost: Production cost for firms
        gamma_multiplier: Multiplier for Market 2's discount factor
        seed: Random seed for reproducibility
        verbose: Print progress during training

    Returns:
        Dict with unified convergence price and individual market prices
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Start timer
    start_time = time.time()

    # Calculate price step from num_price_levels
    price_step = (price_max - price_min) / (num_price_levels - 1) if num_price_levels > 1 else 0.1

    # Create shared configuration for both markets
    market_config = MarketConfig(
        episode_length=max_steps,
        firm_a_cost=marginal_cost,
        firm_b_cost=marginal_cost,
        price_min=price_min,
        price_max=price_max,
        price_step=price_step,
    )

    # Calculate discount factors for each market
    # Market 1: uses the input discount_factor directly (δ)
    # Market 2: uses gamma_multiplier × δ (always less than Market 1)
    discount_factor_m1 = discount_factor
    discount_factor_m2 = gamma_multiplier * discount_factor

    # Create environments for both markets
    env1 = BertrandMultiAgentEnvironment(market_config)
    env2 = BertrandMultiAgentEnvironment(market_config)

    # Create 2 shared agents (used in both markets)
    # Use average discount factor for optimistic initialization
    avg_discount_factor = (discount_factor_m1 + discount_factor_m2) / 2
    
    agent_a = MATLABQLearningAgent(
        name="FirmA",
        actions=env1.get_available_actions("A"),
        learning_rate=learning_rate,
        discount_factor=avg_discount_factor,  # Used for optimistic init only
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        step_beta=step_beta,
        optimistic_init=True,
        environment=env1,
        firm_id="A",
    )
    agent_b = MATLABQLearningAgent(
        name="FirmB",
        actions=env1.get_available_actions("B"),
        learning_rate=learning_rate,
        discount_factor=avg_discount_factor,  # Used for optimistic init only
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        step_beta=step_beta,
        optimistic_init=True,
        environment=env1,
        firm_id="B",
    )

    if verbose:
        log(f"\n{'='*70}")
        log(f"DUAL MARKET WITH SHARED AGENTS")
        log(f"Market 1 γ = {discount_factor_m1:.4f}, Market 2 γ = {discount_factor_m2:.4f}")
        log(f"Agents share Q-tables across both markets")
        log(f"{'='*70}")

    # Run interleaved training with shared agents
    results = train_dual_market_interleaved(
        env1=env1,
        env2=env2,
        agent_a=agent_a,
        agent_b=agent_b,
        discount_factor_m1=discount_factor_m1,
        discount_factor_m2=discount_factor_m2,
        max_steps=max_steps,
        chunk_size=10_000,
        log_every=100_000 if verbose else max_steps + 1,
        price_convergence_count=price_convergence_count,
        stop_immediately=True,
        verbose=verbose,
    )

    # End timer
    end_time = time.time()
    time_to_converge = end_time - start_time

    # Extract converged prices
    stable_prices_m1 = results.get('stable_prices_m1')
    stable_prices_m2 = results.get('stable_prices_m2')
    converged_price = results.get('converged_price')

    sim_results = {
        "discount_factor_a": discount_factor_m1,  # Market 1's discount factor (δ)
        "discount_factor_b": discount_factor_m2,  # Market 2's discount factor (0.7 × δ)
        "time_to_converge": time_to_converge,
        
        # Unified convergence (requires both markets to independently stabilize)
        "converged": results['converged'],
        "convergence_reason": results.get('convergence_reason', ''),
        "convergence_step": results['convergence_step'] if results['converged'] else max_steps,
        "converged_price": converged_price if converged_price is not None else np.nan,
        
        # Per-market stability counts
        "m1_stable_count": results.get('m1_stable_count', 0),
        "m2_stable_count": results.get('m2_stable_count', 0),
        
        # Individual market prices at convergence (for reference)
        "price_1a": stable_prices_m1[0] if stable_prices_m1 else env1.last_price_a,
        "price_1b": stable_prices_m1[1] if stable_prices_m1 else env1.last_price_b,
        "price_2a": stable_prices_m2[0] if stable_prices_m2 else env2.last_price_a,
        "price_2b": stable_prices_m2[1] if stable_prices_m2 else env2.last_price_b,
    }

    # Get Q-values from shared agents (only 2 Q-values now)
    if stable_prices_m1:
        converged_state_a = stable_prices_m1
        converged_state_b = (stable_prices_m1[1], stable_prices_m1[0])
        sim_results["converged_q_value_a"] = agent_a.get_q_value(converged_state_a, stable_prices_m1[0])
        sim_results["converged_q_value_b"] = agent_b.get_q_value(converged_state_b, stable_prices_m1[1])
    else:
        sim_results["converged_q_value_a"] = np.nan
        sim_results["converged_q_value_b"] = np.nan

    # Run undercut experiments for both markets (using shared agents)
    # Market 1 undercut experiment
    if stable_prices_m1:
        undercut_m1 = run_undercut_experiment(
            environment=env1,
            agent_a=agent_a,
            agent_b=agent_b,
            converged_price_a=stable_prices_m1[0],
            converged_price_b=stable_prices_m1[1],
            undercut_amount=0.1,
            steps_after_undercut=15,
        )
        sim_results["m1_undercut_price_b"] = undercut_m1["undercut_price_b"]
        # Add trajectory columns: m1_uc_pa_1, m1_uc_pb_1, ..., m1_uc_pa_15, m1_uc_pb_15
        for i, (pa, pb) in enumerate(undercut_m1["trajectory"], start=1):
            sim_results[f"m1_uc_pa_{i}"] = pa
            sim_results[f"m1_uc_pb_{i}"] = pb
    else:
        # No convergence - fill with NaN
        sim_results["m1_undercut_price_b"] = np.nan
        for i in range(1, 16):
            sim_results[f"m1_uc_pa_{i}"] = np.nan
            sim_results[f"m1_uc_pb_{i}"] = np.nan

    # Market 2 undercut experiment
    if stable_prices_m2:
        undercut_m2 = run_undercut_experiment(
            environment=env2,
            agent_a=agent_a,
            agent_b=agent_b,
            converged_price_a=stable_prices_m2[0],
            converged_price_b=stable_prices_m2[1],
            undercut_amount=0.1,
            steps_after_undercut=15,
        )
        sim_results["m2_undercut_price_b"] = undercut_m2["undercut_price_b"]
        # Add trajectory columns: m2_uc_pa_1, m2_uc_pb_1, ..., m2_uc_pa_15, m2_uc_pb_15
        for i, (pa, pb) in enumerate(undercut_m2["trajectory"], start=1):
            sim_results[f"m2_uc_pa_{i}"] = pa
            sim_results[f"m2_uc_pb_{i}"] = pb
    else:
        # No convergence - fill with NaN
        sim_results["m2_undercut_price_b"] = np.nan
        for i in range(1, 16):
            sim_results[f"m2_uc_pa_{i}"] = np.nan
            sim_results[f"m2_uc_pb_{i}"] = np.nan

    if verbose:
        log(f"\n{'='*70}")
        log(f"SIMULATION COMPLETE - SHARED AGENTS")
        log(f"{'='*70}")
        log(f"Discount factors: Market 1 γ={discount_factor_m1:.4f}, Market 2 γ={discount_factor_m2:.4f}")
        log(f"\nConvergence (both markets must stabilize independently):")
        log(f"  Converged: {sim_results['converged']} ({sim_results['convergence_reason']})")
        log(f"  Steps: {sim_results['convergence_step']:,}")
        log(f"  Unified Price: {sim_results['converged_price']:.4f}")
        log(f"\nPer-Market Stability:")
        log(f"  Market 1 stable count: {sim_results['m1_stable_count']:,}")
        log(f"  Market 2 stable count: {sim_results['m2_stable_count']:,}")
        log(f"\nIndividual Prices:")
        log(f"  Market 1: A={sim_results['price_1a']:.2f}, B={sim_results['price_1b']:.2f}")
        log(f"  Market 2: A={sim_results['price_2a']:.2f}, B={sim_results['price_2b']:.2f}")
        log(f"\nShared Q-values (from Market 1 converged state):")
        log(f"  Agent A: {sim_results['converged_q_value_a']:.4f}")
        log(f"  Agent B: {sim_results['converged_q_value_b']:.4f}")
        log(f"\nTotal time: {time_to_converge:.2f}s")

    return sim_results

def run_batch_simulations(
    discount_factors: List[float],
    n_simulations: int,
    max_steps: int = 10_000_000,
    price_convergence_count: int = 1000,
    output_file: str = "simulation_results.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run batch simulations across multiple discount factors.

    Args:
        discount_factors: List of discount factors to test (e.g., [0.05, 0.10, ..., 0.95, 1.0])
        n_simulations: Number of simulations per discount factor
        max_steps: Maximum steps per simulation
        price_convergence_count: Consecutive steps for price convergence
        output_file: Path to save results CSV
        verbose: Print progress

    Returns:
        DataFrame with all simulation results
    """
    all_results = []
    total_sims = len(discount_factors) * n_simulations

    if verbose:
        log("="*70)
        log("BATCH SIMULATION CONFIGURATION")
        log("="*70)
        log(f"Discount factors: {discount_factors}")
        log(f"Simulations per factor: {n_simulations}")
        log(f"Total simulations: {total_sims:,}")
        log(f"Max steps per sim: {max_steps:,}")
        log(f"Price convergence threshold: {price_convergence_count}")
        log(f"Output file: {output_file}")
        log("="*70)

    batch_start_time = time.time()
    sim_count = 0

    for gamma in discount_factors:
        if verbose:
            log(f"\n{'='*70}")
            log(f"Running {n_simulations} simulations for γ = {gamma}")
            log(f"{'='*70}")

        for sim_idx in range(n_simulations):
            sim_count += 1
            seed = int(time.time() * 1000000) % (2**31)  # Generate unique seed

            if verbose:
                log(f"\nSimulation {sim_count}/{total_sims} (γ={gamma}, run {sim_idx+1}/{n_simulations})")

            result = run_single_simulation(
                discount_factor=gamma,
                max_steps=max_steps,
                price_convergence_count=price_convergence_count,
                seed=seed,
                verbose=verbose,
            )

            result["simulation_id"] = sim_count
            result["gamma_index"] = sim_idx
            all_results.append(result)

            # Save intermediate results every 10 simulations
            if sim_count % 10 == 0:
                df_temp = pd.DataFrame(all_results)
                df_temp.to_csv(output_file, index=False)
                if verbose:
                    log(f"\n[Checkpoint] Saved {sim_count} results to {output_file}")

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    # Save final results
    df_results.to_csv(output_file, index=False)

    batch_end_time = time.time()
    total_time = batch_end_time - batch_start_time

    if verbose:
        log(f"\n{'='*70}")
        log("BATCH SIMULATION COMPLETE")
        log(f"{'='*70}")
        log(f"Total simulations: {total_sims:,}")
        log(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        log(f"Average time per sim: {total_time/total_sims:.2f}s")
        log(f"Results saved to: {output_file}")
        log(f"{'='*70}")

        # Print summary statistics for unified convergence
        log("\nSUMMARY STATISTICS - UNIFIED CONVERGENCE")
        log(f"{'='*70}")
        summary_unified = df_results.groupby('discount_factor_a').agg({
            'converged': ['sum', 'mean'],
            'convergence_step': ['mean', 'std'],
            'converged_price': ['mean', 'std'],
        }).round(4)
        log(summary_unified.to_string())

        # Print summary statistics for individual market prices
        log("\nINDIVIDUAL MARKET PRICES")
        log(f"{'='*70}")
        summary_prices = df_results.groupby('discount_factor_a').agg({
            'price_1a': ['mean', 'std'],
            'price_1b': ['mean', 'std'],
            'price_2a': ['mean', 'std'],
            'price_2b': ['mean', 'std'],
        }).round(4)
        log(summary_prices.to_string())

        # Print Q-value statistics (shared agents - only 2 Q-values)
        log("\nSHARED Q-VALUE STATISTICS")
        log(f"{'='*70}")
        summary_q = df_results.groupby('discount_factor_a').agg({
            'converged_q_value_a': ['mean', 'std'],
            'converged_q_value_b': ['mean', 'std'],
        }).round(4)
        log(summary_q.to_string())

        # Print overall timing
        log("\nTIMING STATISTICS")
        log(f"{'='*70}")
        timing_summary = df_results.groupby('discount_factor_a').agg({
            'time_to_converge': ['mean', 'std'],
        }).round(4)
        log(timing_summary.to_string())
        
        # Print convergence reason breakdown
        log("\nCONVERGENCE REASON BREAKDOWN")
        log(f"{'='*70}")
        reason_counts = df_results.groupby(['discount_factor_a', 'convergence_reason']).size().unstack(fill_value=0)
        log(reason_counts.to_string())

    return df_results


def main():
    parser = argparse.ArgumentParser(
        description="Run batch simulations for multi-agent Q-learning"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "full"],
        default="test",
        help="Simulation mode: test (10 sims, δ=0.05,0.5,1) or full (100k sims, δ=0.05,0.10,...,1)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If not provided, generates a timestamped filename."
    )

    args = parser.parse_args()

    # Fixed parameters
    max_steps = 1_000_000  # Maximum steps if convergence not reached
    price_convergence_count = 100  # Converge when prices stable for 100 consecutive steps

    # Configure based on mode
    # Note: Discount factors start at 0.05 (not 0) so Market 2 has meaningful values
    if args.mode == "test":
        # Test mode: 10 simulations for discount factors 0.05, 0.5, 1.0
        discount_factors = [0.05, 0.5, 1.0]
        n_simulations = 10
        output_file = "simulation_results_test.csv"
    else:  # full
        # Full scale: 100,000 simulations for discount factors [0.05, 0.10, ..., 0.95, 1.0]
        discount_factors = [round(i * 0.05, 2) for i in range(1, 21)]  # [0.05, 0.10, ..., 1.0]
        n_simulations = 100_000
        output_file = "simulation_results_full.csv"

    # Set up logging (writes to both console and file)
    log_file = setup_logging(log_file=args.log_file, verbose=True)
    log(f"Logging to file: {log_file}")

    # Run batch simulations
    results_df = run_batch_simulations(
        discount_factors=discount_factors,
        n_simulations=n_simulations,
        max_steps=max_steps,
        price_convergence_count=price_convergence_count,
        output_file=output_file,
        verbose=True,
    )

    log(f"\nResults shape: {results_df.shape}")
    log(f"\nFirst few rows:")
    log(results_df.head(10).to_string())
    log(f"\n{'='*70}")
    log(f"Log saved to: {log_file}")
    log(f"Results saved to: {output_file}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()

