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
    train_matlab_multiagent,
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


def run_single_market(
    config: MarketConfig,
    discount_factor: float,
    max_steps: int,
    price_convergence_count: int,
    market_id: int,
    learning_rate: float = 0.15,
    verbose: bool = False,
) -> Dict:
    """
    Run a single market simulation.

    Args:
        config: Market configuration
        discount_factor: The discount factor (gamma) for Q-learning
        max_steps: Maximum training steps
        price_convergence_count: Consecutive steps for price convergence
        market_id: Market identifier (1 or 2)
        verbose: Print progress during training

    Returns:
        Dict with market results
    """
    environment = BertrandMultiAgentEnvironment(config)

    agent_a = MATLABQLearningAgent(
        name=f"Market{market_id}_FirmA",
        actions=environment.get_available_actions("A"),
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        step_beta=config.STEP_BETA,
        optimistic_init=True,
        environment=environment,
        firm_id="A",
    )

    agent_b = MATLABQLearningAgent(
        name=f"Market{market_id}_FirmB",
        actions=environment.get_available_actions("B"),
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        step_beta=config.STEP_BETA,
        optimistic_init=True,
        environment=environment,
        firm_id="B",
    )

    # Run training using core module's train function
    results = train_matlab_multiagent(
        environment=environment,
        agent_a=agent_a,
        agent_b=agent_b,
        max_steps=max_steps,
        chunk_size=10_000,
        log_every=100_000 if verbose else max_steps + 1,
        price_convergence_count=price_convergence_count,
        stop_immediately=True,
        verbose=verbose,
    )

    return {
        "environment": environment,
        "results": results,
        "agent_a": agent_a,
        "agent_b": agent_b,
    }


def run_single_simulation(
    discount_factor: float,
    max_steps: int = 10_000_000,
    price_convergence_count: int = 1000,
    learning_rate: float = 0.15,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """
    Run a single simulation with 2 independent markets.

    Args:
        discount_factor: The discount factor (delta) for Market 1
                        Market 2 uses: 0.7 × delta (always less than Market 1)
        max_steps: Maximum training steps
        price_convergence_count: Consecutive steps for price convergence
        seed: Random seed for reproducibility
        verbose: Print progress during training

    Returns:
        Dict with keys for both markets:
            - discount_factor_a: float (Market 1's gamma = δ)
            - discount_factor_b: float (Market 2's gamma = 0.7 × δ)
            - Market 1: converged_m1, convergence_step_m1, converged_price_1a, converged_price_1b, etc.
            - Market 2: converged_m2, convergence_step_m2, converged_price_2a, converged_price_2b, etc.
            - time_to_converge: float (seconds)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Start timer
    start_time = time.time()

    # Create shared configuration for both markets
    config = MarketConfig(episode_length=max_steps)

    # Calculate discount factors for each market
    # Market 1: uses the input discount_factor directly (δ)
    # Market 2: uses γ × δ where γ = GAMMA_MULTIPLIER (always less than Market 1)
    discount_factor_m1 = discount_factor
    discount_factor_m2 = config.GAMMA_MULTIPLIER * discount_factor

    # Run Market 1
    if verbose:
        log(f"\n{'='*40}")
        log(f"MARKET 1 (γ = {discount_factor_m1:.4f})")
        log(f"{'='*40}")
    market1 = run_single_market(
        config=config,
        discount_factor=discount_factor_m1,
        max_steps=max_steps,
        price_convergence_count=price_convergence_count,
        market_id=1,
        learning_rate=learning_rate,
        verbose=verbose,
    )

    # Run Market 2
    if verbose:
        log(f"\n{'='*40}")
        log(f"MARKET 2 (γ = {discount_factor_m2:.4f})")
        log(f"{'='*40}")
    market2 = run_single_market(
        config=config,
        discount_factor=discount_factor_m2,
        max_steps=max_steps,
        price_convergence_count=price_convergence_count,
        market_id=2,
        learning_rate=learning_rate,
        verbose=verbose,
    )

    # End timer
    end_time = time.time()
    time_to_converge = end_time - start_time

    # Extract results for Market 1
    results_m1 = market1["results"]
    env_m1 = market1["environment"]
    stable_price_m1 = results_m1.get('stable_price_pair')

    # Extract results for Market 2
    results_m2 = market2["results"]
    env_m2 = market2["environment"]
    stable_price_m2 = results_m2.get('stable_price_pair')

    sim_results = {
        "discount_factor_a": discount_factor_m1,  # Market 1's discount factor (δ)
        "discount_factor_b": discount_factor_m2,  # Market 2's discount factor (0.7 × δ)
        "time_to_converge": time_to_converge,
        
        # Market 1 results
        "converged_m1": results_m1['converged'],
        "convergence_reason_m1": results_m1.get('convergence_reason', ''),
        "convergence_step_m1": results_m1['convergence_step'] if results_m1['converged'] else max_steps,
        "converged_price_1a": stable_price_m1[0] if stable_price_m1 else env_m1.last_price_a,
        "converged_price_1b": stable_price_m1[1] if stable_price_m1 else env_m1.last_price_b,
        
        # Market 2 results
        "converged_m2": results_m2['converged'],
        "convergence_reason_m2": results_m2.get('convergence_reason', ''),
        "convergence_step_m2": results_m2['convergence_step'] if results_m2['converged'] else max_steps,
        "converged_price_2a": stable_price_m2[0] if stable_price_m2 else env_m2.last_price_a,
        "converged_price_2b": stable_price_m2[1] if stable_price_m2 else env_m2.last_price_b,
    }

    # Get Q-values at converged state for Market 1
    if stable_price_m1:
        converged_state_1a = stable_price_m1
        converged_state_1b = (stable_price_m1[1], stable_price_m1[0])
        sim_results["converged_q_value_1a"] = results_m1['agent_a'].get_q_value(converged_state_1a, stable_price_m1[0])
        sim_results["converged_q_value_1b"] = results_m1['agent_b'].get_q_value(converged_state_1b, stable_price_m1[1])
    else:
        sim_results["converged_q_value_1a"] = np.nan
        sim_results["converged_q_value_1b"] = np.nan

    # Get Q-values at converged state for Market 2
    if stable_price_m2:
        converged_state_2a = stable_price_m2
        converged_state_2b = (stable_price_m2[1], stable_price_m2[0])
        sim_results["converged_q_value_2a"] = results_m2['agent_a'].get_q_value(converged_state_2a, stable_price_m2[0])
        sim_results["converged_q_value_2b"] = results_m2['agent_b'].get_q_value(converged_state_2b, stable_price_m2[1])
    else:
        sim_results["converged_q_value_2a"] = np.nan
        sim_results["converged_q_value_2b"] = np.nan

    if verbose:
        log(f"\n{'='*70}")
        log(f"SIMULATION COMPLETE - BOTH MARKETS")
        log(f"{'='*70}")
        log(f"Discount factors: Market 1 γ={discount_factor_m1:.4f}, Market 2 γ={discount_factor_m2:.4f}")
        log(f"\nMarket 1 (γ = {discount_factor_m1:.4f}):")
        log(f"  Converged: {sim_results['converged_m1']} ({sim_results['convergence_reason_m1']})")
        log(f"  Steps: {sim_results['convergence_step_m1']:,}")
        log(f"  Price: 1A={sim_results['converged_price_1a']:.2f}, 1B={sim_results['converged_price_1b']:.2f}")
        log(f"  Q-values: 1A={sim_results['converged_q_value_1a']:.4f}, 1B={sim_results['converged_q_value_1b']:.4f}")
        log(f"\nMarket 2 (γ = {discount_factor_m2:.4f}):")
        log(f"  Converged: {sim_results['converged_m2']} ({sim_results['convergence_reason_m2']})")
        log(f"  Steps: {sim_results['convergence_step_m2']:,}")
        log(f"  Price: 2A={sim_results['converged_price_2a']:.2f}, 2B={sim_results['converged_price_2b']:.2f}")
        log(f"  Q-values: 2A={sim_results['converged_q_value_2a']:.4f}, 2B={sim_results['converged_q_value_2b']:.4f}")
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

        # Print summary statistics for Market 1
        log("\nSUMMARY STATISTICS - MARKET 1")
        log(f"{'='*70}")
        summary_m1 = df_results.groupby('discount_factor_a').agg({
            'converged_m1': ['sum', 'mean'],
            'convergence_step_m1': ['mean', 'std'],
            'converged_price_1a': ['mean', 'std'],
            'converged_price_1b': ['mean', 'std'],
            'converged_q_value_1a': ['mean', 'std'],
            'converged_q_value_1b': ['mean', 'std'],
        }).round(4)
        log(summary_m1.to_string())

        # Print summary statistics for Market 2
        log("\nSUMMARY STATISTICS - MARKET 2")
        log(f"{'='*70}")
        summary_m2 = df_results.groupby('discount_factor_a').agg({
            'converged_m2': ['sum', 'mean'],
            'convergence_step_m2': ['mean', 'std'],
            'converged_price_2a': ['mean', 'std'],
            'converged_price_2b': ['mean', 'std'],
            'converged_q_value_2a': ['mean', 'std'],
            'converged_q_value_2b': ['mean', 'std'],
        }).round(4)
        log(summary_m2.to_string())

        # Print overall timing
        log("\nTIMING STATISTICS")
        log(f"{'='*70}")
        timing_summary = df_results.groupby('discount_factor_a').agg({
            'time_to_converge': ['mean', 'std'],
        }).round(4)
        log(timing_summary.to_string())
        
        # Print convergence reason breakdown for both markets
        log("\nCONVERGENCE REASON BREAKDOWN - MARKET 1")
        log(f"{'='*70}")
        reason_counts_m1 = df_results.groupby(['discount_factor_a', 'convergence_reason_m1']).size().unstack(fill_value=0)
        log(reason_counts_m1.to_string())

        log("\nCONVERGENCE REASON BREAKDOWN - MARKET 2")
        log(f"{'='*70}")
        reason_counts_m2 = df_results.groupby(['discount_factor_a', 'convergence_reason_m2']).size().unstack(fill_value=0)
        log(reason_counts_m2.to_string())

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

