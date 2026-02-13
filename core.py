"""
Core Q-Learning Module for Multi-Agent Bertrand Competition

This module contains all the shared classes and functions used by both
the interactive notebook and the batch simulator.

Classes:
    - MarketConfig: Configuration for the market environment
    - BertrandMultiAgentEnvironment: Two-firm Bertrand competition environment
    - MATLABQLearningAgent: Q-learning agent replicating MATLAB's logic

Functions:
    - train_dual_market_interleaved: Train dual-market system with shared agents
    - run_undercut_experiment: Test post-convergence deviation behavior
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Import constants from config
import config


@dataclass(frozen=True)
class MarketConfig:
    """Market configuration with parameters from config.py.
    
    Default configuration uses values from config.py:
        - Price range: [PRICE_MIN, PRICE_MAX]
        - Production cost: MARGINAL_COST
        - Price levels: NUM_PRICE_LEVELS
    """
    demand_intercept: float = config.DEMAND_INTERCEPT
    demand_slope: float = config.DEMAND_SLOPE
    firm_a_cost: float = config.MARGINAL_COST
    firm_b_cost: float = config.MARGINAL_COST
    price_min: float = config.PRICE_MIN
    price_max: float = config.PRICE_MAX
    price_step: float = config.PRICE_STEP
    episode_length: int = 50
    reward_scale: float = config.REWARD_SCALE


class BertrandMultiAgentEnvironment:
    """Two-firm Bertrand competition with simultaneous price setting."""

    def __init__(self, config: MarketConfig):
        self.config = config
        # Generate price levels using numpy for float range
        self.price_levels = list(
            np.arange(config.price_min, config.price_max + config.price_step/2, config.price_step)
        )
        # Round to avoid floating point precision issues
        self.price_levels = [round(p, 2) for p in self.price_levels]
        
        if not self.price_levels:
            raise ValueError("Price grid is empty; adjust configuration bounds.")

        self.actions_a = [p for p in self.price_levels if p >= config.firm_a_cost]
        self.actions_b = [p for p in self.price_levels if p >= config.firm_b_cost]
        if not self.actions_a or not self.actions_b:
            raise ValueError("Action spaces are empty; check costs vs price bounds.")

        # Start at marginal cost (or minimum price if cost is 0)
        self.start_price_a = min(self.actions_a)
        self.start_price_b = min(self.actions_b)
        
        self.history: List[Dict[str, float]] = []
        self.step_count = 0
        self.last_price_a = self.start_price_a
        self.last_price_b = self.start_price_b

    def reset(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        self.history.clear()
        self.step_count = 0
        self.last_price_a = self.start_price_a
        self.last_price_b = self.start_price_b
        return self._states()

    def get_available_actions(self, firm: str) -> List[float]:
        if firm.upper() == "A":
            return self.actions_a
        if firm.upper() == "B":
            return self.actions_b
        raise ValueError("firm must be 'A' or 'B'.")

    def step(
        self,
        price_a: float,
        price_b: float,
    ) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[float, float], bool, Dict[str, float]]:
        if price_a not in self.actions_a:
            raise ValueError(f"Firm A chose invalid price {price_a}.")
        if price_b not in self.actions_b:
            raise ValueError(f"Firm B chose invalid price {price_b}.")

        profit_a = self._profit(price_a, price_b, self.config.firm_a_cost)
        profit_b = self._profit(price_b, price_a, self.config.firm_b_cost)

        # Scale rewards
        profit_a *= self.config.reward_scale
        profit_b *= self.config.reward_scale

        self.step_count += 1
        self.last_price_a = price_a
        self.last_price_b = price_b

        self.history.append(
            {
                "step": float(self.step_count),
                "price_a": float(price_a),
                "price_b": float(price_b),
                "profit_a": profit_a,
                "profit_b": profit_b,
            }
        )

        done = self.step_count >= self.config.episode_length
        joint_reward = profit_a + profit_b
        return self._states(), (profit_a, profit_b), done, {
            "profit_a": profit_a,
            "profit_b": profit_b,
            "joint_reward": joint_reward,
        }

    def _states(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        state_a = (self.last_price_a, self.last_price_b)
        state_b = (self.last_price_b, self.last_price_a)
        return state_a, state_b

    def _demand(self, price: float) -> float:
        return max(0.0, self.config.demand_intercept - self.config.demand_slope * price)

    def _profit(self, my_price: float, rival_price: float, my_cost: float) -> float:
        if my_price < rival_price:
            quantity = self._demand(my_price)
            return max(0.0, (my_price - my_cost) * quantity)
        if my_price == rival_price:
            quantity = self._demand(my_price) / 2.0
            return max(0.0, (my_price - my_cost) * quantity)
        return 0.0


class MATLABQLearningAgent:
    """Q-learner that exactly replicates MATLAB's update and convergence logic."""

    def __init__(
        self,
        name: str,
        actions: Iterable[float],
        learning_rate: float = config.LEARNING_RATE,
        discount_factor: float = 0.95,
        epsilon_start: float = config.EPSILON_START,
        epsilon_min: float = config.EPSILON_MIN,
        step_beta: float = config.STEP_BETA,
        optimistic_init: bool = True,
        environment: Optional['BertrandMultiAgentEnvironment'] = None,
        firm_id: str = "A",
    ):
        self.name = name
        self.actions = list(actions)
        if not self.actions:
            raise ValueError("Action space cannot be empty.")

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.step_beta = step_beta

        self.q_table: Dict[Tuple[float, float], Dict[float, float]] = defaultdict(lambda: defaultdict(float))
        self.epsilon = epsilon_start
        self.global_step = 0

        # MATLAB: Optimistic initialization
        if optimistic_init and environment is not None:
            self._init_optimistic_q(environment, firm_id)

    def _init_optimistic_q(self, environment: 'BertrandMultiAgentEnvironment', firm_id: str) -> None:
        """Initialize Q-values optimistically: Q(s,a) = π(s,a) / (1 - γ)"""
        my_cost = environment.config.firm_a_cost if firm_id == "A" else environment.config.firm_b_cost
        all_prices = environment.price_levels
        
        for my_last_price in all_prices:
            for rival_last_price in all_prices:
                state = (my_last_price, rival_last_price)
                for action_price in self.actions:
                    profit = environment._profit(action_price, rival_last_price, my_cost)
                    profit *= environment.config.reward_scale
                    if self.discount_factor < 1.0:
                        optimistic_q = profit / (1.0 - self.discount_factor)
                    else:
                        # For γ=1, use a large but finite value
                        optimistic_q = profit * 1000.0
                    self.q_table[state][action_price] = optimistic_q

    def start_episode(self) -> None:
        self.global_step = 0
        self.epsilon = self.epsilon_start

    def _epsilon_for_step(self, step: int) -> float:
        """MATLAB: pr_explore = exp(-t * beta)"""
        return max(self.epsilon_min, self.epsilon_start * math.exp(-self.step_beta * step))

    def select_action(self, state: Tuple[float, float]) -> float:
        """MATLAB: epsilon-greedy with step-based decay"""
        self.epsilon = self._epsilon_for_step(self.global_step)
        
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.greedy_action(state)
        
        self.global_step += 1
        return action

    def update(
        self,
        state: Tuple[float, float],
        action: float,
        reward: float,
        next_state: Tuple[float, float],
        done: bool,
        discount_factor: float,
    ) -> bool:
        """MATLAB-style Q-update. Returns True if greedy policy changed.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            discount_factor: The discount factor (gamma) to use for this update
        """
        # Get argmax BEFORE update
        old_argmax = self.greedy_action(state)
        
        # MATLAB Q-update
        old_value = self.q_table[state][action]
        if done:
            new_value = reward
        else:
            max_q_next = max([self.q_table[next_state][a] for a in self.actions])
            new_value = reward + discount_factor * max_q_next
        
        self.q_table[state][action] = (1 - self.learning_rate) * old_value + self.learning_rate * new_value
        
        # Get argmax AFTER update
        new_argmax = self.greedy_action(state)
        
        # Return whether policy changed
        return old_argmax != new_argmax

    def greedy_action(self, state: Tuple[float, float]) -> float:
        """Return action with highest Q-value."""
        q_values = [self.q_table[state][a] for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def get_q_value(self, state: Tuple[float, float], action: float) -> float:
        """Get Q-value for a specific state-action pair."""
        return self.q_table[state][action]


def train_dual_market_interleaved(
    env1: BertrandMultiAgentEnvironment,
    env2: BertrandMultiAgentEnvironment,
    agent_a: MATLABQLearningAgent,
    agent_b: MATLABQLearningAgent,
    discount_factor_m1: float,
    discount_factor_m2: float,
    max_steps: int = 10_000_000,
    chunk_size: int = 10_000,
    log_every: int = 100_000,
    price_convergence_count: int = 1000,
    stop_immediately: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Train two markets in interleaved steps with shared agents.
    
    Both markets share the same 2 agents (Firm A and Firm B). Each market uses
    its own discount factor during Q-updates. Convergence requires BOTH markets
    to independently stabilize their prices for price_convergence_count steps.
    
    Args:
        env1: Environment for Market 1
        env2: Environment for Market 2
        agent_a: Shared Q-learning agent for Firm A (used in both markets)
        agent_b: Shared Q-learning agent for Firm B (used in both markets)
        discount_factor_m1: Discount factor for Market 1 updates
        discount_factor_m2: Discount factor for Market 2 updates
        max_steps: Maximum training steps
        chunk_size: Steps per chunk for statistics
        log_every: Logging frequency
        price_convergence_count: Consecutive steps each market must be stable
        stop_immediately: Stop immediately upon convergence
        verbose: Print progress
    
    Returns:
        Dict with training results including unified convergence price
    """
    # Reset both environments
    state_1a, state_1b = env1.reset()
    state_2a, state_2b = env2.reset()
    
    # Start episodes for shared agents
    agent_a.start_episode()
    agent_b.start_episode()

    # Tracking variables
    joint_rewards_m1: List[float] = []
    joint_rewards_m2: List[float] = []
    unified_prices: List[float] = []
    episode_stats: List[Dict[str, float]] = []
    convergence_metrics: List[Dict[str, float]] = []
    
    chunk_reward_m1 = 0.0
    chunk_reward_m2 = 0.0
    
    # Convergence tracking
    converged = False
    convergence_step = 0
    convergence_reason = ""
    
    # Independent market convergence tracking
    m1_stable_count = 0
    m2_stable_count = 0
    last_prices_m1: Optional[Tuple[float, float]] = None
    last_prices_m2: Optional[Tuple[float, float]] = None
    stable_unified_price = None
    stable_prices_m1 = None
    stable_prices_m2 = None
    
    for step in range(max_steps):
        # Market 1: Select actions (using shared agents)
        action_1a = agent_a.select_action(state_1a)
        action_1b = agent_b.select_action(state_1b)
        
        # Market 2: Select actions (using same shared agents)
        action_2a = agent_a.select_action(state_2a)
        action_2b = agent_b.select_action(state_2b)

        # Market 1: Environment step
        (next_state_1a, next_state_1b), (reward_1a, reward_1b), done_1, _ = env1.step(action_1a, action_1b)
        
        # Market 2: Environment step
        (next_state_2a, next_state_2b), (reward_2a, reward_2b), done_2, _ = env2.step(action_2a, action_2b)

        # Calculate combined rewards across both markets
        combined_reward_a = reward_1a + reward_2a
        combined_reward_b = reward_1b + reward_2b

        # Update Q-values: Market 1 state-action with combined reward, using discount_factor_m1
        agent_a.update(state_1a, action_1a, combined_reward_a, next_state_1a, done_1, discount_factor_m1)
        agent_b.update(state_1b, action_1b, combined_reward_b, next_state_1b, done_1, discount_factor_m1)
        
        # Update Q-values: Market 2 state-action with combined reward, using discount_factor_m2
        agent_a.update(state_2a, action_2a, combined_reward_a, next_state_2a, done_2, discount_factor_m2)
        agent_b.update(state_2b, action_2b, combined_reward_b, next_state_2b, done_2, discount_factor_m2)

        # Track current prices for each market
        current_prices_m1 = (env1.last_price_a, env1.last_price_b)
        current_prices_m2 = (env2.last_price_a, env2.last_price_b)
        
        # Calculate unified price: ((M1_A + M1_B)/2 + (M2_A + M2_B)/2) / 2
        market1_avg = (env1.last_price_a + env1.last_price_b) / 2
        market2_avg = (env2.last_price_a + env2.last_price_b) / 2
        unified_price = (market1_avg + market2_avg) / 2
        
        # Track Market 1 stability independently
        if current_prices_m1 == last_prices_m1:
            m1_stable_count += 1
        else:
            m1_stable_count = 1
            last_prices_m1 = current_prices_m1
        
        # Track Market 2 stability independently
        if current_prices_m2 == last_prices_m2:
            m2_stable_count += 1
        else:
            m2_stable_count = 1
            last_prices_m2 = current_prices_m2
        
        # Converge when BOTH markets are independently stable
        if m1_stable_count >= price_convergence_count and m2_stable_count >= price_convergence_count and not converged:
            converged = True
            convergence_step = step + 1
            stable_unified_price = unified_price
            stable_prices_m1 = current_prices_m1
            stable_prices_m2 = current_prices_m2
            convergence_reason = "both-markets-stable"
            if verbose:
                print(f"\n{'='*70}")
                print(f"BOTH MARKETS CONVERGED at step {convergence_step:,}")
                print(f"Market 1 stable for {m1_stable_count:,} steps: A={stable_prices_m1[0]:.2f}, B={stable_prices_m1[1]:.2f}")
                print(f"Market 2 stable for {m2_stable_count:,} steps: A={stable_prices_m2[0]:.2f}, B={stable_prices_m2[1]:.2f}")
                print(f"Unified price: {stable_unified_price:.4f}")
                print(f"{'='*70}\n")
            
            if stop_immediately:
                break

        # Update states
        state_1a, state_1b = next_state_1a, next_state_1b
        state_2a, state_2b = next_state_2a, next_state_2b
        chunk_reward_m1 += reward_1a + reward_1b
        chunk_reward_m2 += reward_2a + reward_2b

        # Record chunks
        if (step + 1) % chunk_size == 0:
            joint_rewards_m1.append(chunk_reward_m1)
            joint_rewards_m2.append(chunk_reward_m2)
            unified_prices.append(unified_price)
            
            episode_stats.append({
                "episode": float(len(joint_rewards_m1)),
                "price_1a": float(env1.last_price_a),
                "price_1b": float(env1.last_price_b),
                "price_2a": float(env2.last_price_a),
                "price_2b": float(env2.last_price_b),
                "unified_price": float(unified_price),
                "joint_profit_m1": float(chunk_reward_m1),
                "joint_profit_m2": float(chunk_reward_m2),
            })
            
            convergence_metrics.append({
                "episode": float(len(joint_rewards_m1)),
                "m1_stable_count": float(m1_stable_count),
                "m2_stable_count": float(m2_stable_count),
                "converged": float(converged),
                "unified_price": float(unified_price),
                "steps": float(step + 1),
            })
            
            chunk_reward_m1 = 0.0
            chunk_reward_m2 = 0.0

        # Logging
        if verbose and (step + 1) % log_every == 0:
            print(f"Step {step + 1:8d} | Unified: {unified_price:.4f} | M1 stable: {m1_stable_count:6d} | M2 stable: {m2_stable_count:6d}")

        # Stop if converged and not stopping immediately
        if converged and not stop_immediately:
            if step >= convergence_step + 100_000:
                break

    # Handle partial chunk
    if chunk_reward_m1 > 0 or chunk_reward_m2 > 0:
        joint_rewards_m1.append(chunk_reward_m1)
        joint_rewards_m2.append(chunk_reward_m2)

    return {
        "joint_rewards_m1": joint_rewards_m1,
        "joint_rewards_m2": joint_rewards_m2,
        "unified_prices": unified_prices,
        "episode_stats": episode_stats,
        "convergence_metrics": convergence_metrics,
        "converged": converged,
        "convergence_step": convergence_step,
        "convergence_reason": convergence_reason,
        "converged_price": stable_unified_price,
        "stable_prices_m1": stable_prices_m1,
        "stable_prices_m2": stable_prices_m2,
        "m1_stable_count": m1_stable_count,
        "m2_stable_count": m2_stable_count,
        "total_steps": step + 1,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "env1": env1,
        "env2": env2,
    }


def run_undercut_experiment(
    environment: BertrandMultiAgentEnvironment,
    agent_a: MATLABQLearningAgent,
    agent_b: MATLABQLearningAgent,
    converged_price_a: float,
    converged_price_b: float,
    undercut_amount: float = 0.1,
    steps_after_undercut: int = 15,
) -> Dict:
    """
    Run undercut experiment where Firm B undercuts Firm A.

    After convergence, Firm B deviates by setting a price below Firm A's converged price.
    Then both firms use greedy policies for the next steps.

    Args:
        environment: The Bertrand competition environment
        agent_a: Trained Q-learning agent for Firm A
        agent_b: Trained Q-learning agent for Firm B
        converged_price_a: Firm A's price at convergence
        converged_price_b: Firm B's price at convergence
        undercut_amount: How much Firm B undercuts (default 0.1 = one price step)
        steps_after_undercut: Number of steps to track after undercut (default 15)

    Returns:
        Dict with:
            - undercut_price_b: The price Firm B used to undercut (NaN if skipped)
            - trajectory: List of (price_a, price_b) tuples for each step
            - undercut_performed: Whether the undercut was actually executed
            - skip_reason: Why the undercut was skipped (empty string if performed)
    """
    valid_prices = environment.actions_b
    if not valid_prices:
        return {
            "undercut_price_b": np.nan,
            "trajectory": [],
            "undercut_performed": False,
            "skip_reason": "no_valid_prices",
        }

    min_valid_b = min(valid_prices)

    # Guard condition 1: Only undercut if B is NOT strictly below A.
    # If B is already cheaper than A, B is already "winning" — no deviation needed.
    if converged_price_b < converged_price_a:
        return {
            "undercut_price_b": np.nan,
            "trajectory": [],
            "undercut_performed": False,
            "skip_reason": "b_already_below_a",
        }

    # Guard condition 2: Only undercut if B is at least one price level above
    # the minimum valid price.  If B is already at the cost floor there is no
    # valid lower price to deviate to.
    if converged_price_b <= min_valid_b:
        return {
            "undercut_price_b": np.nan,
            "trajectory": [],
            "undercut_performed": False,
            "skip_reason": "b_at_min_price",
        }

    # --- Undercut is feasible: proceed ---

    # Reset environment to converged state
    environment.reset()
    environment.last_price_a = converged_price_a
    environment.last_price_b = converged_price_b

    # Calculate undercut price: Firm B undercuts Firm A's converged price
    target_undercut_price = converged_price_a - undercut_amount

    # Find closest valid price to target
    undercut_price_b = min(valid_prices, key=lambda x: abs(x - target_undercut_price))

    # Ensure undercut price is at least at cost (minimum valid price)
    undercut_price_b = max(undercut_price_b, min_valid_b)

    trajectory: List[Tuple[float, float]] = []

    # Set up initial state based on converged prices
    state_a = (converged_price_a, converged_price_b)
    state_b = (converged_price_b, converged_price_a)

    # Step 1: Firm B undercuts, Firm A uses greedy policy
    action_a = agent_a.greedy_action(state_a)
    action_b = undercut_price_b  # Forced undercut

    (next_state_a, next_state_b), (reward_a, reward_b), done, _ = environment.step(action_a, action_b)
    trajectory.append((environment.last_price_a, environment.last_price_b))

    state_a, state_b = next_state_a, next_state_b

    # Steps 2 to steps_after_undercut: Both firms use greedy policies
    for step in range(2, steps_after_undercut + 1):
        action_a = agent_a.greedy_action(state_a)
        action_b = agent_b.greedy_action(state_b)

        (next_state_a, next_state_b), (reward_a, reward_b), done, _ = environment.step(action_a, action_b)
        trajectory.append((environment.last_price_a, environment.last_price_b))

        state_a, state_b = next_state_a, next_state_b

    return {
        "undercut_price_b": undercut_price_b,
        "trajectory": trajectory,
        "undercut_performed": True,
        "skip_reason": "",
    }
