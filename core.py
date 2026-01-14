"""
Core Q-Learning Module for Multi-Agent Bertrand Competition

This module contains all the shared classes and functions used by both
the interactive notebook and the batch simulator.

Classes:
    - MarketConfig: Configuration for the market environment
    - BertrandMultiAgentEnvironment: Two-firm Bertrand competition environment
    - MATLABQLearningAgent: Q-learning agent replicating MATLAB's logic

Functions:
    - train_matlab_multiagent: Train multi-agent system
    - run_greedy_episode: Test learned policies
    - plot_matlab_results: Visualize training results
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
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
    ) -> bool:
        """MATLAB-style Q-update. Returns True if greedy policy changed."""
        # Get argmax BEFORE update
        old_argmax = self.greedy_action(state)
        
        # MATLAB Q-update
        old_value = self.q_table[state][action]
        if done:
            new_value = reward
        else:
            max_q_next = max([self.q_table[next_state][a] for a in self.actions])
            new_value = reward + self.discount_factor * max_q_next
        
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


def train_matlab_multiagent(
    environment: BertrandMultiAgentEnvironment,
    agent_a: MATLABQLearningAgent,
    agent_b: MATLABQLearningAgent,
    max_steps: int = 10_000_000,
    chunk_size: int = 10_000,
    log_every: int = 100_000,
    price_convergence_count: int = 1000,
    stop_immediately: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Train multi-agent system with price-based convergence.
    
    Convergence: Stops when prices remain constant for `price_convergence_count` consecutive steps.
    
    Args:
        environment: The Bertrand competition environment
        agent_a: Q-learning agent for Firm A
        agent_b: Q-learning agent for Firm B
        max_steps: Maximum training steps
        chunk_size: Steps per chunk for statistics
        log_every: Logging frequency
        price_convergence_count: Consecutive steps with same prices to trigger convergence (default: 1000)
        stop_immediately: Stop immediately upon convergence
        verbose: Print progress
    
    Returns:
        Dict with training results including convergence info and statistics
    """
    state_a, state_b = environment.reset()
    agent_a.start_episode()
    agent_b.start_episode()

    # Tracking variables
    rewards_a: List[float] = []
    rewards_b: List[float] = []
    joint_rewards: List[float] = []
    episode_stats: List[Dict[str, float]] = []
    convergence_metrics: List[Dict[str, float]] = []
    
    chunk_reward_a = 0.0
    chunk_reward_b = 0.0
    
    # Convergence tracking
    converged = False
    convergence_step = 0
    convergence_reason = ""
    
    # Price-based convergence tracking
    price_consecutive_count = 0
    last_price_pair = None
    stable_price_pair = None
    
    for step in range(max_steps):
        # Select actions
        action_a = agent_a.select_action(state_a)
        action_b = agent_b.select_action(state_b)

        # Environment step
        (next_state_a, next_state_b), (reward_a, reward_b), done, _ = environment.step(action_a, action_b)

        # Update Q-values
        agent_a.update(state_a, action_a, reward_a, next_state_a, done)
        agent_b.update(state_b, action_b, reward_b, next_state_b, done)

        # Price convergence tracking
        current_price_pair = (environment.last_price_a, environment.last_price_b)
        
        if current_price_pair == last_price_pair:
            price_consecutive_count += 1
        else:
            price_consecutive_count = 1
            last_price_pair = current_price_pair
        
        if price_consecutive_count >= price_convergence_count and not converged:
            converged = True
            convergence_step = step + 1
            stable_price_pair = current_price_pair
            convergence_reason = "price-based"
            if verbose:
                print(f"\n{'='*70}")
                print(f"PRICE CONVERGENCE at step {convergence_step:,}")
                print(f"Stable price: A={stable_price_pair[0]:.2f}, B={stable_price_pair[1]:.2f}")
                print(f"{'='*70}\n")
            
            if stop_immediately:
                break

        # Update state
        state_a, state_b = next_state_a, next_state_b
        chunk_reward_a += reward_a
        chunk_reward_b += reward_b

        # Record chunks
        if (step + 1) % chunk_size == 0:
            rewards_a.append(chunk_reward_a)
            rewards_b.append(chunk_reward_b)
            joint_rewards.append(chunk_reward_a + chunk_reward_b)
            
            last_entry = environment.history[-1] if environment.history else {
                "price_a": float(environment.last_price_a),
                "price_b": float(environment.last_price_b),
                "profit_a": 0.0,
                "profit_b": 0.0,
            }
            episode_stats.append({
                "episode": float(len(rewards_a)),
                "price_a": float(last_entry["price_a"]),
                "price_b": float(last_entry["price_b"]),
                "profit_a": float(last_entry["profit_a"]),
                "profit_b": float(last_entry["profit_b"]),
                "joint_profit": float(last_entry["profit_a"] + last_entry["profit_b"]),
                "margin_a": float(last_entry["price_a"]) - environment.config.firm_a_cost,
                "margin_b": float(last_entry["price_b"]) - environment.config.firm_b_cost,
                "price_gap": float(last_entry["price_a"]) - float(last_entry["price_b"]),
            })
            
            window = min(10, len(joint_rewards))
            reward_window = joint_rewards[-window:]
            convergence_metrics.append({
                "episode": float(len(rewards_a)),
                "price_counter": float(price_consecutive_count) if price_convergence_count else 0.0,
                "converged": float(converged),
                "joint_reward_mean": float(np.mean(reward_window)) if reward_window else 0.0,
                "joint_reward_std": float(np.std(reward_window)) if reward_window else 0.0,
                "states_a": float(len(agent_a.q_table)),
                "states_b": float(len(agent_b.q_table)),
                "steps": float(step + 1),
            })
            
            chunk_reward_a = 0.0
            chunk_reward_b = 0.0

        # Logging
        if verbose and (step + 1) % log_every == 0:
            recent = joint_rewards[-min(5, len(joint_rewards)):] if joint_rewards else [0.0]
            avg_joint = float(np.mean(recent))
            print(f"Step {step + 1:8d} | Joint: {avg_joint:10.2f} | Price count: {price_consecutive_count:6d}")

        # Stop if converged and not stopping immediately
        if converged and not stop_immediately:
            if step >= convergence_step + 100_000:
                break

    # Handle partial chunk
    if chunk_reward_a > 0 or chunk_reward_b > 0:
        rewards_a.append(chunk_reward_a)
        rewards_b.append(chunk_reward_b)
        joint_rewards.append(chunk_reward_a + chunk_reward_b)

    return {
        "rewards_a": rewards_a,
        "rewards_b": rewards_b,
        "joint_rewards": joint_rewards,
        "episode_stats": episode_stats,
        "convergence_metrics": convergence_metrics,
        "converged": converged,
        "convergence_step": convergence_step,
        "convergence_reason": convergence_reason,
        "stable_price_pair": stable_price_pair,
        "final_price_count": price_consecutive_count,
        "total_steps": step + 1,
        "agent_a": agent_a,
        "agent_b": agent_b,
    }

def run_greedy_episode(
    environment: BertrandMultiAgentEnvironment,
    agent_a: MATLABQLearningAgent,
    agent_b: MATLABQLearningAgent,
    max_steps: int = 50,
) -> List[Dict[str, float]]:
    """Run episode with greedy policies to test learned behavior."""
    state_a, state_b = environment.reset()
    trajectory: List[Dict[str, float]] = []

    for _ in range(max_steps):
        action_a = agent_a.greedy_action(state_a)
        action_b = agent_b.greedy_action(state_b)
        (next_state_a, next_state_b), (reward_a, reward_b), done, info = environment.step(action_a, action_b)
        trajectory.append({
            "price_a": float(environment.last_price_a),
            "price_b": float(environment.last_price_b),
            "reward_a": reward_a,
            "reward_b": reward_b,
            "joint_reward": info["joint_reward"],
        })
        state_a, state_b = next_state_a, next_state_b
        if done:
            break

    return trajectory


def plot_matlab_results(results: Dict, greedy_trajectory: List[Dict], nash_price: float = 0.5) -> None:
    """Plot training results with 4 key metrics."""
    fig = plt.figure(figsize=(20, 5))

    # Subplot 1: Joint rewards over time
    ax1 = plt.subplot(1, 4, 1)
    chunks = range(1, len(results['joint_rewards']) + 1)
    ax1.plot(chunks, results['joint_rewards'], label="Joint reward", alpha=0.7)
    window = min(20, len(results['joint_rewards']))
    if len(results['joint_rewards']) >= window:
        mov_avg = np.convolve(results['joint_rewards'], np.ones(window) / window, mode="valid")
        ax1.plot(range(window, len(results['joint_rewards']) + 1), mov_avg,
                label=f"MA ({window})", linewidth=2)
    if results['converged']:
        conv_chunk = results['convergence_step'] // 10000
        ax1.axvline(conv_chunk, color='red', linestyle='--', label=f"Converged at chunk {conv_chunk}")
    ax1.set_xlabel("Chunk index")
    ax1.set_ylabel("Total reward per chunk")
    ax1.set_title("Joint Rewards Over Time")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Subplot 2: Price margins
    ax2 = plt.subplot(1, 4, 2)
    stats = results['episode_stats']
    if stats:
        episodes = [s['episode'] for s in stats]
        margin_a = [s['margin_a'] for s in stats]
        margin_b = [s['margin_b'] for s in stats]
        gap = [s['price_gap'] for s in stats]
        ax2.plot(episodes, margin_a, label="Firm A margin", linewidth=2)
        ax2.plot(episodes, margin_b, label="Firm B margin", linewidth=2)
        ax2.plot(episodes, gap, label="Price gap", linestyle="--")
        ax2.axhline(0, color='black', linewidth=1, alpha=0.6)
        ax2.set_xlabel("Chunk index")
        ax2.set_ylabel("Price difference")
        ax2.set_title("Margins and Price Gap")
        ax2.grid(alpha=0.3)
        ax2.legend()

    # Subplot 3: Greedy trajectory prices
    ax3 = plt.subplot(1, 4, 3)
    if greedy_trajectory:
        steps = range(1, len(greedy_trajectory) + 1)
        prices_a = [t['price_a'] for t in greedy_trajectory]
        prices_b = [t['price_b'] for t in greedy_trajectory]
        ax3.plot(steps, prices_a, 'o-', label="Firm A", markersize=4)
        ax3.plot(steps, prices_b, 'x-', label="Firm B", markersize=4)
        ax3.axhline(nash_price, color='red', linestyle='--', alpha=0.5, label=f"Nash ({nash_price})")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Price")
        ax3.set_title("Greedy Policy Prices (Post-Training)")
        ax3.grid(alpha=0.3)
        ax3.legend()

    # Subplot 4: Epsilon decay
    ax4 = plt.subplot(1, 4, 4)
    metrics = results['convergence_metrics']
    if metrics:
        steps = [m['steps'] for m in metrics]
        # Approximate epsilon from step count
        epsilons = [max(0.001, 1.0 * np.exp(-4e-6 * s)) for s in steps]
        ax4.semilogy(steps, epsilons, linewidth=2)
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Epsilon (log scale)")
        ax4.set_title("Exploration Rate Decay")
        ax4.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()


