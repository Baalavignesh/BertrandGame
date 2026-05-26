"""
Core Q-learning module for two-agent, dual-market Bertrand competition.

State:  (pA_m1, pA_m2, pB_m1, pB_m2) — last-period prices in both markets.
Action: (price_m1, price_m2) — each firm picks prices for both markets simultaneously.
Reward: Combined profit π_m1 + π_m2; next state = actions taken (s' = a).
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

State = Tuple[float, float, float, float]    # (pA_m1, pA_m2, pB_m1, pB_m2)
ActionPair = Tuple[float, float]             # (price_m1, price_m2)


@dataclass(frozen=True)
class MarketConfig:
    """Market configuration with parameters from config.py."""
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
    """Two-firm Bertrand competition with simultaneous price setting.

    Used for price grid generation and per-market profit computation.
    """

    def __init__(self, config: MarketConfig):
        self.config = config
        self.price_levels = list(
            np.arange(config.price_min, config.price_max + config.price_step / 2, config.price_step)
        )
        self.price_levels = [round(p, 2) for p in self.price_levels]

        if not self.price_levels:
            raise ValueError("Price grid is empty; adjust configuration bounds.")

        self.actions_a = [p for p in self.price_levels if p >= config.firm_a_cost]
        self.actions_b = [p for p in self.price_levels if p >= config.firm_b_cost]
        if not self.actions_a or not self.actions_b:
            raise ValueError("Action spaces are empty; check costs vs price bounds.")

        self.start_price_a = min(self.actions_a)
        self.start_price_b = min(self.actions_b)

        self.last_price_a = self.start_price_a
        self.last_price_b = self.start_price_b

    def get_available_actions(self, firm: str) -> List[float]:
        if firm.upper() == "A":
            return self.actions_a
        if firm.upper() == "B":
            return self.actions_b
        raise ValueError("firm must be 'A' or 'B'.")

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
    """Q-learner with 4-tuple state (both markets) and 2-tuple action (price pair).

    Sees full market state: (pA_m1, pA_m2, pB_m1, pB_m2)
    Chooses prices for both markets simultaneously: (price_m1, price_m2)
    Receives combined reward: π_m1 + π_m2
    """

    def __init__(
        self,
        name: str,
        valid_prices: List[float],
        learning_rate: float = config.LEARNING_RATE,
        discount_factor: float = 0.95,
        epsilon_start: float = config.EPSILON_START,
        epsilon_min: float = config.EPSILON_MIN,
        step_beta: float = config.STEP_BETA,
        optimistic_init: bool = True,
        env1: Optional[BertrandMultiAgentEnvironment] = None,
        env2: Optional[BertrandMultiAgentEnvironment] = None,
    ):
        self.name = name
        self.valid_prices = list(valid_prices)
        if not self.valid_prices:
            raise ValueError("Valid price list cannot be empty.")

        # Action pairs: all (price_m1, price_m2) combinations
        self.action_pairs: List[ActionPair] = [
            (p1, p2) for p1 in self.valid_prices for p2 in self.valid_prices
        ]

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.step_beta = step_beta

        self.q_table: Dict[State, Dict[ActionPair, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.epsilon = epsilon_start
        self.global_step = 0

        if optimistic_init and env1 is not None and env2 is not None:
            self._init_optimistic_q(env1, env2)

    def _init_optimistic_q(
        self,
        env1: BertrandMultiAgentEnvironment,
        env2: BertrandMultiAgentEnvironment,
    ) -> None:
        """MATLAB-style optimistic init: Q(s, any_a) = avg_profit(s) / (1 - δ).

        For each state, compute combined profits for both firms, average over
        firms, and divide by (1 - δ). All action pairs in a given state get the
        same initial value.
        """
        cost_a = env1.config.firm_a_cost
        cost_b = env1.config.firm_b_cost
        scale = env1.config.reward_scale
        vp = self.valid_prices

        for pA_m1 in vp:
            for pA_m2 in vp:
                for pB_m1 in vp:
                    for pB_m2 in vp:
                        state: State = (pA_m1, pA_m2, pB_m1, pB_m2)

                        # Per-market profits at this state
                        pi1_a = env1._profit(pA_m1, pB_m1, cost_a) * scale
                        pi1_b = env1._profit(pB_m1, pA_m1, cost_b) * scale
                        pi2_a = env2._profit(pA_m2, pB_m2, cost_a) * scale
                        pi2_b = env2._profit(pB_m2, pA_m2, cost_b) * scale

                        # Average combined profit over both firms
                        combined_a = pi1_a + pi2_a
                        combined_b = pi1_b + pi2_b
                        avg_pi = (combined_a + combined_b) / 2.0

                        if self.discount_factor < 1.0:
                            init_val = avg_pi / (1.0 - self.discount_factor)
                        else:
                            init_val = avg_pi * 1000.0

                        for action in self.action_pairs:
                            self.q_table[state][action] = init_val

    def start_episode(self) -> None:
        self.global_step = 0
        self.epsilon = self.epsilon_start

    def _epsilon_for_step(self, step: int) -> float:
        """MATLAB: pr_explore = exp(-t * beta)"""
        return max(self.epsilon_min, self.epsilon_start * math.exp(-self.step_beta * step))

    def select_action(self, state: State) -> ActionPair:
        """Epsilon-greedy action selection. Returns (price_m1, price_m2)."""
        self.epsilon = self._epsilon_for_step(self.global_step)

        if random.random() < self.epsilon:
            action = random.choice(self.action_pairs)
        else:
            action = self.greedy_action(state)

        self.global_step += 1
        return action

    def update(
        self,
        state: State,
        action: ActionPair,
        reward: float,
        next_state: State,
        done: bool,
        discount_factor: float,
    ) -> bool:
        """Q-learning update with combined reward. Returns True if greedy policy changed."""
        old_argmax = self.greedy_action(state)

        old_value = self.q_table[state][action]
        if done:
            new_value = reward
        else:
            max_q_next = max(self.q_table[next_state][a] for a in self.action_pairs)
            new_value = reward + discount_factor * max_q_next

        self.q_table[state][action] = (
            (1 - self.learning_rate) * old_value + self.learning_rate * new_value
        )

        new_argmax = self.greedy_action(state)
        return old_argmax != new_argmax

    def greedy_action(self, state: State) -> ActionPair:
        """Return action pair with highest Q-value.

        Ties are broken deterministically (first match), matching MATLAB's
        max() which returns the first index of the maximum value.
        """
        best_action = self.action_pairs[0]
        best_q = self.q_table[state][best_action]
        for a in self.action_pairs[1:]:
            q = self.q_table[state][a]
            if q > best_q:
                best_q = q
                best_action = a
        return best_action

    def get_q_value(self, state: State, action: ActionPair) -> float:
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
    """Train two agents across two markets with combined rewards.

    At each step:
    1. Both agents see the same 4-tuple state: (pA_m1, pA_m2, pB_m1, pB_m2)
    2. Each agent picks an action pair: (price_m1, price_m2)
    3. Per-market profits are computed and summed into combined reward per firm
    4. Q-tables updated with average discount factor
    5. Next state = actions taken (s' = a)
    6. Convergence tracked independently per market
    """
    cost_a = env1.config.firm_a_cost
    cost_b = env1.config.firm_b_cost
    scale = env1.config.reward_scale
    avg_discount = (discount_factor_m1 + discount_factor_m2) / 2.0

    min_price_a = min(env1.actions_a)
    min_price_b = min(env1.actions_b)
    state: State = (min_price_a, min_price_a, min_price_b, min_price_b)

    agent_a.start_episode()
    agent_b.start_episode()

    # Tracking
    joint_rewards_m1: List[float] = []
    joint_rewards_m2: List[float] = []
    unified_prices: List[float] = []
    episode_stats: List[Dict[str, float]] = []
    convergence_metrics: List[Dict[str, float]] = []

    chunk_reward_m1 = 0.0
    chunk_reward_m2 = 0.0

    # Convergence
    converged = False
    convergence_step = 0
    convergence_reason = ""

    m1_stable_count = 0
    m2_stable_count = 0
    last_prices_m1: Optional[Tuple[float, float]] = None
    last_prices_m2: Optional[Tuple[float, float]] = None
    stable_unified_price = None
    stable_prices_m1 = None
    stable_prices_m2 = None
    stable_state: Optional[State] = None

    for step in range(max_steps):
        action_a: ActionPair = agent_a.select_action(state)
        action_b: ActionPair = agent_b.select_action(state)

        pi1_a = env1._profit(action_a[0], action_b[0], cost_a) * scale
        pi1_b = env1._profit(action_b[0], action_a[0], cost_b) * scale
        pi2_a = env2._profit(action_a[1], action_b[1], cost_a) * scale
        pi2_b = env2._profit(action_b[1], action_a[1], cost_b) * scale

        reward_a = pi1_a + pi2_a
        reward_b = pi1_b + pi2_b

        # Next state = actions taken (s' = a)
        next_state: State = (action_a[0], action_a[1], action_b[0], action_b[1])

        agent_a.update(state, action_a, reward_a, next_state, False, avg_discount)
        agent_b.update(state, action_b, reward_b, next_state, False, avg_discount)

        current_prices_m1 = (action_a[0], action_b[0])
        current_prices_m2 = (action_a[1], action_b[1])

        # Update environment tracking for external inspection
        env1.last_price_a = action_a[0]
        env1.last_price_b = action_b[0]
        env2.last_price_a = action_a[1]
        env2.last_price_b = action_b[1]

        market1_avg = (action_a[0] + action_b[0]) / 2.0
        market2_avg = (action_a[1] + action_b[1]) / 2.0
        unified_price = (market1_avg + market2_avg) / 2.0

        # Market stability tracking
        if current_prices_m1 == last_prices_m1:
            m1_stable_count += 1
        else:
            m1_stable_count = 1
            last_prices_m1 = current_prices_m1

        if current_prices_m2 == last_prices_m2:
            m2_stable_count += 1
        else:
            m2_stable_count = 1
            last_prices_m2 = current_prices_m2

        if (
            m1_stable_count >= price_convergence_count
            and m2_stable_count >= price_convergence_count
            and not converged
        ):
            converged = True
            convergence_step = step + 1
            stable_unified_price = unified_price
            stable_prices_m1 = current_prices_m1
            stable_prices_m2 = current_prices_m2
            stable_state = next_state
            convergence_reason = "both-markets-stable"
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"BOTH MARKETS CONVERGED at step {convergence_step:,}")
                print(f"Market 1 stable for {m1_stable_count:,} steps: A={stable_prices_m1[0]:.2f}, B={stable_prices_m1[1]:.2f}")
                print(f"Market 2 stable for {m2_stable_count:,} steps: A={stable_prices_m2[0]:.2f}, B={stable_prices_m2[1]:.2f}")
                print(f"Unified price: {stable_unified_price:.4f}")
                print(f"{'=' * 70}\n")

            if stop_immediately:
                break

        state = next_state
        chunk_reward_m1 += pi1_a + pi1_b
        chunk_reward_m2 += pi2_a + pi2_b

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

        if verbose and (step + 1) % log_every == 0:
            print(
                f"Step {step + 1:8d} | Unified: {unified_price:.4f} "
                f"| M1 stable: {m1_stable_count:6d} | M2 stable: {m2_stable_count:6d}"
            )

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
        "stable_state": stable_state,
        "m1_stable_count": m1_stable_count,
        "m2_stable_count": m2_stable_count,
        "total_steps": step + 1,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "env1": env1,
        "env2": env2,
    }


def run_undercut_experiment(
    agent_a: MATLABQLearningAgent,
    agent_b: MATLABQLearningAgent,
    env1: BertrandMultiAgentEnvironment,
    env2: BertrandMultiAgentEnvironment,
    converged_state: State,
    market: int,
    steps_after_undercut: int = 15,
) -> Dict:
    """Run undercut experiment where Firm B undercuts in one market.

    Firm B deviates by setting a price one level below Firm A's converged price
    in the specified market, while keeping the other market price unchanged.
    After the forced undercut, both firms use greedy policies (no Q-updates).

    Design notes (deviations from MATLAB reference):
    - **Frozen policy**: Post-deviation steps use greedy_action with no Q-learning
      updates, isolating the punishment response from learning effects. The MATLAB
      reference continues Q-updates during deviation (qlearningmm.update_q), which
      conflates adaptation with the equilibrium punishment path.
    - **Per-market undercut**: B deviates in one market at a time. The MATLAB
      reference deviates in both markets simultaneously. Per-market undercut is
      more informative for studying cross-market punishment spillovers.

    Args:
        agent_a: Trained agent for Firm A
        agent_b: Trained agent for Firm B
        env1: Market 1 environment (for profit computation)
        env2: Market 2 environment (for profit computation)
        converged_state: The converged 4-tuple state (pA_m1, pA_m2, pB_m1, pB_m2)
        market: Which market to undercut in (1 or 2)
        steps_after_undercut: Number of steps to track after undercut

    Returns:
        Dict with undercut_price_b, trajectory (per-market prices), and status.
    """
    pA_m1, pA_m2, pB_m1, pB_m2 = converged_state
    valid_prices = agent_b.valid_prices

    if not valid_prices:
        return {
            "undercut_price_b": np.nan,
            "trajectory": [],
            "undercut_performed": False,
            "skip_reason": "no_valid_prices",
        }

    if market == 1:
        conv_price_a = pA_m1
        conv_price_b = pB_m1
    else:
        conv_price_a = pA_m2
        conv_price_b = pB_m2

    min_valid_b = min(valid_prices)

    # Guard: B already below A
    if conv_price_b < conv_price_a:
        return {
            "undercut_price_b": np.nan,
            "trajectory": [],
            "undercut_performed": False,
            "skip_reason": "b_already_below_a",
        }

    # Guard: B at minimum price
    if conv_price_b <= min_valid_b:
        return {
            "undercut_price_b": np.nan,
            "trajectory": [],
            "undercut_performed": False,
            "skip_reason": "b_at_min_price",
        }

    # Find undercut price: highest valid price strictly below A's converged price
    prices_below_a = [p for p in valid_prices if p < conv_price_a - 1e-9]
    if not prices_below_a:
        return {
            "undercut_price_b": np.nan,
            "trajectory": [],
            "undercut_performed": False,
            "skip_reason": "no_valid_price_below_a",
        }
    undercut_price = max(prices_below_a)

    state = converged_state
    trajectory: List[Tuple[float, float]] = []

    # Step 1: A greedy, B undercuts in specified market
    action_a = agent_a.greedy_action(state)
    greedy_b = agent_b.greedy_action(state)

    if market == 1:
        action_b = (undercut_price, greedy_b[1])
    else:
        action_b = (greedy_b[0], undercut_price)

    next_state: State = (action_a[0], action_a[1], action_b[0], action_b[1])

    if market == 1:
        trajectory.append((action_a[0], action_b[0]))
    else:
        trajectory.append((action_a[1], action_b[1]))

    state = next_state

    # Steps 2+: both firms use greedy policies
    for _ in range(2, steps_after_undercut + 1):
        action_a = agent_a.greedy_action(state)
        action_b = agent_b.greedy_action(state)

        next_state = (action_a[0], action_a[1], action_b[0], action_b[1])

        if market == 1:
            trajectory.append((action_a[0], action_b[0]))
        else:
            trajectory.append((action_a[1], action_b[1]))

        state = next_state

    return {
        "undercut_price_b": undercut_price,
        "trajectory": trajectory,
        "undercut_performed": True,
        "skip_reason": "",
    }
