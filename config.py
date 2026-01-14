"""
Configuration Constants for Bertrand Q-Learning Simulation

"""

# =============================================================================
# MARKET PARAMETERS
# =============================================================================

# Cost and Price Settings
MARGINAL_COST = 0.1          # Production cost for both firms
PRICE_MIN = 0.0              # Minimum price
PRICE_MAX = 0.65             # Maximum price
NUM_PRICE_LEVELS = 6         # k = 6 price levels

# Calculate price step to get exactly k levels from min to max
# Levels: {0, 0.13, 0.26, 0.39, 0.52, 0.65} for k=6
PRICE_STEP = (PRICE_MAX - PRICE_MIN) / (NUM_PRICE_LEVELS - 1)

# Demand function parameters
DEMAND_INTERCEPT = 100.0
DEMAND_SLOPE = 1.0

# Reward scaling
REWARD_SCALE = 0.01


# =============================================================================
# Q-LEARNING HYPERPARAMETERS
# =============================================================================

# Learning rate (alpha)
LEARNING_RATE = 0.15

# Discount factor multiplier for Market 2 (Market 2 uses: GAMMA_MULTIPLIER × γ)
GAMMA_MULTIPLIER = 0.7

# Epsilon-greedy exploration parameters
EPSILON_START = 1.0
EPSILON_MIN = 0.001
STEP_BETA = 2e-4             # β for epsilon decay: ε = exp(-β × t)


# =============================================================================
# DISCOUNT FACTORS
# =============================================================================

# Fixed set of discount factors to test: {0, 0.05, 0.1, ..., 0.95}
DISCOUNT_FACTORS = [round(i * 0.05, 2) for i in range(20)]


# =============================================================================
# SIMULATION DEFAULTS
# =============================================================================

DEFAULT_MAX_STEPS = 1_000_000
DEFAULT_PRICE_CONVERGENCE_COUNT = 100
DEFAULT_N_SIMULATIONS = 10