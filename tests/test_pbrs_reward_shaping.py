"""Unit tests for Potential-Based Reward Shaping (PBRS) in microgrid RL."""

import gymnasium as gym
import numpy as np
import pytest

from src.rl.shaping import (
    PotentialBasedRewardShapingWrapper,
    REFERENCE_VALUE_USD_PER_KWH_DC,
    DEFAULT_BATTERY_CAPACITY_KWH,
)
from src.rl.pbrs_experiment import load_pbrs_config


class DummyEnv(gym.Env):
    """Mock environment mimicking MicrogridGymEnvironment."""

    def __init__(self, initial_soc=0.5):
        super().__init__()
        self.soc = float(initial_soc)
        self.observation = np.zeros(15, dtype=np.float32)
        self.observation[0] = self.soc
        self.steps = 0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.price = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc = 0.5
        self.observation[0] = self.soc
        self.steps = 0
        self.price = None
        return self.observation.copy(), {"step_metrics": {"values": {"net_operating_cost": 0.0}}}

    def step(self, action):
        self.steps += 1
        # Action is normalized: -1 (charge), 0 (idle), +1 (discharge)
        # AC rating: 150 kW
        if action < 0: # Charge
            p = 0.12 if self.price is None else self.price
            delta_e = 150.0 * 0.95 # +142.5 kWh DC
            cost = 150.0 * p + 150.0 * 0.02 # 21.00 USD at 0.12
        elif action > 0: # Discharge
            p = 0.18 if self.price is None else self.price
            delta_e = -150.0 / 0.95 # -157.8947 kWh DC
            cost = -(150.0 * p - 150.0 * 0.02)
        else: # Idle
            delta_e = 0.0
            cost = 0.0

        self.soc = float(np.clip(self.soc + delta_e / 500.0, 0.1, 0.9))
        self.observation[0] = self.soc
        raw_reward = -cost
        terminated = False
        truncated = bool(self.steps >= 168)
        info = {
            "step_metrics": {"values": {"net_operating_cost": cost}},
            "TimeLimit.truncated": truncated,
        }
        return self.observation.copy(), raw_reward, terminated, truncated, info


def test_01_reference_valuation_constant():
    expected = (0.12 + 0.02) / 0.95
    assert np.isclose(REFERENCE_VALUE_USD_PER_KWH_DC, expected, atol=1e-14)
    assert np.isclose(REFERENCE_VALUE_USD_PER_KWH_DC, 0.14736842105263157, atol=1e-14)


def test_02_wrapper_initialization_and_validation():
    dummy = DummyEnv()
    wrapper = PotentialBasedRewardShapingWrapper(dummy, gamma=0.999)
    assert wrapper.gamma == 0.999
    assert wrapper.capacity_kwh == 500.0
    assert wrapper.v_ref == REFERENCE_VALUE_USD_PER_KWH_DC

    with pytest.raises(ValueError):
        PotentialBasedRewardShapingWrapper(dummy, gamma=-0.1)
    with pytest.raises(ValueError):
        PotentialBasedRewardShapingWrapper(dummy, gamma=1.5)
    with pytest.raises(ValueError):
        PotentialBasedRewardShapingWrapper(dummy, reference_value_usd_per_kwh_dc=0)
    with pytest.raises(ValueError):
        PotentialBasedRewardShapingWrapper(dummy, capacity_kwh=-100)


def test_03_one_step_off_peak_charging_numerical_derivation():
    dummy = DummyEnv(initial_soc=0.5)
    wrapper = PotentialBasedRewardShapingWrapper(dummy, gamma=0.999)
    obs, info = wrapper.reset()
    assert wrapper.phi_current == REFERENCE_VALUE_USD_PER_KWH_DC * 250.0
    assert info["inventory_potential_usd"] == wrapper.phi_current

    # Step: Charge 150 kW AC
    obs, shaped_reward, terminated, truncated, info = wrapper.step(-1.0)
    assert info["unshaped_reward_usd"] == -21.00
    assert np.isclose(info["shaping_reward_usd"], 20.9422, atol=1e-3)
    assert np.isclose(shaped_reward, -0.0578, atol=1e-3)
    assert np.isclose(info["shaped_reward_usd"], shaped_reward, atol=1e-12)


def test_04_one_step_peak_discharging_numerical_derivation():
    dummy = DummyEnv(initial_soc=0.5)
    wrapper = PotentialBasedRewardShapingWrapper(dummy, gamma=0.999)
    obs, info = wrapper.reset()

    # Step: Discharge 150 kW AC
    obs, shaped_reward, terminated, truncated, info = wrapper.step(1.0)
    assert info["unshaped_reward_usd"] == 24.00
    assert np.isclose(info["shaping_reward_usd"], -23.2823, atol=1e-3)
    assert np.isclose(shaped_reward, 0.7177, atol=1e-3)
    assert np.isclose(info["shaped_reward_usd"], shaped_reward, atol=1e-12)


def test_05_one_step_idling():
    dummy = DummyEnv(initial_soc=0.5)
    wrapper = PotentialBasedRewardShapingWrapper(dummy, gamma=0.999)
    obs, info = wrapper.reset()

    # Step: Idle 0 kW AC
    obs, shaped_reward, terminated, truncated, info = wrapper.step(0.0)
    assert info["unshaped_reward_usd"] == 0.0
    # Shaping: (gamma - 1) * Phi(250) = -0.001 * 0.147368 * 250 = -0.036842 USD
    expected_idle_shaping = (0.999 - 1.0) * REFERENCE_VALUE_USD_PER_KWH_DC * 250.0
    assert np.isclose(info["shaping_reward_usd"], expected_idle_shaping, atol=1e-6)
    assert np.isclose(shaped_reward, expected_idle_shaping, atol=1e-6)


def test_06_telescoping_discounted_sum_over_random_trajectory():
    dummy = DummyEnv(initial_soc=0.5)
    gamma = 0.999
    wrapper = PotentialBasedRewardShapingWrapper(dummy, gamma=gamma)
    obs, info = wrapper.reset()
    phi_0 = wrapper.initial_phi

    T = 20
    np.random.seed(123)
    actions = np.random.choice([-1.0, 0.0, 1.0], size=T)

    shaping_sum = 0.0
    for t in range(T):
        obs, shaped_reward, terminated, truncated, info = wrapper.step(actions[t])
        shaping_sum += (gamma ** t) * info["shaping_reward_usd"]

    phi_T = wrapper.phi_current
    expected_sum = -phi_0 + (gamma ** T) * phi_T
    assert np.isclose(shaping_sum, expected_sum, atol=1e-11)


def test_07_telescoping_identity_over_168_step_episode():
    dummy = DummyEnv(initial_soc=0.5)
    gamma = 0.999
    wrapper = PotentialBasedRewardShapingWrapper(dummy, gamma=gamma)
    obs, info = wrapper.reset()
    phi_0 = wrapper.initial_phi

    T = 168
    np.random.seed(456)
    actions = np.random.choice([-1.0, 0.0, 1.0], size=T)

    shaping_sum = 0.0
    for t in range(T):
        obs, shaped_reward, terminated, truncated, info = wrapper.step(actions[t])
        shaping_sum += (gamma ** t) * info["shaping_reward_usd"]
        if t == T - 1:
            assert truncated is True
            assert terminated is False

    phi_168 = wrapper.phi_current
    expected_sum = -phi_0 + (gamma ** 168) * phi_168
    assert np.isclose(shaping_sum, expected_sum, atol=1e-11)


def test_08_flat_rate_roundtrip_cycling_penalty():
    dummy = DummyEnv(initial_soc=0.5)
    gamma = 0.999
    wrapper = PotentialBasedRewardShapingWrapper(dummy, gamma=gamma)
    wrapper.reset()

    # Step 1: Charge at flat 0.12 rate
    dummy.price = 0.12
    obs1, r1, term1, trunc1, info1 = wrapper.step(-1.0)
    # Step 2: Discharge at flat 0.12 rate (no peak arbitrage spread)
    dummy.price = 0.12
    obs2, r2, term2, trunc2, info2 = wrapper.step(1.0)

    # In flat-rate cycling, roundtrip AC efficiency 0.9025 causes net cash loss
    two_step_shaped_return = r1 + gamma * r2
    assert two_step_shaped_return < 0
    assert np.isclose(two_step_shaped_return, -8.35, atol=0.1)


def test_09_sb3_timeout_bootstrapping_cancellation():
    gamma = 0.999
    reward_scale = 0.0025
    v_ref = REFERENCE_VALUE_USD_PER_KWH_DC

    # Terminal state at T=168
    energy_T = 300.0
    energy_T_minus_1 = 200.0
    phi_T = v_ref * energy_T
    phi_T_minus_1 = v_ref * energy_T_minus_1

    # Raw cost at step 167
    c_167 = 25.0
    r_raw_167 = -c_167

    # Shaped step reward
    shaping_167 = gamma * phi_T - phi_T_minus_1
    r_shaped_167 = r_raw_167 + shaping_167
    r_shaped_scaled_167 = reward_scale * r_shaped_167

    # Critic value prediction for state s_T in shaped MDP
    # V'_*(s) = V_*(s) - reward_scale * Phi(s)
    v_unshaped_s_T = -12.5 # unshaped critic value in scaled units
    v_shaped_s_T = v_unshaped_s_T - reward_scale * phi_T

    # SB3 timeout bootstrapping evaluates:
    # bootstrapped_reward = r_shaped_scaled_167 + gamma * v_shaped_s_T
    bootstrapped_buffer_reward = r_shaped_scaled_167 + gamma * v_shaped_s_T

    # Direct analytical cancellation:
    # reward_scale * (r_raw_167 - phi_T_minus_1) + gamma * v_unshaped_s_T
    analytical_cancellation = reward_scale * (r_raw_167 - phi_T_minus_1) + gamma * v_unshaped_s_T

    assert np.isclose(bootstrapped_buffer_reward, analytical_cancellation, atol=1e-14)


def test_10_pbrs_config_validation():
    config = load_pbrs_config()
    assert config["stage"] == "pbrs_diagnostic"
    assert config["gamma"] == 0.999
    assert config["reward_scale"] == 0.0025
    assert config["total_timesteps"] == 16384
    assert config["seed"] == 0
    assert config["action_space_type"] == "discrete_3"
    assert config["reward_shaping"]["enabled"] is True
    assert np.isclose(config["reward_shaping"]["reference_value_usd_per_kwh_dc"], (0.12 + 0.02) / 0.95)
