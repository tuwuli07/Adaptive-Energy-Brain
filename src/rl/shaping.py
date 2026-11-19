"""Potential-Based Reward Shaping (PBRS) for microgrid battery inventory valuation.

Implements policy-invariant potential-based reward shaping following Ng, Harada,
and Russell (1999):
    F(s, a, s') = gamma * Phi(s') - Phi(s)
where:
    Phi(s) = reference_value_usd_per_kwh_dc * stored_energy_kwh(s)

Shaping is applied strictly to the training reward signal. Raw physical accounting,
simulator dispatch, and evaluation metrics remain untouched.
"""

import gymnasium as gym
import numpy as np

# Reference replacement valuation of stored DC energy:
# (c_import_offpeak + c_deg) / eta_chg = (0.12 + 0.02) / 0.95 USD / kWh DC
REFERENCE_VALUE_USD_PER_KWH_DC = (0.12 + 0.02) / 0.95
DEFAULT_BATTERY_CAPACITY_KWH = 500.0


class PotentialBasedRewardShapingWrapper(gym.Wrapper):
    """Wrap a microgrid environment with potential-based battery inventory shaping.

    Adds F(s, a, s') = gamma * Phi(s') - Phi(s) in nominal USD to the raw step reward.
    Downstream RewardScaleVecEnv scales the shaped sum exactly once.
    """

    def __init__(
        self,
        env,
        gamma=0.999,
        reference_value_usd_per_kwh_dc=REFERENCE_VALUE_USD_PER_KWH_DC,
        capacity_kwh=DEFAULT_BATTERY_CAPACITY_KWH,
    ):
        super().__init__(env)
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if reference_value_usd_per_kwh_dc <= 0:
            raise ValueError("reference_value_usd_per_kwh_dc must be positive")
        if capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")

        self.gamma = float(gamma)
        self.v_ref = float(reference_value_usd_per_kwh_dc)
        self.capacity_kwh = float(capacity_kwh)
        self.phi_current = None
        self.episode_step = 0
        self.initial_phi = None

    def _extract_stored_energy_kwh(self, observation):
        """Extract battery stored DC energy from observation feature 0 (SOC)."""
        soc = float(observation[0])
        return soc * self.capacity_kwh

    def potential(self, stored_energy_kwh):
        """Compute the nominal USD inventory potential Phi(s)."""
        return self.v_ref * stored_energy_kwh

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        stored_energy = self._extract_stored_energy_kwh(observation)
        self.phi_current = self.potential(stored_energy)
        self.initial_phi = self.phi_current
        self.episode_step = 0
        info["inventory_potential_usd"] = self.phi_current
        return observation, info

    def step(self, action):
        if self.phi_current is None:
            raise RuntimeError("Call reset() before step() in PotentialBasedRewardShapingWrapper.")

        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1

        stored_energy_next = self._extract_stored_energy_kwh(observation)
        phi_next = self.potential(stored_energy_next)

        # F(s, a, s') = gamma * Phi(s') - Phi(s)
        shaping_usd = self.gamma * phi_next - self.phi_current
        shaped_reward = raw_reward + shaping_usd

        info["unshaped_reward_usd"] = float(raw_reward)
        info["shaping_reward_usd"] = float(shaping_usd)
        info["shaped_reward_usd"] = float(shaped_reward)
        info["inventory_potential_usd"] = float(phi_next)

        self.phi_current = phi_next
        return observation, shaped_reward, terminated, truncated, info
