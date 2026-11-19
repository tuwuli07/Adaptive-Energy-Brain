"""Fixed-window Gymnasium adapter; physical dispatch remains entirely in the core."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from src.rl.observations import (
    DEFAULT_CONFIG, FEATURE_NAMES, POWER_REFERENCE_KW, PRICE_REFERENCE_PER_KWH,
    SCHEMA, ObservationEncoder, validate_adapter_config,
)
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.environment import MicrogridEnvironment
from src.simulation.state import SimulationState


class RLAdapterError(RuntimeError):
    """Wrapper lifecycle failure; a post-commit failure requires reset."""


def map_normalized_action(action, config):
    """Validate and map a normalized command through static AC-side ratings."""
    values = np.asarray(action)
    if (values.shape != (1,) or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all() or np.any(values < -1) or np.any(values > 1)):
        raise ValueError("Action must be one finite real value in shape (1,) within [-1, 1].")
    value = float(values[0])
    rating = config.battery.charge_limit_kw if value < 0 else config.battery.discharge_limit_kw
    return value, BatteryAction(value * rating)


class MicrogridGymEnvironment(gym.Env):
    """Own a core environment for exactly N-1 scored transitions from N rows.

    Supply one validated, split-local SimulationData window and physical
    components. No sampling, forecasting, training, or terminal valuation occurs.
    Nonempty reset options are unsupported. No rendering resources are owned.
    """

    metadata = {"render_modes": []}

    def __init__(self, data, renewables, demand, config=None, costs=None, *,
                 price_provenance=None, adapter_config=DEFAULT_CONFIG):
        validate_adapter_config(adapter_config)
        config = load_simulation_config() if config is None else config
        self._core = MicrogridEnvironment(data, renewables, demand, config, costs,
                                         price_provenance=price_provenance)
        self._encoder = ObservationEncoder(config)
        self.observation_space = self._encoder.space
        self.action_space = Box(-1, 1, shape=(1,), dtype=np.float32)
        self._config = config
        # Preflight representation only. These rows never become policy inputs
        # until returned by the core after their declared availability boundary.
        for row in data:
            boundary = (datetime.fromisoformat(row.timestamp_utc.replace("Z", "+00:00"))
                        + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
            state = SimulationState(
                row.timestamp_utc, boundary, row.pv_available_kw, row.wind_available_kw,
                row.campus_load_kw, config.battery.initial_soc,
                row.import_price, row.export_price, config)
            self._encoder.encode(state.observation(decision_timestamp_utc=boundary))
        self._ready = False
        self._broken = False
        self._steps = 0

    def _info(self, observation):
        return {
            "observation_schema": SCHEMA, "feature_names": FEATURE_NAMES,
            "power_reference_kw": POWER_REFERENCE_KW,
            "price_reference_per_kwh": PRICE_REFERENCE_PER_KWH,
            "currency_code": observation.currency_code,
            "observed_interval_utc": observation.state.timestamp_utc,
            "decision_timestamp_utc": observation.decision_timestamp_utc,
            "episode": self._core.metadata, "scored_steps": self._steps,
            "cumulative_metrics": asdict(self._core.cumulative_metrics),
            "core_terminated": self._core.terminated,
            "completion_reason": "replay_window_exhausted" if self._core.terminated else None,
        }

    def reset(self, *, seed=None, options=None):
        if options is not None and (not isinstance(options, dict) or options):
            raise ValueError("Only None or empty reset options are supported.")
        # Mark unusable until the whole reset response has been constructed.
        self._ready = False
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        observation = self._core.reset()
        self._steps = 0
        encoded = self._encoder.encode(observation)
        info = self._info(observation)
        self._broken, self._ready = False, True
        return encoded, info

    def _action(self, action):
        return map_normalized_action(action, self._config)

    def step(self, action):
        if self._broken:
            raise RLAdapterError("Wrapper failed after a committed transition; reset required.")
        if not self._ready or self._core.terminated:
            raise RLAdapterError("Call reset before stepping or after episode completion.")
        normalized, physical = self._action(action)
        # Core failures are atomic and remain retryable; wrapper has not advanced.
        step = self._core.step(physical)
        self._steps += 1
        try:
            observation = self._encoder.encode(step.observation)
            reward = -float(step.metrics.values.net_operating_cost)
            if not np.isfinite(reward):
                raise ValueError("Nonfinite committed reward.")
            info = self._info(step.observation)
            info.update({
                "normalized_action": normalized,
                "requested_power_kw": physical.requested_power_kw,
                "executed_power_kw": step.dispatch.actual_battery_power_kw,
                "step_metrics": asdict(step.metrics),
                "power_balance_residual_kw": step.dispatch.balance_residual_kw,
                "energy_balance_residual_kwh": step.dispatch.energy_balance_residual_kwh,
            })
            return observation, reward, False, bool(step.terminated), deepcopy(info)
        except Exception as exc:
            self._broken, self._ready = True, False
            raise RLAdapterError(
                "Wrapper failed after a committed transition; physical step was not rolled back. "
                "Reset required."
            ) from exc

    def close(self):
        self._ready = False


DISCRETE_ACTION_MAP = {
    0: np.array([-1.0], dtype=np.float32),
    1: np.array([0.0], dtype=np.float32),
    2: np.array([1.0], dtype=np.float32),
}


class DiscreteActionAdapter(gym.ActionWrapper):
    """Minimal discrete action adapter wrapping MicrogridGymEnvironment.

    Maps Discrete(3) actions to the existing continuous Box(-1, 1) interface:
      - Action 0 -> -1.0 (requested -charge_limit_kw, e.g. -150 kW)
      - Action 1 ->  0.0 (requested    0 kW idle)
      - Action 2 -> +1.0 (requested +discharge_limit_kw, e.g. +150 kW)

    Preserves underlying observation semantics, reward calculations, episode boundaries,
    SOC accounting, grid constraints, battery efficiency, and degradation costs identically.
    """

    def __init__(self, env: MicrogridGymEnvironment):
        if not isinstance(env, gym.Env):
            raise TypeError("env must be a Gymnasium environment.")
        super().__init__(env)
        self.action_space = Discrete(3)

    def action(self, action):
        """Map discrete index (0, 1, 2) to shape (1,) float32 continuous command."""
        try:
            val = int(action) if np.ndim(action) == 0 else int(np.asarray(action).item())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid discrete action: {action}") from exc
        if val not in DISCRETE_ACTION_MAP:
            raise ValueError(f"Discrete action must be 0, 1, or 2; received {action}.")
        return DISCRETE_ACTION_MAP[val].copy()

    def step(self, action):
        continuous = self.action(action)
        obs, reward, terminated, truncated, info = self.env.step(continuous)
        val = int(action) if np.ndim(action) == 0 else int(np.asarray(action).item())
        info["discrete_action"] = val
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)

