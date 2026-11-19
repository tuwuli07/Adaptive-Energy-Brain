"""Preflight tests for the DiscreteActionAdapter wrapping MicrogridGymEnvironment."""

import builtins
import io
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("gymnasium")
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.utils.env_checker import check_env

pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import CategoricalDistribution

from src.rl.environment import (
    DISCRETE_ACTION_MAP,
    DiscreteActionAdapter,
    MicrogridGymEnvironment,
    map_normalized_action,
)
from src.rl.observations import FEATURE_NAMES
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.renewables import SolarPV, WindGeneration
from src.evaluation.ppo_validation import FrozenPPOController


ROOT = Path(__file__).resolve().parents[1]


def provenance():
    return {
        "source_id": "rl_discrete_fixture",
        "origin": "simulated",
        "synthetic": True,
        "source_metadata": {"note": "Hypothetical discrete test fixture"},
    }


def dataset(hours=6, export=0.04):
    from datetime import datetime, timedelta, timezone
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        HourlyObservation(
            (start + timedelta(hours=h)).isoformat().replace("+00:00", "Z"),
            20.0, 10.0, 100.0 + h * 10.0, 0.12, export,
            {"B01": 100.0 + h * 10.0}, {"PV01": 20.0}, {"W01": 10.0},
        )
        for h in range(hours)
    ]
    return SimulationData(
        rows,
        {
            "currency_code": "USD",
            "units": {"power": "kW", "prices": "USD/kWh"},
            "sources": {"price": {"record_type": "price", "metadata": provenance()}},
        },
    )


def make_continuous(data=None, config=None):
    return MicrogridGymEnvironment(
        dataset() if data is None else data,
        [
            SolarPV("PV01", 300, provenance(), capacity_origin="hypothetical_example"),
            WindGeneration("W01", 150, provenance(), capacity_origin="hypothetical_example"),
        ],
        CampusDemand(provenance(), building_ids=["B01"]),
        config=config,
    )


def make_discrete(data=None, config=None):
    return DiscreteActionAdapter(make_continuous(data=data, config=config))


def test_1_discrete_action_space_contains_exactly_three_valid_actions():
    """Requirement 1: The discrete action space contains exactly three valid actions."""
    env = make_discrete()
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 3

    # Exactly 0, 1, 2 are valid
    for a in (0, 1, 2):
        assert env.action_space.contains(a)

    for invalid in (-1, 3, 10, 1.5, "0", None):
        assert not env.action_space.contains(invalid)


def test_2_each_action_maps_to_correct_continuous_request():
    """Requirement 2: Each action maps to the correct continuous request (-1.0, 0.0, +1.0)."""
    env = make_discrete()
    np.testing.assert_array_equal(env.action(0), np.array([-1.0], dtype=np.float32))
    np.testing.assert_array_equal(env.action(1), np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(env.action(2), np.array([1.0], dtype=np.float32))

    # Also supports numpy integer scalars
    np.testing.assert_array_equal(env.action(np.int64(0)), np.array([-1.0], dtype=np.float32))
    np.testing.assert_array_equal(env.action(np.int32(1)), np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(env.action(np.int64(2)), np.array([1.0], dtype=np.float32))


def test_3_corresponding_requested_power_is_minus_150_zero_plus_150_kw():
    """Requirement 3: The corresponding requested power is −150, 0, or +150 kW."""
    config = load_simulation_config()
    assert config.battery.charge_limit_kw == 150.0
    assert config.battery.discharge_limit_kw == 150.0

    env = make_discrete(config=config)
    env.reset(seed=42)

    # Action 0 -> -150 kW
    _, _, _, _, info0 = env.step(0)
    assert info0["requested_power_kw"] == -150.0
    assert info0["normalized_action"] == -1.0
    assert info0["discrete_action"] == 0

    # Action 1 -> 0 kW
    _, _, _, _, info1 = env.step(1)
    assert info1["requested_power_kw"] == 0.0
    assert info1["normalized_action"] == 0.0
    assert info1["discrete_action"] == 1

    # Action 2 -> +150 kW
    _, _, _, _, info2 = env.step(2)
    assert info2["requested_power_kw"] == 150.0
    assert info2["normalized_action"] == 1.0
    assert info2["discrete_action"] == 2


def test_4_physical_dispatch_still_respects_soc_and_power_constraints():
    """Requirement 4: Physical dispatch still respects SOC and power constraints."""
    config = load_simulation_config()
    # 500 kWh battery, charge limit 150 kW, max SOC 0.9 (450 kWh)
    # Initial SOC 0.5 (250 kWh). At 150 kW charge:
    # Step 1: 250 + 150*0.95 = 392.5 kWh (SOC 0.785)
    # Step 2: 392.5 + 150*0.95 = 535 -> clamped to 450 kWh (SOC 0.9)
    env = make_discrete(config=config)
    env.reset(seed=1)

    obs1, _, _, _, info1 = env.step(0)  # charge
    assert info1["executed_power_kw"] == pytest.approx(-150.0)

    obs2, _, _, _, info2 = env.step(0)  # charge again, should be clamped by SOC max
    assert info2["executed_power_kw"] < 0  # still charging
    assert abs(info2["executed_power_kw"]) < 150.0  # curtailed to hit exactly 450 kWh
    assert obs2[0] == pytest.approx(config.battery.soc_max)


def test_5_idle_requests_produce_zero_battery_power():
    """Requirement 5: Idle requests produce zero battery power."""
    env = make_discrete()
    obs_before, _ = env.reset(seed=2)
    soc_before = obs_before[0]

    obs_after, reward, terminated, truncated, info = env.step(1)
    assert info["discrete_action"] == 1
    assert info["requested_power_kw"] == 0.0
    assert info["executed_power_kw"] == 0.0
    assert obs_after[0] == pytest.approx(soc_before)


def test_6_no_out_of_range_continuous_actions_are_generated():
    """Requirement 6: No out-of-range continuous actions are generated."""
    env = make_discrete()
    for a in (0, 1, 2):
        cont = env.action(a)
        assert cont.shape == (1,)
        assert cont.dtype == np.float32
        assert -1.0 <= cont[0] <= 1.0

    # Invalid actions raise ValueError
    for invalid in (-1, 3, 4, 99, "invalid"):
        with pytest.raises(ValueError):
            env.action(invalid)
        with pytest.raises(ValueError):
            env.step(invalid)


def test_7_discrete_adapter_does_not_modify_reward_values():
    """Requirement 7: The discrete adapter does not modify reward values."""
    data = dataset(hours=4)
    cont_env = make_continuous(data=data)
    disc_env = make_discrete(data=data)

    cont_env.reset(seed=10)
    disc_env.reset(seed=10)

    actions_disc = [0, 1, 2]
    actions_cont = [np.array([-1.0], dtype=np.float32), np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)]

    for a_d, a_c in zip(actions_disc, actions_cont):
        obs_d, rew_d, term_d, trunc_d, info_d = disc_env.step(a_d)
        obs_c, rew_c, term_c, trunc_c, info_c = cont_env.step(a_c)

        assert rew_d == pytest.approx(rew_c)
        assert info_d["step_metrics"] == info_c["step_metrics"]
        assert info_d["executed_power_kw"] == info_c["executed_power_kw"]


def test_8_adapter_preserves_observation_shape_and_dtype():
    """Requirement 8: The adapter preserves observation shape and dtype."""
    env = make_discrete()
    obs, info = env.reset(seed=5)
    assert obs.shape == (15,)
    assert obs.dtype == np.float32
    assert env.observation_space.shape == (15,)
    assert env.observation_space.dtype == np.float32

    obs_step, _, _, _, _ = env.step(1)
    assert obs_step.shape == (15,)
    assert obs_step.dtype == np.float32


def test_9_episode_termination_and_truncation_behavior_unchanged():
    """Requirement 9: Episode termination and truncation behavior is unchanged."""
    data = dataset(hours=4)  # 4 rows -> exactly 3 scored transitions
    cont_env = make_continuous(data=data)
    disc_env = make_discrete(data=data)

    cont_env.reset(seed=7)
    disc_env.reset(seed=7)

    for i in range(2):
        _, _, term_d, trunc_d, _ = disc_env.step(1)
        _, _, term_c, trunc_c, _ = cont_env.step(np.array([0.0], dtype=np.float32))
        assert not term_d and not trunc_d
        assert not term_c and not trunc_c

    # Step 3 (final transition: replay window exhausted)
    _, _, term_d, trunc_d, info_d = disc_env.step(1)
    _, _, term_c, trunc_c, info_c = cont_env.step(np.array([0.0], dtype=np.float32))
    assert term_d is False and term_c is False
    assert trunc_d is True and trunc_c is True
    assert info_d["completion_reason"] == "replay_window_exhausted"
    assert info_c["completion_reason"] == "replay_window_exhausted"


def test_10_existing_continuous_ppo_behavior_remains_unchanged():
    """Requirement 10: Existing continuous PPO behavior remains unchanged."""
    cont_env = make_continuous()
    obs_init, _ = cont_env.reset(seed=123)
    assert isinstance(cont_env.action_space, Box)
    assert cont_env.action_space.shape == (1,)
    assert cont_env.action_space.low[0] == -1.0
    assert cont_env.action_space.high[0] == 1.0

    # Verify FrozenPPOController with mock continuous model
    config = load_simulation_config()

    class MockContinuousModel:
        action_space = Box(-1, 1, shape=(1,), dtype=np.float32)
        policy = type("P", (), {"set_training_mode": lambda self, m: None})()
        def predict(self, obs, deterministic=True):
            return np.array([0.5], dtype=np.float32), None

    controller_cont = FrozenPPOController(MockContinuousModel(), config)
    obs = cont_env._core.observation
    action_cont = controller_cont._select_action(obs)
    assert action_cont.requested_power_kw == pytest.approx(0.5 * 150.0)

    # Verify FrozenPPOController with mock discrete model
    class MockDiscreteModel:
        action_space = Discrete(3)
        policy = type("P", (), {"set_training_mode": lambda self, m: None})()
        def predict(self, obs, deterministic=True):
            return 0, None  # action 0 = charge

    controller_disc = FrozenPPOController(MockDiscreteModel(), config)
    action_disc = controller_disc._select_action(obs)
    assert action_disc.requested_power_kw == pytest.approx(-150.0)


def test_11_no_access_to_held_out_test_data(monkeypatch):
    """Requirement 11: No access to held-out test data."""
    accessed_test = []
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        path_str = str(file).replace("\\", "/")
        if "/splits_hypothetical_campus_v1/test" in path_str:
            accessed_test.append(path_str)
            raise AssertionError(f"Illegal access to held-out test partition: {path_str}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    env = make_discrete()
    env.reset(seed=99)
    env.step(0)
    assert len(accessed_test) == 0


def test_12_gymnasium_and_stable_baselines3_compatibility():
    """Verify Gymnasium check_env and Stable-Baselines3 compatibility."""
    env = make_discrete()
    # Check env with Gymnasium checker
    check_env(env, skip_render_check=True)

    # Verify SB3 PPO model creation and execution with standard log_interval=None
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=8, n_epochs=1, seed=0)
    assert isinstance(model.policy.action_dist, CategoricalDistribution)
    assert model.action_space == Discrete(3)

    # Run minimal learning step
    model.learn(total_timesteps=16, log_interval=None)
    obs, _ = env.reset(seed=42)
    pred, _ = model.predict(obs, deterministic=True)
    assert int(pred) in (0, 1, 2)
