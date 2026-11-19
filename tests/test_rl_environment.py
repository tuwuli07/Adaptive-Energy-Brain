"""Synthetic adapter tests; no saved datasets, forecast models, or training."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env

from src.rl.environment import MicrogridGymEnvironment, RLAdapterError
from src.rl.observations import FEATURE_NAMES, ObservationEncoder
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import MetricsTracker
from src.simulation.state import AgentObservation, ForecastFeature
from src.simulation.renewables import SolarPV, WindGeneration


def provenance():
    return {"source_id": "rl_fixture", "origin": "simulated", "synthetic": True,
            "source_metadata": {"note": "Hypothetical test fixture"}}


def dataset(hours=5, export=.04):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [HourlyObservation(
        (start + timedelta(hours=h)).isoformat().replace("+00:00", "Z"),
        20, 10, 100 + h * 10, .1 + h * .01, export,
        {"B01": 100 + h * 10}, {"PV01": 20}, {"W01": 10}) for h in range(hours)]
    return SimulationData(rows, {
        "currency_code": "USD", "units": {"power": "kW", "prices": "USD/kWh"},
        "sources": {"price": {"record_type": "price", "metadata": provenance()}},
    })


def make(data=None, config=None, cls=MicrogridGymEnvironment):
    return cls(dataset() if data is None else data, [
        SolarPV("PV01", 300, provenance(), capacity_origin="hypothetical_example"),
        WindGeneration("W01", 150, provenance(), capacity_origin="hypothetical_example"),
    ], CampusDemand(provenance(), building_ids=["B01"]), config)


def test_schema_order_scaling_bounds_and_timing():
    env = make()
    obs, info = env.reset(seed=10)
    assert obs.shape == (15,) and obs.dtype == np.float32
    assert np.isfinite(obs).all() and env.observation_space.contains(obs)
    assert info["feature_names"] == FEATURE_NAMES
    np.testing.assert_allclose(obs[:11], [.5, .15, .15, .1, .02, .01, .1, .04, 1, 2, .5])
    np.testing.assert_allclose(obs[11:], [np.sin(2*np.pi/24), np.cos(2*np.pi/24),
                                       np.sin(6*np.pi/7), np.cos(6*np.pi/7)])
    assert info["decision_timestamp_utc"] == "2026-01-01T01:00:00Z"
    assert info["observed_interval_utc"] == "2026-01-01T00:00:00Z"
    assert info["scored_steps"] == 0


@pytest.mark.parametrize("policy,limit", [("disable_export", 0), ("zero_compensation", .5)])
def test_export_missing_mask_and_effective_limit(policy, limit):
    cfg = load_simulation_config()
    cfg = replace(cfg, grid=replace(cfg.grid, missing_export_price_handling=policy))
    env = make(dataset(export=None), cfg)
    obs, _ = env.reset()
    assert tuple(obs[7:9]) == (0, 0) and obs[10] == limit
    assert env.step([0])[0][8] == 0
    real_zero, _ = make(dataset(export=0), cfg).reset()
    assert tuple(real_zero[7:9]) == (0, 1)


def test_negative_prices_and_unseen_large_valid_observations():
    cfg = load_simulation_config()
    cfg = replace(cfg, grid=replace(cfg.grid, allow_negative_prices=True))
    data = dataset()
    rows = [replace(r, import_price=-2, export_price=-1, campus_load_kw=5000,
                    building_loads_kw={"B01": 5000}) for r in data]
    env = make(SimulationData(rows, data.metadata), cfg)
    obs, _ = env.reset()
    assert obs[3] == 5 and tuple(obs[6:8]) == (-2, -1)
    assert env.observation_space.contains(obs)
    assert np.isneginf(env.observation_space.low[6:8]).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e100, 1e-100])
def test_bad_representation_rejected_at_construction(value):
    data = dataset()
    rows = list(data)
    rows[-1] = replace(rows[-1], import_price=value)
    with pytest.raises(ValueError):
        make(SimulationData(rows, data.metadata))


@pytest.mark.parametrize("a,expected", [(-1, -80), (-.5, -40), (0, 0), (.5, 60), (1, 120)])
def test_asymmetric_action_mapping(a, expected):
    cfg = load_simulation_config()
    cfg = replace(cfg, battery=replace(cfg.battery, charge_limit_kw=80, discharge_limit_kw=120))
    env = make(config=cfg)
    env.reset()
    _, reward, _, _, info = env.step(np.array([a], dtype=np.float32))
    assert info["requested_power_kw"] == expected
    assert reward == -info["step_metrics"]["values"]["net_operating_cost"]


def test_disabled_direction_and_physical_soc_projection():
    cfg = load_simulation_config()
    cfg = replace(cfg, battery=replace(cfg.battery, charge_limit_kw=0, initial_soc=.1))
    env = make(config=cfg)
    env.reset()
    assert env.step([-1])[4]["requested_power_kw"] == 0
    info = env.step([1])[4]
    assert info["requested_power_kw"] == 150 and info["executed_power_kw"] == 0
    assert info["step_metrics"]["adjustments"]


@pytest.mark.parametrize("action", [None, True, 0, [], [True], [1, 0], [[0]],
                                    [float("nan")], [float("inf")], [1.01], [-1.01],
                                    ["0"], [1j], {}, np.array([0], dtype=object)])
def test_invalid_actions_do_not_call_core_or_advance(action, monkeypatch):
    env = make()
    env.reset()
    def forbidden(*args):
        pytest.fail("Core must not be called for an invalid action")
    monkeypatch.setattr(env._core, "step", forbidden)
    with pytest.raises(ValueError):
        env.step(action)
    assert env._steps == 0 and env._core.state.step_count == 0


def test_lifecycle_count_final_boundary_and_reproducibility():
    env = make()
    with pytest.raises(RLAdapterError, match="reset"):
        env.step([0])
    def run(seed):
        first, _ = env.reset(seed=seed)
        actions = env.action_space.sample()
        results = [env.step([a]) for a in [-.5, .3, 0, 1]]
        return first, actions, results
    first, actions, results = run(20)
    again, again_actions, repeated = run(20)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(actions, again_actions)
    for a, b in zip(results, repeated):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
    assert all(not r[2] for r in results)
    assert [r[3] for r in results] == [False, False, False, True]
    assert results[-1][4]["scored_steps"] == 4
    assert results[-1][4]["core_terminated"]
    assert results[-1][4]["decision_timestamp_utc"] == "2026-01-01T05:00:00Z"
    assert sum(r[1] for r in results) == pytest.approx(
        -results[-1][4]["cumulative_metrics"]["net_operating_cost"])
    with pytest.raises(RLAdapterError, match="reset"):
        env.step([0])
    env.close()
    with pytest.raises(RLAdapterError):
        env.step([0])
    env.reset(seed=21)
    np.testing.assert_array_equal(env.step([-.5])[0], results[0][0])


@pytest.mark.parametrize("options", [{"start": 1}, [], 1, "invalid"])
def test_bad_reset_options_preserve_current_episode(options):
    env = make()
    env.reset(options={})
    env.step([0])
    before = env._core.state
    with pytest.raises(ValueError, match="options"):
        env.reset(options=options)
    assert env._core.state == before and env._steps == 1
    env.step([0])


def test_dispatch_and_accounting_failures_are_retryable(monkeypatch):
    env = make()
    env.reset()
    before = env._core.state
    original = MetricsTracker.record
    def fail(*args):
        raise ValueError("injected accounting failure")
    monkeypatch.setattr(MetricsTracker, "record", fail)
    with pytest.raises(ValueError, match="accounting"):
        env.step([.2])
    assert env._core.state == before and env._steps == 0
    monkeypatch.setattr(MetricsTracker, "record", original)
    env.step([.2])
    cfg = load_simulation_config()
    cfg = replace(cfg, battery=replace(cfg.battery, initial_soc=.1),
                  grid=replace(cfg.grid, import_limit_kw=0))
    short = make(config=cfg)
    short.reset()
    for _ in range(2):
        with pytest.raises(ValueError, match="shortfall"):
            short.step([0])
        assert short._steps == short._core.state.step_count == 0


@pytest.mark.parametrize("where", ["encode", "info"])
def test_postcommit_failure_requires_reset(where, monkeypatch):
    env = make()
    env.reset()
    target, name = (env._encoder, "encode") if where == "encode" else (env, "_info")
    original = getattr(target, name)
    def fail(*args):
        raise ValueError("injected wrapper failure")
    monkeypatch.setattr(target, name, fail)
    with pytest.raises(RLAdapterError, match="not rolled back"):
        env.step([0])
    assert env._core.state.step_count == env._steps == 1
    with pytest.raises(RLAdapterError, match="reset"):
        env.step([0])
    monkeypatch.setattr(target, name, original)
    env.reset()
    assert env.step([0])[4]["scored_steps"] == 1


@pytest.mark.parametrize("kind", ["demand", "solar", "wind", "import", "export"])
def test_future_realizations_do_not_change_earlier_observations(kind):
    data = dataset()
    rows = list(data)
    changes = {
        "demand": {"campus_load_kw": 250, "building_loads_kw": {"B01": 250}},
        "solar": {"pv_available_kw": 50, "pv_asset_powers_kw": {"PV01": 50}},
        "wind": {"wind_available_kw": 40, "wind_asset_powers_kw": {"W01": 40}},
        "import": {"import_price": 5}, "export": {"export_price": 3},
    }
    rows[2] = replace(rows[2], **changes[kind])
    a, b = make(data), make(SimulationData(rows, data.metadata))
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
    np.testing.assert_array_equal(a.step([0])[0], b.step([0])[0])
    assert not np.array_equal(a.step([0])[0], b.step([0])[0])


def test_direct_core_equivalence_and_independent_results():
    core, wrapped = make(cls=MicrogridEnvironment), make()
    obs = core.reset()
    encoded, info = wrapped.reset()
    encoder = ObservationEncoder(obs.state.config)
    np.testing.assert_array_equal(encoded, encoder.encode(obs))
    info["episode"]["environment"]["action_steps"] = 999
    for a in [-1, 1, .2, 0]:
        step = core.step(a * 150)
        numeric, reward, terminated, truncated, info = wrapped.step([a])
        np.testing.assert_array_equal(numeric, encoder.encode(step.observation))
        assert wrapped._core.dispatch_results == core.dispatch_results
        assert wrapped._core.state == core.state
        assert wrapped._core.step_metrics == core.step_metrics
        assert wrapped._core.cumulative_metrics == core.cumulative_metrics
        assert reward == -step.metrics.values.net_operating_cost
        assert not terminated and truncated == step.terminated
        assert info["episode"]["environment"]["action_steps"] == 4
        numeric[:] = 99


@pytest.mark.parametrize("hours", [0, 1])
def test_requires_history_and_dispatch_rows(hours):
    with pytest.raises(ValueError, match="two hourly"):
        make(dataset(hours))


def test_gymnasium_checker():
    # Infinite price/power bounds are intentional; rendering is out of scope.
    check_env(make(), skip_render_check=True)


def test_one_scored_step_window():
    env = make(dataset(2))
    env.reset()
    result = env.step([0])
    assert result[2:4] == (False, True)
    assert result[4]["scored_steps"] == 1
    assert len(env._core.dispatch_results) == 1


def test_encoder_rejects_forecasts_and_changed_configuration():
    core = make(cls=MicrogridEnvironment)
    obs = core.reset()
    encoder = ObservationEncoder(obs.state.config)
    feature = ForecastFeature("campus_demand_kw", 50, "2026-01-01T02:00:00Z",
                              "2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z", "fixture")
    with pytest.raises(ValueError, match="Forecasts"):
        encoder.encode(AgentObservation(obs.state, obs.decision_timestamp_utc, (feature,)))
    cfg = replace(obs.state.config, battery=replace(obs.state.config.battery, discharge_limit_kw=100))
    with pytest.raises(ValueError, match="configuration"):
        encoder.encode(replace(obs, state=replace(obs.state, config=cfg)))


def test_negative_prices_require_physical_opt_in():
    data = dataset()
    rows = list(data)
    rows[0] = replace(rows[0], import_price=-1)
    with pytest.raises(ValueError, match="negative"):
        make(SimulationData(rows, data.metadata))


def test_config_rejects_unapproved_scale_and_duplicate_keys(tmp_path):
    path = tmp_path / "adapter.yaml"
    path.write_text("version: 1\nobservation_schema: drl_observation_v1\n"
                    "power_reference_kw: 500\nprice_reference_per_kwh: 1\n")
    from src.rl.observations import validate_adapter_config
    with pytest.raises(ValueError, match="scales"):
        validate_adapter_config(path)
    path.write_text("version: 1\nversion: 1\n")
    with pytest.raises(ValueError):
        validate_adapter_config(path)


def test_source_mutation_cannot_change_owned_core():
    data = dataset()
    env = make(data)
    data[1].building_loads_kw["B01"] = 999
    env.reset()
    assert env.step([0])[0][3] == np.float32(.11)
