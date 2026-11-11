"""Hourly episode lifecycle, causality, state persistence and transactional failure."""

from dataclasses import replace

import pytest

from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.environment import EnvironmentError, MicrogridEnvironment
from src.simulation.metrics import MetricsTracker, load_cost_config
from src.simulation.renewables import SolarPV, WindGeneration


def provenance():
    return {"source_id": "environment_fixture", "origin": "simulated", "synthetic": True,
            "source_metadata": {"note": "Hypothetical unit test only"}}


def data(hours=4):
    rows = [HourlyObservation(f"2026-01-01T{h:02}:00:00Z", 20, 10, 100 + 10 * h,
                               0.1, 0.04, {"B01": 100 + 10 * h}, {"PV01": 20}, {"W01": 10})
            for h in range(hours)]
    return SimulationData(rows, {
        "currency_code": "USD", "units": {"power": "kW", "prices": "USD/kWh"},
        "sources": {"price": {"record_type": "price", "metadata": provenance()}},
    })


def environment(dataset=None, config=None, costs=None):
    return MicrogridEnvironment(data() if dataset is None else dataset, [
        SolarPV("PV01", 300, provenance(), capacity_origin="hypothetical_example"),
        WindGeneration("W01", 150, provenance(), capacity_origin="hypothetical_example"),
    ], CampusDemand(provenance(), building_ids=["B01"]), config, costs)


def test_reset_requires_explicit_lifecycle_and_restores_everything():
    env = environment()
    with pytest.raises(EnvironmentError, match="reset"):
        env.step(0)
    initial = env.reset()
    assert initial.state.timestamp_utc == "2026-01-01T00:00:00Z"
    assert initial.decision_timestamp_utc == "2026-01-01T01:00:00Z"
    assert env.state.step_count == 0
    assert env.state.battery.battery_energy_kwh == 250
    env.step(-100)
    env.step(50)
    assert env.state.step_count == 2
    assert env.reset() == initial
    assert env.state.step_count == env.state.grid.step_count == 0
    assert env.state.battery.battery_energy_kwh == 250
    assert env.cumulative_metrics.net_operating_cost == 0
    assert env.step_metrics == ()
    assert not env.terminated


def test_one_interval_per_step_battery_and_cost_persistence():
    env = environment()
    env.reset()
    first = env.step(BatteryAction(-100))
    assert first.dispatch.timestamp_utc == "2026-01-01T01:00:00Z"
    assert first.observation.decision_timestamp_utc == "2026-01-01T02:00:00Z"
    assert first.observation.state.battery_soc == pytest.approx(345 / 500)
    assert first.metrics.values.import_cost == 18  # (110 - 30 + 100) * .1
    assert first.metrics.values.battery_degradation_cost == 2
    second = env.step(50)
    assert second.dispatch.timestamp_utc == "2026-01-01T02:00:00Z"
    assert second.dispatch.battery.state_before == first.dispatch.battery.state_after
    assert env.state.battery.battery_energy_kwh == pytest.approx(345 - 50 / .95)
    assert env.cumulative_metrics.net_operating_cost == pytest.approx(20 + 4 + 1)
    assert second.cumulative_metrics == env.cumulative_metrics
    assert len(env.step_metrics) == 2


def test_terminal_boundary_no_out_of_range_reads_or_extra_transactions():
    env = environment(data(2))
    env.reset()
    result = env.step(0)
    assert result.terminated and env.terminated
    assert not result.truncated
    assert result.observation.state.timestamp_utc == "2026-01-01T01:00:00Z"
    assert result.observation.decision_timestamp_utc == "2026-01-01T02:00:00Z"
    before = env.state
    with pytest.raises(EnvironmentError, match="complete"):
        env.step(0)
    assert env.state == before
    assert env.metadata["environment"]["action_steps"] == 1
    assert not env.metadata["environment"]["history_row_in_episode_metrics"]
    env.reset()
    assert env.step(0) == result


@pytest.mark.parametrize("action", [None, True, float("nan"), float("inf"), "100", {}, [100]])
def test_invalid_actions_are_atomic(action):
    env = environment()
    initial = env.reset()
    state, totals = env.state, env.cumulative_metrics
    with pytest.raises(ValueError):
        env.step(action)
    assert env.state == state
    assert env.observation == initial
    assert env.cumulative_metrics == totals
    assert env.step_metrics == ()


def test_oversized_finite_action_is_constrained_and_reported():
    env = environment()
    env.reset()
    result = env.step(1000)
    assert result.dispatch.requested_battery_power_kw == 1000
    assert result.dispatch.actual_battery_power_kw == 150
    assert result.metrics.adjustments


def test_failed_dispatch_can_be_retried_without_progress():
    config = load_simulation_config()
    config = replace(config, battery=replace(config.battery, initial_soc=.1),
                     grid=replace(config.grid, import_limit_kw=0))
    env = environment(config=config)
    env.reset()
    before = env.state
    for _ in range(2):
        with pytest.raises(ValueError, match="shortfall"):
            env.step(0)
        assert env.state == before
        assert env.step_metrics == ()


def test_metrics_failure_does_not_commit_dispatch_or_advance(monkeypatch):
    env = environment()
    initial = env.reset()
    before = env.state
    original = MetricsTracker.record

    def fail(*args):
        raise ValueError("Injected accounting failure")

    monkeypatch.setattr(MetricsTracker, "record", fail)
    with pytest.raises(ValueError, match="accounting failure"):
        env.step(-100)
    assert env.state == before
    assert env.observation == initial
    assert env.cumulative_metrics.net_operating_cost == 0
    monkeypatch.setattr(MetricsTracker, "record", original)
    assert env.step(-100).dispatch.timestamp_utc == "2026-01-01T01:00:00Z"


def test_identical_actions_resets_and_inputs_reproduce_whole_episode():
    env = environment()

    def run():
        initial = env.reset()
        steps = tuple(env.step(a) for a in (-100, 50, 0))
        return initial, steps, env.state, env.cumulative_metrics

    assert run() == run()


def test_observation_failure_leaves_episode_and_grid_ledger_unchanged(monkeypatch):
    env = environment()
    env.reset()
    state = env.state

    def fail(*args):
        raise ValueError("Injected observation failure")

    monkeypatch.setattr(env, "_observe", fail)
    with pytest.raises(ValueError, match="observation failure"):
        env.step(50)
    assert env.state == state
    assert env.dispatch_results == ()
    assert env.step_metrics == ()


def test_agent_does_not_see_next_interval_realizations():
    original = data()
    changed = list(original)
    changed[1] = replace(changed[1], campus_load_kw=999, building_loads_kw={"B01": 999})
    first, second = environment(original), environment(SimulationData(changed, original.metadata))
    assert first.reset() == second.reset()
    # Only after settlement does the changed current-hour demand become visible.
    assert first.step(0).observation != second.step(0).observation


def test_dataset_snapshot_and_metadata_cannot_be_changed_by_caller():
    original = data()
    env = environment(original)
    original[1].building_loads_kw["B01"] = 999
    original[1].pv_asset_powers_kw["PV01"] = 999
    metadata = env.metadata
    metadata["environment"]["action_steps"] = 999
    env.reset()
    assert env.step(0).dispatch.demand.campus_load_kw == 110
    assert env.metadata["environment"]["action_steps"] == 3


@pytest.mark.parametrize("hours", [0, 1])
def test_requires_history_and_dispatch_rows(hours):
    with pytest.raises(EnvironmentError, match="two hourly"):
        environment(data(hours))


def test_rejects_discontinuous_duplicate_or_incompatible_dataset():
    original = data()
    for rows in ([original[0], original[0]], [original[0], original[2]]):
        with pytest.raises(ValueError, match="consecutive"):
            environment(SimulationData(rows, original.metadata))
    for field, value in (("currency_code", "EUR"), ("units", {"power": "MW"})):
        metadata = original.metadata
        metadata[field] = value
        with pytest.raises(EnvironmentError):
            environment(SimulationData(list(original), metadata))
    with pytest.raises(EnvironmentError, match="currency"):
        environment(costs=replace(load_cost_config(), currency_code="EUR"))
