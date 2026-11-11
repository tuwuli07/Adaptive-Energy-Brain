"""State validation, causal feature contracts and dispatch action adapters."""

from dataclasses import FrozenInstanceError, replace

import pytest

from src.simulation.actions import BatteryAction, ExecutedBatteryAction
from src.simulation.battery import Battery
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation
from src.simulation.demand import CampusDemand
from src.simulation.dispatch import DispatchEngine
from src.simulation.renewables import SolarPV, WindGeneration
from src.simulation.state import ForecastFeature, SimulationState


T0 = "2026-01-01T00:00:00Z"
T1 = "2026-01-01T01:00:00Z"
T2 = "2026-01-01T02:00:00Z"


def snapshot(**changes):
    state = SimulationState(T0, T1, 100, 50, 300, 0.5, 0.1, None,
                            load_simulation_config())
    return replace(state, **changes)


def forecast(**changes):
    return replace(ForecastFeature("campus_demand_kw", 310, T2, T1, T0,
                                   "test_fixture_model"), **changes)


def test_state_and_observation_units_limits_and_missing_tariff():
    state = snapshot()
    obs = state.observation(decision_timestamp_utc=T1)
    features = obs.features()
    assert features["battery_energy_kwh"] == 250
    assert features["charge_limit_kw"] == 150
    assert features["grid_export_price"] is None
    assert obs.price_unit == "USD/kWh"
    features["battery_soc"] = 0
    assert obs.features()["battery_soc"] == 0.5
    assert not {"next_state", "reward", "future_demand", "datasets"} & obs.features().keys()
    with pytest.raises(FrozenInstanceError):
        state.battery_soc = 0


@pytest.mark.parametrize("field,value", [
    ("solar_generation_kw", -1), ("wind_generation_kw", float("nan")),
    ("campus_demand_kw", True), ("battery_soc", 0.99),
    ("grid_import_price", -0.1), ("grid_export_price", float("inf")),
    ("timestamp_utc", "2026-01-01"), ("available_at_utc", "2025-12-31T23:00:00Z"),
])
def test_invalid_state_rejected(field, value):
    with pytest.raises(ValueError):
        snapshot(**{field: value})


def test_soc_and_tariff_configuration_policies():
    config = load_simulation_config()
    config = replace(config, grid=replace(config.grid, allow_negative_prices=True,
                                         missing_export_price_handling="disable_export"))
    state = snapshot(config=config, grid_import_price=-0.1, battery_soc=0.9)
    features = state.observation(decision_timestamp_utc=T1).features()
    assert features["charge_limit_kw"] == 0
    assert features["grid_export_limit_kw"] == 0
    assert features["grid_import_price"] == -0.1


def test_future_measurement_cannot_enter_earlier_observation():
    with pytest.raises(ValueError, match="not yet available"):
        snapshot().observation(decision_timestamp_utc=T0)
    assert snapshot().observation(decision_timestamp_utc=T1).state.timestamp_utc == T0


def test_forecasts_separate_and_deterministically_ordered():
    first = forecast()
    second = forecast(variable="solar_generation_kw")
    state = snapshot()
    obs = state.observation(decision_timestamp_utc=T1, forecasts=[second, first])
    assert obs.forecasts == (first, second)
    assert obs.features() == state.observation(decision_timestamp_utc=T1).features()
    assert obs == state.observation(decision_timestamp_utc=T1, forecasts=[first, second])


@pytest.mark.parametrize("changes", [
    {"issued_at_utc": T2}, {"data_cutoff_utc": T2},
    {"target_timestamp_utc": T1}, {"variable": "future_observed_demand"},
    {"value_kw": -1}, {"model_id": ""},
])
def test_invalid_or_future_forecast_rejected(changes):
    with pytest.raises(ValueError):
        snapshot().observation(decision_timestamp_utc=T1, forecasts=[forecast(**changes)])


def test_observation_rejects_untyped_or_duplicate_forecasts():
    for features in ([{"future_observation": 100}], [forecast(), forecast()]):
        with pytest.raises(ValueError):
            snapshot().observation(decision_timestamp_utc=T1, forecasts=features)


@pytest.mark.parametrize("value", [True, "150", float("nan"), float("inf"), None])
def test_invalid_action_rejected(value):
    with pytest.raises(ValueError):
        BatteryAction(value)


def test_action_sign_and_direction_validation():
    assert BatteryAction.from_directions(charge_kw=100).requested_power_kw == -100
    assert BatteryAction.from_directions(discharge_kw=100).requested_power_kw == 100
    assert BatteryAction().requested_power_kw == 0
    for args in ({"charge_kw": -1}, {"charge_kw": 1, "discharge_kw": 1}):
        with pytest.raises(ValueError):
            BatteryAction.from_directions(**args)


def engine():
    provenance = {"source_id": "state_test", "origin": "simulated", "synthetic": True,
                  "source_metadata": {"note": "Test fixture only"}}
    return DispatchEngine([
        SolarPV("PV01", 300, provenance, capacity_origin="hypothetical_example"),
        WindGeneration("W01", 150, provenance, capacity_origin="hypothetical_example"),
    ], CampusDemand(provenance, building_ids=["B01"]))


def test_action_adapter_execution_and_deterministic_state_transition():
    first, second = engine(), engine()
    observation = HourlyObservation(T0, 100, 50, 300, 0.1, 0.04,
                                    {"B01": 300}, {"PV01": 100}, {"W01": 50})
    before = first.state
    state = SimulationState.from_observation(
        observation, before.battery, load_simulation_config(), available_at_utc=T1)
    assert state.battery_soc == 0.5
    action = BatteryAction(1000)
    result = first.step(observation, **action.dispatch_kwargs())
    assert result == second.step(observation, **action.dispatch_kwargs())
    executed = ExecutedBatteryAction.from_dispatch(result)
    assert executed.requested_power_kw == 1000
    assert executed.actual_power_kw == 150
    assert executed.adjustment_codes
    assert first.state == second.state
    assert state.battery_soc == 0.5  # Snapshot does not advance with engine.
    assert before != first.state


def test_snapshot_rejects_inconsistent_battery_energy():
    config = load_simulation_config()
    battery = replace(Battery(config).state, battery_energy_kwh=1)
    observation = HourlyObservation(T0, 0, 0, 0, 0.1, None, {}, {}, {})
    with pytest.raises(ValueError, match="disagrees"):
        SimulationState.from_observation(observation, battery, config, available_at_utc=T1)
