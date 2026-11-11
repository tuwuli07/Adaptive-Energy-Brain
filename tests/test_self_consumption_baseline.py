"""Causal self-consumption actions and simulator-enforced feasibility."""

import pytest

from src.baselines import (
    SelfConsumptionController,
    available_controllers,
    create_controller,
)
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.renewables import SolarPV, WindGeneration
from src.simulation.state import SimulationState


def observation(*, solar=0, wind=0, demand=100, soc=0.5):
    config = load_simulation_config()
    state = SimulationState(
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z",
        solar,
        wind,
        demand,
        soc,
        0.1,
        0.04,
        config,
    )
    return state.observation(decision_timestamp_utc="2026-01-01T01:00:00Z")


def make_environment(*, history_solar=200, history_demand=100):
    provenance = {
        "source_id": "self_consumption_fixture",
        "origin": "simulated",
        "synthetic": True,
        "source_metadata": {"note": "Unit test only"},
    }
    rows = [
        HourlyObservation(
            f"2026-01-01T{hour:02}:00:00Z",
            solar,
            0,
            demand,
            0.1,
            0.04,
            {"B01": demand},
            {"PV01": solar},
            {"W01": 0},
        )
        for hour, solar, demand in (
            (0, history_solar, history_demand),
            (1, 0, 100),
            (2, 0, 100),
        )
    ]
    data = SimulationData(
        rows,
        {
            "currency_code": "USD",
            "units": {"power": "kW", "prices": "USD/kWh"},
            "sources": {
                "price": {
                    "record_type": "price",
                    "metadata": provenance,
                }
            },
        },
    )
    return MicrogridEnvironment(
        data,
        [
            SolarPV("PV01", 300, provenance, capacity_origin="hypothetical_example"),
            WindGeneration("W01", 150, provenance, capacity_origin="hypothetical_example"),
        ],
        CampusDemand(provenance, building_ids=["B01"]),
    )


def test_surplus_deficit_and_balance_use_existing_action_sign():
    controller = SelfConsumptionController()
    assert controller.select_action(observation(solar=150, wind=50, demand=100)).requested_power_kw == -100
    assert controller.select_action(observation(solar=20, wind=30, demand=100)).requested_power_kw == 50
    assert controller.select_action(observation(solar=40, wind=60, demand=100)).requested_power_kw == 0


def test_deadband_suppresses_small_surplus_and_deficit_and_reset_is_supported():
    controller = SelfConsumptionController(power_deadband_kw=10)
    controller.reset()
    assert controller.select_action(observation(solar=106, demand=100)).requested_power_kw == 0
    assert controller.select_action(observation(solar=111, demand=100)).requested_power_kw == -11
    assert controller.select_action(observation(solar=89, demand=100)).requested_power_kw == 11


@pytest.mark.parametrize("deadband", [-1, True, float("nan"), float("inf"), "1", None])
def test_invalid_deadband_is_rejected(deadband):
    with pytest.raises(ValueError, match="power_deadband_kw"):
        SelfConsumptionController(power_deadband_kw=deadband)


def test_registered_controller_uses_causal_observation_not_forecast_targets():
    assert "self_consumption" in available_controllers()
    controller = create_controller("self_consumption")
    current = observation(solar=150, demand=100)
    assert controller.select_action(current).requested_power_kw == -50


def test_dispatch_enforces_limits_and_preserves_energy_balance():
    environment = make_environment()
    controller = SelfConsumptionController()
    controller.reset()
    environment.reset()
    surplus_step = environment.step(controller.select_action(environment.observation))
    assert surplus_step.dispatch.requested_battery_power_kw == -100
    assert surplus_step.dispatch.actual_battery_power_kw == -100
    assert abs(surplus_step.dispatch.balance_residual_kw) <= 1e-6
    assert abs(surplus_step.dispatch.energy_balance_residual_kwh) <= 1e-6

    deficit_step = environment.step(controller.select_action(environment.observation))
    assert deficit_step.dispatch.requested_battery_power_kw == 100
    assert deficit_step.dispatch.actual_battery_power_kw == 100
    assert abs(deficit_step.dispatch.balance_residual_kw) <= 1e-6
    assert abs(deficit_step.dispatch.energy_balance_residual_kwh) <= 1e-6


def test_oversized_request_is_constrained_by_dispatch():
    controller = SelfConsumptionController()
    environment = make_environment(history_solar=300, history_demand=0)
    environment.reset()
    action = controller.select_action(environment.observation)
    assert action.requested_power_kw == -300
    result = environment.step(action).dispatch
    assert result.requested_battery_power_kw == -300
    assert result.actual_battery_power_kw == -150
    assert abs(result.balance_residual_kw) <= 1e-6
