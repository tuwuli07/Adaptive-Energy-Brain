"""Zero-action behavior through the existing hourly simulator."""

import pytest

from src.baselines import (
    ZeroActionController,
    available_controllers,
    create_controller,
)
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.renewables import SolarPV, WindGeneration


def make_environment():
    provenance = {
        "source_id": "zero_action_fixture",
        "origin": "simulated",
        "synthetic": True,
        "source_metadata": {"note": "Unit test only"},
    }
    rows = [
        HourlyObservation(
            f"2026-01-01T{hour:02}:00:00Z",
            solar,
            wind,
            demand,
            0.1,
            0.04,
            {"B01": demand},
            {"PV01": solar},
            {"W01": wind},
        )
        for hour, solar, wind, demand in (
            (0, 20, 10, 100),
            (1, 150, 25, 100),
            (2, 0, 0, 100),
            (3, 300, 150, 100),
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
    config = load_simulation_config()
    environment = MicrogridEnvironment(
        data,
        [
            SolarPV("PV01", 300, provenance, capacity_origin="hypothetical_example"),
            WindGeneration("W01", 150, provenance, capacity_origin="hypothetical_example"),
        ],
        CampusDemand(provenance, building_ids=["B01"]),
        config,
    )
    return environment


def run_episode():
    environment = make_environment()
    controller = create_controller("zero_action")
    assert isinstance(controller, ZeroActionController)
    controller.reset()
    initial = environment.reset()
    initial_soc = initial.state.battery_soc
    dispatches = []
    while not environment.terminated:
        step = environment.step(controller.select_action(environment.observation))
        dispatch = step.dispatch
        assert dispatch.requested_battery_power_kw == 0
        assert step.observation.state.battery_soc == pytest.approx(initial_soc)
        assert abs(dispatch.balance_residual_kw) <= 1e-6
        assert abs(dispatch.energy_balance_residual_kwh) <= 1e-6
        assert dispatch.demand.campus_load_kw == pytest.approx(
            dispatch.served_demand_kw + dispatch.unserved_demand_kw
        )
        supply = (
            dispatch.grid.import_power_kw
            + dispatch.renewables.total_used_kw
            + dispatch.battery.discharge_power_kw
        )
        consumed = (
            dispatch.served_demand_kw
            + dispatch.battery.charge_power_kw
            + dispatch.grid.export_power_kw
        )
        assert supply == pytest.approx(consumed, abs=1e-6)
        dispatches.append(dispatch)
    return tuple(dispatches), environment.cumulative_metrics, environment.state


def test_zero_action_is_registered_and_always_requests_zero():
    assert "zero_action" in available_controllers()
    controller = create_controller("zero_action")
    observation = make_environment().reset()
    controller.reset()
    assert controller.select_action(observation).requested_power_kw == 0
    assert controller.select_action(observation).requested_power_kw == 0


def test_zero_action_preserves_idle_battery_and_dispatches_physics():
    dispatches, _, final_state = run_episode()
    assert len(dispatches) == 3
    assert all(result.actual_battery_power_kw == 0 for result in dispatches)
    assert all(
        result.battery.state_after.battery_soc
        == pytest.approx(result.battery.state_before.battery_soc)
        for result in dispatches
    )
    assert all(
        result.battery.state_after.battery_energy_kwh
        == pytest.approx(result.battery.state_before.battery_energy_kwh)
        for result in dispatches
    )
    assert final_state.battery.battery_soc == pytest.approx(0.5)


def test_zero_action_episode_is_deterministic():
    first = run_episode()
    second = run_episode()
    assert first == second
