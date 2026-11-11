"""Conservative SOC-band controller decisions and physical integration."""

import pytest

from src.baselines import (
    BatteryPreservingController,
    available_controllers,
    create_controller,
)
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.renewables import SolarPV, WindGeneration
from src.simulation.state import ForecastFeature, SimulationState


def observation(*, solar=0, wind=0, demand=100, soc=0.5, forecasts=()):
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
    return state.observation(
        decision_timestamp_utc="2026-01-01T01:00:00Z",
        forecasts=forecasts,
    )


def make_controller(**overrides):
    settings = {
        "preferred_soc_min": 0.3,
        "preferred_soc_max": 0.8,
        "power_deadband_kw": 5,
        "max_charge_power_kw": 50,
        "max_discharge_power_kw": 50,
    }
    settings.update(overrides)
    return BatteryPreservingController(**settings)


def test_registry_and_configured_soc_band_defaults():
    assert "battery_preserving" in available_controllers()
    controller = create_controller("battery_preserving")
    assert isinstance(controller, BatteryPreservingController)
    assert controller.preferred_soc_min == pytest.approx(0.3)
    assert controller.preferred_soc_max == pytest.approx(0.8)
    assert controller.power_deadband_kw == pytest.approx(5)


def test_surplus_charges_conservatively_toward_preferred_upper_soc():
    controller = make_controller()
    controller.reset()
    action = controller.select_action(observation(solar=500, demand=100, soc=0.5))
    assert action.requested_power_kw == -50
    near_upper = controller.select_action(
        observation(solar=500, demand=100, soc=0.79)
    )
    assert near_upper.requested_power_kw == pytest.approx(-5 / 0.95)
    assert controller.select_action(
        observation(solar=500, demand=100, soc=0.8)
    ).requested_power_kw == 0


def test_deficit_respects_preferred_soc_floor_and_power_cap():
    controller = make_controller()
    assert controller.select_action(
        observation(solar=0, demand=1000, soc=0.5)
    ).requested_power_kw == 50
    assert controller.select_action(
        observation(solar=0, demand=1000, soc=0.31)
    ).requested_power_kw == pytest.approx(4.75)
    assert controller.select_action(
        observation(solar=0, demand=1000, soc=0.3)
    ).requested_power_kw == 0


def test_deadband_prevents_small_surplus_or_deficit_cycling():
    controller = make_controller()
    assert controller.select_action(
        observation(solar=104, demand=100)
    ).requested_power_kw == 0
    assert controller.select_action(
        observation(solar=95, demand=100)
    ).requested_power_kw == 0
    assert controller.select_action(
        observation(solar=106, demand=100)
    ).requested_power_kw < 0
    assert controller.select_action(
        observation(solar=95, demand=100)
    ).requested_power_kw == 0


@pytest.mark.parametrize(
    "values",
    [
        {"preferred_soc_min": -0.1},
        {"preferred_soc_min": 0.8, "preferred_soc_max": 0.8},
        {"preferred_soc_max": 1.1},
        {"preferred_soc_min": True},
        {"power_deadband_kw": -1},
        {"power_deadband_kw": float("nan")},
        {"max_charge_power_kw": -1},
        {"max_discharge_power_kw": float("inf")},
    ],
)
def test_invalid_controller_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        make_controller(**values)


def test_preferred_range_must_fit_physical_soc_limits():
    controller = make_controller(preferred_soc_min=0.05)
    with pytest.raises(ValueError, match="within simulator SOC bounds"):
        controller.select_action(observation())


def test_only_causal_observation_features_are_used_and_decisions_repeat():
    controller = make_controller()
    base = observation(solar=200, demand=100)
    forecast = ForecastFeature(
        "campus_demand_kw",
        10000,
        "2026-01-01T02:00:00Z",
        "2026-01-01T01:00:00Z",
        "2026-01-01T01:00:00Z",
        "test_model",
    )
    with_forecast = observation(
        solar=200,
        demand=100,
        forecasts=(forecast,),
    )
    assert controller.select_action(base) == controller.select_action(with_forecast)
    assert controller.select_action(base) == controller.select_action(base)


def make_environment():
    provenance = {
        "source_id": "battery_preserving_fixture",
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
            (0, 200, 100),
            (1, 0, 500),
            (2, 0, 500),
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


def test_dispatch_enforces_physical_soc_and_balance_for_controller_actions():
    environment = make_environment()
    controller = make_controller()
    controller.reset()
    environment.reset()
    charge = environment.step(controller.select_action(environment.observation))
    assert charge.dispatch.requested_battery_power_kw == -50
    assert charge.dispatch.actual_battery_power_kw == -50
    assert charge.observation.state.battery_soc == pytest.approx(0.595)
    assert charge.dispatch.balance_residual_kw == pytest.approx(0, abs=1e-6)

    discharge = environment.step(controller.select_action(environment.observation))
    assert discharge.dispatch.requested_battery_power_kw == 50
    assert discharge.observation.state.battery_soc >= controller.preferred_soc_min
    assert discharge.observation.state.battery_soc <= 0.9
    assert discharge.dispatch.balance_residual_kw == pytest.approx(0, abs=1e-6)
    assert discharge.dispatch.energy_balance_residual_kwh == pytest.approx(0, abs=1e-6)


def test_no_renewable_surplus_means_no_grid_assisted_charge():
    controller = make_controller()
    low_state = observation(solar=0, wind=0, demand=100, soc=0.1)
    assert controller.select_action(low_state).requested_power_kw == 0
