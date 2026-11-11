"""Price-aware controller thresholds, SOC reserve, and simulator integration."""

from dataclasses import replace

import pytest

from src.baselines import (
    PriceAwareController,
    available_controllers,
    create_controller,
)
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.renewables import SolarPV, WindGeneration
from src.simulation.state import SimulationState


def observation(*, price=0.1, soc=0.5):
    config = load_simulation_config()
    state = SimulationState(
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z",
        0,
        0,
        100,
        soc,
        price,
        0.04,
        config,
    )
    return state.observation(decision_timestamp_utc="2026-01-01T01:00:00Z")


def controller(**kwargs):
    settings = {
        "low_price_threshold": 0.12,
        "high_price_threshold": 0.18,
        "minimum_soc_reserve": 0.2,
        "degradation_cost_per_kwh_throughput": 0.02,
    }
    settings.update(kwargs)
    return PriceAwareController(
        **settings,
    )


def test_default_config_and_registry_selection():
    assert "price_aware" in available_controllers()
    selected = create_controller("price_aware")
    assert isinstance(selected, PriceAwareController)
    assert selected.low_price_threshold == pytest.approx(0.12)
    assert selected.high_price_threshold == pytest.approx(0.18)
    assert selected.minimum_soc_reserve == pytest.approx(0.2)


def test_low_high_and_neutral_price_actions():
    policy = controller()
    policy.reset()
    assert policy.select_action(observation(price=0.12)).requested_power_kw == -150
    assert policy.select_action(observation(price=0.15)).requested_power_kw == 0
    assert policy.select_action(observation(price=0.18)).requested_power_kw == pytest.approx(142.5)


def test_degradation_and_round_trip_losses_must_be_recovered():
    expensive_wear = controller(degradation_cost_per_kwh_throughput=1.0)
    assert expensive_wear.select_action(observation(price=0.1)).requested_power_kw == 0
    assert expensive_wear.select_action(observation(price=0.5)).requested_power_kw == 0

    narrow_spread = PriceAwareController(
        low_price_threshold=0.12,
        high_price_threshold=0.13,
        minimum_soc_reserve=0.2,
        degradation_cost_per_kwh_throughput=0.0,
    )
    assert narrow_spread.select_action(observation(price=0.12)).requested_power_kw == 0
    assert narrow_spread.select_action(observation(price=0.13)).requested_power_kw == 0


def test_minimum_soc_reserve_limits_discharge():
    policy = controller()
    assert policy.select_action(observation(price=0.2, soc=0.2)).requested_power_kw == 0
    assert policy.select_action(observation(price=0.2, soc=0.21)).requested_power_kw == pytest.approx(4.75)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low_price_threshold": -0.1},
        {"high_price_threshold": float("nan")},
        {"low_price_threshold": 0.2, "high_price_threshold": 0.2},
        {"minimum_soc_reserve": -0.1},
        {"minimum_soc_reserve": 1.1},
        {"minimum_soc_reserve": True},
        {"degradation_cost_per_kwh_throughput": -1},
        {"degradation_cost_per_kwh_throughput": float("inf")},
    ],
)
def test_invalid_price_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        controller(**kwargs)


def test_currency_mismatch_is_rejected():
    policy = controller()
    state = observation().state
    euro_config = replace(
        state.config,
        grid=replace(state.config.grid, currency_code="EUR"),
    )
    euro_state = replace(state, config=euro_config)
    with pytest.raises(ValueError, match="currency"):
        policy.select_action(
            euro_state.observation(decision_timestamp_utc="2026-01-01T01:00:00Z")
        )


def make_environment():
    provenance = {
        "source_id": "price_aware_fixture",
        "origin": "simulated",
        "synthetic": True,
        "source_metadata": {"note": "Unit test only"},
    }
    rows = [
        HourlyObservation(
            f"2026-01-01T{hour:02}:00:00Z",
            0,
            0,
            100,
            import_price,
            0.04,
            {"B01": 100},
            {"PV01": 0},
            {"W01": 0},
        )
        for hour, import_price in ((0, 0.1), (1, 0.5))
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


def test_grid_assisted_charge_uses_observed_price_and_dispatch_enforces_limits():
    environment = make_environment()
    policy = controller()
    policy.reset()
    environment.reset()
    action = policy.select_action(environment.observation)
    assert action.requested_power_kw == -150
    result = environment.step(action).dispatch
    assert result.requested_battery_power_kw == -150
    assert result.actual_battery_power_kw == -150
    assert result.grid.import_power_kw == pytest.approx(250)
    assert result.grid.import_power_kw <= 2000
    assert result.balance_residual_kw == pytest.approx(0, abs=1e-6)
    assert result.energy_balance_residual_kwh == pytest.approx(0, abs=1e-6)
    assert result.timestamp_utc == "2026-01-01T01:00:00Z"
