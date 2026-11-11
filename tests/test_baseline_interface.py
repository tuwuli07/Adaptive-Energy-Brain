"""Controller lifecycle, registration, and simulator action compatibility."""

from dataclasses import dataclass

import pytest

from src.baselines import (
    BaselineController,
    BaselineControllerError,
    ControllerRegistry,
    create_controller,
    register_controller,
)
from src.baselines.registry import ControllerRegistryError
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.renewables import SolarPV, WindGeneration
from src.simulation.state import AgentObservation, SimulationState


@dataclass
class ConstantController(BaselineController):
    requested_power_kw: float = 0.0
    reset_count: int = 0

    def _reset(self) -> None:
        self.reset_count = 0

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        self.reset_count += 1
        return BatteryAction(self.requested_power_kw)


class InvalidActionController(BaselineController):
    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        return 0.0


def observation() -> AgentObservation:
    config = load_simulation_config()
    state = SimulationState(
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z",
        20,
        10,
        100,
        config.battery.initial_soc,
        0.1,
        0.04,
        config,
    )
    return state.observation(decision_timestamp_utc="2026-01-01T01:00:00Z")


def test_reset_and_select_action_use_existing_causal_observation_and_action():
    controller = ConstantController(25)
    controller.select_action(observation())
    assert controller.reset_count == 1
    controller.reset()
    assert controller.reset_count == 0
    action = controller.select_action(observation())
    assert action == BatteryAction(25)
    assert action.dispatch_kwargs() == {"requested_battery_power_kw": 25}


def test_input_and_returned_action_are_validated():
    controller = ConstantController()
    with pytest.raises(BaselineControllerError, match="AgentObservation"):
        controller.select_action({})
    with pytest.raises(BaselineControllerError, match="BatteryAction"):
        InvalidActionController().select_action(observation())


def test_registry_registers_creates_and_rejects_unknown_or_invalid_entries():
    registry = ControllerRegistry()
    registry.register("example", lambda: ConstantController(10))
    assert registry.names() == ("example",)
    assert registry.create("example").select_action(observation()) == BatteryAction(10)
    with pytest.raises(ControllerRegistryError, match="already registered"):
        registry.register("example", ConstantController)
    with pytest.raises(ControllerRegistryError, match="Unknown controller"):
        registry.create("missing")
    with pytest.raises(ControllerRegistryError, match="Controller names"):
        registry.register("Bad Name", ConstantController)
    with pytest.raises(ControllerRegistryError, match="callable"):
        registry.register("invalid", None)
    registry.register("wrong-factory", lambda: object())
    with pytest.raises(ControllerRegistryError, match="must return"):
        registry.create("wrong-factory")


def test_package_registry_can_select_registered_controller():
    name = "test-interface-controller"
    register_controller(name, lambda: ConstantController(0))
    try:
        controller = create_controller(name)
        assert controller.select_action(observation()) == BatteryAction(0)
    finally:
        from src.baselines.registry import _registry

        _registry._factories.pop(name, None)


def test_controller_action_is_accepted_by_microgrid_environment():
    provenance = {
        "source_id": "baseline_interface_fixture",
        "origin": "simulated",
        "synthetic": True,
        "source_metadata": {"note": "Unit test only"},
    }
    rows = [
        HourlyObservation(
            f"2026-01-01T{hour:02}:00:00Z",
            20,
            10,
            100,
            0.1,
            0.04,
            {"B01": 100},
            {"PV01": 20},
            {"W01": 10},
        )
        for hour in range(2)
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
    environment = MicrogridEnvironment(
        data,
        [
            SolarPV("PV01", 300, provenance, capacity_origin="hypothetical_example"),
            WindGeneration("W01", 150, provenance, capacity_origin="hypothetical_example"),
        ],
        CampusDemand(provenance, building_ids=["B01"]),
    )
    controller = ConstantController(-25)
    controller.reset()
    environment.reset()
    result = environment.step(controller.select_action(environment.observation))
    assert result.dispatch.requested_battery_power_kw == -25
    assert result.dispatch.timestamp_utc == "2026-01-01T01:00:00Z"
