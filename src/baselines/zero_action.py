"""Baseline that requests no discretionary battery charge or discharge."""

from src.baselines.base import BaselineController
from src.simulation.actions import BatteryAction
from src.simulation.state import AgentObservation


class ZeroActionController(BaselineController):
    """Always request zero battery power and defer dispatch to the simulator."""

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        return BatteryAction(0.0)
