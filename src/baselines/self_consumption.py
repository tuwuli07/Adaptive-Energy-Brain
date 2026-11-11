"""Renewable self-consumption baseline using causal lagged observations."""

import math
from numbers import Real

from src.baselines.base import BaselineController
from src.simulation.actions import BatteryAction
from src.simulation.state import AgentObservation


class SelfConsumptionController(BaselineController):
    """Request charging on observed surplus and discharging on observed deficit.

    The request magnitude is the absolute observed net-power difference outside
    the deadband. Dispatch remains responsible for physical limits and balance.
    """

    def __init__(self, power_deadband_kw: float = 0.0) -> None:
        if (
            isinstance(power_deadband_kw, bool)
            or not isinstance(power_deadband_kw, Real)
            or not math.isfinite(power_deadband_kw)
            or power_deadband_kw < 0
        ):
            raise ValueError("power_deadband_kw must be a finite nonnegative number.")
        self.power_deadband_kw = float(power_deadband_kw)

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        features = observation.features()
        net_renewable_power_kw = (
            features["solar_generation_kw"]
            + features["wind_generation_kw"]
            - features["campus_demand_kw"]
        )
        if abs(net_renewable_power_kw) <= self.power_deadband_kw:
            return BatteryAction(0.0)
        return BatteryAction(-net_renewable_power_kw)
