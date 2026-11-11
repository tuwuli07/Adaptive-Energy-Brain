"""Shared simulator-independent interface for baseline and future controllers."""

from abc import ABC, abstractmethod

from src.simulation.actions import BatteryAction
from src.simulation.state import AgentObservation


class BaselineControllerError(ValueError):
    """Invalid controller input or returned simulator action."""


class BaselineController(ABC):
    """Choose one physical battery request from a causal observation.

    Implementations receive no simulator, replay-data, or metric object. Stateful
    implementations should clear episode-local state in ``_reset``.
    """

    def reset(self) -> None:
        """Reset episode-local controller state."""
        self._reset()

    def _reset(self) -> None:
        """Optional reset hook for stateful controllers."""

    def select_action(self, observation: AgentObservation) -> BatteryAction:
        """Return a validated action accepted directly by ``env.step``."""
        if not isinstance(observation, AgentObservation):
            raise BaselineControllerError(
                "observation must be an AgentObservation from the simulator."
            )
        action = self._select_action(observation)
        if not isinstance(action, BatteryAction):
            raise BaselineControllerError(
                "Controllers must return a BatteryAction."
            )
        try:
            return BatteryAction(action.requested_power_kw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BaselineControllerError(
                f"Controller returned an invalid BatteryAction: {exc}"
            ) from exc

    @abstractmethod
    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        """Implement a decision using only this causal observation."""
