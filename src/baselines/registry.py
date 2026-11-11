"""Name-based construction for independently implemented controllers."""

import re
from collections.abc import Callable

from src.baselines.base import BaselineController
from src.baselines.battery_preserving import BatteryPreservingController
from src.baselines.price_aware import PriceAwareController
from src.baselines.self_consumption import SelfConsumptionController
from src.baselines.zero_action import ZeroActionController


class ControllerRegistryError(ValueError):
    """Invalid controller name, registration, or factory result."""


class ControllerRegistry:
    """Map stable names to zero-argument controller factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., BaselineController]] = {}

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ControllerRegistryError(
                "Controller names must match [a-z][a-z0-9_-]*."
            )
        return name

    def register(
        self,
        name: str,
        factory: Callable[..., BaselineController],
        *,
        replace: bool = False,
    ) -> None:
        """Register a factory; duplicate names fail unless replacement is explicit."""
        name = self._validate_name(name)
        if not callable(factory):
            raise ControllerRegistryError("Controller factory must be callable.")
        if name in self._factories and not replace:
            raise ControllerRegistryError(
                f"Controller {name!r} is already registered."
            )
        self._factories[name] = factory

    def create(self, name: str, **kwargs) -> BaselineController:
        """Create a fresh controller by name and validate its interface."""
        name = self._validate_name(name)
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "(none)"
            raise ControllerRegistryError(
                f"Unknown controller {name!r}; registered controllers: {available}."
            ) from exc
        controller = factory(**kwargs)
        if not isinstance(controller, BaselineController):
            raise ControllerRegistryError(
                f"Factory for {name!r} must return a BaselineController."
            )
        return controller

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._factories))


_registry = ControllerRegistry()
_registry.register("battery_preserving", BatteryPreservingController)
_registry.register("price_aware", PriceAwareController)
_registry.register("self_consumption", SelfConsumptionController)
_registry.register("zero_action", ZeroActionController)


def register_controller(
    name: str,
    factory: Callable[..., BaselineController],
    *,
    replace: bool = False,
) -> None:
    """Register a controller factory in the package-level registry."""
    _registry.register(name, factory, replace=replace)


def create_controller(name: str, **kwargs) -> BaselineController:
    """Create a fresh controller from the package-level registry."""
    return _registry.create(name, **kwargs)


def available_controllers() -> tuple[str, ...]:
    """Return names available from the package-level registry."""
    return _registry.names()
