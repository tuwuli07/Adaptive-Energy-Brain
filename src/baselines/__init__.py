"""Baseline controller interfaces, registry, and implemented strategies."""

from src.baselines.base import BaselineController, BaselineControllerError
from src.baselines.battery_preserving import (
    BatteryPreservingConfig,
    BatteryPreservingController,
    load_battery_preserving_config,
)
from src.baselines.price_aware import PriceAwareController, PriceAwareConfig, load_price_aware_config
from src.baselines.registry import (
    ControllerRegistry,
    available_controllers,
    create_controller,
    register_controller,
)
from src.baselines.self_consumption import SelfConsumptionController
from src.baselines.zero_action import ZeroActionController

__all__ = [
    "BaselineController",
    "BaselineControllerError",
    "BatteryPreservingConfig",
    "BatteryPreservingController",
    "ControllerRegistry",
    "PriceAwareConfig",
    "PriceAwareController",
    "SelfConsumptionController",
    "ZeroActionController",
    "available_controllers",
    "create_controller",
    "load_price_aware_config",
    "load_battery_preserving_config",
    "register_controller",
]
