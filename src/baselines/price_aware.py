"""Causal threshold-based battery arbitrage controller."""

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path

import yaml

from src.baselines.base import BaselineController
from src.simulation.actions import BatteryAction, finite_number
from src.simulation.config import _UniqueKeyLoader
from src.simulation.metrics import load_cost_config
from src.simulation.state import AgentObservation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_CONFIG_PATH = ROOT / "configs" / "baselines.yaml"


@dataclass(frozen=True)
class PriceAwareConfig:
    version: int
    parameter_origin: str
    currency_code: str
    low_price_threshold: float
    high_price_threshold: float
    minimum_soc_reserve: float

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Baseline configuration version must be 1.")
        if self.parameter_origin not in {
            "hypothetical_example",
            "documented_specification",
            "mixed",
        }:
            raise ValueError("Baseline parameter_origin must be explicit.")
        if (
            not isinstance(self.currency_code, str)
            or len(self.currency_code) != 3
            or not self.currency_code.isascii()
            or not self.currency_code.isalpha()
            or self.currency_code.upper() != self.currency_code
        ):
            raise ValueError("currency_code must be an explicit three-letter code.")
        for name in ("low_price_threshold", "high_price_threshold"):
            value = finite_number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)
        if self.low_price_threshold >= self.high_price_threshold:
            raise ValueError("low_price_threshold must be less than high_price_threshold.")
        reserve = finite_number(self.minimum_soc_reserve, "minimum_soc_reserve")
        if not 0 <= reserve <= 1:
            raise ValueError("minimum_soc_reserve must be in [0, 1].")
        object.__setattr__(self, "minimum_soc_reserve", reserve)


def load_price_aware_config(path=None) -> PriceAwareConfig:
    """Load exact versioned price-aware settings with duplicate-key rejection."""
    config_path = Path(path) if path is not None else DEFAULT_BASELINE_CONFIG_PATH
    try:
        values = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        expected = {"version", "parameter_origin", "currency_code", "price_aware"}
        if not isinstance(values, dict) or frozenset(values) not in {
            frozenset(expected),
            frozenset(expected | {"battery_preserving"}),
        }:
            raise ValueError("Baseline configuration requires the documented fields.")
        price = values["price_aware"]
        if not isinstance(price, dict) or set(price) != {
            "low_price_threshold",
            "high_price_threshold",
            "minimum_soc_reserve",
        }:
            raise ValueError("price_aware requires exactly the documented fields.")
        return PriceAwareConfig(
            version=values["version"],
            parameter_origin=values["parameter_origin"],
            currency_code=values["currency_code"],
            **price,
        )
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid baseline configuration {config_path}: {exc}") from exc


class PriceAwareController(BaselineController):
    """Use lagged observed prices to request profitable charge/discharge.

    A charge/discharge pair is considered worthwhile only when high-price
    revenue after round-trip losses exceeds low-price energy cost plus the
    modeled AC-throughput degradation cost. Dispatch enforces actual limits.
    """

    def __init__(
        self,
        low_price_threshold: float | None = None,
        high_price_threshold: float | None = None,
        minimum_soc_reserve: float | None = None,
        *,
        degradation_cost_per_kwh_throughput: float | None = None,
        currency_code: str | None = None,
        config_path=None,
    ) -> None:
        config = load_price_aware_config(config_path)
        self.low_price_threshold = self._setting(
            low_price_threshold, config.low_price_threshold, "low_price_threshold"
        )
        self.high_price_threshold = self._setting(
            high_price_threshold, config.high_price_threshold, "high_price_threshold"
        )
        self.minimum_soc_reserve = self._setting(
            minimum_soc_reserve, config.minimum_soc_reserve, "minimum_soc_reserve"
        )
        self.currency_code = config.currency_code if currency_code is None else currency_code
        self._validate_currency(self.currency_code)
        if (
            self.low_price_threshold < 0
            or self.high_price_threshold < 0
            or self.low_price_threshold >= self.high_price_threshold
        ):
            raise ValueError(
                "Price thresholds must be nonnegative and low must be less than high."
            )
        if not 0 <= self.minimum_soc_reserve <= 1:
            raise ValueError("minimum_soc_reserve must be in [0, 1].")

        if degradation_cost_per_kwh_throughput is None:
            costs = load_cost_config()
            if costs.currency_code != self.currency_code:
                raise ValueError(
                    "Baseline and simulation cost currencies must match."
                )
            degradation_cost_per_kwh_throughput = (
                costs.battery_degradation_per_kwh_throughput
            )
        self.degradation_cost_per_kwh_throughput = finite_number(
            degradation_cost_per_kwh_throughput,
            "degradation_cost_per_kwh_throughput",
        )
        if self.degradation_cost_per_kwh_throughput < 0:
            raise ValueError(
                "degradation_cost_per_kwh_throughput must be nonnegative."
            )

    @staticmethod
    def _setting(value, default, name):
        if value is None:
            return default
        return finite_number(value, name)

    @staticmethod
    def _validate_currency(currency_code):
        if (
            not isinstance(currency_code, str)
            or len(currency_code) != 3
            or not currency_code.isascii()
            or not currency_code.isalpha()
            or currency_code.upper() != currency_code
        ):
            raise ValueError("currency_code must be an explicit three-letter code.")

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        if observation.currency_code != self.currency_code:
            raise ValueError(
                "Controller threshold/cost currency must match the observation currency."
            )
        features = observation.features()
        price = finite_number(features["grid_import_price"], "grid_import_price")
        efficiency = (
            features["charge_efficiency"] * features["discharge_efficiency"]
        )
        replacement_cost = (
            self.low_price_threshold
            + self.degradation_cost_per_kwh_throughput * (1 + efficiency)
        )
        if price <= self.low_price_threshold and self.high_price_threshold * efficiency > replacement_cost:
            return BatteryAction(-features["charge_limit_kw"])
        if price >= self.high_price_threshold and price * efficiency > replacement_cost:
            reserve = max(self.minimum_soc_reserve, features["soc_min"])
            available_energy_above_reserve = max(
                0.0,
                features["battery_energy_kwh"]
                - reserve * features["battery_capacity_kwh"],
            )
            reserve_limited_discharge_kw = (
                available_energy_above_reserve
                * features["discharge_efficiency"]
                / features["time_step_hours"]
            )
            discharge_kw = min(
                features["discharge_limit_kw"],
                reserve_limited_discharge_kw,
            )
            return BatteryAction(discharge_kw)
        return BatteryAction(0.0)
