"""Conservative SOC-range controller with renewable-only charging."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.baselines.base import BaselineController
from src.simulation.actions import BatteryAction, finite_number
from src.simulation.config import _UniqueKeyLoader
from src.simulation.state import AgentObservation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_CONFIG_PATH = ROOT / "configs" / "baselines.yaml"


@dataclass(frozen=True)
class BatteryPreservingConfig:
    preferred_soc_min: float
    preferred_soc_max: float
    power_deadband_kw: float
    max_charge_power_kw: float
    max_discharge_power_kw: float

    def __post_init__(self):
        for name in ("preferred_soc_min", "preferred_soc_max"):
            value = finite_number(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
            object.__setattr__(self, name, value)
        if self.preferred_soc_min >= self.preferred_soc_max:
            raise ValueError("preferred_soc_min must be less than preferred_soc_max.")
        for name in ("power_deadband_kw", "max_charge_power_kw", "max_discharge_power_kw"):
            value = finite_number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)


def load_battery_preserving_config(path=None) -> BatteryPreservingConfig:
    """Load battery-preserving settings from the shared baseline YAML."""
    config_path = Path(path) if path is not None else DEFAULT_BASELINE_CONFIG_PATH
    try:
        values = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if not isinstance(values, dict) or set(values) != {
            "version",
            "parameter_origin",
            "currency_code",
            "price_aware",
            "battery_preserving",
        }:
            raise ValueError(
                "Baseline configuration requires the documented top-level fields."
            )
        if type(values["version"]) is not int or values["version"] != 1:
            raise ValueError("Baseline configuration version must be 1.")
        if values["parameter_origin"] not in {
            "hypothetical_example",
            "documented_specification",
            "mixed",
        }:
            raise ValueError("Baseline parameter_origin must be explicit.")
        currency = values["currency_code"]
        if (
            not isinstance(currency, str)
            or len(currency) != 3
            or not currency.isascii()
            or not currency.isalpha()
            or currency.upper() != currency
        ):
            raise ValueError("currency_code must be an explicit three-letter code.")
        settings = values["battery_preserving"]
        required = {
            "preferred_soc_min",
            "preferred_soc_max",
            "power_deadband_kw",
            "max_charge_power_kw",
            "max_discharge_power_kw",
        }
        if not isinstance(settings, dict) or set(settings) != required:
            raise ValueError(
                "battery_preserving requires exactly the documented fields."
            )
        return BatteryPreservingConfig(**settings)
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid baseline configuration {config_path}: {exc}"
        ) from exc


class BatteryPreservingController(BaselineController):
    """Avoid cycling except for meaningful observed surplus/deficit.

    Charging is requested only from observed renewable surplus. Discharge is
    limited to energy above the preferred SOC floor. Requests are conservatively
    capped; simulator dispatch remains authoritative for physical feasibility.
    """

    def __init__(
        self,
        preferred_soc_min: float | None = None,
        preferred_soc_max: float | None = None,
        power_deadband_kw: float | None = None,
        max_charge_power_kw: float | None = None,
        max_discharge_power_kw: float | None = None,
        *,
        config_path=None,
    ) -> None:
        config = load_battery_preserving_config(config_path)
        self.preferred_soc_min = self._setting(
            preferred_soc_min, config.preferred_soc_min, "preferred_soc_min"
        )
        self.preferred_soc_max = self._setting(
            preferred_soc_max, config.preferred_soc_max, "preferred_soc_max"
        )
        self.power_deadband_kw = self._setting(
            power_deadband_kw, config.power_deadband_kw, "power_deadband_kw"
        )
        self.max_charge_power_kw = self._setting(
            max_charge_power_kw, config.max_charge_power_kw, "max_charge_power_kw"
        )
        self.max_discharge_power_kw = self._setting(
            max_discharge_power_kw,
            config.max_discharge_power_kw,
            "max_discharge_power_kw",
        )
        BatteryPreservingConfig(
            self.preferred_soc_min,
            self.preferred_soc_max,
            self.power_deadband_kw,
            self.max_charge_power_kw,
            self.max_discharge_power_kw,
        )

    @staticmethod
    def _setting(value, default, name):
        if value is None:
            return default
        return finite_number(value, name)

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        features = observation.features()
        if (
            self.preferred_soc_min < features["soc_min"]
            or self.preferred_soc_max > features["soc_max"]
        ):
            raise ValueError(
                "Preferred SOC range must remain within simulator SOC bounds."
            )

        net_renewable_power_kw = (
            features["solar_generation_kw"]
            + features["wind_generation_kw"]
            - features["campus_demand_kw"]
        )
        if abs(net_renewable_power_kw) <= self.power_deadband_kw:
            return BatteryAction(0.0)

        soc = features["battery_soc"]
        duration_hours = features["time_step_hours"]
        if net_renewable_power_kw > self.power_deadband_kw:
            if soc >= self.preferred_soc_max:
                return BatteryAction(0.0)
            target_headroom_kw = (
                (self.preferred_soc_max * features["battery_capacity_kwh"]
                 - features["battery_energy_kwh"])
                / (features["charge_efficiency"] * duration_hours)
            )
            charge_kw = min(
                net_renewable_power_kw,
                features["charge_limit_kw"],
                self.max_charge_power_kw,
                max(0.0, target_headroom_kw),
            )
            return BatteryAction(-charge_kw)

        if soc <= self.preferred_soc_min:
            return BatteryAction(0.0)
        reserve_energy_kwh = (
            self.preferred_soc_min * features["battery_capacity_kwh"]
        )
        reserve_limited_discharge_kw = (
            max(0.0, features["battery_energy_kwh"] - reserve_energy_kwh)
            * features["discharge_efficiency"]
            / duration_hours
        )
        discharge_kw = min(
            abs(net_renewable_power_kw),
            features["discharge_limit_kw"],
            self.max_discharge_power_kw,
            reserve_limited_discharge_kw,
        )
        return BatteryAction(discharge_kw)
