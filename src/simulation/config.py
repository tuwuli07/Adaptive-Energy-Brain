"""Typed, immutable simulator hardware/policy configuration; no physics logic.

load_simulation_config() reads configs/simulation.yaml using the established
versioned YAML convention. validate_simulation_config(mapping) supports callers
with an already-parsed mapping. Both return SimulationConfig or raise
SimulationConfigError with a field path. No defaults conceal missing fields.

Positive finite time steps, proportional curtailment allocation and explicit
error/record unserved-load policies are supported. Part 2 replay inputs remain hourly; a future
engine must match its step to the input interval, never silently resample.
Runtime episode feasibility, dispatch,
SOC transitions, tariffs and source authenticity are outside this loader.
source_reference is provenance text, not an instruction to load another file.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import re

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/simulation.yaml"


class SimulationConfigError(ValueError):
    """Configuration cannot safely be used by a future simulator."""


@dataclass(frozen=True)
class BatteryConfig:
    asset_id: str
    capacity_kwh: float
    initial_soc: float
    soc_min: float
    soc_max: float
    charge_limit_kw: float
    discharge_limit_kw: float
    charge_efficiency: float
    discharge_efficiency: float


@dataclass(frozen=True)
class GridConfig:
    asset_id: str
    import_limit_kw: float
    export_limit_kw: float
    currency_code: str
    missing_export_price_handling: str
    allow_negative_prices: bool


@dataclass(frozen=True)
class CurtailmentConfig:
    enabled: bool
    allocation: str


@dataclass(frozen=True)
class UnservedLoadConfig:
    handling: str


@dataclass(frozen=True)
class SimulationConfig:
    version: int
    parameter_origin: str
    source_reference: str
    time_step_hours: float
    battery: BatteryConfig
    grid: GridConfig
    renewable_curtailment: CurtailmentConfig
    unserved_load: UnservedLoadConfig


class _UniqueKeyLoader(yaml.SafeLoader):
    """Refuse YAML duplicate keys rather than silently taking the last value."""


def _unique_mapping(loader, node):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            if key in result:
                raise SimulationConfigError(
                    f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}."
                )
            result[key] = loader.construct_object(value_node, deep=True)
        except TypeError as exc:
            raise SimulationConfigError("YAML configuration keys must be scalar text.") from exc
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _mapping(value, fields, path):
    if not isinstance(value, Mapping):
        raise SimulationConfigError(f"{path}: expected a mapping.")
    missing = set(fields) - set(value)
    if missing:
        raise SimulationConfigError(
            f"{path}: missing required parameters {sorted(missing)}."
        )
    unknown = set(value) - set(fields)
    if unknown:
        raise SimulationConfigError(
            f"{path}: unknown parameters {sorted(unknown, key=str)}."
        )
    return value


def _number(value, path, *, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationConfigError(f"{path}: expected a finite number, not {value!r}.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise SimulationConfigError(f"{path}: number must be finite.") from exc
    if not math.isfinite(number):
        raise SimulationConfigError(f"{path}: number must be finite.")
    if positive and number <= 0:
        raise SimulationConfigError(f"{path}: must be greater than zero.")
    if minimum is not None and number < minimum:
        raise SimulationConfigError(f"{path}: must be >= {minimum}.")
    if maximum is not None and number > maximum:
        raise SimulationConfigError(f"{path}: must be <= {maximum}.")
    return number


def _text(value, path):
    if not isinstance(value, str) or not value.strip():
        raise SimulationConfigError(f"{path}: expected nonempty text.")
    return value


def validate_simulation_config(config):
    """Validate types/ranges/relationships without modifying the source mapping.

    Capacity must be positive; power limits may be zero. SOC is fractional and
    endpoints are inclusive for initial SOC, while min must be strictly below
    max. Efficiencies are in (0, 1]. Numeric strings, booleans used as numbers,
    NaN and infinities are rejected. Zero grid limits do not imply that an
    arbitrary demand profile will be feasible; check that at episode reset.
    """
    root = _mapping(config, (
        "version", "parameter_origin", "source_reference", "time_step_hours",
        "battery", "grid", "renewable_curtailment", "unserved_load",
    ), "simulation")
    if type(root["version"]) is not int or root["version"] != 1:
        raise SimulationConfigError("simulation.version: expected integer 1.")
    origin = _text(root["parameter_origin"], "simulation.parameter_origin")
    if origin not in {"hypothetical_example", "documented_specification", "mixed"}:
        raise SimulationConfigError(
            "simulation.parameter_origin: expected hypothetical_example, "
            "documented_specification or mixed."
        )
    reference = _text(root["source_reference"], "simulation.source_reference")
    step = _number(root["time_step_hours"], "simulation.time_step_hours", positive=True)
    battery = _mapping(root["battery"], (
        "asset_id", "capacity_kwh", "initial_soc", "soc_min", "soc_max",
        "charge_limit_kw", "discharge_limit_kw", "charge_efficiency",
        "discharge_efficiency",
    ), "simulation.battery")
    battery_id = _text(battery["asset_id"], "simulation.battery.asset_id")
    capacity = _number(battery["capacity_kwh"], "simulation.battery.capacity_kwh", positive=True)
    soc = {field: _number(battery[field], f"simulation.battery.{field}", minimum=0, maximum=1)
           for field in ("initial_soc", "soc_min", "soc_max")}
    if soc["soc_min"] >= soc["soc_max"]:
        raise SimulationConfigError(
            "simulation.battery.soc_min: must be strictly less than soc_max."
        )
    if not soc["soc_min"] <= soc["initial_soc"] <= soc["soc_max"]:
        raise SimulationConfigError(
            "simulation.battery.initial_soc: must be within [soc_min, soc_max]."
        )
    powers = {field: _number(battery[field], f"simulation.battery.{field}", minimum=0)
              for field in ("charge_limit_kw", "discharge_limit_kw")}
    efficiencies = {field: _number(battery[field], f"simulation.battery.{field}", positive=True, maximum=1)
                    for field in ("charge_efficiency", "discharge_efficiency")}
    grid = _mapping(root["grid"], (
        "asset_id", "import_limit_kw", "export_limit_kw", "currency_code",
        "missing_export_price_handling", "allow_negative_prices",
    ), "simulation.grid")
    grid_id = _text(grid["asset_id"], "simulation.grid.asset_id")
    if battery_id == grid_id:
        raise SimulationConfigError("simulation.grid.asset_id: must differ from battery.asset_id.")
    imports = _number(grid["import_limit_kw"], "simulation.grid.import_limit_kw", minimum=0)
    exports = _number(grid["export_limit_kw"], "simulation.grid.export_limit_kw", minimum=0)
    currency = _text(grid["currency_code"], "simulation.grid.currency_code")
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise SimulationConfigError("simulation.grid.currency_code: expected a three-uppercase-letter currency code.")
    missing_price = grid["missing_export_price_handling"]
    if missing_price not in ("error", "disable_export", "zero_compensation"):
        raise SimulationConfigError("simulation.grid.missing_export_price_handling: expected error, disable_export or zero_compensation.")
    if type(grid["allow_negative_prices"]) is not bool:
        raise SimulationConfigError("simulation.grid.allow_negative_prices: expected a boolean.")
    curtailment = _mapping(root["renewable_curtailment"], ("enabled", "allocation"), "simulation.renewable_curtailment")
    if type(curtailment["enabled"]) is not bool:
        raise SimulationConfigError("simulation.renewable_curtailment.enabled: expected a boolean.")
    if curtailment["allocation"] != "proportional":
        raise SimulationConfigError(
            "simulation.renewable_curtailment.allocation: only proportional is supported."
        )
    unserved = _mapping(root["unserved_load"], ("handling",), "simulation.unserved_load")
    if unserved["handling"] not in ("error", "record"):
        raise SimulationConfigError(
            "simulation.unserved_load.handling: expected error or record; "
            "no service priorities or monetary shortage penalty is inferred."
        )
    return SimulationConfig(
        version=1, parameter_origin=origin, source_reference=reference,
        time_step_hours=step,
        battery=BatteryConfig(asset_id=battery_id, capacity_kwh=capacity, **soc,
                              **powers, **efficiencies),
        grid=GridConfig(asset_id=grid_id, import_limit_kw=imports, export_limit_kw=exports,
                        currency_code=currency, missing_export_price_handling=missing_price,
                        allow_negative_prices=grid["allow_negative_prices"]),
        renewable_curtailment=CurtailmentConfig(**curtailment),
        unserved_load=UnservedLoadConfig(**unserved),
    )


def load_simulation_config(path=DEFAULT_CONFIG_PATH):
    """Read safe, duplicate-key-checked YAML and return immutable configuration."""
    try:
        contents = Path(path).read_text(encoding="utf-8")
        config = yaml.load(contents, Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SimulationConfigError(f"Cannot load simulation configuration {path}: {exc}") from exc
    return validate_simulation_config(config)
