"""Post-dispatch physical and economic accounting; no rewards or control logic.

Available renewable energy = available AC kW * hours (includes curtailed potential).
Utilized renewable energy = (available - curtailed) * hours delivered to the bus,
including export/charging. This is NOT campus self-consumption or renewable load
share: battery discharge is never counted again as renewable generation. No stored
energy source attribution is attempted. Utilization is None when availability is
zero, including an empty episode; episode ratios use summed energies, not means.

Degradation proxy = (AC charge kWh + AC discharge kWh) * configured coefficient.
This linear throughput proxy is NOT validated electrochemical aging; it ignores
calendar age, temperature, depth of discharge and chemistry. Tariffs come from
grid transactions; separate labeled assumptions add degradation and optional
unserved-demand accounting costs. Net = purchase - revenue + modeled costs.
Adjustments are safety interventions, not automatically physical violations.
Only committed results are accepted; rejected dispatch attempts must be logged
by the caller and are never represented as successfully executed energy flows.
"""

from dataclasses import dataclass, fields
from copy import copy
from datetime import datetime, timedelta
from pathlib import Path
import re

import yaml

from src.simulation._validation import power_kw, timestamp_utc
from src.simulation.actions import finite_number
from src.simulation.config import _UniqueKeyLoader
from src.simulation.dispatch import DispatchResult


@dataclass(frozen=True)
class CostConfig:
    version: int
    currency_code: str
    parameter_origin: str
    battery_degradation_per_kwh_throughput: float
    unserved_demand_per_kwh: float

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Cost configuration version must be 1.")
        if not isinstance(self.currency_code, str) or not re.fullmatch(r"[A-Z]{3}", self.currency_code):
            raise ValueError("currency_code must be an explicit three-letter currency code.")
        if self.parameter_origin not in {"hypothetical_example", "documented_specification", "mixed"}:
            raise ValueError("Economic parameter_origin must be explicit.")
        for name in ("battery_degradation_per_kwh_throughput", "unserved_demand_per_kwh"):
            value = finite_number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative currency/kWh.")
            object.__setattr__(self, name, value)


def load_cost_config(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "configs/simulation_costs.yaml"
    try:
        values = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if not isinstance(values, dict) or set(values) != {f.name for f in fields(CostConfig)}:
            raise ValueError("Cost configuration requires exactly the documented fields.")
        return CostConfig(**values)
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cost configuration {path}: {exc}") from exc


@dataclass(frozen=True)
class MetricValues:
    """All energies in kWh, all costs in the report's explicit currency."""

    import_cost: float = 0.0
    export_revenue: float = 0.0
    battery_degradation_cost: float = 0.0
    unserved_demand_cost: float = 0.0
    net_operating_cost: float = 0.0
    solar_available_energy_kwh: float = 0.0
    wind_available_energy_kwh: float = 0.0
    renewable_available_energy_kwh: float = 0.0
    renewable_utilized_energy_kwh: float = 0.0
    renewable_curtailed_energy_kwh: float = 0.0
    battery_charge_energy_kwh: float = 0.0
    battery_discharge_energy_kwh: float = 0.0
    battery_throughput_kwh: float = 0.0
    grid_import_energy_kwh: float = 0.0
    grid_export_energy_kwh: float = 0.0
    demand_energy_kwh: float = 0.0
    served_demand_energy_kwh: float = 0.0
    unserved_demand_energy_kwh: float = 0.0

    @property
    def renewable_utilization(self):
        if self.renewable_available_energy_kwh == 0:
            return None
        return self.renewable_utilized_energy_kwh / self.renewable_available_energy_kwh


@dataclass(frozen=True)
class ConstraintViolation:
    code: str
    actual: float
    limit: float
    unit: str


@dataclass(frozen=True)
class StepMetrics:
    timestamp_utc: str
    time_step_hours: float
    currency_code: str
    cost_assumptions: CostConfig
    values: MetricValues
    adjustments: tuple
    violations: tuple[ConstraintViolation, ...]
    power_balance_residual_kw: float
    energy_balance_residual_kwh: float


def calculate_step_metrics(result, costs=None):
    """Calculate once from executed flows, without mutating simulation state."""
    costs = load_cost_config() if costs is None else costs
    if not isinstance(costs, CostConfig) or not isinstance(result, DispatchResult):
        raise ValueError("Expected CostConfig and committed DispatchResult values.")
    grid, battery, renewable = result.grid, result.battery, result.renewables
    if grid.currency_code != costs.currency_code or grid.price_unit != f"{costs.currency_code}/kWh":
        raise ValueError("Grid and economic currencies/price units must match; no implicit FX.")
    dt = power_kw(grid.time_step_hours, "time_step_hours", positive=True)
    if dt != battery.time_step_hours:
        raise ValueError("Battery and grid time-step durations must agree.")
    if timestamp_utc(result.timestamp_utc) != timestamp_utc(grid.timestamp_utc):
        raise ValueError("Grid and dispatch timestamps must agree.")
    tolerance = 1e-6

    def close(actual, expected, name):
        if abs(finite_number(actual, name) - expected) > tolerance:
            raise ValueError(f"Inconsistent {name}: {actual} versus expected {expected}.")

    for name, energy, power in (
        ("grid import", grid.imported_energy_kwh, grid.import_power_kw),
        ("grid export", grid.exported_energy_kwh, grid.export_power_kw),
        ("battery charge", battery.charge_energy_kwh, battery.charge_power_kw),
        ("battery discharge", battery.discharge_energy_kwh, battery.discharge_power_kw),
    ):
        close(power_kw(energy, name), power_kw(power, name) * dt, name)
    available = power_kw(renewable.total_available_kw, "renewable availability") * dt
    curtailed = power_kw(renewable.total_curtailed_kw, "curtailment") * dt
    utilized = power_kw(renewable.total_used_kw, "renewable utilization") * dt
    if curtailed > available or utilized > available:
        raise ValueError("Renewable utilization/curtailment exceeds availability.")
    close(available, utilized + curtailed, "renewable accounting")
    solar = sum(o.available_generation_kw * dt for o in result.renewable_outputs if o.generation_type == "solar")
    wind = sum(o.available_generation_kw * dt for o in result.renewable_outputs if o.generation_type == "wind")
    close(available, solar + wind, "solar/wind total")
    throughput = battery.charge_energy_kwh + battery.discharge_energy_kwh
    close(battery.throughput_increment_kwh, throughput, "battery throughput")
    unserved = power_kw(result.unserved_demand_kw, "unserved demand") * dt
    served = power_kw(result.served_demand_kw, "served demand") * dt
    demand = power_kw(result.demand.campus_load_kw, "campus demand") * dt
    close(result.unserved_energy_kwh, unserved, "unserved energy")
    close(demand, served + unserved, "demand accounting")
    purchase = finite_number(grid.import_cost, "import cost")
    revenue = finite_number(grid.export_revenue, "export revenue")
    close(purchase, grid.imported_energy_kwh * finite_number(grid.import_price, "import price")
          if grid.imported_energy_kwh else 0, "import tariff cost")
    close(revenue, grid.exported_energy_kwh * finite_number(grid.effective_export_price, "export price")
          if grid.exported_energy_kwh else 0, "export tariff revenue")
    close(grid.net_cost, purchase - revenue, "grid net cost")
    degradation = throughput * costs.battery_degradation_per_kwh_throughput
    shortage_cost = unserved * costs.unserved_demand_per_kwh
    values = MetricValues(purchase, revenue, degradation, shortage_cost,
                          purchase - revenue + degradation + shortage_cost,
                          solar, wind, available, utilized, curtailed,
                          battery.charge_energy_kwh, battery.discharge_energy_kwh,
                          throughput, grid.imported_energy_kwh, grid.exported_energy_kwh,
                          demand, served, unserved)
    for f in fields(values):
        finite_number(getattr(values, f.name), f.name)
    violations = []

    def upper(code, actual, limit, unit, slack=tolerance):
        finite_number(actual, code)
        if actual > limit + slack:
            violations.append(ConstraintViolation(code, actual, limit, unit))

    config = result.metadata["config"]
    upper("grid_import_limit", grid.import_power_kw, config["grid"]["import_limit_kw"], "kW")
    upper("grid_export_limit", grid.export_power_kw, config["grid"]["export_limit_kw"], "kW")
    upper("battery_charge_limit", battery.charge_power_kw, config["battery"]["charge_limit_kw"], "kW")
    upper("battery_discharge_limit", battery.discharge_power_kw, config["battery"]["discharge_limit_kw"], "kW")
    for state in (battery.state_before, battery.state_after):
        upper("soc_max", state.battery_soc, config["battery"]["soc_max"], "fraction")
        if state.battery_soc < config["battery"]["soc_min"] - tolerance:
            violations.append(ConstraintViolation("soc_min", state.battery_soc,
                                                  config["battery"]["soc_min"], "fraction"))
        upper("soc_energy_consistency", abs(
            state.battery_energy_kwh - state.battery_soc * config["battery"]["capacity_kwh"]),
            tolerance, "kWh", slack=0)
    residual = grid.import_power_kw + renewable.total_used_kw + battery.discharge_power_kw - result.served_demand_kw - battery.charge_power_kw - grid.export_power_kw
    energy_residual = (grid.imported_energy_kwh + utilized + battery.state_before.battery_energy_kwh
                       + battery.rounding_adjustment_kwh - served - grid.exported_energy_kwh
                       - battery.state_after.battery_energy_kwh - battery.efficiency_loss_kwh)
    close(result.balance_residual_kw, residual, "reported power residual")
    close(result.energy_balance_residual_kwh, energy_residual, "reported energy residual")
    upper("power_balance", abs(residual), result.metadata["power_balance_tolerance_kw"], "kW", slack=0)
    upper("energy_balance", abs(energy_residual), result.metadata["energy_balance_tolerance_kwh"], "kWh", slack=0)
    for code, first, second in (("simultaneous_grid_flow", grid.import_power_kw, grid.export_power_kw),
                                ("simultaneous_battery_flow", battery.charge_power_kw, battery.discharge_power_kw)):
        upper(code, min(first, second), 0, "kW")
    return StepMetrics(timestamp_utc(result.timestamp_utc), dt, costs.currency_code, costs,
                       values, tuple(result.adjustments), tuple(violations), residual, energy_residual)


class MetricsTracker:
    """Append chronological executed steps once; failed appends leave totals intact."""

    def __init__(self, costs=None):
        self.costs = load_cost_config() if costs is None else costs
        if not isinstance(self.costs, CostConfig):
            raise ValueError("costs must be a CostConfig.")
        self._steps = []
        self._totals = MetricValues()

    @property
    def steps(self):
        return tuple(self._steps)

    def fork(self):
        """Independent append candidate; immutable records/totals may be shared."""
        candidate = copy(self)
        candidate._steps = self._steps.copy()
        return candidate

    @property
    def totals(self):
        return self._totals

    @property
    def constraint_records(self):
        return tuple((s.timestamp_utc, s.adjustments, s.violations) for s in self._steps
                     if s.adjustments or s.violations)

    def record(self, result):
        step = calculate_step_metrics(result, self.costs)
        if self._steps:
            previous = self._steps[-1]
            expected = datetime.fromisoformat(previous.timestamp_utc.replace("Z", "+00:00")) + timedelta(hours=previous.time_step_hours)
            if datetime.fromisoformat(step.timestamp_utc.replace("Z", "+00:00")) != expected:
                raise ValueError("Metrics require consecutive steps; duplicate or discontinuous timestamp.")
        totals = MetricValues(**{
            f.name: finite_number(getattr(self._totals, f.name) + getattr(step.values, f.name), f.name)
            for f in fields(MetricValues)
        })
        self._steps.append(step)
        self._totals = totals
        return step
