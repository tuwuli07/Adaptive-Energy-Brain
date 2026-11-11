"""Deterministic hourly feasibility dispatch, without AI or cost optimization.

Battery request sign: positive discharges, negative charges. Grid sign: positive
imports, negative exports. Requests need not equal applied actions; every change
is recorded. Fixed building demand and available renewable data are immutable.

Priority: retain requested renewable curtailment when feasible, then choose the
nearest feasible battery command. Restore voluntary curtailment to relieve a
deficit. If all supply is still inadequate, error (default) or record unmet load
with maximum feasible discharge/import. Curtail unavoidable surplus only when
enabled, proportionally across used asset powers in sorted asset-ID order.
No automatic battery charging occurs when the requested action and grid export
already balance the system; safety constraints can force a different action.

Bus balance: import + used renewables + discharge = served demand + charge + export.
Unserved = original demand - served demand, never a fictitious power source.
Power and full-system energy residuals are returned, not silently zeroed. Absolute
defaults are 1e-6 kW and 1e-6 kWh. Battery rounding is explicitly accounted for.
Hourly observations require dt=1 h; standalone battery/grid may use other dt.
Only checked private working copies are committed; failures preserve all state.
record mode reports reliability separately from grid cost: no reward, shortage
penalty, building-level service allocation or learning controller is implemented.
"""

from collections.abc import Mapping
from copy import deepcopy
from copy import copy
from dataclasses import asdict, dataclass, field
import math
from numbers import Real
from threading import RLock
from types import MappingProxyType

from src.simulation._validation import ComponentValidationError, power_kw, total_power
from src.simulation.battery import Battery, BatteryState, BatteryTransition
from src.simulation.config import SimulationConfig, load_simulation_config, validate_simulation_config
from src.simulation.demand import CampusDemand, DemandOutput
from src.simulation.grid import Grid, GridState, GridTransaction
from src.simulation.renewables import RenewableGenerator, RenewableOutput, RenewableTotals, combine_renewables


class DispatchError(ComponentValidationError):
    """Invalid or unbalanced dispatch; no state/counter/transaction committed."""


class DispatchInfeasibleError(DispatchError):
    def __init__(self, message, *, unserved_demand_kw=0.0):
        super().__init__(message)
        self.unserved_demand_kw = unserved_demand_kw


@dataclass(frozen=True)
class DispatchAdjustment:
    code: str
    component: str
    requested: float | None
    actual: float | None


@dataclass(frozen=True)
class DispatchState:
    battery: BatteryState
    grid: GridState
    step_count: int
    unserved_energy_kwh: float


@dataclass(frozen=True)
class DispatchResult:
    timestamp_utc: str
    requested_battery_power_kw: float
    actual_battery_power_kw: float
    requested_curtailment_by_asset_kw: Mapping[str, float]
    renewable_outputs: tuple[RenewableOutput, ...]
    renewables: RenewableTotals
    demand: DemandOutput
    battery: BatteryTransition
    grid: GridTransaction
    served_demand_kw: float
    unserved_demand_kw: float
    unserved_energy_kwh: float
    balance_residual_kw: float
    energy_balance_residual_kwh: float
    adjustments: tuple[DispatchAdjustment, ...]
    _metadata: dict = field(repr=False, compare=False)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def step_cost(self):
        """Grid operating cost only; not a reliability-adjusted reward."""
        return self.grid.net_cost


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise DispatchError(f"{name} must be a positive finite number.")
    return float(value)


def _share(amount, weights):
    """Bounded proportional allocation; final remainder uses stable ID order."""
    total = total_power(weights.values(), "allocation_total_kw")
    if amount <= 0:
        return {key: 0.0 for key in weights}
    if total <= 0 or amount > total + 1e-9:
        raise DispatchError("Insufficient renewable power for curtailment/restoration.")
    amount = min(amount, total)
    remaining, result = amount, {}
    keys = sorted(weights)
    for index, key in enumerate(keys):
        portion = remaining if index == len(keys) - 1 else amount * (weights[key] / total)
        portion = min(weights[key], max(0.0, min(portion, remaining)))
        result[key] = portion
        remaining -= portion
    if remaining > 1e-9:
        raise DispatchError("Proportional allocation failed to account for renewable power.")
    return result


class DispatchEngine:
    """Own battery/grid working state; public properties expose immutable snapshots."""

    def __init__(self, renewables, demand, config=None, *,
                 power_balance_tolerance_kw=1e-6, energy_balance_tolerance_kwh=1e-6):
        config = load_simulation_config() if config is None else config
        if not isinstance(config, SimulationConfig):
            raise DispatchError("config must be a validated SimulationConfig.")
        self._config = validate_simulation_config(asdict(config))
        if self._config.time_step_hours != 1:
            raise DispatchError("Dispatch of Part 3 hourly observations requires a one-hour time step; no resampling is implicit.")
        if not isinstance(renewables, (list, tuple)) or not renewables or not all(isinstance(item, RenewableGenerator) for item in renewables):
            raise DispatchError("Provide registered solar/wind components.")
        if len({item.asset_id for item in renewables}) != len(renewables):
            raise DispatchError("Duplicate renewable asset IDs.")
        if {item.generation_type for item in renewables} != {"solar", "wind"}:
            raise DispatchError("Register both solar and wind; zero observed generation is allowed.")
        if not isinstance(demand, CampusDemand):
            raise DispatchError("demand must be a CampusDemand component.")
        registry = [item.asset_id for item in renewables] + list(demand.building_ids or ()) + [self._config.battery.asset_id, self._config.grid.asset_id]
        if len(registry) != len(set(registry)):
            raise DispatchError("Building, renewable, battery and grid IDs must be distinct.")
        self._renewables = tuple(sorted(renewables, key=lambda item: item.asset_id))
        self._demand = demand
        self._power_tolerance = _positive(power_balance_tolerance_kw, "power_balance_tolerance_kw")
        self._energy_tolerance = _positive(energy_balance_tolerance_kwh, "energy_balance_tolerance_kwh")
        self._battery, self._grid = Battery(self._config), Grid(self._config)
        self._results = []
        self._unserved_energy = 0.0
        self._lock = RLock()

    @property
    def state(self):
        with self._lock:
            return DispatchState(self._battery.state, self._grid.state, len(self._results), self._unserved_energy)

    @property
    def results(self):
        with self._lock:
            return tuple(self._results)

    def fork(self):
        """Independent episode transaction candidate, including component ledgers."""
        with self._lock:
            candidate = copy(self)
            candidate._battery = self._battery.fork()
            candidate._grid = self._grid.fork()
            candidate._results = self._results.copy()
            candidate._lock = RLock()
            return candidate

    def reset(self, *, initial_soc=None, start_timestamp_utc=None):
        with self._lock:
            battery, grid = self._battery.fork(), self._grid.fork()
            battery.reset(initial_soc)
            grid.reset(start_timestamp_utc=start_timestamp_utc)
            self._battery, self._grid = battery, grid
            self._results, self._unserved_energy = [], 0.0
            return self.state

    def step(self, observation, requested_battery_power_kw=None, *, charge_kw=None,
             discharge_kw=None, curtailment_by_asset_kw=None, currency_code=None,
             price_provenance=None):
        with self._lock:
            try:
                return self._step(observation, requested_battery_power_kw, charge_kw, discharge_kw,
                                  curtailment_by_asset_kw, currency_code, price_provenance)
            except DispatchError:
                raise
            except (ComponentValidationError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
                raise DispatchError(f"Dispatch rejected without changing state: {exc}") from exc

    def _step(self, observation, requested, charge_kw, discharge_kw, curtailment_request, currency, price_provenance):
        battery, grid = self._battery.fork(), self._grid.fork()
        dt, cfg = self._config.time_step_hours, self._config
        adjustments = []

        def note(code, component, before=None, after=None):
            adjustments.append(DispatchAdjustment(code, component, before, after))

        request_preview = battery.preview(requested, charge_kw=charge_kw, discharge_kw=discharge_kw)
        original_request = request_preview.requested_power_kw
        for reason in request_preview.limiting_reasons:
            note(reason, cfg.battery.asset_id, original_request, request_preview.actual_power_kw)
        demand = self._demand.read(observation)
        initial = tuple(item.read(observation) for item in self._renewables)
        for kind, field_name in (("solar", "pv_asset_powers_kw"), ("wind", "wind_asset_powers_kw")):
            if set(getattr(observation, field_name)) != {item.asset_id for item in self._renewables if item.generation_type == kind}:
                raise DispatchError(f"Observed {kind} asset IDs disagree with component registry.")
        for kind, observed in (("solar", observation.pv_available_kw), ("wind", observation.wind_available_kw)):
            aggregate = total_power((item.available_generation_kw for item in initial if item.generation_type == kind), "renewable_available_kw")
            if abs(aggregate - power_kw(observed, f"observation.{kind}_available_kw")) > self._power_tolerance:
                raise DispatchError("Observed renewable aggregate disagrees with asset readings.")
        available = {item.asset_id: item.available_generation_kw for item in initial}
        if curtailment_request is None:
            curtailment_request = {}
        if not isinstance(curtailment_request, Mapping) or not set(curtailment_request) <= available.keys():
            raise DispatchError("Curtailment requests must map registered asset IDs to nonnegative kW.")
        requests = {key: power_kw(curtailment_request.get(key, 0), f"{key}.requested_curtailment_kw") for key in available}
        if not cfg.renewable_curtailment.enabled and any(requests.values()):
            raise DispatchError("Renewable curtailment is disabled by configuration.")
        curtailed = {key: min(requests[key], value) for key, value in available.items()}
        for key in available:
            if curtailed[key] != requests[key]:
                note("curtailment_limited_by_availability", key, requests[key], curtailed[key])
        prices = {"import_price": observation.import_price, "export_price": observation.export_price,
                  "currency_code": currency, "timestamp_utc": observation.timestamp_utc,
                  "price_provenance": price_provenance}
        grid.preview(0, **prices) # Validate tariffs/clock even before a transfer is planned.
        export_limit = cfg.grid.export_limit_kw
        if observation.export_price is None and cfg.grid.missing_export_price_handling == "disable_export":
            export_limit = 0.0
        limits = battery.feasible_limits()

        def residual_demand():
            return demand.campus_load_kw - total_power((available[key] - curtailed[key] for key in available), "renewables_used_kw")

        residual = residual_demand()
        shortage = residual - cfg.grid.import_limit_kw - limits.discharge_limit_kw
        if shortage > self._power_tolerance and any(curtailed.values()):
            restoration = _share(min(shortage, total_power(curtailed.values(), "curtailed_kw")), curtailed)
            for key, restored in restoration.items():
                if restored:
                    previous = curtailed[key]
                    curtailed[key] = max(0.0, previous - restored)
                    note("curtailment_restored_for_supply", key, previous, curtailed[key])
            residual = residual_demand()
        shortage = residual - cfg.grid.import_limit_kw - limits.discharge_limit_kw
        unserved = 0.0
        if shortage > self._power_tolerance:
            if cfg.unserved_load.handling == "error":
                raise DispatchInfeasibleError(f"Supply shortfall of {shortage} kW; unserved_load.handling is error.", unserved_demand_kw=shortage)
            unserved = min(demand.campus_load_kw, shortage)
            note("unserved_demand_recorded", "campus", demand.campus_load_kw, demand.campus_load_kw - unserved)
        surplus = -limits.charge_limit_kw - export_limit - residual
        if surplus > self._power_tolerance:
            if not cfg.renewable_curtailment.enabled:
                raise DispatchInfeasibleError("Surplus cannot be absorbed/exported and renewable curtailment is disabled.")
            additional = _share(surplus, {key: available[key] - curtailed[key] for key in available})
            for key, value in additional.items():
                if value:
                    previous = curtailed[key]
                    curtailed[key] = min(available[key], previous + value)
                    note("excess_renewable_curtailed", key, previous, curtailed[key])
            residual = residual_demand()
        lower = max(-limits.charge_limit_kw, residual - unserved - cfg.grid.import_limit_kw)
        upper = min(limits.discharge_limit_kw, residual - unserved + export_limit)
        if lower > upper + self._power_tolerance:
            raise DispatchInfeasibleError("No feasible battery/grid exchange remains after safety adjustments.")
        projected = min(max(original_request, lower), upper)
        battery_result = battery.preview(projected)
        if battery_result.actual_power_kw != request_preview.actual_power_kw:
            note("battery_adjusted_for_grid_feasibility", cfg.battery.asset_id,
                 request_preview.actual_power_kw, battery_result.actual_power_kw)
        outputs = tuple(component.read(observation, curtailment_kw=curtailed[component.asset_id]) for component in self._renewables)
        renewable_total = combine_renewables(outputs)
        served = demand.campus_load_kw - unserved
        grid_request = served - renewable_total.total_used_kw - battery_result.actual_power_kw
        if 0 < abs(grid_request) <= self._power_tolerance:
            note("numerical_grid_roundoff", cfg.grid.asset_id, grid_request, 0.0)
            grid_request = 0.0
        grid_result = grid.preview(grid_request, **prices)
        if export_limit == 0 and cfg.grid.export_limit_kw > 0 and observation.export_price is None and (residual - original_request < 0):
            note("missing_export_price_disables_export", cfg.grid.asset_id)
        if grid_result.power_limited:
            for reason in grid_result.limiting_reasons:
                note(reason, cfg.grid.asset_id, grid_result.requested_power_kw, grid_result.actual_power_kw)
        balance = math.fsum([grid_result.import_power_kw, renewable_total.total_used_kw,
                            battery_result.discharge_power_kw, -served, -battery_result.charge_power_kw, -grid_result.export_power_kw])
        energy_balance = math.fsum([grid_result.imported_energy_kwh, renewable_total.total_used_kw * dt,
                                   battery_result.state_before.battery_energy_kwh, battery_result.rounding_adjustment_kwh,
                                   -served * dt, -grid_result.exported_energy_kwh,
                                   -battery_result.state_after.battery_energy_kwh, -battery_result.efficiency_loss_kwh])
        if not math.isfinite(balance) or abs(balance) > self._power_tolerance:
            raise DispatchError(f"Power balance residual {balance} kW exceeds tolerance.")
        if not math.isfinite(energy_balance) or abs(energy_balance) > self._energy_tolerance:
            raise DispatchError(f"Energy balance residual {energy_balance} kWh exceeds tolerance.")
        cumulative_unserved = self._unserved_energy + unserved * dt
        if not math.isfinite(cumulative_unserved):
            raise DispatchError("Unserved energy accounting overflow.")
        result = DispatchResult(observation.timestamp_utc, original_request, battery_result.actual_power_kw,
                                MappingProxyType(dict(requests)), outputs, renewable_total, demand,
                                battery_result, grid_result, served, unserved, unserved * dt,
                                balance, energy_balance, tuple(adjustments), {
                                    "origin": "derived", "derived_kind": "simulator",
                                    "config": asdict(cfg), "power_balance_tolerance_kw": self._power_tolerance,
                                    "energy_balance_tolerance_kwh": self._energy_tolerance,
                                    "unserved_allocation": "Campus total only; no building-level priorities or allocations inferred.",
                                    "accounting": "Balance uses served demand; original demand and unmet kW/kWh remain separate.",
                                    "internal_battery_request": "Battery transition request is the projected command; root requested_battery_power_kw preserves the original request.",
                                    "cost_scope": "Grid operating cost only; no shortage penalty, reward, degradation or AI policy."})
        # Commits affect independent forks. Publish only after both succeed, so
        # even a component commit failure cannot leave a half-advanced engine.
        battery.commit(battery_result)
        grid.commit(grid_result)
        self._battery, self._grid = battery, grid
        self._unserved_energy = cumulative_unserved
        self._results.append(result)
        return result
