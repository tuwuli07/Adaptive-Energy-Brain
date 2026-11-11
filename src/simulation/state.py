"""Validated physical snapshots and causal, model-independent observations.

An hourly replay row is NOT automatically information available at interval start.
Callers must supply its actual available_at_utc and a decision timestamp. An
interval-mean row normally becomes available at interval END. Using current-row
power before then would require a documented persistence/measurement assumption.
Availability declarations are trusted input contracts, not authenticity proofs.

Observations expose an explicit scalar allowlist, not replay datasets, next states,
dispatch outcomes, or arbitrary metadata. Forecasts are separate modeled features:
issue time AND training/input data cutoff must precede the decision; targets must
follow it. The caller remains responsible for truthful forecast lineage and for
fitting models only on permitted training data. No forecasting is performed here.
"""

from dataclasses import asdict, dataclass
import math

from src.simulation._validation import power_kw, timestamp_utc
from src.simulation.actions import finite_number
from src.simulation.battery import Battery
from src.simulation.config import SimulationConfig, validate_simulation_config


@dataclass(frozen=True)
class SimulationState:
    """Current exogenous row plus PRE-action battery state; kW, kWh and SOC [0,1]."""

    timestamp_utc: str
    available_at_utc: str
    solar_generation_kw: float
    wind_generation_kw: float
    campus_demand_kw: float
    battery_soc: float
    grid_import_price: float
    grid_export_price: float | None
    config: SimulationConfig

    def __post_init__(self):
        for name in ("timestamp_utc", "available_at_utc"):
            object.__setattr__(self, name, timestamp_utc(getattr(self, name)))
        if self.available_at_utc < self.timestamp_utc:
            raise ValueError("available_at_utc cannot precede the observed interval.")
        if not isinstance(self.config, SimulationConfig):
            raise ValueError("config must be a SimulationConfig.")
        config = validate_simulation_config(asdict(self.config))
        if config.time_step_hours != 1:
            raise ValueError("Hourly simulation states require a one-hour time step.")
        object.__setattr__(self, "config", config)
        for name in ("solar_generation_kw", "wind_generation_kw", "campus_demand_kw"):
            object.__setattr__(self, name, power_kw(getattr(self, name), name))
        soc = finite_number(self.battery_soc, "battery_soc")
        if not config.battery.soc_min <= soc <= config.battery.soc_max:
            raise ValueError("battery_soc must lie within configured SOC limits.")
        object.__setattr__(self, "battery_soc", soc)
        for name in ("grid_import_price", "grid_export_price"):
            if name == "grid_export_price" and getattr(self, name) is None:
                continue
            price = finite_number(getattr(self, name), name)
            if price < 0 and not config.grid.allow_negative_prices:
                raise ValueError(f"{name}: negative prices are disabled.")
            object.__setattr__(self, name, price)

    @classmethod
    def from_observation(cls, observation, battery_state, config, *, available_at_utc):
        """Copy scalar fields only; never retain a replay iterator or future rows."""
        if not math.isclose(battery_state.battery_energy_kwh,
                            battery_state.battery_soc * config.battery.capacity_kwh,
                            rel_tol=0, abs_tol=1e-6):
            raise ValueError("Battery stored energy disagrees with SOC and capacity.")
        return cls(observation.timestamp_utc, available_at_utc,
                   observation.pv_available_kw, observation.wind_available_kw,
                   observation.campus_load_kw, battery_state.battery_soc,
                   observation.import_price, observation.export_price, config)

    @property
    def battery_energy_kwh(self):
        return self.battery_soc * self.config.battery.capacity_kwh

    def observation(self, *, decision_timestamp_utc, forecasts=()):
        return AgentObservation(self, decision_timestamp_utc, forecasts)


@dataclass(frozen=True)
class ForecastFeature:
    """Explicit modeled prediction, never a future observed target value."""

    variable: str
    value_kw: float
    target_timestamp_utc: str
    issued_at_utc: str
    data_cutoff_utc: str
    model_id: str

    def __post_init__(self):
        if self.variable not in {"solar_generation_kw", "wind_generation_kw", "campus_demand_kw"}:
            raise ValueError("Unsupported forecast variable; expected demand, solar or wind power.")
        object.__setattr__(self, "value_kw", power_kw(self.value_kw, "forecast.value_kw"))
        for name in ("target_timestamp_utc", "issued_at_utc", "data_cutoff_utc"):
            object.__setattr__(self, name, timestamp_utc(getattr(self, name)))
        if self.data_cutoff_utc > self.issued_at_utc:
            raise ValueError("Forecast data cutoff cannot follow issue time.")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("Forecast model_id must be explicit.")


@dataclass(frozen=True)
class AgentObservation:
    """No Gym dependency. Feature names/order/units are explicit and deterministic.

    Export price remains None when unavailable. A future numeric adapter must
    use a missingness mask or explicit policy; no implicit zero-price filling.
    Forecast features retain their own target times and are not physical inputs.
    """

    state: SimulationState
    decision_timestamp_utc: str
    forecasts: tuple[ForecastFeature, ...] = ()

    def __post_init__(self):
        if not isinstance(self.state, SimulationState):
            raise ValueError("state must be a SimulationState.")
        decision = timestamp_utc(self.decision_timestamp_utc)
        object.__setattr__(self, "decision_timestamp_utc", decision)
        if self.state.available_at_utc > decision:
            raise ValueError("Observed values are not yet available at the decision timestamp.")
        forecasts = tuple(self.forecasts)
        keys = set()
        for feature in forecasts:
            if not isinstance(feature, ForecastFeature):
                raise ValueError("Only explicit modeled ForecastFeature values are permitted.")
            if feature.issued_at_utc > decision or feature.data_cutoff_utc > decision:
                raise ValueError("Forecast uses information unavailable at the decision timestamp.")
            if feature.target_timestamp_utc <= decision:
                raise ValueError("Forecast target must follow the decision timestamp.")
            key = (feature.variable, feature.target_timestamp_utc)
            if key in keys:
                raise ValueError("Duplicate forecast variable/target timestamp.")
            keys.add(key)
        object.__setattr__(self, "forecasts", tuple(sorted(
            forecasts, key=lambda f: (f.target_timestamp_utc, f.variable))))

    def features(self):
        """Fresh allowlisted mapping in stable order; prices use currency/kWh."""
        s, config = self.state, self.state.config
        battery = Battery(config)
        battery.reset(s.battery_soc)
        limits = battery.feasible_limits()
        return {
            "solar_generation_kw": s.solar_generation_kw,
            "wind_generation_kw": s.wind_generation_kw,
            "campus_demand_kw": s.campus_demand_kw,
            "battery_soc": s.battery_soc,
            "battery_energy_kwh": s.battery_energy_kwh,
            "grid_import_price": s.grid_import_price,
            "grid_export_price": s.grid_export_price,
            "battery_capacity_kwh": config.battery.capacity_kwh,
            "soc_min": config.battery.soc_min,
            "soc_max": config.battery.soc_max,
            "charge_limit_kw": limits.charge_limit_kw,
            "discharge_limit_kw": limits.discharge_limit_kw,
            "grid_import_limit_kw": config.grid.import_limit_kw,
            "grid_export_limit_kw": (0.0 if s.grid_export_price is None and
                                     config.grid.missing_export_price_handling == "disable_export"
                                     else config.grid.export_limit_kw),
            "charge_efficiency": config.battery.charge_efficiency,
            "discharge_efficiency": config.battery.discharge_efficiency,
            "time_step_hours": config.time_step_hours,
        }

    @property
    def currency_code(self):
        return self.state.config.grid.currency_code

    @property
    def price_unit(self):
        return f"{self.currency_code}/kWh"
