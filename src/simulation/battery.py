"""Bounded battery storage with explicit AC-side energy accounting.

Sign convention: positive requested/actual kW DISCHARGES to the campus bus;
negative kW CHARGES from the bus. charge()/discharge() accept nonnegative kW.
step(charge_kw=..., discharge_kw=...) is an alternative to a signed request;
simultaneous positive directions or mixed request styles are rejected.

E_next = E + eta_charge * charge_kw * dt - discharge_kw * dt / eta_discharge.
Loss = AC charging energy minus retained energy, plus withdrawn stored energy
minus AC discharge energy. Throughput is tracked both AC-side (charge +
discharge energy, as specified in the architecture) and internal stored-energy
movement. Neither is a degradation cost or a cycle-life model.

Finite excessive requests saturate to feasible power/SOC limits and return
the original request, actual exchange and limiting reasons. Invalid values
raise BatteryError atomically. This component does not implement grid dispatch;
a future engine may reject out-of-action-space requests before invoking it.
preview() does not mutate. commit() rejects stale or altered physical results;
step() is an atomic convenience. Default parameters come from simulation.yaml.
"""

from copy import deepcopy
from copy import copy
from dataclasses import asdict, dataclass, field
import math
from numbers import Real

from src.simulation._validation import ComponentValidationError
from src.simulation.config import (
    DEFAULT_CONFIG_PATH,
    SimulationConfig,
    SimulationConfigError,
    load_simulation_config,
    validate_simulation_config,
)


class BatteryError(ComponentValidationError):
    """An invalid battery request or numerical transition left state unchanged."""


@dataclass(frozen=True)
class BatteryState:
    battery_energy_kwh: float
    battery_soc: float
    charge_throughput_kwh: float = 0.0
    discharge_throughput_kwh: float = 0.0
    energy_throughput_kwh: float = 0.0
    internal_energy_throughput_kwh: float = 0.0
    elapsed_hours: float = 0.0
    step_count: int = 0


@dataclass(frozen=True)
class BatteryPowerLimits:
    charge_limit_kw: float
    discharge_limit_kw: float


@dataclass(frozen=True)
class BatteryTransition:
    asset_id: str
    time_step_hours: float
    requested_power_kw: float
    actual_power_kw: float
    charge_power_kw: float
    discharge_power_kw: float
    charge_energy_kwh: float
    discharge_energy_kwh: float
    stored_energy_change_kwh: float
    efficiency_loss_kwh: float
    throughput_increment_kwh: float
    internal_throughput_increment_kwh: float
    rounding_adjustment_kwh: float
    power_limited: bool
    limiting_reasons: tuple[str, ...]
    state_before: BatteryState
    state_after: BatteryState
    _revision: int = field(repr=False)
    _metadata: dict = field(repr=False, compare=False)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def updated_soc(self):
        return self.state_after.battery_soc


def _finite(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BatteryError(f"{name}: expected a finite numeric value.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise BatteryError(f"{name}: value must be finite.") from exc
    if not math.isfinite(result):
        raise BatteryError(f"{name}: value must be finite.")
    if positive and result <= 0:
        raise BatteryError(f"{name}: must be positive.")
    if nonnegative and result < 0:
        raise BatteryError(f"{name}: must be nonnegative.")
    return result


class Battery:
    """Stateful storage component; all public state snapshots are immutable."""

    def __init__(self, config=None):
        try:
            if config is None:
                config = load_simulation_config()
            if not isinstance(config, SimulationConfig):
                raise BatteryError("config must be a validated SimulationConfig.")
            self._config = validate_simulation_config(asdict(config))
        except SimulationConfigError as exc:
            raise BatteryError(f"Invalid battery configuration: {exc}") from exc
        self._parameters = self._config.battery
        self._min_energy = self._parameters.capacity_kwh * self._parameters.soc_min
        self._max_energy = self._parameters.capacity_kwh * self._parameters.soc_max
        if not self._max_energy > self._min_energy:
            raise BatteryError("Battery energy bounds are not distinct at floating-point precision.")
        self._revision = 0
        self.reset()

    @classmethod
    def from_config(cls, path=DEFAULT_CONFIG_PATH):
        try:
            return cls(load_simulation_config(path))
        except SimulationConfigError as exc:
            raise BatteryError(f"Cannot initialize battery: {exc}") from exc

    @property
    def state(self):
        return self._state

    def fork(self):
        """Independent working copy sharing only immutable configuration/state."""
        return copy(self)

    @property
    def energy_kwh(self):
        return self._state.battery_energy_kwh

    @property
    def soc(self):
        return self._state.battery_soc

    @property
    def throughput_kwh(self):
        """Cumulative AC charging plus AC discharging kWh."""
        return self._state.energy_throughput_kwh

    def reset(self, initial_soc=None):
        value = self._parameters.initial_soc if initial_soc is None else _finite(initial_soc, "initial_soc")
        if not self._parameters.soc_min <= value <= self._parameters.soc_max:
            raise BatteryError("initial_soc must lie within [soc_min, soc_max].")
        candidate = BatteryState(self._parameters.capacity_kwh * value, value)
        self._state = candidate
        self._revision += 1
        return candidate

    def _duration(self, value):
        return _finite(self._config.time_step_hours if value is None else value,
                       "time_step_hours", positive=True)

    def feasible_limits(self, time_step_hours=None):
        """Maximum nonnegative AC powers respecting energy bounds for duration dt."""
        dt = self._duration(time_step_hours)
        p = self._parameters
        charge = ((self._max_energy - self.energy_kwh) / p.charge_efficiency) / dt
        discharge = ((self.energy_kwh - self._min_energy) * p.discharge_efficiency) / dt
        return BatteryPowerLimits(min(p.charge_limit_kw, max(0.0, charge)),
                                  min(p.discharge_limit_kw, max(0.0, discharge)))

    def preview(self, requested_power_kw=None, *, charge_kw=None, discharge_kw=None,
                time_step_hours=None):
        """Calculate a candidate without advancing state, time or throughput."""
        if requested_power_kw is not None and (charge_kw is not None or discharge_kw is not None):
            raise BatteryError("Choose either signed requested_power_kw or separate charge_kw/discharge_kw.")
        if charge_kw is not None or discharge_kw is not None:
            charge_request = _finite(0 if charge_kw is None else charge_kw, "charge_kw", nonnegative=True)
            discharge_request = _finite(0 if discharge_kw is None else discharge_kw, "discharge_kw", nonnegative=True)
            if charge_request > 0 and discharge_request > 0:
                raise BatteryError("Simultaneous charging and discharging is not allowed.")
            request = discharge_request - charge_request
        else:
            request = _finite(0 if requested_power_kw is None else requested_power_kw, "requested_power_kw")
        dt = self._duration(time_step_hours)
        p, before = self._parameters, self.state
        limits = self.feasible_limits(dt)
        requested_charge, requested_discharge = max(-request, 0.0), max(request, 0.0)
        charge = min(requested_charge, limits.charge_limit_kw)
        discharge = min(requested_discharge, limits.discharge_limit_kw)
        reasons = []
        for requested, rating, dynamic, power_reason, soc_reason in (
            (requested_charge, p.charge_limit_kw, ((self._max_energy - self.energy_kwh) / p.charge_efficiency) / dt, "charge_power_limit", "max_soc"),
            (requested_discharge, p.discharge_limit_kw, ((self.energy_kwh - self._min_energy) * p.discharge_efficiency) / dt, "discharge_power_limit", "min_soc"),
        ):
            if requested > min(rating, dynamic):
                if rating <= dynamic:
                    reasons.append(power_reason)
                if dynamic <= rating:
                    reasons.append(soc_reason)
        charge_energy, discharge_energy = charge * dt, discharge * dt
        retained = p.charge_efficiency * charge_energy
        withdrawn = discharge_energy / p.discharge_efficiency
        loss = (charge_energy - retained) + (withdrawn - discharge_energy)
        candidate_energy = self.energy_kwh + retained - withdrawn
        for name, value in (("charge_energy_kwh", charge_energy), ("discharge_energy_kwh", discharge_energy),
                            ("stored_energy_kwh", candidate_energy), ("efficiency_loss_kwh", loss)):
            _finite(value, name)
        # Only representational rounding may be clamped; report it separately
        # from physical losses. Material violations fail without committing.
        tolerance = 32 * max(math.ulp(self.energy_kwh), math.ulp(self._min_energy), math.ulp(self._max_energy))
        if candidate_energy < self._min_energy - tolerance or candidate_energy > self._max_energy + tolerance:
            raise BatteryError("Candidate energy violates SOC bounds beyond numerical rounding.")
        bounded = min(self._max_energy, max(self._min_energy, candidate_energy))
        rounding = bounded - candidate_energy
        ac_throughput, internal_throughput = charge_energy + discharge_energy, retained + withdrawn
        cumulative = (
            before.charge_throughput_kwh + charge_energy,
            before.discharge_throughput_kwh + discharge_energy,
            before.energy_throughput_kwh + ac_throughput,
            before.internal_energy_throughput_kwh + internal_throughput,
            before.elapsed_hours + dt,
        )
        for value in cumulative:
            _finite(value, "Cumulative battery accounting", nonnegative=True)
        next_soc = before.battery_soc if charge == discharge == 0 else min(p.soc_max, max(p.soc_min, bounded / p.capacity_kwh))
        after = BatteryState(bounded, next_soc,
                             *cumulative, before.step_count + 1)
        metadata = {"origin": "derived", "derived_kind": "simulator",
                    "parameter_origin": self._config.parameter_origin,
                    "source_reference": self._config.source_reference,
                    "battery_parameters": asdict(p), "power_unit": "kW", "energy_unit": "kWh",
                    "sign_convention": "positive discharge; negative charge; AC-side power",
                    "throughput_basis": "AC charge plus discharge; internal movement recorded separately",
                    "rounding_policy": "only ULP-sized boundary correction, reported as rounding_adjustment_kwh"}
        actual = discharge - charge
        return BatteryTransition(p.asset_id, dt, request, actual, charge, discharge,
                                 charge_energy, discharge_energy, bounded - self.energy_kwh,
                                 loss, ac_throughput, internal_throughput, rounding,
                                 actual != request, tuple(reasons), before, after, self._revision, metadata)

    def commit(self, transition):
        """Commit only a current, reproducible candidate from this configuration."""
        if not isinstance(transition, BatteryTransition) or transition.state_before != self.state or transition._revision != self._revision:
            raise BatteryError("Stale or invalid battery transition; state unchanged.")
        expected = self.preview(transition.requested_power_kw, time_step_hours=transition.time_step_hours)
        if transition != expected or transition.metadata != expected.metadata:
            raise BatteryError("Altered or incompatible battery transition; state unchanged.")
        self._state = transition.state_after
        self._revision += 1
        return transition

    def step(self, requested_power_kw=None, *, charge_kw=None, discharge_kw=None,
             time_step_hours=None):
        return self.commit(self.preview(requested_power_kw, charge_kw=charge_kw,
                                        discharge_kw=discharge_kw, time_step_hours=time_step_hours))

    def charge(self, power_kw, *, time_step_hours=None):
        return self.step(charge_kw=power_kw, time_step_hours=time_step_hours)

    def discharge(self, power_kw, *, time_step_hours=None):
        return self.step(discharge_kw=power_kw, time_step_hours=time_step_hours)
