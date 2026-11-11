"""Deterministic, causal hourly replay environment; no Gym or RL rewards.

Usage: env = MicrogridEnvironment(data, renewables, demand); obs = env.reset();
then result = env.step(BatteryAction(...)) until result.terminated.
result contains observation, metrics, cumulative_metrics and dispatch diagnostics.

Timing: row t describes interval [t,t+1h). It is observable at t+1h. The action
chosen at t+1h controls row t+1; only row t is exposed to the agent, with the
current PRE-action battery SOC. Therefore the first row is explicitly reserved
as history, not dispatched or charged to episode metrics. N rows give N-1 steps.
Prices in observations are also lagged, not assumed known future tariffs. The
internal dispatcher uses realized current-hour data for settlement and physical
balancing. This is an hourly control abstraction, not sub-hourly closed-loop
physics or perfect foresight. No initial history is invented or filled with zero.

After each step the just-completed row becomes visible, with POST-step SOC, at
the next boundary. A terminal observation is returned without reading beyond the
dataset. Failed actions, dispatch, accounting or observation construction publish
no state changes. reset creates fresh ledgers and counters; no RNG is needed.
ForecastFeature can be attached by a future observation adapter using the existing
issue-time/data-cutoff checks; the physical dispatch interface stays unchanged.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import math
from threading import RLock
from types import MappingProxyType

from src.simulation._validation import source_provenance, timestamp_utc, total_power
from src.simulation.actions import BatteryAction
from src.simulation.config import (
    SimulationConfig, load_simulation_config, validate_simulation_config,
)
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.dispatch import DispatchEngine, DispatchResult
from src.simulation.metrics import (
    CostConfig, MetricValues, MetricsTracker, StepMetrics, load_cost_config,
)
from src.simulation.state import AgentObservation, SimulationState


class EnvironmentError(ValueError):
    """Invalid environment operation; the episode remains unchanged."""


def _next_hour(stamp):
    value = datetime.fromisoformat(timestamp_utc(stamp).replace("Z", "+00:00"))
    return (value + timedelta(hours=1)).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EnvironmentStep:
    observation: AgentObservation
    metrics: StepMetrics
    cumulative_metrics: MetricValues
    terminated: bool
    dispatch: DispatchResult

    @property
    def truncated(self):
        """No independent time-limit truncation is implemented."""
        return False


class MicrogridEnvironment:
    """Own a fixed read-only dataset snapshot and private episode components.

    Data must be SimulationData from the Part 3 loader, in physical units with
    source metadata. Components require explicit ratings/provenance; no campus
    topology, ratings, tariffs, or measured/synthetic origins are inferred here.
    reset() is required before step(). Actions may be BatteryAction or signed
    finite scalar kW (positive discharge, negative charge). Finite oversized
    requests use dispatch's constrained execution policy; invalid inputs fail.
    """

    def __init__(self, data, renewables, demand, config=None, costs=None, *,
                 price_provenance=None):
        if not isinstance(data, SimulationData) or len(data) < 2:
            raise EnvironmentError("Provide at least two hourly rows: one history row and one dispatch row.")
        config = load_simulation_config() if config is None else config
        if not isinstance(config, SimulationConfig):
            raise EnvironmentError("config must be a SimulationConfig.")
        self._config = validate_simulation_config(asdict(config))
        self._costs = load_cost_config() if costs is None else costs
        if not isinstance(self._costs, CostConfig):
            raise EnvironmentError("costs must be a CostConfig.")
        metadata = data.metadata
        currency = self._config.grid.currency_code
        if metadata.get("currency_code") != currency or self._costs.currency_code != currency:
            raise EnvironmentError("Dataset, grid and economic currency codes must match.")
        if metadata.get("units") != {"power": "kW", "prices": f"{currency}/kWh"}:
            raise EnvironmentError("Use loader-validated physical kW and currency/kWh data.")
        self._renewables, self._demand = tuple(renewables), demand
        # Validate the component registry and hourly configuration once.
        initial_battery = DispatchEngine(list(self._renewables), demand, self._config).state.battery
        rows = []
        for row in data:
            if not isinstance(row, HourlyObservation):
                raise EnvironmentError("Dataset rows must be HourlyObservation values.")
            stamp = timestamp_utc(row.timestamp_utc)
            if rows and stamp != _next_hour(rows[-1].timestamp_utc):
                raise EnvironmentError("Dataset timestamps must be consecutive hourly UTC intervals.")
            copied = HourlyObservation(
                stamp, row.pv_available_kw, row.wind_available_kw, row.campus_load_kw,
                row.import_price, row.export_price,
                MappingProxyType(dict(row.building_loads_kw)),
                MappingProxyType(dict(row.pv_asset_powers_kw)),
                MappingProxyType(dict(row.wind_asset_powers_kw)))
            # Validate observations/ratings without changing state or dispatching.
            demand.read(copied)
            for component in self._renewables:
                component.read(copied)
            for kind, readings, declared in (
                ("solar", copied.pv_asset_powers_kw, copied.pv_available_kw),
                ("wind", copied.wind_asset_powers_kw, copied.wind_available_kw),
            ):
                registered = {c.asset_id for c in self._renewables if c.generation_type == kind}
                if set(readings) != registered or not math.isclose(
                    total_power(readings.values(), f"{kind} aggregate"), declared,
                    rel_tol=0, abs_tol=1e-6,
                ):
                    raise EnvironmentError(f"Observed {kind} registry/aggregate disagrees with asset readings.")
            SimulationState.from_observation(
                copied, initial_battery,
                self._config, available_at_utc=_next_hour(stamp))
            rows.append(copied)
        if price_provenance is None:
            price_sources = [s["metadata"] for s in metadata.get("sources", {}).values()
                             if s.get("record_type") == "price"]
            if len(price_sources) != 1:
                raise EnvironmentError("Provide explicit price_provenance when price sources are absent or multiple.")
            price_provenance = price_sources[0]
        self._price_provenance = source_provenance(price_provenance)
        self._rows = tuple(rows)
        self._metadata = deepcopy(metadata)
        self._metadata["environment"] = {
            "observation_policy": "previous_completed_hour",
            "history_timestamp_utc": rows[0].timestamp_utc,
            "dispatch_start_utc": rows[1].timestamp_utc,
            "end_exclusive_utc": _next_hour(rows[-1].timestamp_utc),
            "action_steps": len(rows) - 1,
            "history_row_in_episode_metrics": False,
        }
        self._engine = self._tracker = self._observation = None
        self._index = 1
        self._lock = RLock()

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def observation(self):
        with self._lock:
            self._require_reset()
            return self._observation

    @property
    def state(self):
        with self._lock:
            self._require_reset()
            return self._engine.state

    @property
    def cumulative_metrics(self):
        with self._lock:
            self._require_reset()
            return self._tracker.totals

    @property
    def step_metrics(self):
        with self._lock:
            self._require_reset()
            return self._tracker.steps

    @property
    def dispatch_results(self):
        """Immutable step records, including executed grid transactions."""
        with self._lock:
            self._require_reset()
            return self._engine.results

    @property
    def terminated(self):
        with self._lock:
            return self._engine is not None and self._index == len(self._rows)

    def _require_reset(self):
        if self._engine is None:
            raise EnvironmentError("Call reset() before accessing or advancing the episode.")

    def _observe(self, row, engine):
        boundary = _next_hour(row.timestamp_utc)
        state = SimulationState.from_observation(
            row, engine.state.battery, self._config, available_at_utc=boundary)
        return state.observation(decision_timestamp_utc=boundary)

    def reset(self):
        """Return the first causal observation; discard prior episode accounting."""
        with self._lock:
            engine = DispatchEngine(list(self._renewables), self._demand, self._config)
            engine.reset(start_timestamp_utc=self._rows[1].timestamp_utc)
            tracker = MetricsTracker(self._costs)
            observation = self._observe(self._rows[0], engine)
            self._engine, self._tracker = engine, tracker
            self._index, self._observation = 1, observation
            return observation

    def step(self, action):
        """Advance exactly one hour; return EnvironmentStep, with no RL reward."""
        with self._lock:
            self._require_reset()
            if self.terminated:
                raise EnvironmentError("Episode is complete; call reset() before stepping again.")
            action = action if isinstance(action, BatteryAction) else BatteryAction(action)
            engine, tracker = self._engine.fork(), self._tracker.fork()
            row = self._rows[self._index]
            dispatch = engine.step(row, **action.dispatch_kwargs(),
                                   currency_code=self._config.grid.currency_code,
                                   price_provenance=self._price_provenance)
            metrics = tracker.record(dispatch)
            if metrics.violations:
                raise EnvironmentError("Physical violations detected; step was not committed.")
            observation = self._observe(row, engine)
            index = self._index + 1
            result = EnvironmentStep(observation, metrics, tracker.totals,
                                     index == len(self._rows), dispatch)
            self._engine, self._tracker = engine, tracker
            self._index, self._observation = index, observation
            return result
