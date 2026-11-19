"""Causal 24-hour seasonal-persistence MPC controller.

Phase 7.3 implementation: generates causal 24-hour seasonal-persistence forecasts
for campus load, solar, and wind generation using strictly completed historical
observations, plans against the deterministic published time-of-use tariff schedule,
and solves the linear program at each hourly simulation step.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from src.baselines.base import BaselineController
from src.control.mpc_solver import (
    MPCInputProfile,
    MPCSolverConfig,
    MPCSolverResult,
    load_mpc_config,
    solve_mpc_linear_program,
)
from src.simulation.actions import BatteryAction
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.data_loader import HourlyObservation
from src.simulation.metrics import CostConfig, load_cost_config
from src.simulation.state import AgentObservation, SimulationState


def get_scheduled_tariff(timestamp_utc: str) -> tuple[float, float]:
    """Calculate the deterministic published time-of-use tariff for a UTC timestamp.

    Scenario specification (configs/synthetic_scenario.yaml):
      - Base import price: 0.12 USD/kWh
      - Peak import price: 0.18 USD/kWh
      - Export price: 0.04 USD/kWh
      - Campus timezone: Asia/Dhaka (UTC+6)
      - Peak hours in local time: [17, 22), corresponding to [11, 16) UTC.
    Unannounced stochastic price spikes (0.48 or 0.72 USD/kWh) are NOT published
    in advance and are not included in the forward planning schedule.
    """
    dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    utc_hour = dt.hour
    # Local hour in Asia/Dhaka (UTC+6)
    local_hour = (utc_hour + 6) % 24
    is_peak = 17 <= local_hour < 22  # UTC 11:00 to 15:59
    import_price = 0.18 if is_peak else 0.12
    export_price = 0.04
    return import_price, export_price


@dataclass(frozen=True)
class CausalHistoricalRecord:
    """Historical completed observation values retained causally."""

    timestamp_utc: str
    campus_demand_kw: float
    solar_generation_kw: float
    wind_generation_kw: float


@dataclass(frozen=True)
class SeasonalPersistenceStepDiagnostic:
    """Per-step diagnostics for the causal seasonal-persistence MPC controller."""

    decision_timestamp_utc: str
    success: bool
    status: int
    solve_time_seconds: float
    first_requested_power_kw: float
    horizon_hours: int
    fallback_used: bool
    fallback_hours_count: int
    has_simultaneous_battery_flow: bool
    forecast_demand_kw: tuple[float, ...]
    forecast_solar_kw: tuple[float, ...]
    forecast_wind_kw: tuple[float, ...]
    target_timestamps_utc: tuple[str, ...]
    source_timestamps_utc: tuple[str, ...]


class SeasonalPersistenceMPCController(BaselineController):
    """24-Hour rolling MPC controller driven by causal seasonal-persistence forecasts.

    Strict causality rules:
      - Uses completed historical intervals only: observations completed at or before t-1.
      - Never receives or accesses future realized validation partition rows.
      - Forecast: forecast(t + k) = observed(t + k - 24 hours).
      - For any missing history, uses causal 1-step persistence from the latest completed observation.
      - Plans against the deterministic published TOU tariff schedule without future stochastic spikes.
    """

    def __init__(
        self,
        initial_history: Sequence[HourlyObservation] | None = None,
        sim_config: SimulationConfig | None = None,
        cost_config: CostConfig | None = None,
        mpc_config: MPCSolverConfig | None = None,
        *,
        episode_end_utc: str | None = None,
    ) -> None:
        if sim_config is None:
            sim_config = load_simulation_config()
        if cost_config is None:
            cost_config = load_cost_config()
        if mpc_config is None:
            mpc_config = load_mpc_config()

        self.sim_config = sim_config
        self.cost_config = cost_config
        self.mpc_config = mpc_config
        self.episode_end_utc = episode_end_utc

        self._initial_history_records: dict[str, CausalHistoricalRecord] = {}
        if initial_history is not None:
            for r in initial_history:
                self._initial_history_records[r.timestamp_utc] = CausalHistoricalRecord(
                    timestamp_utc=r.timestamp_utc,
                    campus_demand_kw=float(r.campus_load_kw),
                    solar_generation_kw=float(r.pv_available_kw),
                    wind_generation_kw=float(r.wind_available_kw),
                )

        self._history: dict[str, CausalHistoricalRecord] = {}
        self._latest_record: CausalHistoricalRecord | None = None

        self.step_diagnostics: list[SeasonalPersistenceStepDiagnostic] = []
        self.solve_times: list[float] = []
        self.step_actions: list[BatteryAction] = []
        self.fallback_count: int = 0
        self.total_history_fallback_steps: int = 0

        self._reset()

    def _reset(self) -> None:
        """Reset episode state, restoring initial history buffer."""
        self._history = dict(self._initial_history_records)
        if self._initial_history_records:
            # Sort by timestamp to find latest
            latest_ts = max(self._initial_history_records)
            self._latest_record = self._initial_history_records[latest_ts]
        else:
            self._latest_record = None

        self.step_diagnostics.clear()
        self.solve_times.clear()
        self.step_actions.clear()
        self.fallback_count = 0
        self.total_history_fallback_steps = 0

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        """Causal decision step using seasonal-persistence forecasts and scheduled tariffs."""
        # 1. Ingest just-completed observation into causal history
        completed_state = observation.state
        completed_ts = completed_state.timestamp_utc
        record = CausalHistoricalRecord(
            timestamp_utc=completed_ts,
            campus_demand_kw=float(completed_state.campus_demand_kw),
            solar_generation_kw=float(completed_state.solar_generation_kw),
            wind_generation_kw=float(completed_state.wind_generation_kw),
        )
        self._history[completed_ts] = record
        self._latest_record = record

        # 2. Decision timestamp
        dec_ts_str = observation.decision_timestamp_utc
        dec_dt = datetime.fromisoformat(dec_ts_str.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )

        # 3. Determine horizon length
        max_h = self.mpc_config.horizon_hours
        if self.episode_end_utc is not None:
            end_dt = datetime.fromisoformat(
                self.episode_end_utc.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            rem_hours = int((end_dt - dec_dt).total_seconds() // 3600)
            horizon = max(1, min(max_h, rem_hours))
        else:
            horizon = max_h

        # 4. Construct 24-hour seasonal-persistence forecast
        forecast_demands: list[float] = []
        forecast_solars: list[float] = []
        forecast_winds: list[float] = []
        target_timestamps: list[str] = []
        source_timestamps: list[str] = []
        missing_history_count = 0

        for k in range(horizon):
            target_dt = dec_dt + timedelta(hours=k)
            target_ts = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            target_timestamps.append(target_ts)

            # Seasonal persistence formula: observation from 24 hours prior
            source_dt = target_dt - timedelta(hours=24)
            source_ts = source_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            source_timestamps.append(source_ts)

            # Assert strict causality: source timestamp must strictly precede decision time
            if source_dt >= dec_dt:
                raise RuntimeError(
                    f"Causality violation: source timestamp {source_ts} >= decision timestamp {dec_ts_str}"
                )

            if source_ts in self._history:
                src_rec = self._history[source_ts]
                forecast_demands.append(src_rec.campus_demand_kw)
                forecast_solars.append(src_rec.solar_generation_kw)
                forecast_winds.append(src_rec.wind_generation_kw)
            else:
                # Documented causal fallback: 1-step persistence from latest available observation
                missing_history_count += 1
                if self._latest_record is not None:
                    forecast_demands.append(self._latest_record.campus_demand_kw)
                    forecast_solars.append(self._latest_record.solar_generation_kw)
                    forecast_winds.append(self._latest_record.wind_generation_kw)
                else:
                    forecast_demands.append(250.0)  # Nominal default
                    forecast_solars.append(0.0)
                    forecast_winds.append(0.0)

        if missing_history_count > 0:
            self.total_history_fallback_steps += 1

        # 5. Published scheduled tariffs for planning horizon (no stochastic spikes)
        import_prices = []
        export_prices = []
        for target_ts in target_timestamps:
            p_imp, p_exp = get_scheduled_tariff(target_ts)
            import_prices.append(p_imp)
            export_prices.append(p_exp)

        # 6. Current pre-action battery state
        capacity_kwh = self.sim_config.battery.capacity_kwh
        initial_energy_kwh = completed_state.battery_soc * capacity_kwh

        # 7. Formulate and solve LP
        profile = MPCInputProfile(
            initial_energy_kwh=initial_energy_kwh,
            demand_kw=tuple(forecast_demands),
            solar_generation_kw=tuple(forecast_solars),
            wind_generation_kw=tuple(forecast_winds),
            import_price=tuple(import_prices),
            export_price=tuple(export_prices),
        )

        res: MPCSolverResult = solve_mpc_linear_program(
            profile,
            sim_config=self.sim_config,
            cost_config=self.cost_config,
            mpc_config=self.mpc_config,
        )

        solve_time = res.diagnostics.solve_time_seconds
        self.solve_times.append(solve_time)

        if not res.success:
            self.fallback_count += 1
            action = BatteryAction(0.0)
        else:
            action = res.first_action

        self.step_actions.append(action)
        self.step_diagnostics.append(
            SeasonalPersistenceStepDiagnostic(
                decision_timestamp_utc=dec_ts_str,
                success=res.success,
                status=res.diagnostics.status,
                solve_time_seconds=solve_time,
                first_requested_power_kw=action.requested_power_kw,
                horizon_hours=horizon,
                fallback_used=missing_history_count > 0,
                fallback_hours_count=missing_history_count,
                has_simultaneous_battery_flow=res.diagnostics.has_simultaneous_battery_flow,
                forecast_demand_kw=tuple(forecast_demands),
                forecast_solar_kw=tuple(forecast_solars),
                forecast_wind_kw=tuple(forecast_winds),
                target_timestamps_utc=tuple(target_timestamps),
                source_timestamps_utc=tuple(source_timestamps),
            )
        )

        return action
