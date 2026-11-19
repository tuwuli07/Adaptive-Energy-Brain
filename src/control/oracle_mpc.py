"""24-Hour rolling perfect-foresight MPC controller for diagnostic benchmarking.

Part 7.2 implementation: extracts future realized profile rows from a trusted
partition snapshot (e.g., validation split) for a 24-hour lookahead window and
solves the linear program at each hourly step.
"""

from collections.abc import Sequence
from dataclasses import dataclass
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
from src.simulation.data_loader import HourlyObservation, SimulationData
from src.simulation.metrics import CostConfig, load_cost_config
from src.simulation.state import AgentObservation


@dataclass(frozen=True)
class OracleStepDiagnostic:
    """Per-step diagnostics for the rolling Oracle MPC controller."""

    timestamp_utc: str
    success: bool
    status: int
    solve_time_seconds: float
    first_requested_power_kw: float
    planned_charge_kw_step0: float
    planned_discharge_kw_step0: float
    has_simultaneous_battery_flow: bool
    has_simultaneous_grid_flow: bool
    horizon_hours: int


class OracleMPCController(BaselineController):
    """Rolling-horizon MPC controller with perfect foresight into a given partition.

    At each decision step, extracts the upcoming H hours (default 24) of realized
    demand, solar, wind, and prices from the provided partition data, constructs
    an MPCInputProfile with the current pre-action battery state, and solves the LP.
    If the solver fails or strict exclusivity is violated, falls back safely to idle (0 kW).
    """

    def __init__(
        self,
        partition_data: SimulationData | Sequence[HourlyObservation],
        sim_config: SimulationConfig | None = None,
        cost_config: CostConfig | None = None,
        mpc_config: MPCSolverConfig | None = None,
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

        self._rows: tuple[HourlyObservation, ...] = tuple(partition_data)
        self._timestamp_to_index: dict[str, int] = {
            r.timestamp_utc: i for i, r in enumerate(self._rows)
        }

        self.step_diagnostics: list[OracleStepDiagnostic] = []
        self.solve_times: list[float] = []
        self.step_actions: list[BatteryAction] = []
        self.fallback_count: int = 0

    def _reset(self) -> None:
        """Reset episode-local tracking state."""
        self.step_diagnostics.clear()
        self.solve_times.clear()
        self.step_actions.clear()
        self.fallback_count = 0

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        """Formulate and solve LP for the current rolling window, returning first action."""
        decision_ts = observation.decision_timestamp_utc
        if decision_ts not in self._timestamp_to_index:
            self.fallback_count += 1
            action = BatteryAction(0.0)
            self.step_actions.append(action)
            return action

        idx = self._timestamp_to_index[decision_ts]
        horizon = min(self.mpc_config.horizon_hours, len(self._rows) - idx)
        future_rows = self._rows[idx : idx + horizon]

        capacity_kwh = self.sim_config.battery.capacity_kwh
        initial_energy_kwh = observation.state.battery_soc * capacity_kwh

        profile = MPCInputProfile(
            initial_energy_kwh=initial_energy_kwh,
            demand_kw=tuple(r.campus_load_kw for r in future_rows),
            solar_generation_kw=tuple(r.pv_available_kw for r in future_rows),
            wind_generation_kw=tuple(r.wind_available_kw for r in future_rows),
            import_price=tuple(r.import_price for r in future_rows),
            export_price=tuple(
                r.export_price if r.export_price is not None else 0.0 for r in future_rows
            ),
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
            OracleStepDiagnostic(
                timestamp_utc=decision_ts,
                success=res.success,
                status=res.diagnostics.status,
                solve_time_seconds=solve_time,
                first_requested_power_kw=action.requested_power_kw,
                planned_charge_kw_step0=(
                    res.planned_charge_kw[0] if res.planned_charge_kw else 0.0
                ),
                planned_discharge_kw_step0=(
                    res.planned_discharge_kw[0] if res.planned_discharge_kw else 0.0
                ),
                has_simultaneous_battery_flow=res.diagnostics.has_simultaneous_battery_flow,
                has_simultaneous_grid_flow=res.diagnostics.has_simultaneous_grid_flow,
                horizon_hours=horizon,
            )
        )

        return action
