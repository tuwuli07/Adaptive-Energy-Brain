"""Phase 8.3 Reliability and Fallback Evaluation Module.

Evaluates controller robustness under synthetic stress conditions and fault-injection fixtures:
- Solver infeasibility and timeout fallback handling.
- Non-finite (NaN/Inf) forecast input rejection.
- Forecast horizon shape mismatch validation.
- Battery SOC boundary clamping and physical power limit enforcement.
- Safe default dispatch verification (idle 0 kW).
Keeps all fault-injection experiments strictly isolated from authoritative benchmarks.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.control.gnn_transformer_mpc import GNNTransformerMPCController
from src.control.mpc_solver import (
    MPCInputProfile,
    MPCSolverConfig,
    MPCSolverDiagnostics,
    MPCSolverResult,
    load_mpc_config,
    solve_mpc_linear_program,
)
from src.control.seasonal_persistence_mpc import SeasonalPersistenceMPCController
from src.simulation.actions import BatteryAction
from src.simulation.battery import Battery, BatteryState
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.data_loader import HourlyObservation, load_simulation_split
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config
from src.simulation.state import AgentObservation


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReliabilityTestResult:
    """Summary of a reliability / fault-injection audit case."""

    test_case_id: str
    scenario_description: str
    expected_behavior: str
    actual_behavior: str
    safe_fallback_verified: bool
    exceptions_trapped: str
    notes: str


def audit_solver_infeasibility_fallback() -> ReliabilityTestResult:
    """Test solver handling when constraints cannot be satisfied."""
    mpc_config = load_mpc_config()
    sim_config = load_simulation_config()

    # Create an impossible profile where demand is huge and import is disabled or limits violated
    h = mpc_config.horizon_hours
    profile = MPCInputProfile(
        initial_energy_kwh=250.0,
        demand_kw=tuple([10000.0] * h),  # Exceeds max grid import (2000 kW) + battery (150 kW)
        solar_generation_kw=tuple([0.0] * h),
        wind_generation_kw=tuple([0.0] * h),
        import_price=tuple([0.12] * h),
        export_price=tuple([0.04] * h),
    )

    result = solve_mpc_linear_program(profile, sim_config=sim_config, mpc_config=mpc_config)

    # Infeasibility: LP solver terminates with non-zero status and success=False
    is_safe = not result.success and (result.first_requested_power_kw == 0.0 or result.first_action.requested_power_kw == 0.0)

    return ReliabilityTestResult(
        test_case_id="REL-01-INFEASIBILITY",
        scenario_description="Massive campus demand (10,000 kW) exceeding grid and battery capacity",
        expected_behavior="Solver returns success=False; does not crash unhandled",
        actual_behavior=f"Status: {result.diagnostics.status}, Message: {result.diagnostics.message}, Success: {result.success}",
        safe_fallback_verified=is_safe,
        exceptions_trapped="None (handled within solver)",
        notes=f"HiGHS solver iteration count: {result.diagnostics.iterations}",
    )


def audit_non_finite_forecast_protection() -> ReliabilityTestResult:
    """Test rejection of NaN and Inf in forecast profiles."""
    h = 24
    trapped_nan = False
    trapped_inf = False

    # 1. NaN in demand
    try:
        MPCInputProfile(
            initial_energy_kwh=250.0,
            demand_kw=tuple([float("nan")] * h),
            solar_generation_kw=tuple([0.0] * h),
            wind_generation_kw=tuple([0.0] * h),
            import_price=tuple([0.12] * h),
            export_price=tuple([0.04] * h),
        )
    except ValueError as e:
        if "finite" in str(e).lower() or "nan" in str(e).lower():
            trapped_nan = True

    # 2. Inf in solar
    try:
        MPCInputProfile(
            initial_energy_kwh=250.0,
            demand_kw=tuple([100.0] * h),
            solar_generation_kw=tuple([float("inf")] * h),
            wind_generation_kw=tuple([0.0] * h),
            import_price=tuple([0.12] * h),
            export_price=tuple([0.04] * h),
        )
    except ValueError as e:
        if "finite" in str(e).lower() or "inf" in str(e).lower():
            trapped_inf = True

    safe = trapped_nan and trapped_inf

    return ReliabilityTestResult(
        test_case_id="REL-02-NON-FINITE-PROTECTION",
        scenario_description="Injection of NaN and Inf into forecast profiles",
        expected_behavior="Strict rejection via ValueError ('must be finite')",
        actual_behavior=f"NaN Trapped: {trapped_nan}, Inf Trapped: {trapped_inf}",
        safe_fallback_verified=safe,
        exceptions_trapped="ValueError (finite_number check)",
        notes="Prevents numerical poisoning of linear program matrices",
    )


def audit_horizon_mismatch_protection() -> ReliabilityTestResult:
    """Test validation when forecast horizon does not match configured H."""
    sim_config = load_simulation_config()
    mpc_config = load_mpc_config()

    # Pass profile with length mismatch across components
    mismatch_trapped = False
    try:
        MPCInputProfile(
            initial_energy_kwh=250.0,
            demand_kw=tuple([100.0] * 12),
            solar_generation_kw=tuple([0.0] * 24),  # Mismatch with demand length
            wind_generation_kw=tuple([0.0] * 12),
            import_price=tuple([0.12] * 12),
            export_price=tuple([0.04] * 12),
        )
    except ValueError as e:
        if "mismatch" in str(e).lower():
            mismatch_trapped = True

    return ReliabilityTestResult(
        test_case_id="REL-03-HORIZON-MISMATCH",
        scenario_description="Passing mismatched profile lengths across forecast series",
        expected_behavior="Rejection via ValueError identifying profile length mismatch",
        actual_behavior=f"Length Mismatch Trapped: {mismatch_trapped}",
        safe_fallback_verified=mismatch_trapped,
        exceptions_trapped="ValueError (length mismatch check)",
        notes="Ensures dimension alignment across decision horizons",
    )


def audit_soc_boundary_clamping() -> ReliabilityTestResult:
    """Test physical battery clamping at extreme SOC boundaries."""
    sim_config = load_simulation_config()
    battery = Battery(sim_config)

    # 1. Battery at SOC max (0.90) requesting full charge (-150 kW)
    battery.reset(initial_soc=sim_config.battery.soc_max)
    t1 = battery.step(requested_power_kw=-150.0)

    # 2. Battery at SOC min (0.10) requesting full discharge (+150 kW)
    battery.reset(initial_soc=sim_config.battery.soc_min)
    t2 = battery.step(requested_power_kw=150.0)

    clamped_charge = abs(t1.actual_power_kw) < 1e-6 and t1.power_limited
    clamped_discharge = abs(t2.actual_power_kw) < 1e-6 and t2.power_limited
    safe = clamped_charge and clamped_discharge

    return ReliabilityTestResult(
        test_case_id="REL-04-SOC-BOUNDARY-CLAMPING",
        scenario_description="Battery at SOC=0.90 requesting -150 kW charge; at SOC=0.10 requesting +150 kW discharge",
        expected_behavior="Physical battery model safely clamps actual power to 0 kW with power_limited=True",
        actual_behavior=(
            f"At SOC {sim_config.battery.soc_max} actual: {t1.actual_power_kw:.2f} kW (reasons={t1.limiting_reasons}); "
            f"At SOC {sim_config.battery.soc_min} actual: {t2.actual_power_kw:.2f} kW (reasons={t2.limiting_reasons})"
        ),
        safe_fallback_verified=safe,
        exceptions_trapped="None (handled via physical saturation)",
        notes="Confirms battery physical state invariants are strictly preserved",
    )


def audit_excess_power_limit_clamping() -> ReliabilityTestResult:
    """Test physical battery clamping when requested power exceeds inverter limits."""
    sim_config = load_simulation_config()
    battery = Battery(sim_config)
    battery.reset(initial_soc=0.50)

    # Request extreme +500 kW discharge (hardware inverter limit is 150 kW)
    t = battery.step(requested_power_kw=500.0)
    actual_p = t.actual_power_kw
    safe = abs(actual_p - sim_config.battery.discharge_limit_kw) < 1e-6 and t.power_limited

    return ReliabilityTestResult(
        test_case_id="REL-05-POWER-LIMIT-CLAMPING",
        scenario_description="Requesting +500 kW discharge (exceeding 150 kW hardware limit)",
        expected_behavior=f"Clamped to {sim_config.battery.discharge_limit_kw:.1f} kW maximum limit with power_limited=True",
        actual_behavior=f"Actual power dispatched: {actual_p:.2f} kW, Reasons: {t.limiting_reasons}",
        safe_fallback_verified=safe,
        exceptions_trapped="None (handled via hardware rate limits)",
        notes="Guarantees battery inverter hardware protection",
    )


def audit_controller_fallback_to_idle() -> ReliabilityTestResult:
    """Test that MPC controller falls back strictly to 0.0 kW idle on solver failure."""
    split_root = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
    val_data = load_simulation_split(split_root, "validation")
    from scripts.run_simulation import _components

    sim_config = load_simulation_config()
    cost_config = load_cost_config()
    from src.simulation.data_loader import SimulationData
    val_slice = SimulationData(val_data[:5], val_data.metadata)
    renewables, demand = _components(val_slice, ROOT / "configs" / "synthetic_scenario.yaml", "hypothetical_example")
    env = MicrogridEnvironment(
        val_slice,
        renewables,
        demand,
        sim_config,
        cost_config,
    )
    obs = env.reset()

    sp_ctrl = SeasonalPersistenceMPCController(initial_history=val_data[:24])

    diag = MPCSolverDiagnostics(
        success=False,
        status=2,
        message="Model infeasible",
        iterations=0,
        solve_time_seconds=0.001,
        has_simultaneous_battery_flow=False,
        max_simultaneous_battery_kw=0.0,
        has_simultaneous_grid_flow=False,
        max_simultaneous_grid_kw=0.0,
    )
    failed_result = MPCSolverResult(
        success=False,
        first_action=BatteryAction(0.0),
        first_requested_power_kw=0.0,
        planned_charge_kw=(0.0,) * 24,
        planned_discharge_kw=(0.0,) * 24,
        planned_net_battery_kw=(0.0,) * 24,
        planned_energy_kwh=(250.0,) * 24,
        planned_soc=(0.5,) * 24,
        planned_grid_import_kw=(0.0,) * 24,
        planned_grid_export_kw=(0.0,) * 24,
        planned_curtailment_kw=(0.0,) * 24,
        total_cost_usd=0.0,
        import_cost_usd=0.0,
        export_revenue_usd=0.0,
        degradation_cost_usd=0.0,
        terminal_inventory_credit_usd=0.0,
        net_objective_usd=0.0,
        diagnostics=diag,
    )

    with patch("src.control.seasonal_persistence_mpc.solve_mpc_linear_program", return_value=failed_result):
        action = sp_ctrl.select_action(obs)

    is_safe = (action.requested_power_kw == 0.0) and (sp_ctrl.fallback_count > 0)

    return ReliabilityTestResult(
        test_case_id="REL-06-CONTROLLER-IDLE-FALLBACK",
        scenario_description="Seasonal Persistence MPC solver fails or returns success=False",
        expected_behavior="Controller emits safe idle action BatteryAction(0.0 kW) and increments fallback_count",
        actual_behavior=f"Requested power: {action.requested_power_kw:.2f} kW, Fallback count: {sp_ctrl.fallback_count}",
        safe_fallback_verified=is_safe,
        exceptions_trapped="None (graceful fallback to 0.0 kW)",
        notes="Prevents runaway or uncontrolled action on solver failure",
    )


def run_all_reliability_audits() -> list[ReliabilityTestResult]:
    """Execute all reliability audit test cases."""
    return [
        audit_solver_infeasibility_fallback(),
        audit_non_finite_forecast_protection(),
        audit_horizon_mismatch_protection(),
        audit_soc_boundary_clamping(),
        audit_excess_power_limit_clamping(),
        audit_controller_fallback_to_idle(),
    ]
