"""Standalone rolling-horizon Linear Programming solver for battery dispatch.

This module provides the core mathematical optimization formulation for Model
Predictive Control (MPC). It solves an H-step linear program over continuous
dispatch variables (charge, discharge, grid import, grid export, curtailment)
and battery energy state variables using scipy.optimize.linprog with the HiGHS
solver.
"""

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
import yaml

from src.simulation.actions import BatteryAction, finite_number
from src.simulation.config import SimulationConfig, _UniqueKeyLoader, load_simulation_config
from src.simulation.metrics import CostConfig, load_cost_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MPC_CONFIG_PATH = ROOT / "configs" / "mpc_controller.yaml"


@dataclass(frozen=True)
class MPCSolverConfig:
    version: int
    horizon_hours: int
    time_step_hours: float
    solver_method: str
    terminal_inventory_value_usd_per_kwh_dc: float
    power_balance_tolerance_kw: float
    simultaneous_flow_tolerance_kw: float
    allow_curtailment: bool
    strict_exclusivity: bool = True

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("MPC configuration version must be 1.")
        if type(self.horizon_hours) is not int or self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be a positive integer.")
        step = finite_number(self.time_step_hours, "time_step_hours")
        if step <= 0:
            raise ValueError("time_step_hours must be positive.")
        object.__setattr__(self, "time_step_hours", step)
        if self.solver_method != "highs":
            raise ValueError("solver_method must be 'highs'.")
        v_term = finite_number(
            self.terminal_inventory_value_usd_per_kwh_dc,
            "terminal_inventory_value_usd_per_kwh_dc",
        )
        if v_term < 0:
            raise ValueError("terminal_inventory_value_usd_per_kwh_dc must be nonnegative.")
        object.__setattr__(self, "terminal_inventory_value_usd_per_kwh_dc", v_term)
        tol_p = finite_number(self.power_balance_tolerance_kw, "power_balance_tolerance_kw")
        if tol_p <= 0:
            raise ValueError("power_balance_tolerance_kw must be positive.")
        object.__setattr__(self, "power_balance_tolerance_kw", tol_p)
        tol_s = finite_number(self.simultaneous_flow_tolerance_kw, "simultaneous_flow_tolerance_kw")
        if tol_s <= 0:
            raise ValueError("simultaneous_flow_tolerance_kw must be positive.")
        object.__setattr__(self, "simultaneous_flow_tolerance_kw", tol_s)
        if not isinstance(self.allow_curtailment, bool):
            raise ValueError("allow_curtailment must be a boolean.")
        if not isinstance(self.strict_exclusivity, bool):
            raise ValueError("strict_exclusivity must be a boolean.")


def load_mpc_config(path=None) -> MPCSolverConfig:
    """Load validated MPC configuration with duplicate key rejection."""
    config_path = Path(path) if path is not None else DEFAULT_MPC_CONFIG_PATH
    try:
        values = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        expected = {
            "version",
            "horizon_hours",
            "time_step_hours",
            "solver_method",
            "terminal_inventory_value_usd_per_kwh_dc",
            "power_balance_tolerance_kw",
            "simultaneous_flow_tolerance_kw",
            "allow_curtailment",
            "strict_exclusivity",
        }
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"MPC configuration requires exactly fields: {sorted(expected)}")
        return MPCSolverConfig(**values)
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MPC configuration {config_path}: {exc}") from exc


@dataclass(frozen=True)
class MPCInputProfile:
    """Forecast and initial state profile over the planning horizon H."""

    initial_energy_kwh: float
    demand_kw: tuple[float, ...]
    solar_generation_kw: tuple[float, ...]
    wind_generation_kw: tuple[float, ...]
    import_price: tuple[float, ...]
    export_price: tuple[float, ...]

    def __post_init__(self):
        init_e = finite_number(self.initial_energy_kwh, "initial_energy_kwh")
        if init_e < 0:
            raise ValueError("initial_energy_kwh must be nonnegative.")
        object.__setattr__(self, "initial_energy_kwh", init_e)

        h = len(self.demand_kw)
        if h <= 0:
            raise ValueError("demand_kw profile cannot be empty.")

        for name in (
            "solar_generation_kw",
            "wind_generation_kw",
            "import_price",
            "export_price",
        ):
            seq = getattr(self, name)
            if len(seq) != h:
                raise ValueError(f"Profile length mismatch: {name} length {len(seq)} != horizon {h}.")

        object.__setattr__(
            self,
            "demand_kw",
            tuple(finite_number(x, "demand_kw") for x in self.demand_kw),
        )
        object.__setattr__(
            self,
            "solar_generation_kw",
            tuple(finite_number(x, "solar_generation_kw") for x in self.solar_generation_kw),
        )
        object.__setattr__(
            self,
            "wind_generation_kw",
            tuple(finite_number(x, "wind_generation_kw") for x in self.wind_generation_kw),
        )
        object.__setattr__(
            self,
            "import_price",
            tuple(finite_number(x, "import_price") for x in self.import_price),
        )
        object.__setattr__(
            self,
            "export_price",
            tuple(finite_number(x, "export_price") for x in self.export_price),
        )

    @property
    def horizon_hours(self) -> int:
        return len(self.demand_kw)


@dataclass(frozen=True)
class MPCSolverDiagnostics:
    """Diagnostic details regarding solver execution and flow exclusivity."""

    success: bool
    status: int
    message: str
    iterations: int
    solve_time_seconds: float
    has_simultaneous_battery_flow: bool
    max_simultaneous_battery_kw: float
    has_simultaneous_grid_flow: bool
    max_simultaneous_grid_kw: float


@dataclass(frozen=True)
class MPCSolverResult:
    """Typed result from solving an MPC horizon instance."""

    success: bool
    first_action: BatteryAction
    first_requested_power_kw: float
    planned_charge_kw: tuple[float, ...]
    planned_discharge_kw: tuple[float, ...]
    planned_net_battery_kw: tuple[float, ...]
    planned_energy_kwh: tuple[float, ...]
    planned_soc: tuple[float, ...]
    planned_grid_import_kw: tuple[float, ...]
    planned_grid_export_kw: tuple[float, ...]
    planned_curtailment_kw: tuple[float, ...]
    total_cost_usd: float
    import_cost_usd: float
    export_revenue_usd: float
    degradation_cost_usd: float
    terminal_inventory_credit_usd: float
    net_objective_usd: float
    diagnostics: MPCSolverDiagnostics


def solve_mpc_linear_program(
    profile: MPCInputProfile,
    sim_config: SimulationConfig | None = None,
    cost_config: CostConfig | None = None,
    mpc_config: MPCSolverConfig | None = None,
    *,
    override_terminal_value: float | None = None,
) -> MPCSolverResult:
    """Formulate and solve the rolling-horizon LP battery dispatch problem.

    Variables per time step k (0 <= k < H):
      - x[5k + 0]: P_charge[k] >= 0
      - x[5k + 1]: P_discharge[k] >= 0
      - x[5k + 2]: P_import[k] >= 0
      - x[5k + 3]: P_export[k] >= 0
      - x[5k + 4]: P_curtail[k] >= 0
    State variables:
      - x[5H + k]: E[k+1] >= 0 for k in 0 <= k < H

    Total variables: n = 6 * H.
    """
    start_time = time.perf_counter()

    if sim_config is None:
        sim_config = load_simulation_config()
    if cost_config is None:
        cost_config = load_cost_config()
    if mpc_config is None:
        mpc_config = load_mpc_config()

    h = profile.horizon_hours
    dt = mpc_config.time_step_hours

    # Physical battery parameters
    bat = sim_config.battery
    capacity = bat.capacity_kwh
    e_min = capacity * bat.soc_min
    e_max = capacity * bat.soc_max
    p_charge_max = bat.charge_limit_kw
    p_discharge_max = bat.discharge_limit_kw
    eta_c = bat.charge_efficiency
    eta_d = bat.discharge_efficiency

    # Grid parameters
    grid = sim_config.grid
    p_import_max = grid.import_limit_kw
    p_export_max = grid.export_limit_kw

    # Economic parameters
    c_deg = cost_config.battery_degradation_per_kwh_throughput
    v_term = (
        override_terminal_value
        if override_terminal_value is not None
        else mpc_config.terminal_inventory_value_usd_per_kwh_dc
    )
    v_term = finite_number(v_term, "terminal_inventory_value")

    # Initial state
    e_0 = profile.initial_energy_kwh

    # Total variables: 5 * H dispatch variables + H state variables
    n_vars = 6 * h
    c_obj = np.zeros(n_vars, dtype=np.float64)

    # Variable bounds
    bounds = []
    for k in range(h):
        # Charge
        bounds.append((0.0, p_charge_max))
        # Discharge
        bounds.append((0.0, p_discharge_max))
        # Import
        bounds.append((0.0, p_import_max))
        # Export
        bounds.append((0.0, p_export_max))
        # Curtailment
        p_ren = max(0.0, profile.solar_generation_kw[k] + profile.wind_generation_kw[k])
        if mpc_config.allow_curtailment and sim_config.renewable_curtailment.enabled:
            bounds.append((0.0, p_ren))
        else:
            bounds.append((0.0, 0.0))

        # Objective coefficients for dispatch step k:
        # Import cost
        c_obj[5 * k + 2] = profile.import_price[k] * dt
        # Export revenue (negative cost)
        c_obj[5 * k + 3] = -profile.export_price[k] * dt
        # Battery degradation cost on AC throughput
        c_obj[5 * k + 0] = c_deg * dt
        c_obj[5 * k + 1] = c_deg * dt

    # State bounds E[k+1]
    for k in range(h):
        bounds.append((e_min, e_max))

    # Terminal inventory credit: - v_term * (E_H - E_0)
    # The term + v_term * E_0 is constant; the variable coefficient on E_H is -v_term.
    c_obj[5 * h + (h - 1)] = -v_term

    # Constraints:
    # 1. Power balance (H equality constraints):
    #    P_load[k] - P_solar[k] - P_wind[k] = - P_c[k] + P_d[k] + P_imp[k] - P_exp[k] - P_curt[k]
    # 2. Battery energy dynamics (H equality constraints):
    #    E[k+1] - E[k] = eta_c * dt * P_c[k] - (dt / eta_d) * P_d[k]
    n_eq = 2 * h
    a_eq = np.zeros((n_eq, n_vars), dtype=np.float64)
    b_eq = np.zeros(n_eq, dtype=np.float64)

    for k in range(h):
        # Power balance row k
        a_eq[k, 5 * k + 0] = -1.0  # - P_c
        a_eq[k, 5 * k + 1] = +1.0  # + P_d
        a_eq[k, 5 * k + 2] = +1.0  # + P_imp
        a_eq[k, 5 * k + 3] = -1.0  # - P_exp
        a_eq[k, 5 * k + 4] = -1.0  # - P_curt

        p_net_load = profile.demand_kw[k] - (profile.solar_generation_kw[k] + profile.wind_generation_kw[k])
        b_eq[k] = p_net_load

        # Energy dynamics row H + k
        row = h + k
        a_eq[row, 5 * k + 0] = -eta_c * dt  # - eta_c * dt * P_c
        a_eq[row, 5 * k + 1] = +(dt / eta_d)  # + (dt / eta_d) * P_d
        a_eq[row, 5 * h + k] = +1.0  # + E[k+1]

        if k == 0:
            b_eq[row] = e_0
        else:
            a_eq[row, 5 * h + (k - 1)] = -1.0  # - E[k]
            b_eq[row] = 0.0

    # Solve LP using HiGHS
    res = linprog(
        c=c_obj,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=mpc_config.solver_method,
    )
    solve_duration = time.perf_counter() - start_time

    if not res.success or res.x is None:
        # Graceful failure handling
        diag = MPCSolverDiagnostics(
            success=False,
            status=int(res.status),
            message=str(res.message),
            iterations=int(getattr(res, "nit", 0)),
            solve_time_seconds=float(solve_duration),
            has_simultaneous_battery_flow=False,
            max_simultaneous_battery_kw=0.0,
            has_simultaneous_grid_flow=False,
            max_simultaneous_grid_kw=0.0,
        )
        return MPCSolverResult(
            success=False,
            first_action=BatteryAction(0.0),
            first_requested_power_kw=0.0,
            planned_charge_kw=tuple([0.0] * h),
            planned_discharge_kw=tuple([0.0] * h),
            planned_net_battery_kw=tuple([0.0] * h),
            planned_energy_kwh=tuple([e_0] * (h + 1)),
            planned_soc=tuple([e_0 / capacity] * (h + 1)),
            planned_grid_import_kw=tuple([0.0] * h),
            planned_grid_export_kw=tuple([0.0] * h),
            planned_curtailment_kw=tuple([0.0] * h),
            total_cost_usd=0.0,
            import_cost_usd=0.0,
            export_revenue_usd=0.0,
            degradation_cost_usd=0.0,
            terminal_inventory_credit_usd=0.0,
            net_objective_usd=float("nan"),
            diagnostics=diag,
        )

    # Extract solution
    x = res.x
    p_c = np.array([max(0.0, float(x[5 * k + 0])) for k in range(h)])
    p_d = np.array([max(0.0, float(x[5 * k + 1])) for k in range(h)])
    p_imp = np.array([max(0.0, float(x[5 * k + 2])) for k in range(h)])
    p_exp = np.array([max(0.0, float(x[5 * k + 3])) for k in range(h)])
    p_curt = np.array([max(0.0, float(x[5 * k + 4])) for k in range(h)])
    e_trajectory = np.array([e_0] + [max(0.0, float(x[5 * h + k])) for k in range(h)])
    soc_trajectory = e_trajectory / capacity

    # First action: positive kW discharge, negative kW charge
    net_first_battery_kw = float(p_d[0] - p_c[0])
    first_action = BatteryAction(net_first_battery_kw)

    # Simultaneous flow audits
    simult_battery = np.minimum(p_c, p_d)
    max_simult_battery = float(np.max(simult_battery))
    has_simult_battery = max_simult_battery > mpc_config.simultaneous_flow_tolerance_kw

    simult_grid = np.minimum(p_imp, p_exp)
    max_simult_grid = float(np.max(simult_grid))
    has_simult_grid = max_simult_grid > mpc_config.simultaneous_flow_tolerance_kw

    # Cost breakdown (in USD)
    imp_cost = float(np.sum(p_imp * np.array(profile.import_price) * dt))
    exp_rev = float(np.sum(p_exp * np.array(profile.export_price) * dt))
    deg_cost = float(np.sum((p_c + p_d) * c_deg * dt))
    term_credit = float(v_term * (e_trajectory[-1] - e_0))
    tot_cost = imp_cost - exp_rev + deg_cost
    net_obj = tot_cost - term_credit

    # Enforce strict exclusivity if configured
    if mpc_config.strict_exclusivity and has_simult_battery:
        diag = MPCSolverDiagnostics(
            success=False,
            status=-1,
            message=(
                f"Strict exclusivity violation: material simultaneous battery flow detected "
                f"(max={max_simult_battery:.4f} kW > {mpc_config.simultaneous_flow_tolerance_kw:.4f} kW); "
                f"safe fallback to idle."
            ),
            iterations=int(getattr(res, "nit", 0)),
            solve_time_seconds=float(solve_duration),
            has_simultaneous_battery_flow=True,
            max_simultaneous_battery_kw=max_simult_battery,
            has_simultaneous_grid_flow=has_simult_grid,
            max_simultaneous_grid_kw=max_simult_grid,
        )
        return MPCSolverResult(
            success=False,
            first_action=BatteryAction(0.0),
            first_requested_power_kw=0.0,
            planned_charge_kw=tuple(p_c.tolist()),
            planned_discharge_kw=tuple(p_d.tolist()),
            planned_net_battery_kw=tuple((p_d - p_c).tolist()),
            planned_energy_kwh=tuple(e_trajectory.tolist()),
            planned_soc=tuple(soc_trajectory.tolist()),
            planned_grid_import_kw=tuple(p_imp.tolist()),
            planned_grid_export_kw=tuple(p_exp.tolist()),
            planned_curtailment_kw=tuple(p_curt.tolist()),
            total_cost_usd=tot_cost,
            import_cost_usd=imp_cost,
            export_revenue_usd=exp_rev,
            degradation_cost_usd=deg_cost,
            terminal_inventory_credit_usd=term_credit,
            net_objective_usd=net_obj,
            diagnostics=diag,
        )

    diag = MPCSolverDiagnostics(
        success=True,
        status=int(res.status),
        message=str(res.message),
        iterations=int(getattr(res, "nit", 0)),
        solve_time_seconds=float(solve_duration),
        has_simultaneous_battery_flow=has_simult_battery,
        max_simultaneous_battery_kw=max_simult_battery,
        has_simultaneous_grid_flow=has_simult_grid,
        max_simultaneous_grid_kw=max_simult_grid,
    )

    return MPCSolverResult(
        success=True,
        first_action=first_action,
        first_requested_power_kw=net_first_battery_kw,
        planned_charge_kw=tuple(p_c.tolist()),
        planned_discharge_kw=tuple(p_d.tolist()),
        planned_net_battery_kw=tuple((p_d - p_c).tolist()),
        planned_energy_kwh=tuple(e_trajectory.tolist()),
        planned_soc=tuple(soc_trajectory.tolist()),
        planned_grid_import_kw=tuple(p_imp.tolist()),
        planned_grid_export_kw=tuple(p_exp.tolist()),
        planned_curtailment_kw=tuple(p_curt.tolist()),
        total_cost_usd=tot_cost,
        import_cost_usd=imp_cost,
        export_revenue_usd=exp_rev,
        degradation_cost_usd=deg_cost,
        terminal_inventory_credit_usd=term_credit,
        net_objective_usd=net_obj,
        diagnostics=diag,
    )
