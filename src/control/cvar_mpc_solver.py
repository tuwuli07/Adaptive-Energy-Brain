"""Multi-scenario Linear Programming solver for CVaR Risk-Aware Model Predictive Control.

Formulates and solves a multi-scenario risk-averse LP using the Rockafellar-Uryasev
formulation with HiGHS. Jointly optimizes expected operating cost and Conditional
Value-at-Risk (CVaR_alpha) with strict nonanticipativity on the immediate dispatch action.
"""

from dataclasses import dataclass
import time
from typing import Sequence
import numpy as np
from scipy.optimize import linprog

from src.simulation.actions import BatteryAction, finite_number
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.metrics import CostConfig, load_cost_config
from src.control.scenario_generator import UncertaintyScenario


@dataclass(frozen=True)
class CVaRSolverConfig:
    """Configuration for the CVaR-MPC optimization problem."""

    alpha: float = 0.95
    risk_aversion_lambda: float = 1.0
    solver_method: str = "highs"
    time_step_hours: float = 1.0
    terminal_inventory_value_usd_per_kwh_dc: float = 0.14736842105263157
    power_balance_tolerance_kw: float = 1e-4
    allow_curtailment: bool = True

    def __post_init__(self):
        a = finite_number(self.alpha, "alpha")
        if a <= 0.0 or a >= 1.0:
            raise ValueError(f"CVaR confidence level alpha must be in (0, 1), got {a}")
        object.__setattr__(self, "alpha", a)

        lam = finite_number(self.risk_aversion_lambda, "risk_aversion_lambda")
        if lam < 0.0:
            raise ValueError(f"Risk aversion lambda must be nonnegative, got {lam}")
        object.__setattr__(self, "risk_aversion_lambda", lam)

        if self.solver_method != "highs":
            raise ValueError("solver_method must be 'highs'")


@dataclass(frozen=True)
class CVaRSolverDiagnostics:
    """Execution diagnostics for CVaR LP solver."""

    success: bool
    status: int
    message: str
    iterations: int
    solve_time_seconds: float
    num_scenarios: int
    num_variables: int
    num_constraints: int
    var_threshold_eta: float
    excess_losses: tuple[float, ...]
    scenario_losses: tuple[float, ...]


@dataclass(frozen=True)
class CVaRSolverResult:
    """Typed result from solving a multi-scenario CVaR-MPC instance."""

    success: bool
    first_action: BatteryAction
    first_requested_power_kw: float
    expected_operating_cost_usd: float
    cvar_loss_usd: float
    var_loss_usd: float
    worst_case_scenario_cost_usd: float
    best_case_scenario_cost_usd: float
    scenario_costs_usd: tuple[float, ...]
    planned_charge_kw: tuple[float, ...]
    planned_discharge_kw: tuple[float, ...]
    planned_net_battery_kw: tuple[float, ...]
    planned_energy_kwh: tuple[float, ...]
    planned_soc: tuple[float, ...]
    planned_grid_import_kw: tuple[float, ...]
    planned_grid_export_kw: tuple[float, ...]
    planned_curtailment_kw: tuple[float, ...]
    total_objective_usd: float
    diagnostics: CVaRSolverDiagnostics


def solve_cvar_mpc_linear_program(
    initial_energy_kwh: float,
    scenarios: Sequence[UncertaintyScenario],
    sim_config: SimulationConfig | None = None,
    cost_config: CostConfig | None = None,
    cvar_config: CVaRSolverConfig | None = None,
) -> CVaRSolverResult:
    """Formulate and solve the multi-scenario CVaR risk-aware battery dispatch LP.

    Decision variables structure:
      For each scenario s in 0..S-1 (6*H variables per scenario):
        - x[s * 6H + 5k + 0]: P_charge[s, k] >= 0
        - x[s * 6H + 5k + 1]: P_discharge[s, k] >= 0
        - x[s * 6H + 5k + 2]: P_import[s, k] >= 0
        - x[s * 6H + 5k + 3]: P_export[s, k] >= 0
        - x[s * 6H + 5k + 4]: P_curtail[s, k] >= 0
        - x[s * 6H + 5H + k]: E[s, k+1] in [E_min, E_max]
      Auxiliary CVaR variables:
        - x[6HS + 0]: eta (Value-at-Risk threshold, free variable)
        - x[6HS + 1 + s]: z_s (Excess loss for scenario s >= 0)
    Total variables: N = 6 * H * S + 1 + S.
    """
    start_time = time.perf_counter()

    if sim_config is None:
        sim_config = load_simulation_config()
    if cost_config is None:
        cost_config = load_cost_config()
    if cvar_config is None:
        cvar_config = CVaRSolverConfig()

    num_scenarios = len(scenarios)
    if num_scenarios <= 0:
        raise ValueError("At least one uncertainty scenario is required.")

    h = scenarios[0].horizon_hours
    dt = cvar_config.time_step_hours
    alpha = cvar_config.alpha
    lam = cvar_config.risk_aversion_lambda

    # Battery limits
    bat = sim_config.battery
    capacity = bat.capacity_kwh
    e_min = capacity * bat.soc_min
    e_max = capacity * bat.soc_max
    p_charge_max = bat.charge_limit_kw
    p_discharge_max = bat.discharge_limit_kw
    eta_c = bat.charge_efficiency
    eta_d = bat.discharge_efficiency

    # Grid limits
    grid = sim_config.grid
    p_import_max = grid.import_limit_kw
    p_export_max = grid.export_limit_kw

    c_deg = cost_config.battery_degradation_per_kwh_throughput
    v_term = cvar_config.terminal_inventory_value_usd_per_kwh_dc
    e_0 = finite_number(initial_energy_kwh, "initial_energy_kwh")

    # Dimensions
    vars_per_scenario = 6 * h
    cvar_start_idx = num_scenarios * vars_per_scenario
    total_vars = cvar_start_idx + 1 + num_scenarios
    eta_idx = cvar_start_idx
    z_start_idx = cvar_start_idx + 1

    c_obj = np.zeros(total_vars, dtype=np.float64)
    bounds = []

    # 1. Build variable bounds and objective coefficients
    for s_idx, sc in enumerate(scenarios):
        p_s = sc.probability
        s_offset = s_idx * vars_per_scenario

        for k in range(h):
            # Dispatch bounds
            bounds.append((0.0, p_charge_max))
            bounds.append((0.0, p_discharge_max))
            bounds.append((0.0, p_import_max))
            bounds.append((0.0, p_export_max))

            p_ren = max(0.0, sc.solar_generation_kw[k] + sc.wind_generation_kw[k])
            if cvar_config.allow_curtailment and sim_config.renewable_curtailment.enabled:
                bounds.append((0.0, p_ren))
            else:
                bounds.append((0.0, 0.0))

            # Expected cost objective coefficients (weighted by p_s):
            # Degradation cost
            c_obj[s_offset + 5 * k + 0] += p_s * c_deg * dt
            c_obj[s_offset + 5 * k + 1] += p_s * c_deg * dt
            # Import cost
            c_obj[s_offset + 5 * k + 2] += p_s * sc.import_price[k] * dt
            # Export revenue (negative cost)
            c_obj[s_offset + 5 * k + 3] += -p_s * sc.export_price[k] * dt

        # State bounds E[s, k+1]
        for k in range(h):
            bounds.append((e_min, e_max))

        # Terminal inventory credit in expected cost: - p_s * v_term * (E[s, H] - E_0)
        c_obj[s_offset + 5 * h + (h - 1)] += -p_s * v_term

    # Bounds & objective for auxiliary CVaR variables:
    # eta is free (unbounded real)
    bounds.append((-1e7, 1e7))
    c_obj[eta_idx] = lam  # lambda * eta

    # z_s >= 0
    for s_idx, sc in enumerate(scenarios):
        bounds.append((0.0, None))
        c_obj[z_start_idx + s_idx] = (lam * sc.probability) / (1.0 - alpha)

    # 2. Equality Constraints (A_eq x = b_eq):
    # - Power balance per scenario (H * S)
    # - Battery energy dynamics per scenario (H * S)
    # - Nonanticipativity on step 0: charge and discharge must be equal across all scenarios (2 * (S-1))
    n_pb = num_scenarios * h
    n_dyn = num_scenarios * h
    n_na = 2 * (num_scenarios - 1)
    total_eq = n_pb + n_dyn + n_na

    a_eq = np.zeros((total_eq, total_vars), dtype=np.float64)
    b_eq = np.zeros(total_eq, dtype=np.float64)

    eq_row = 0
    # Power balance & Dynamics per scenario
    for s_idx, sc in enumerate(scenarios):
        s_offset = s_idx * vars_per_scenario

        for k in range(h):
            # Power balance: -P_c + P_d + P_imp - P_exp - P_curt = P_demand - P_solar - P_wind
            row = eq_row
            a_eq[row, s_offset + 5 * k + 0] = -1.0
            a_eq[row, s_offset + 5 * k + 1] = +1.0
            a_eq[row, s_offset + 5 * k + 2] = +1.0
            a_eq[row, s_offset + 5 * k + 3] = -1.0
            a_eq[row, s_offset + 5 * k + 4] = -1.0
            b_eq[row] = sc.demand_kw[k] - (sc.solar_generation_kw[k] + sc.wind_generation_kw[k])
            eq_row += 1

            # Dynamics: -eta_c dt P_c + (dt/eta_d) P_d + E[k+1] - E[k] = 0
            d_row = eq_row
            a_eq[d_row, s_offset + 5 * k + 0] = -eta_c * dt
            a_eq[d_row, s_offset + 5 * k + 1] = +(dt / eta_d)
            a_eq[d_row, s_offset + 5 * h + k] = +1.0
            if k == 0:
                b_eq[d_row] = e_0
            else:
                a_eq[d_row, s_offset + 5 * h + (k - 1)] = -1.0
                b_eq[d_row] = 0.0
            eq_row += 1

    # Nonanticipativity: link scenario s (s >= 1) to scenario 0 at step k=0
    for s_idx in range(1, num_scenarios):
        s_offset = s_idx * vars_per_scenario
        # Charge equality: P_charge[s, 0] - P_charge[0, 0] = 0
        a_eq[eq_row, s_offset + 0] = 1.0
        a_eq[eq_row, 0] = -1.0
        b_eq[eq_row] = 0.0
        eq_row += 1

        # Discharge equality: P_discharge[s, 0] - P_discharge[0, 0] = 0
        a_eq[eq_row, s_offset + 1] = 1.0
        a_eq[eq_row, 1] = -1.0
        b_eq[eq_row] = 0.0
        eq_row += 1

    # 3. Inequality Constraints (A_ub x <= b_ub):
    # CVaR excess loss constraints for each scenario s:
    # L_s - eta - z_s <= 0
    # where L_s = sum_k dt [ c_imp P_imp - c_exp P_exp + c_deg (P_c + P_d) ] - v_term (E_H - E_0)
    # => sum_k dt [ c_imp P_imp - c_exp P_exp + c_deg (P_c + P_d) ] - v_term E_H - eta - z_s <= - v_term E_0
    total_ub = num_scenarios
    a_ub = np.zeros((total_ub, total_vars), dtype=np.float64)
    b_ub = np.zeros(total_ub, dtype=np.float64)

    for s_idx, sc in enumerate(scenarios):
        s_offset = s_idx * vars_per_scenario
        for k in range(h):
            a_ub[s_idx, s_offset + 5 * k + 0] = c_deg * dt
            a_ub[s_idx, s_offset + 5 * k + 1] = c_deg * dt
            a_ub[s_idx, s_offset + 5 * k + 2] = sc.import_price[k] * dt
            a_ub[s_idx, s_offset + 5 * k + 3] = -sc.export_price[k] * dt

        # Terminal state coefficient: -v_term * E_H
        a_ub[s_idx, s_offset + 5 * h + (h - 1)] = -v_term
        # -eta
        a_ub[s_idx, eta_idx] = -1.0
        # -z_s
        a_ub[s_idx, z_start_idx + s_idx] = -1.0
        # Constant on RHS: - v_term * E_0
        b_ub[s_idx] = -v_term * e_0

    # 4. Solve with HiGHS
    res = linprog(
        c=c_obj,
        A_eq=a_eq,
        b_eq=b_eq,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method=cvar_config.solver_method,
    )
    solve_duration = time.perf_counter() - start_time

    if not res.success or res.x is None:
        # Graceful failure handling
        diag = CVaRSolverDiagnostics(
            success=False,
            status=int(res.status),
            message=str(res.message),
            iterations=int(getattr(res, "nit", 0)),
            solve_time_seconds=float(solve_duration),
            num_scenarios=num_scenarios,
            num_variables=total_vars,
            num_constraints=total_eq + total_ub,
            var_threshold_eta=0.0,
            excess_losses=tuple([0.0] * num_scenarios),
            scenario_losses=tuple([0.0] * num_scenarios),
        )
        return CVaRSolverResult(
            success=False,
            first_action=BatteryAction(0.0),
            first_requested_power_kw=0.0,
            expected_operating_cost_usd=0.0,
            cvar_loss_usd=0.0,
            var_loss_usd=0.0,
            worst_case_scenario_cost_usd=0.0,
            best_case_scenario_cost_usd=0.0,
            scenario_costs_usd=tuple([0.0] * num_scenarios),
            planned_charge_kw=tuple([0.0] * h),
            planned_discharge_kw=tuple([0.0] * h),
            planned_net_battery_kw=tuple([0.0] * h),
            planned_energy_kwh=tuple([e_0] * (h + 1)),
            planned_soc=tuple([e_0 / capacity] * (h + 1)),
            planned_grid_import_kw=tuple([0.0] * h),
            planned_grid_export_kw=tuple([0.0] * h),
            planned_curtailment_kw=tuple([0.0] * h),
            total_objective_usd=float("nan"),
            diagnostics=diag,
        )

    # Extract solution
    x = res.x
    eta_val = float(x[eta_idx])
    z_vals = tuple(max(0.0, float(x[z_start_idx + s])) for s in range(num_scenarios))

    # Compute scenario operating costs L_s
    scenario_costs = []
    for s_idx, sc in enumerate(scenarios):
        s_offset = s_idx * vars_per_scenario
        raw_cost = 0.0
        for k in range(h):
            p_c = max(0.0, float(x[s_offset + 5 * k + 0]))
            p_d = max(0.0, float(x[s_offset + 5 * k + 1]))
            p_imp = max(0.0, float(x[s_offset + 5 * k + 2]))
            p_exp = max(0.0, float(x[s_offset + 5 * k + 3]))

            raw_cost += sc.import_price[k] * dt * p_imp
            raw_cost -= sc.export_price[k] * dt * p_exp
            raw_cost += c_deg * dt * (p_c + p_d)

        e_term = max(0.0, float(x[s_offset + 5 * h + (h - 1)]))
        term_credit = v_term * (e_term - e_0)
        adj_cost = raw_cost - term_credit
        scenario_costs.append(adj_cost)

    scenario_costs = tuple(scenario_costs)
    expected_cost = sum(sc.probability * c for sc, c in zip(scenarios, scenario_costs))
    cvar_val = eta_val + (1.0 / (1.0 - alpha)) * sum(sc.probability * z for sc, z in zip(scenarios, z_vals))
    worst_cost = max(scenario_costs)
    best_cost = min(scenario_costs)

    # First action (from scenario 0, strictly identical across all s)
    first_p_c = max(0.0, float(x[0]))
    first_p_d = max(0.0, float(x[1]))
    first_net_p = first_p_d - first_p_c

    # Nominal trajectory (from scenario 0)
    p_c_nom = tuple(max(0.0, float(x[5 * k + 0])) for k in range(h))
    p_d_nom = tuple(max(0.0, float(x[5 * k + 1])) for k in range(h))
    p_net_nom = tuple(p_d_nom[k] - p_c_nom[k] for k in range(h))
    p_imp_nom = tuple(max(0.0, float(x[5 * k + 2])) for k in range(h))
    p_exp_nom = tuple(max(0.0, float(x[5 * k + 3])) for k in range(h))
    p_curt_nom = tuple(max(0.0, float(x[5 * k + 4])) for k in range(h))
    e_nom = tuple([e_0] + [max(0.0, float(x[5 * h + k])) for k in range(h)])
    soc_nom = tuple(e / capacity for e in e_nom)

    diag = CVaRSolverDiagnostics(
        success=True,
        status=int(res.status),
        message=str(res.message),
        iterations=int(getattr(res, "nit", 0)),
        solve_time_seconds=float(solve_duration),
        num_scenarios=num_scenarios,
        num_variables=total_vars,
        num_constraints=total_eq + total_ub,
        var_threshold_eta=eta_val,
        excess_losses=z_vals,
        scenario_losses=scenario_costs,
    )

    return CVaRSolverResult(
        success=True,
        first_action=BatteryAction(requested_power_kw=first_net_p),
        first_requested_power_kw=first_net_p,
        expected_operating_cost_usd=float(expected_cost),
        cvar_loss_usd=float(cvar_val),
        var_loss_usd=float(eta_val),
        worst_case_scenario_cost_usd=float(worst_cost),
        best_case_scenario_cost_usd=float(best_cost),
        scenario_costs_usd=scenario_costs,
        planned_charge_kw=p_c_nom,
        planned_discharge_kw=p_d_nom,
        planned_net_battery_kw=p_net_nom,
        planned_energy_kwh=e_nom,
        planned_soc=soc_nom,
        planned_grid_import_kw=p_imp_nom,
        planned_grid_export_kw=p_exp_nom,
        planned_curtailment_kw=p_curt_nom,
        total_objective_usd=float(res.fun),
        diagnostics=diag,
    )
