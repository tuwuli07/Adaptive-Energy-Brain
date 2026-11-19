"""Phase 11: Distributionally Robust Optimization (DRO) MPC Linear Program Solver.

Solves the minimax distributionally robust energy management problem:
    min_u max_{q in Q} sum_{s=1}^S q_s * L_s(u)
where Q is the L1 total-variation probability ambiguity set:
    Q = { q >= 0, sum_s q_s = 1, ||q - p||_1 <= epsilon }

Using LP strong duality, the inner maximization is reformulated as:
    min_{u, theta, gamma, mu} theta + sum_s p_s * gamma_s + epsilon * mu
    s.t.  theta + gamma_s >= L_s(u)              forall s in {1, ..., S}
         -mu <= gamma_s <= mu                    forall s in {1, ..., S}
          mu >= 0
plus physical power balance, battery dynamics, and causal nonanticipativity:
          P_charge[s, 0] = P_charge[0, 0]        forall s >= 1
          P_discharge[s, 0] = P_discharge[0, 0]  forall s >= 1
"""

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linprog

from src.control.scenario_generator import UncertaintyScenario
from src.simulation.actions import BatteryAction
from src.simulation.config import SimulationConfig, load_simulation_config
from src.simulation.metrics import CostConfig, load_cost_config


@dataclass(frozen=True)
class DROSolverConfig:
    """Configuration for Distributionally Robust Optimization LP solver."""

    ambiguity_radius_epsilon: float = 0.5
    solver_method: str = "highs"
    time_step_hours: float = 1.0
    terminal_inventory_value_usd_per_kwh_dc: float = 0.14736842105263157
    power_balance_tolerance_kw: float = 1e-4
    allow_curtailment: bool = True


@dataclass(frozen=True)
class DROSolverDiagnostics:
    """Convergence and timing telemetry for the DRO solver."""

    solve_time_seconds: float
    iterations: int
    solver_status: int
    solver_message: str
    num_scenarios: int
    horizon_hours: int
    total_variables: int
    total_constraints: int
    ambiguity_radius_epsilon: float


@dataclass(frozen=True)
class DROSolverResult:
    """Results from solving the DRO MPC linear program."""

    success: bool
    first_action: BatteryAction
    first_requested_power_kw: float
    worst_case_robust_cost_usd: float
    nominal_expected_cost_usd: float
    worst_case_scenario_cost_usd: float
    theta_val: float
    mu_val: float
    gamma_vals: list[float]
    scenario_losses_usd: list[float]
    optimal_worst_case_probabilities: list[float]
    scenario_dispatches: list[dict[str, list[float]]]
    diagnostics: DROSolverDiagnostics


def solve_dro_mpc_linear_program(
    initial_energy_kwh: float,
    scenarios: Sequence[UncertaintyScenario],
    sim_config: SimulationConfig | None = None,
    cost_config: CostConfig | None = None,
    dro_config: DROSolverConfig | None = None,
) -> DROSolverResult:
    """Formulate and solve the L1-Wasserstein/TV Distributionally Robust Linear Program.

    Variables vector layout:
      For each scenario s in {0, ..., S-1}:
        - x[s * 6H + 5k + 0]: P_charge[s, k] >= 0
        - x[s * 6H + 5k + 1]: P_discharge[s, k] >= 0
        - x[s * 6H + 5k + 2]: P_import[s, k] >= 0
        - x[s * 6H + 5k + 3]: P_export[s, k] >= 0
        - x[s * 6H + 5k + 4]: P_curtail[s, k] >= 0
        - x[s * 6H + 5H + k]: E[s, k+1] in [E_min, E_max]
      Auxiliary DRO dual variables:
        - x[6HS + 0]: theta (free)
        - x[6HS + 1 + s]: gamma_s (free) for s in {0, ..., S-1}
        - x[6HS + 1 + S]: mu (>= 0)
    Total variables: N = 6 * H * S + S + 2.
    """
    start_time = time.perf_counter()

    if sim_config is None:
        sim_config = load_simulation_config()
    if cost_config is None:
        cost_config = load_cost_config()
    if dro_config is None:
        dro_config = DROSolverConfig()

    num_scenarios = len(scenarios)
    if num_scenarios <= 0:
        raise ValueError("At least one uncertainty scenario is required.")

    h = scenarios[0].horizon_hours
    dt = dro_config.time_step_hours
    epsilon = max(0.0, float(dro_config.ambiguity_radius_epsilon))

    # Hardware specs from authoritative simulation config
    bat = sim_config.battery
    capacity = bat.capacity_kwh
    e_min = capacity * bat.soc_min
    e_max = capacity * bat.soc_max
    p_charge_max = bat.charge_limit_kw
    p_discharge_max = bat.discharge_limit_kw
    eta_c = bat.charge_efficiency
    eta_d = bat.discharge_efficiency

    grid = sim_config.grid
    p_import_max = grid.import_limit_kw
    p_export_max = grid.export_limit_kw if grid.export_limit_kw is not None else 500.0

    c_deg = cost_config.battery_degradation_per_kwh_throughput
    v_term = dro_config.terminal_inventory_value_usd_per_kwh_dc

    vars_per_scenario = 6 * h
    total_phys_vars = num_scenarios * vars_per_scenario
    theta_idx = total_phys_vars
    gamma_start_idx = theta_idx + 1
    mu_idx = gamma_start_idx + num_scenarios
    total_vars = mu_idx + 1

    # 1. Objective function: c^T x
    # min theta + sum_s p_s * gamma_s + epsilon * mu
    c_obj = np.zeros(total_vars, dtype=np.float64)
    c_obj[theta_idx] = 1.0
    for s_idx, sc in enumerate(scenarios):
        c_obj[gamma_start_idx + s_idx] = float(sc.probability)
    c_obj[mu_idx] = epsilon

    # 2. Variable bounds
    bounds = []
    for s_idx in range(num_scenarios):
        for k in range(h):
            bounds.append((0.0, p_charge_max))
            bounds.append((0.0, p_discharge_max))
            bounds.append((0.0, p_import_max))
            bounds.append((0.0, p_export_max))
            bounds.append((0.0, None if dro_config.allow_curtailment else 0.0))
        for k in range(h):
            bounds.append((e_min, e_max))

    # DRO dual variable bounds:
    bounds.append((None, None))  # theta: free
    for _ in range(num_scenarios):
        bounds.append((None, None))  # gamma_s: free
    bounds.append((0.0, None))  # mu >= 0

    # 3. Equality Constraints: A_eq x = b_eq
    # - Power balance: h * S
    # - Battery energy dynamics: h * S
    # - Nonanticipativity: 2 * (S - 1)
    total_eq = 2 * h * num_scenarios + 2 * (num_scenarios - 1)
    a_eq = np.zeros((total_eq, total_vars), dtype=np.float64)
    b_eq = np.zeros(total_eq, dtype=np.float64)

    eq_row = 0
    for s_idx, sc in enumerate(scenarios):
        s_offset = s_idx * vars_per_scenario
        e_offset = s_offset + 5 * h

        # Power balance
        for k in range(h):
            a_eq[eq_row, s_offset + 5 * k + 0] = -1.0  # -P_charge
            a_eq[eq_row, s_offset + 5 * k + 1] = 1.0   # +P_discharge
            a_eq[eq_row, s_offset + 5 * k + 2] = 1.0   # +P_import
            a_eq[eq_row, s_offset + 5 * k + 3] = -1.0  # -P_export
            a_eq[eq_row, s_offset + 5 * k + 4] = -1.0  # -P_curtail

            p_ren = sc.solar_generation_kw[k] + sc.wind_generation_kw[k]
            d_kw = sc.demand_kw[k] if hasattr(sc, "demand_kw") else sc.campus_demand_kw[k]
            b_eq[eq_row] = d_kw - p_ren
            eq_row += 1

        # Battery dynamics
        for k in range(h):
            a_eq[eq_row, e_offset + k] = 1.0  # +E[k+1]
            if k == 0:
                b_eq[eq_row] += initial_energy_kwh
            else:
                a_eq[eq_row, e_offset + k - 1] = -1.0  # -E[k]

            a_eq[eq_row, s_offset + 5 * k + 0] = -eta_c * dt
            a_eq[eq_row, s_offset + 5 * k + 1] = (1.0 / eta_d) * dt
            eq_row += 1

    # Nonanticipativity at k=0 across scenarios
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

    # 4. Inequality Constraints: A_ub x <= b_ub
    # (a) Robust loss bound: L_s(u) - theta - gamma_s <= 0  (S constraints)
    # (b) Dual L1 bounds: gamma_s - mu <= 0                 (S constraints)
    # (c) Dual L1 bounds: -gamma_s - mu <= 0                (S constraints)
    total_ub = 3 * num_scenarios
    a_ub = np.zeros((total_ub, total_vars), dtype=np.float64)
    b_ub = np.zeros(total_ub, dtype=np.float64)

    # (a) L_s(u) - theta - gamma_s <= 0
    # where L_s(u) = sum_k dt [ c_imp P_imp - c_exp P_exp + c_deg (P_c + P_d) ] - v_term (E_H - E_0)
    for s_idx, sc in enumerate(scenarios):
        s_offset = s_idx * vars_per_scenario
        row_idx = s_idx
        for k in range(h):
            a_ub[row_idx, s_offset + 5 * k + 0] = c_deg * dt
            a_ub[row_idx, s_offset + 5 * k + 1] = c_deg * dt
            a_ub[row_idx, s_offset + 5 * k + 2] = sc.import_price[k] * dt
            a_ub[row_idx, s_offset + 5 * k + 3] = -sc.export_price[k] * dt

        # Terminal state coefficient: -v_term * E_H
        a_ub[row_idx, s_offset + 5 * h + (h - 1)] = -v_term
        # -theta
        a_ub[row_idx, theta_idx] = -1.0
        # -gamma_s
        a_ub[row_idx, gamma_start_idx + s_idx] = -1.0
        # RHS constant: -v_term * E_0
        b_ub[row_idx] = -v_term * initial_energy_kwh

    # (b) gamma_s - mu <= 0
    for s_idx in range(num_scenarios):
        row_idx = num_scenarios + s_idx
        a_ub[row_idx, gamma_start_idx + s_idx] = 1.0
        a_ub[row_idx, mu_idx] = -1.0
        b_ub[row_idx] = 0.0

    # (c) -gamma_s - mu <= 0
    for s_idx in range(num_scenarios):
        row_idx = 2 * num_scenarios + s_idx
        a_ub[row_idx, gamma_start_idx + s_idx] = -1.0
        a_ub[row_idx, mu_idx] = -1.0
        b_ub[row_idx] = 0.0

    # Solve using HiGHS
    res = linprog(
        c=c_obj,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=dro_config.solver_method,
    )

    solve_time = time.perf_counter() - start_time

    diag = DROSolverDiagnostics(
        solve_time_seconds=solve_time,
        iterations=int(getattr(res, "nit", 0)),
        solver_status=int(res.status),
        solver_message=str(res.message),
        num_scenarios=num_scenarios,
        horizon_hours=h,
        total_variables=total_vars,
        total_constraints=total_eq + total_ub,
        ambiguity_radius_epsilon=epsilon,
    )

    if not res.success:
        return DROSolverResult(
            success=False,
            first_action=BatteryAction(0.0),
            first_requested_power_kw=0.0,
            worst_case_robust_cost_usd=float("nan"),
            nominal_expected_cost_usd=float("nan"),
            worst_case_scenario_cost_usd=float("nan"),
            theta_val=float("nan"),
            mu_val=float("nan"),
            gamma_vals=[],
            scenario_losses_usd=[],
            optimal_worst_case_probabilities=[],
            scenario_dispatches=[],
            diagnostics=diag,
        )

    x_sol = res.x
    theta_sol = float(x_sol[theta_idx])
    gamma_sols = [float(x_sol[gamma_start_idx + s]) for s in range(num_scenarios)]
    mu_sol = float(x_sol[mu_idx])

    # Extract scenario-wise physical losses and dispatches
    scenario_losses = []
    scenario_dispatches = []

    for s_idx, sc in enumerate(scenarios):
        s_offset = s_idx * vars_per_scenario
        chg = [float(x_sol[s_offset + 5 * k + 0]) for k in range(h)]
        dis = [float(x_sol[s_offset + 5 * k + 1]) for k in range(h)]
        imp = [float(x_sol[s_offset + 5 * k + 2]) for k in range(h)]
        exp = [float(x_sol[s_offset + 5 * k + 3]) for k in range(h)]
        soc = [float(x_sol[s_offset + 5 * h + k]) / capacity for k in range(h)]

        term_e = float(x_sol[s_offset + 5 * h + (h - 1)])
        term_credit = v_term * (term_e - initial_energy_kwh)

        loss = sum(
            dt * (sc.import_price[k] * imp[k] - sc.export_price[k] * exp[k] + c_deg * (chg[k] + dis[k]))
            for k in range(h)
        ) - term_credit
        scenario_losses.append(loss)

        scenario_dispatches.append({
            "charge_kw": chg,
            "discharge_kw": dis,
            "import_kw": imp,
            "export_kw": exp,
            "soc": soc,
        })

    # Nominal expected cost with scenario probabilities
    nominal_exp_cost = sum(float(sc.probability) * loss for sc, loss in zip(scenarios, scenario_losses))
    worst_case_scenario_cost = max(scenario_losses)

    # Worst-case robust cost from the DRO objective
    worst_case_robust_cost = theta_sol + sum(
        float(sc.probability) * g for sc, g in zip(scenarios, gamma_sols)
    ) + epsilon * mu_sol

    # Compute adversarial probability distribution q* via inner simplex projection
    # Solve inner: max sum q_s L_s s.t. sum q_s = 1, sum |q_s - p_s| <= epsilon, q >= 0
    p_vec = np.array([float(sc.probability) for sc in scenarios], dtype=np.float64)
    l_vec = np.array(scenario_losses, dtype=np.float64)

    c_inner = -np.concatenate([l_vec, np.zeros(2 * num_scenarios)])
    # vars: q (S), delta+ (S), delta- (S)
    a_eq_inner = np.zeros((1 + num_scenarios, 3 * num_scenarios))
    b_eq_inner = np.zeros(1 + num_scenarios)
    a_eq_inner[0, :num_scenarios] = 1.0
    b_eq_inner[0] = 1.0
    for s in range(num_scenarios):
        a_eq_inner[1 + s, s] = 1.0
        a_eq_inner[1 + s, num_scenarios + s] = -1.0
        a_eq_inner[1 + s, 2 * num_scenarios + s] = 1.0
        b_eq_inner[1 + s] = p_vec[s]

    a_ub_inner = np.zeros((1, 3 * num_scenarios))
    a_ub_inner[0, num_scenarios:] = 1.0
    b_ub_inner = np.array([epsilon])

    inner_adv = linprog(
        c=c_inner,
        A_ub=a_ub_inner,
        b_ub=b_ub_inner,
        A_eq=a_eq_inner,
        b_eq=b_eq_inner,
        bounds=[(0.0, 1.0) for _ in range(num_scenarios)] + [(0.0, 1.0) for _ in range(2 * num_scenarios)],
        method="highs",
    )
    if inner_adv.success:
        adversarial_q = [float(x) for x in inner_adv.x[:num_scenarios]]
    else:
        adversarial_q = [float(p) for p in p_vec]

    # First immediate action at k=0 (common across all scenarios due to nonanticipativity)
    first_charge = float(x_sol[0])
    first_discharge = float(x_sol[1])
    first_action_kw = first_discharge - first_charge

    return DROSolverResult(
        success=True,
        first_action=BatteryAction(first_action_kw),
        first_requested_power_kw=first_action_kw,
        worst_case_robust_cost_usd=float(worst_case_robust_cost),
        nominal_expected_cost_usd=float(nominal_exp_cost),
        worst_case_scenario_cost_usd=float(worst_case_scenario_cost),
        theta_val=theta_sol,
        mu_val=mu_sol,
        gamma_vals=gamma_sols,
        scenario_losses_usd=scenario_losses,
        optimal_worst_case_probabilities=adversarial_q,
        scenario_dispatches=scenario_dispatches,
        diagnostics=diag,
    )
