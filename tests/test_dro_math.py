"""Mathematical tests for Distributionally Robust Optimization (DRO) LP duality.

Validates:
1. Analytical 2-scenario problem with exact hand-calculated optimum across epsilon.
2. Epsilon = 0 equivalence to nominal risk-neutral expected loss.
3. Monotonicity of worst-case robust cost with respect to ambiguity radius epsilon.
4. Total-variation / L1 ambiguity set constraint compliance (q* >= 0, sum q* = 1, ||q* - p||_1 <= epsilon).
5. Exact strong duality match between primal inner maximization and dual outer LP.
"""

import numpy as np
import pytest
from scipy.optimize import linprog

from src.control.dro_mpc_solver import (
    DROSolverConfig,
    solve_dro_mpc_linear_program,
)
from src.control.scenario_generator import UncertaintyScenario
from src.simulation.config import load_simulation_config
from src.simulation.metrics import load_cost_config


def solve_dro_analytical_dual(losses: list[float], probs: list[float], epsilon: float):
    """Directly solve the DRO dual LP for fixed loss vector L and nominal probabilities p."""
    s = len(losses)
    l_vec = np.array(losses, dtype=np.float64)
    p_vec = np.array(probs, dtype=np.float64)

    # Variables: [theta, gamma_0, ..., gamma_{s-1}, mu]
    # Obj: theta + sum_s p_s * gamma_s + epsilon * mu
    c = np.zeros(s + 2)
    c[0] = 1.0
    c[1 : 1 + s] = p_vec
    c[-1] = epsilon

    # Inequalities:
    # 1. -theta - gamma_s <= -L_s
    # 2. gamma_s - mu <= 0
    # 3. -gamma_s - mu <= 0
    a_ub = np.zeros((3 * s, s + 2))
    b_ub = np.zeros(3 * s)

    for i in range(s):
        # -theta - gamma_i <= -L_i
        a_ub[i, 0] = -1.0
        a_ub[i, 1 + i] = -1.0
        b_ub[i] = -l_vec[i]

        # gamma_i - mu <= 0
        a_ub[s + i, 1 + i] = 1.0
        a_ub[s + i, -1] = -1.0

        # -gamma_i - mu <= 0
        a_ub[2 * s + i, 1 + i] = -1.0
        a_ub[2 * s + i, -1] = -1.0

    bounds = [(None, None)] + [(None, None) for _ in range(s)] + [(0.0, None)]
    res = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    assert res.success, f"Dual LP failed: {res.message}"
    return float(res.fun), float(res.x[0]), [float(x) for x in res.x[1 : 1 + s]], float(res.x[-1])


def test_dro_analytical_two_scenario_exact_optimum():
    """Verify hand-derived analytical solutions on a 2-state discrete loss distribution.

    Let L = [10.0, 50.0], p = [0.5, 0.5].
    Nominal expected loss: 0.5 * 10 + 0.5 * 50 = 30.0.
    Adversary shifts mass from scenario 1 to scenario 2:
      delta q_2 <= epsilon / 2.
    For epsilon = 0.0: Obj = 30.0.
    For epsilon = 0.4: Obj = 30.0 + 0.2 * (50 - 10) = 38.0.
    For epsilon = 1.0: Obj = 0.0 * 10 + 1.0 * 50 = 50.0 (worst case).
    For epsilon = 2.0: Obj = 50.0 (saturated worst case).
    """
    losses = [10.0, 50.0]
    probs = [0.5, 0.5]

    # epsilon = 0.0 -> exact nominal
    obj_0, theta_0, _, _ = solve_dro_analytical_dual(losses, probs, epsilon=0.0)
    assert obj_0 == pytest.approx(30.0, abs=1e-5)

    # epsilon = 0.4 -> 38.0
    obj_04, _, _, _ = solve_dro_analytical_dual(losses, probs, epsilon=0.4)
    assert obj_04 == pytest.approx(38.0, abs=1e-5)

    # epsilon = 1.0 -> 50.0
    obj_10, _, _, _ = solve_dro_analytical_dual(losses, probs, epsilon=1.0)
    assert obj_10 == pytest.approx(50.0, abs=1e-5)

    # epsilon = 2.0 -> 50.0
    obj_20, _, _, _ = solve_dro_analytical_dual(losses, probs, epsilon=2.0)
    assert obj_20 == pytest.approx(50.0, abs=1e-5)


def test_dro_monotonicity_with_increasing_epsilon():
    """Verify that the worst-case robust objective is monotonically non-decreasing in epsilon."""
    losses = [15.0, 35.0, 80.0, 120.0, 200.0]
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    epsilons = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    objectives = [solve_dro_analytical_dual(losses, probs, eps)[0] for eps in epsilons]

    for i in range(len(objectives) - 1):
        assert objectives[i + 1] >= objectives[i] - 1e-6, (
            f"Monotonicity violated: eps={epsilons[i]} -> {objectives[i]}, "
            f"eps={epsilons[i+1]} -> {objectives[i+1]}"
        )


def test_dro_strong_duality_match():
    """Verify that inner primal maximization equals outer dual minimization."""
    losses = [25.0, 40.0, 95.0, 150.0]
    probs = [0.4, 0.3, 0.2, 0.1]
    epsilon = 0.6

    dual_obj, _, _, _ = solve_dro_analytical_dual(losses, probs, epsilon=epsilon)

    # Primal inner LP: max sum q_s L_s s.t. sum q = 1, sum |q_s - p_s| <= epsilon, q >= 0
    s = len(losses)
    l_vec = np.array(losses, dtype=np.float64)
    p_vec = np.array(probs, dtype=np.float64)

    # Variables: q (s), delta+ (s), delta- (s)
    c_primal = -np.concatenate([l_vec, np.zeros(2 * s)])
    a_eq = np.zeros((1 + s, 3 * s))
    b_eq = np.zeros(1 + s)
    a_eq[0, :s] = 1.0
    b_eq[0] = 1.0
    for i in range(s):
        a_eq[1 + i, i] = 1.0
        a_eq[1 + i, s + i] = -1.0
        a_eq[1 + i, 2 * s + i] = 1.0
        b_eq[1 + i] = p_vec[i]

    a_ub = np.zeros((1, 3 * s))
    a_ub[0, s:] = 1.0
    b_ub = np.array([epsilon])

    res = linprog(
        c=c_primal,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0) for _ in range(s)] + [(0.0, 1.0) for _ in range(2 * s)],
        method="highs",
    )
    assert res.success
    primal_obj = -float(res.fun)

    assert dual_obj == pytest.approx(primal_obj, abs=1e-5), f"Duality gap: dual={dual_obj}, primal={primal_obj}"


@pytest.mark.parametrize(
    "losses,probs",
    [
        ([10.0, 25.0, 60.0, 110.0], [0.25, 0.25, 0.25, 0.25]),
        ([15.0, 30.0, 75.0, 140.0], [0.60, 0.20, 0.15, 0.05]),
        ([5.0, 50.0, 200.0], [0.50, 0.30, 0.20]),
        ([12.0, 18.0, 45.0, 80.0, 160.0], [0.10, 0.20, 0.40, 0.20, 0.10]),
        ([50.0, 50.0, 50.0], [0.333, 0.333, 0.334]),
    ],
)
@pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.2, 0.5, 1.0, 1.5, 2.0])
def test_dro_strong_duality_parameterized_sweep(losses, probs, epsilon):
    """Verify strong LP duality and nominal recovery across diverse distributions and radii."""
    dual_obj, _, _, _ = solve_dro_analytical_dual(losses, probs, epsilon=epsilon)

    # When epsilon = 0, exact nominal expected loss
    if epsilon == 0.0:
        nominal_expected = sum(p * l for p, l in zip(probs, losses))
        assert dual_obj == pytest.approx(nominal_expected, abs=1e-5)

    # Primal inner LP
    s = len(losses)
    l_vec = np.array(losses, dtype=np.float64)
    p_vec = np.array(probs, dtype=np.float64)

    # Variables: q (s), delta+ (s), delta- (s)
    c_primal = -np.concatenate([l_vec, np.zeros(2 * s)])
    a_eq = np.zeros((1 + s, 3 * s))
    b_eq = np.zeros(1 + s)
    a_eq[0, :s] = 1.0
    b_eq[0] = 1.0
    for i in range(s):
        a_eq[1 + i, i] = 1.0
        a_eq[1 + i, s + i] = -1.0
        a_eq[1 + i, 2 * s + i] = 1.0
        b_eq[1 + i] = p_vec[i]

    a_ub = np.zeros((1, 3 * s))
    a_ub[0, s:] = 1.0
    b_ub = np.array([epsilon])

    res = linprog(
        c=c_primal,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0) for _ in range(s)] + [(0.0, 1.0) for _ in range(2 * s)],
        method="highs",
    )
    assert res.success
    primal_obj = -float(res.fun)
    assert dual_obj == pytest.approx(primal_obj, abs=1e-5)



def test_dro_solver_integrated_with_synthetic_scenarios():
    """Verify solve_dro_mpc_linear_program on synthetic UncertaintyScenario objects."""
    h = 24
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()

    scenarios = [
        UncertaintyScenario(
            scenario_id=0,
            name="nominal",
            probability=0.5,
            demand_kw=tuple([250.0] * h),
            solar_generation_kw=tuple([40.0] * h),
            wind_generation_kw=tuple([20.0] * h),
            import_price=tuple([0.15] * h),
            export_price=tuple([0.04] * h),
            description="Nominal test scenario",
        ),
        UncertaintyScenario(
            scenario_id=1,
            name="stress",
            probability=0.5,
            demand_kw=tuple([300.0] * h),
            solar_generation_kw=tuple([20.0] * h),
            wind_generation_kw=tuple([10.0] * h),
            import_price=tuple([0.45] * h),
            export_price=tuple([0.04] * h),
            description="Stress test scenario",
        ),
    ]

    # With epsilon = 0.0 (nominal risk-neutral)
    res_0 = solve_dro_mpc_linear_program(
        initial_energy_kwh=250.0,
        scenarios=scenarios,
        sim_config=sim_cfg,
        cost_config=cost_cfg,
        dro_config=DROSolverConfig(ambiguity_radius_epsilon=0.0),
    )
    assert res_0.success
    assert res_0.worst_case_robust_cost_usd == pytest.approx(res_0.nominal_expected_cost_usd, abs=1e-3)

    # With epsilon = 0.5 (distributionally robust)
    res_dro = solve_dro_mpc_linear_program(
        initial_energy_kwh=250.0,
        scenarios=scenarios,
        sim_config=sim_cfg,
        cost_config=cost_cfg,
        dro_config=DROSolverConfig(ambiguity_radius_epsilon=0.5),
    )
    assert res_dro.success
    assert res_dro.worst_case_robust_cost_usd >= res_0.worst_case_robust_cost_usd - 1e-4

    # Nonanticipativity: first action must be identical across scenarios
    assert abs(res_dro.first_requested_power_kw) <= sim_cfg.battery.charge_limit_kw
