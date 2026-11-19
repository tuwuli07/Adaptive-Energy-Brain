"""Unit tests for CVaR mathematical formulation and LP equivalence."""

import pytest
import numpy as np
from scipy.optimize import linprog

from src.control.cvar_mpc_solver import (
    CVaRSolverConfig,
    solve_cvar_mpc_linear_program,
)
from src.control.scenario_generator import UncertaintyScenario


def compute_discrete_cvar(losses: list[float], probs: list[float], alpha: float) -> tuple[float, float]:
    """Independent mathematical reference computation of discrete VaR and CVaR using LP."""
    losses = np.array(losses, dtype=np.float64)
    probs = np.array(probs, dtype=np.float64)
    s = len(losses)

    # Variables: [eta, z_0, ..., z_{s-1}] (1 + s vars)
    c = np.zeros(1 + s)
    c[0] = 1.0  # eta
    c[1:] = probs / (1.0 - alpha)  # sum p_s z_s / (1 - alpha)

    # Bounds: eta free, z_s >= 0
    bounds = [(-1e7, 1e7)] + [(0.0, None) for _ in range(s)]

    # Constraints: L_s - eta - z_s <= 0 => -eta - z_s <= -L_s
    a_ub = np.zeros((s, 1 + s))
    b_ub = np.zeros(s)
    for i in range(s):
        a_ub[i, 0] = -1.0
        a_ub[i, 1 + i] = -1.0
        b_ub[i] = -losses[i]

    res = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    assert res.success
    eta_opt = res.x[0]
    cvar_opt = res.fun
    return eta_opt, cvar_opt


def test_cvar_rockafellar_uryasev_manual_example():
    """Verify CVaR calculation against known analytical discrete distribution."""
    # Discrete distribution: 5 equally likely outcomes [100, 200, 300, 400, 500]
    losses = [100.0, 200.0, 300.0, 400.0, 500.0]
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    # For alpha = 0.80:
    # 80th percentile tail is the single worst outcome (500)
    # Expected tail loss = 500.0
    eta_80, cvar_80 = compute_discrete_cvar(losses, probs, alpha=0.80)
    assert cvar_80 == pytest.approx(500.0, abs=1e-5)
    assert 400.0 <= eta_80 <= 500.0 + 1e-5

    # For alpha = 0.60:
    # 60th percentile tail covers outcomes 400 and 500 (average = 450)
    eta_60, cvar_60 = compute_discrete_cvar(losses, probs, alpha=0.60)
    assert cvar_60 == pytest.approx(450.0, abs=1e-5)
    assert 300.0 <= eta_60 <= 400.0 + 1e-5


def test_cvar_solver_zero_vs_high_lambda():
    """Test that increasing lambda shifts the objective towards reducing CVaR."""
    # Construct 3 simple scenarios with differing solar generation
    h = 24
    scenarios = [
        UncertaintyScenario(
            scenario_id=0,
            name="Normal",
            probability=0.50,
            demand_kw=tuple([250.0] * h),
            solar_generation_kw=tuple([100.0] * h),
            wind_generation_kw=tuple([20.0] * h),
            import_price=tuple([0.15] * h),
            export_price=tuple([0.04] * h),
            description="Normal",
        ),
        UncertaintyScenario(
            scenario_id=1,
            name="Cloudy",
            probability=0.30,
            demand_kw=tuple([250.0] * h),
            solar_generation_kw=tuple([10.0] * h),
            wind_generation_kw=tuple([20.0] * h),
            import_price=tuple([0.15] * h),
            export_price=tuple([0.04] * h),
            description="Cloudy",
        ),
        UncertaintyScenario(
            scenario_id=2,
            name="Extreme Stress",
            probability=0.20,
            demand_kw=tuple([350.0] * h),
            solar_generation_kw=tuple([0.0] * h),
            wind_generation_kw=tuple([0.0] * h),
            import_price=tuple([0.45] * h),
            export_price=tuple([0.04] * h),
            description="Extreme",
        ),
    ]

    # Risk-neutral solve (lambda = 0)
    cfg_risk_neutral = CVaRSolverConfig(alpha=0.90, risk_aversion_lambda=0.0)
    res_neutral = solve_cvar_mpc_linear_program(
        initial_energy_kwh=250.0,
        scenarios=scenarios,
        cvar_config=cfg_risk_neutral,
    )
    assert res_neutral.success

    # Risk-averse solve (lambda = 5.0)
    cfg_risk_averse = CVaRSolverConfig(alpha=0.90, risk_aversion_lambda=5.0)
    res_averse = solve_cvar_mpc_linear_program(
        initial_energy_kwh=250.0,
        scenarios=scenarios,
        cvar_config=cfg_risk_averse,
    )
    assert res_averse.success

    # Nonanticipativity: first action must be finite and within limits
    assert abs(res_neutral.first_requested_power_kw) <= 250.0
    assert abs(res_averse.first_requested_power_kw) <= 250.0

    # Risk-averse CVaR loss must be less than or equal to risk-neutral CVaR loss (within numerical tolerance)
    assert res_averse.cvar_loss_usd <= res_neutral.cvar_loss_usd + 1e-3
