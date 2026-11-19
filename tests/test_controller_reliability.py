"""Tests for Phase 8.3 Controller Reliability and Fallback Audit."""

import pytest

from src.evaluation.reliability import (
    audit_controller_fallback_to_idle,
    audit_excess_power_limit_clamping,
    audit_horizon_mismatch_protection,
    audit_non_finite_forecast_protection,
    audit_soc_boundary_clamping,
    audit_solver_infeasibility_fallback,
    run_all_reliability_audits,
)


def test_audit_solver_infeasibility_fallback() -> None:
    res = audit_solver_infeasibility_fallback()
    assert res.safe_fallback_verified is True
    assert res.test_case_id == "REL-01-INFEASIBILITY"


def test_audit_non_finite_forecast_protection() -> None:
    res = audit_non_finite_forecast_protection()
    assert res.safe_fallback_verified is True
    assert res.test_case_id == "REL-02-NON-FINITE-PROTECTION"


def test_audit_horizon_mismatch_protection() -> None:
    res = audit_horizon_mismatch_protection()
    assert res.safe_fallback_verified is True
    assert res.test_case_id == "REL-03-HORIZON-MISMATCH"


def test_audit_soc_boundary_clamping() -> None:
    res = audit_soc_boundary_clamping()
    assert res.safe_fallback_verified is True
    assert res.test_case_id == "REL-04-SOC-BOUNDARY-CLAMPING"


def test_audit_excess_power_limit_clamping() -> None:
    res = audit_excess_power_limit_clamping()
    assert res.safe_fallback_verified is True
    assert res.test_case_id == "REL-05-POWER-LIMIT-CLAMPING"


def test_audit_controller_fallback_to_idle() -> None:
    res = audit_controller_fallback_to_idle()
    assert res.safe_fallback_verified is True
    assert res.test_case_id == "REL-06-CONTROLLER-IDLE-FALLBACK"


def test_run_all_reliability_audits() -> None:
    results = run_all_reliability_audits()
    assert len(results) == 6
    assert all(r.safe_fallback_verified for r in results)


def test_warmup_state_isolation() -> None:
    """Verify that warm-up steps do not alter initial state, SOC, index, or replay outcomes."""
    from pathlib import Path
    from scripts.run_simulation import _components
    from src.control.seasonal_persistence_mpc import SeasonalPersistenceMPCController
    from src.simulation.config import load_simulation_config
    from src.simulation.data_loader import SimulationData, load_simulation_split
    from src.simulation.environment import MicrogridEnvironment
    from src.simulation.metrics import load_cost_config

    root = Path(__file__).resolve().parents[1]
    split_root = root / "data" / "processed" / "splits_hypothetical_campus_v1"
    assets_path = root / "configs" / "synthetic_scenario.yaml"

    val_data = load_simulation_split(split_root, "validation")
    sim_config = load_simulation_config()
    cost_config = load_cost_config()
    renewables, demand = _components(val_data, assets_path, "hypothetical_example")

    # Instance A: Clean without warm-up
    ctrl_a = SeasonalPersistenceMPCController(initial_history=val_data[:24])
    env_a = MicrogridEnvironment(val_data, renewables, demand, sim_config, cost_config)
    obs_a = env_a.reset()
    act_a = ctrl_a.select_action(obs_a)
    step_a = env_a.step(act_a)

    # Instance B: Warm-up on throwaway dummy environment, then reset and run
    ctrl_b = SeasonalPersistenceMPCController(initial_history=val_data[:24])
    dummy_slice = SimulationData(val_data[:5], val_data.metadata)
    d_renewables, d_demand = _components(dummy_slice, assets_path, "hypothetical_example")
    dummy_env = MicrogridEnvironment(dummy_slice, d_renewables, d_demand, sim_config, cost_config)
    dummy_obs = dummy_env.reset()
    for _ in range(5):
        ctrl_b.select_action(dummy_obs)

    # Clean reset
    ctrl_b.reset()

    # Benchmark environment created fresh
    env_b = MicrogridEnvironment(val_data, renewables, demand, sim_config, cost_config)
    obs_b = env_b.reset()

    # Assert pristine initial state
    assert env_b._index == 1
    assert env_b.observation.state.battery_soc == 0.50
    assert not env_b.terminated
    assert len(ctrl_b.step_actions) == 0
    assert len(ctrl_b.step_diagnostics) == 0
    assert ctrl_b.fallback_count == 0

    act_b = ctrl_b.select_action(obs_b)
    step_b = env_b.step(act_b)

    # Verify bitwise/exact parity between unwarmed and state-isolated warmed run
    assert act_a.requested_power_kw == act_b.requested_power_kw
    assert step_a.dispatch.actual_battery_power_kw == step_b.dispatch.actual_battery_power_kw
    assert step_a.observation.state.battery_soc == step_b.observation.state.battery_soc
    assert step_a.metrics.values.net_operating_cost == step_b.metrics.values.net_operating_cost
