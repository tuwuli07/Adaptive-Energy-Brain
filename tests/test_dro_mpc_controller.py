"""Integration tests for DistributionallyRobustMPCController (mpc_dro_robust)."""

from datetime import datetime, timezone
import pytest

from src.control.dro_mpc import DistributionallyRobustMPCController
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.state import AgentObservation, SimulationState


def make_dummy_observation(step: int, soc: float = 0.5) -> AgentObservation:
    dec_dt = datetime(2026, 11, 1, 7 + step, 0, tzinfo=timezone.utc)
    dec_ts = dec_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    sim_cfg = load_simulation_config()

    state = SimulationState(
        timestamp_utc=dec_ts,
        available_at_utc=dec_ts,
        solar_generation_kw=50.0,
        wind_generation_kw=20.0,
        campus_demand_kw=280.0,
        battery_soc=soc,
        grid_import_price=0.18,
        grid_export_price=0.04,
        config=sim_cfg,
    )

    return AgentObservation(
        state=state,
        decision_timestamp_utc=dec_ts,
    )


def test_dro_controller_initialization_and_reset():
    controller = DistributionallyRobustMPCController(override_epsilon=0.5)
    assert controller.epsilon == 0.5
    assert controller.model.training is False
    assert len(controller.step_diagnostics) == 0

    controller.reset()
    assert len(controller._history) == 0
    assert controller._latest_record is None


def test_dro_controller_select_action_physical_limits():
    sim_cfg = load_simulation_config()
    controller = DistributionallyRobustMPCController(override_epsilon=1.0)
    obs = make_dummy_observation(step=0, soc=0.5)

    action = controller.select_action(obs)
    assert isinstance(action, BatteryAction)

    # Must adhere to authoritative charge/discharge limits (150 kW)
    assert -sim_cfg.battery.charge_limit_kw <= action.requested_power_kw <= sim_cfg.battery.discharge_limit_kw

    # Verify diagnostics were captured
    assert len(controller.step_diagnostics) == 1
    diag = controller.step_diagnostics[0]
    assert diag.success is True
    assert diag.ambiguity_radius_epsilon == 1.0
    assert diag.worst_case_robust_cost_usd >= diag.nominal_expected_cost_usd - 1e-4
    assert diag.solve_time_seconds > 0.0


def test_dro_controller_epsilon_zero_recovers_nominal():
    controller = DistributionallyRobustMPCController(override_epsilon=0.0)
    obs = make_dummy_observation(step=0, soc=0.5)
    action = controller.select_action(obs)
    assert isinstance(action, BatteryAction)

    diag = controller.step_diagnostics[0]
    assert diag.success is True
    # At epsilon=0, robust cost equals nominal expected cost within numerical tolerance
    assert diag.worst_case_robust_cost_usd == pytest.approx(diag.nominal_expected_cost_usd, abs=1e-2)
