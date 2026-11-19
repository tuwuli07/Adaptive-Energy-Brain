"""Integration tests for CVaRRiskAwareMPCController."""

import pytest
from datetime import datetime, timezone

from src.control.cvar_mpc import CVaRRiskAwareMPCController
from src.simulation.state import AgentObservation, SimulationState
from src.simulation.data_loader import HourlyObservation
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config


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


def test_cvar_controller_initialization_and_reset():
    controller = CVaRRiskAwareMPCController()
    assert controller.config.version == 1
    assert controller.config.alpha == 0.95
    assert controller.risk_lambda == 1.0

    controller.reset()
    assert len(controller.step_diagnostics) == 0


def test_cvar_controller_select_action_physical_limits():
    controller = CVaRRiskAwareMPCController(override_lambda=2.0)
    obs = make_dummy_observation(step=0, soc=0.5)

    action = controller.select_action(obs)
    assert isinstance(action, BatteryAction)

    # Inverter power limit
    assert -250.0 <= action.requested_power_kw <= 250.0

    # Diagnostic recording
    assert len(controller.step_diagnostics) == 1
    diag = controller.step_diagnostics[0]
    assert diag.success is True
    assert diag.num_scenarios == 5
    assert diag.cvar_usd >= diag.var_usd - 1e-4
    assert diag.worst_case_cost_usd >= diag.expected_cost_usd - 1e-4
