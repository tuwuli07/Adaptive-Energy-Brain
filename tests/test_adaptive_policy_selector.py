"""Tests for Phase 12: Adaptive Policy Selector.

Verifies:
1. Causal feature extraction from AgentObservation.
2. Contextual regime classification and controller selection logic.
3. Fallback resilience when candidate controllers encounter exceptions.
4. Inverter power limit compliance ([-150 kW, +150 kW]).
5. Reset lifecycle and state clearing.
"""

from datetime import datetime, timezone
import pytest

from src.baselines.base import BaselineController
from src.control.adaptive_policy_selector import (
    AdaptivePolicySelector,
    AdaptiveSelectorConfig,
    load_adaptive_selector_config,
)
from src.simulation.actions import BatteryAction
from src.simulation.config import load_simulation_config
from src.simulation.state import AgentObservation, SimulationState


class DummyController(BaselineController):
    def __init__(self, name: str, return_kw: float = 0.0, should_fail: bool = False):
        self.name = name
        self.return_kw = return_kw
        self.should_fail = should_fail
        self.call_count = 0

    def _select_action(self, observation: AgentObservation) -> BatteryAction:
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(f"{self.name} deliberate solver failure")
        return BatteryAction(self.return_kw)

    def _reset(self) -> None:
        self.call_count = 0


def make_test_observation(
    dt_str: str = "2026-11-10T12:00:00Z",
    soc: float = 0.5,
    demand_kw: float = 200.0,
    solar_kw: float = 50.0,
    wind_kw: float = 10.0,
    import_price: float = 0.12,
    export_price: float = 0.04,
) -> AgentObservation:
    sim_cfg = load_simulation_config()
    state = SimulationState(
        timestamp_utc=dt_str,
        available_at_utc=dt_str,
        solar_generation_kw=solar_kw,
        wind_generation_kw=wind_kw,
        campus_demand_kw=demand_kw,
        battery_soc=soc,
        grid_import_price=import_price,
        grid_export_price=export_price,
        config=sim_cfg,
    )
    return AgentObservation(
        state=state,
        decision_timestamp_utc=dt_str,
    )


def test_adaptive_selector_config_loader():
    cfg = load_adaptive_selector_config()
    assert cfg.version == 1
    assert cfg.method == "adaptive_regime_routing"
    assert cfg.meta_learning_status == "future_work"
    assert cfg.peak_tariff_threshold_usd == 0.16
    assert cfg.extreme_price_spike_usd == 0.25


def test_feature_extraction_causality():
    selector = AdaptivePolicySelector()
    obs = make_test_observation(
        dt_str="2026-11-10T14:00:00Z",
        soc=0.65,
        demand_kw=280.0,
        solar_kw=75.0,
        wind_kw=15.0,
        import_price=0.18,
    )
    features = selector.extract_observable_features(obs)

    assert features["hour"] == 14
    assert features["soc"] == 0.65
    assert features["demand_kw"] == 280.0
    assert features["solar_kw"] == 75.0
    assert features["wind_kw"] == 15.0
    assert features["renewables_kw"] == 90.0
    assert features["net_deficit_kw"] == 190.0
    assert features["is_peak_tariff"] is True
    assert features["is_extreme_price"] is False
    assert features["has_abundant_solar"] is True


def test_regime_routing_solar_capture():
    """In daytime with abundant solar, selector chooses GNN-Transformer MPC."""
    selector = AdaptivePolicySelector()
    features = {
        "hour": 11,
        "soc": 0.45,
        "import_price_usd": 0.12,
        "is_peak_tariff": False,
        "is_extreme_price": False,
        "has_abundant_solar": True,
        "solar_kw": 85.0,
        "net_deficit_kw": 50.0,
        "renewables_kw": 95.0,
    }
    ctrl_id, regime, rationale = selector.determine_regime_and_policy(features)
    assert ctrl_id == "mpc_gnn_transformer"
    assert regime == "SOLAR_SELF_CONSUMPTION"


def test_regime_routing_peak_tariff_hedging():
    """In evening peak tariff with available SOC, selector chooses CVaR Risk-Aware MPC."""
    selector = AdaptivePolicySelector()
    features = {
        "hour": 17,
        "soc": 0.60,
        "import_price_usd": 0.18,
        "is_peak_tariff": True,
        "is_extreme_price": False,
        "has_abundant_solar": False,
        "solar_kw": 5.0,
        "net_deficit_kw": 150.0,
        "renewables_kw": 15.0,
    }
    ctrl_id, regime, rationale = selector.determine_regime_and_policy(features)
    assert ctrl_id == "mpc_cvar_balanced"
    assert regime == "PEAK_TARIFF_TAIL_HEDGING"


def test_regime_routing_compound_stress_dro():
    """Under compound adverse conditions or extreme price spike, selector chooses DRO Robust MPC."""
    selector = AdaptivePolicySelector()
    features = {
        "hour": 18,
        "soc": 0.35,
        "import_price_usd": 0.54,  # extreme 3x spike
        "is_peak_tariff": True,
        "is_extreme_price": True,
        "has_abundant_solar": False,
        "solar_kw": 0.0,
        "net_deficit_kw": 250.0,
        "renewables_kw": 5.0,
    }
    ctrl_id, regime, rationale = selector.determine_regime_and_policy(features)
    assert ctrl_id == "mpc_dro_robust"
    assert regime == "COMPOUND_STRESS_ROBUST"


def test_regime_routing_nominal_base_dispatch():
    """During nighttime off-peak with nominal load, selector uses nominal predictive MPC."""
    selector = AdaptivePolicySelector()
    features = {
        "hour": 3,
        "soc": 0.50,
        "import_price_usd": 0.12,
        "is_peak_tariff": False,
        "is_extreme_price": False,
        "has_abundant_solar": False,
        "solar_kw": 0.0,
        "net_deficit_kw": 100.0,
        "renewables_kw": 20.0,
    }
    ctrl_id, regime, rationale = selector.determine_regime_and_policy(features)
    assert ctrl_id == "mpc_gnn_transformer"
    assert regime == "NOMINAL_BASE_DISPATCH"


def test_fallback_sequence_on_solver_failure():
    """If primary controller fails, selector automatically cascades to fallback candidate."""
    candidates = {
        "mpc_dro_robust": DummyController("dro", return_kw=50.0, should_fail=True),
        "mpc_gnn_transformer": DummyController("gnn", return_kw=-60.0, should_fail=False),
        "mpc_seasonal_persistence": DummyController("pers", return_kw=-20.0, should_fail=False),
    }
    selector = AdaptivePolicySelector(candidate_controllers=candidates)

    obs = make_test_observation(
        dt_str="2026-11-10T18:00:00Z",
        import_price=0.54,
    )
    action = selector.select_action(obs)

    assert action.requested_power_kw == -60.0
    assert len(selector.selection_history) == 1
    decision = selector.selection_history[0]
    assert decision.fallback_used is True
    assert decision.selected_controller_id == "mpc_gnn_transformer"


def test_physical_inverter_power_clamping():
    """Ensures dispatched action never exceeds the physical 150 kW inverter rating."""
    sim_cfg = load_simulation_config()
    limit = sim_cfg.battery.charge_limit_kw
    assert limit == 150.0

    candidates = {
        "mpc_gnn_transformer": DummyController("gnn", return_kw=300.0),  # exceeds 150 kW
    }
    selector = AdaptivePolicySelector(candidate_controllers=candidates, sim_config=sim_cfg)

    obs = make_test_observation(dt_str="2026-11-10T02:00:00Z")
    action = selector.select_action(obs)

    assert action.requested_power_kw == 150.0


def test_selector_reset_lifecycle():
    candidates = {
        "mpc_gnn_transformer": DummyController("gnn", return_kw=10.0),
    }
    selector = AdaptivePolicySelector(candidate_controllers=candidates)

    obs = make_test_observation(dt_str="2026-11-10T02:00:00Z")
    selector.select_action(obs)
    selector.select_action(obs)
    assert len(selector.selection_history) == 2
    assert candidates["mpc_gnn_transformer"].call_count == 2

    selector.reset()
    assert len(selector.selection_history) == 0
    assert selector._step_counter == 0
    assert candidates["mpc_gnn_transformer"].call_count == 0
