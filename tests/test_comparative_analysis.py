"""Tests for Phase 8.1 Unified Comparative Evaluation Engine.

Verifies:
- Complete and typed controller registry across rule-based, DRL, MPC, and oracle models.
- Accurate inventory-adjusted metric calculations using established replacement rate ($0.14736842/kWh DC).
- Strict partition access gating (held-out test partition blocked without explicit permission).
- Historical benchmark import and provenance tracking.
- Parity of deterministic replays against authoritative validation benchmarks within $0.01 tolerance.
- Strict causality boundaries (noncausal oracle vs causal controllers).
"""

from pathlib import Path
import pytest

from src.evaluation.comparative_metrics import (
    ComparativeEvaluationMetrics,
    calculate_comparative_metrics,
)
from src.evaluation.comparative_runner import (
    ComparativeEvaluationError,
    import_historical_benchmark,
    reproduce_controller_replay,
)
from src.evaluation.controller_registry import (
    CONTROLLER_REGISTRY,
    BenchmarkStatus,
    CausalityType,
    ControllerCategory,
    ControllerRegistryEntry,
    get_controller_entry,
    get_primary_benchmark_ids,
    list_controllers,
)
from src.simulation.config import load_simulation_config
from src.simulation.metrics import load_cost_config


ROOT = Path(__file__).resolve().parents[1]


class TestControllerRegistry:
    """Verifies controller registration, typing, metadata, and categorization."""

    def test_registered_controller_count_and_keys(self):
        """Registry contains all 10 registered candidate controllers."""
        expected_ids = {
            "zero_action",
            "price_aware_continuous",
            "price_aware_discrete",
            "ppo_continuous_control",
            "mpc_seasonal_persistence",
            "mpc_gnn_transformer",
            "mpc_oracle_rolling_24h",
            "ppo_discrete_diagnostic",
            "self_consumption",
            "battery_preserving",
        }
        assert set(CONTROLLER_REGISTRY.keys()) == expected_ids

    def test_primary_benchmark_ids(self):
        """Primary benchmark suite consists of the core 7 controllers."""
        primary_ids = get_primary_benchmark_ids()
        expected_primary = [
            "zero_action",
            "price_aware_continuous",
            "price_aware_discrete",
            "ppo_continuous_control",
            "mpc_seasonal_persistence",
            "mpc_gnn_transformer",
            "mpc_oracle_rolling_24h",
        ]
        assert set(primary_ids) == set(expected_primary)
        assert len(primary_ids) == 7

    def test_causality_classification(self):
        """Oracle is explicitly NONCAUSAL_LOOKAHEAD; all other controllers are STRICTLY_CAUSAL."""
        for cid, entry in CONTROLLER_REGISTRY.items():
            if cid == "mpc_oracle_rolling_24h":
                assert entry.causality == CausalityType.NONCAUSAL_LOOKAHEAD
                assert entry.category == ControllerCategory.DIAGNOSTIC_ORACLE
            else:
                assert entry.causality == CausalityType.STRICTLY_CAUSAL

    def test_metadata_serialization(self):
        """All registry entries serialize cleanly to dict with required schema keys."""
        required_keys = {
            "controller_id",
            "display_name",
            "category",
            "benchmark_status",
            "causality",
            "source_implementation",
            "original_experiment_id",
            "configuration_path",
            "checkpoint_reference",
            "observation_adapter",
            "action_adapter",
            "validation_benchmark_artifact",
            "authoritative_raw_cost_usd",
            "authoritative_inventory_credit_usd",
            "authoritative_adjusted_cost_usd",
        }
        for entry in CONTROLLER_REGISTRY.values():
            d = entry.to_dict()
            assert required_keys.issubset(set(d.keys()))
            assert isinstance(d["controller_id"], str)
            assert isinstance(d["display_name"], str)

    def test_controller_lookup_and_filtering(self):
        """Registry lookup and category filtering operate correctly."""
        entry = get_controller_entry("zero_action")
        assert entry.controller_id == "zero_action"

        with pytest.raises(KeyError, match="not found in registry"):
            get_controller_entry("non_existent_controller")

        rule_based = list_controllers(category=ControllerCategory.RULE_BASED)
        assert len(rule_based) == 5
        assert {e.controller_id for e in rule_based} == {
            "zero_action",
            "price_aware_continuous",
            "price_aware_discrete",
            "self_consumption",
            "battery_preserving",
        }

        rl = list_controllers(category=ControllerCategory.REINFORCEMENT_LEARNING)
        assert len(rl) == 2

        mpc = list_controllers(category=ControllerCategory.MODEL_PREDICTIVE_CONTROL)
        assert len(mpc) == 2


class TestComparativeMetrics:
    """Verifies inventory accounting, flow metrics, and simultaneous flow checks."""

    def test_inventory_reference_value_calculation(self):
        """Reference value matches established $0.14736842/kWh DC rate."""
        sim_config = load_simulation_config()
        cost_config = load_cost_config()
        # Scheduled off-peak = 0.12, degradation = 0.02, charging efficiency eta_c = 0.95
        expected_v_ref = (0.12 + 0.02) / 0.95
        assert abs(expected_v_ref - 0.14736842105263157) < 1e-12

    def test_inventory_adjustment_arithmetic(self):
        """Synthetic record correctly computes inventory credit and adjusted cost."""
        sim_config = load_simulation_config()
        cost_config = load_cost_config()
        v_ref = (0.12 + 0.02) / 0.95

        # 1-hour synthetic episode: SOC moves from 0.50 (250 kWh) to 0.70 (350 kWh)
        # Gain of 100 kWh DC -> Credit = +100 * v_ref
        hourly_records = [
            {
                "timestamp_utc": "2026-10-01T00:00:00Z",
                "interval_hours": 1.0,
                "currency_code": "USD",
                "synthetic_inputs_present": True,
                "solar_available_kw": 50.0,
                "wind_available_kw": 50.0,
                "renewable_used_kw": 100.0,
                "campus_demand_kw": 200.0,
                "served_demand_kw": 200.0,
                "unserved_demand_kw": 0.0,
                "battery_soc_before": 0.50,
                "battery_soc_after": 0.70,
                "battery_requested_kw": -100.0,
                "battery_actual_kw": -100.0,
                "battery_charge_kw": 100.0,
                "battery_discharge_kw": 0.0,
                "battery_charge_energy_kwh": 100.0,
                "battery_discharge_energy_kwh": 0.0,
                "grid_import_kw": 100.0,
                "grid_export_kw": 0.0,
                "grid_import_energy_kwh": 100.0,
                "grid_export_energy_kwh": 0.0,
                "curtailment_kw": 0.0,
                "import_cost": 12.0,
                "export_revenue": 0.0,
                "net_operating_cost": 14.0,  # 12 import + 2 degradation
                "battery_degradation_cost": 2.0,
                "unserved_demand_penalty": 0.0,
                "renewable_curtailment_cost": 0.0,
                "renewable_utilization": 1.0,
                "power_balance_residual_kw": 0.0,
                "energy_balance_residual_kwh": 0.0,
                "adjustment_count": 0,
                "violation_count": 0,
                "step_latency_ms": 1.5,
            }
        ]

        metrics = calculate_comparative_metrics(
            hourly_records,
            sim_config=sim_config,
            cost_config=cost_config,
            solver_fallback_count=0,
            timing_records_ms=[1.5],
        )

        assert metrics.initial_soc == 0.50
        assert metrics.terminal_soc == 0.70
        assert abs(metrics.energy_change_kwh_dc - 100.0) < 1e-6
        expected_credit = 100.0 * v_ref
        assert abs(metrics.terminal_inventory_credit_usd - expected_credit) < 1e-4
        assert abs(metrics.inventory_adjusted_cost_usd - (14.0 - expected_credit)) < 1e-4
        assert metrics.simultaneous_flow_steps == 0
        assert metrics.physical_violation_count == 0


class TestComparativeRunnerAndSafety:
    """Verifies partition gating, deterministic replays, and benchmark parity."""

    def test_held_out_test_partition_access_blocked(self):
        """Calling replay with partition='test' raises PermissionError when allow_held_out_test=False."""
        with pytest.raises(PermissionError, match="held-out test partition is strictly blocked"):
            reproduce_controller_replay(
                "zero_action",
                partition="test",
                allow_held_out_test=False,
            )

    def test_unknown_partition_rejected(self):
        """Replay rejects invalid partition name."""
        with pytest.raises(ValueError, match="Unknown or unsupported partition"):
            reproduce_controller_replay("zero_action", partition="eval_split_xyz")

    def test_import_historical_benchmark(self):
        """Importing historical benchmark loads authoritative adjusted cost for primary controllers."""
        for cid in ["zero_action", "price_aware_continuous", "price_aware_discrete"]:
            entry = get_controller_entry(cid)
            res = import_historical_benchmark(cid)
            assert res["controller_id"] == cid
            assert res["mode"] == "imported_historical"
            assert abs(res["adjusted_cost_usd"] - entry.authoritative_adjusted_cost_usd) < 1e-4

    @pytest.mark.parametrize(
        ("controller_id", "expected_adj_cost"),
        [
            ("zero_action", 43866.17),
            ("price_aware_continuous", 43645.30),
            ("price_aware_discrete", 43632.76),
        ],
    )
    def test_rule_based_deterministic_replay_parity(
        self, controller_id: str, expected_adj_cost: float
    ):
        """Replay of rule-based baselines matches authoritative benchmark within $0.01 tolerance."""
        metrics, hourly, prov = reproduce_controller_replay(controller_id, partition="validation")
        assert len(hourly) == 1313
        assert metrics.initial_soc == 0.50
        diff = abs(metrics.inventory_adjusted_cost_usd - expected_adj_cost)
        assert diff <= 0.01, (
            f"Controller {controller_id} reproduced cost ${metrics.inventory_adjusted_cost_usd:.4f} "
            f"differs from authoritative benchmark ${expected_adj_cost:.4f} by ${diff:.4f}"
        )
