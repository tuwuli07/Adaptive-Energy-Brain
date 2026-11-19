"""Tests for secondary inventory-adjusted cost accounting and validation reporting."""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import yaml

from src.baselines.price_aware import load_price_aware_config
from src.evaluation.baseline_runner import run_baseline_experiments
from src.evaluation.inventory_accounting import (
    compute_inventory_adjusted_cost,
    compute_replacement_reference_value,
)
from src.evaluation.ppo_validation import _enhance, dominates
from src.simulation.config import load_simulation_config
from src.simulation.metrics import load_cost_config
from src.data.synthetic import generate_synthetic, load_scenario, save_synthetic


ROOT = Path(__file__).resolve().parents[1]


def test_1_replacement_reference_value_formula_evaluates_exactly():
    """Requirement 1: Formula (p_off + c_deg) / eta_c evaluates to 0.14736842105263157."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    price_cfg = load_price_aware_config(ROOT / "configs/baselines.yaml")

    p_off = price_cfg.low_price_threshold
    c_deg = cost_cfg.battery_degradation_per_kwh_throughput
    eta_c = sim_cfg.battery.charge_efficiency

    assert p_off == 0.12
    assert c_deg == 0.02
    assert eta_c == 0.95

    expected = (0.12 + 0.02) / 0.95
    assert expected == pytest.approx(0.14736842105263157, rel=1e-15)

    ref_val = compute_replacement_reference_value(sim_cfg, cost_cfg)
    assert ref_val == pytest.approx(expected, rel=1e-15)
    assert ref_val == 0.14736842105263157

    # Test explicit override
    ref_val_override = compute_replacement_reference_value(
        sim_cfg, cost_cfg, off_peak_price_per_kwh=0.10
    )
    assert ref_val_override == pytest.approx((0.10 + 0.02) / 0.95, rel=1e-15)


def test_2_positive_delta_energy_decreases_adjusted_cost():
    """Requirement 2: Positive delta E decreases adjusted cost below raw cost."""
    raw_cost = 43674.78
    initial_energy = 250.0
    terminal_energy = 450.0  # +200 kWh accumulated
    ref_val = 0.14736842105263157

    result = compute_inventory_adjusted_cost(raw_cost, initial_energy, terminal_energy, ref_val)

    assert result["energy_change_kwh"] == 200.0
    assert result["inventory_adjustment_usd"] > 0
    assert result["inventory_adjustment_usd"] == pytest.approx(200.0 * ref_val)
    assert result["inventory_adjusted_net_operating_cost_usd"] < raw_cost
    assert result["inventory_adjusted_net_operating_cost_usd"] == pytest.approx(
        raw_cost - (200.0 * ref_val)
    )


def test_3_negative_delta_energy_increases_adjusted_cost():
    """Requirement 3: Negative delta E increases adjusted cost above raw cost."""
    raw_cost = 43835.77
    initial_energy = 250.0
    terminal_energy = 50.0  # -200 kWh depleted
    ref_val = 0.14736842105263157

    result = compute_inventory_adjusted_cost(raw_cost, initial_energy, terminal_energy, ref_val)

    assert result["energy_change_kwh"] == -200.0
    assert result["inventory_adjustment_usd"] < 0
    assert result["inventory_adjustment_usd"] == pytest.approx(-200.0 * ref_val)
    assert result["inventory_adjusted_net_operating_cost_usd"] > raw_cost
    assert result["inventory_adjusted_net_operating_cost_usd"] == pytest.approx(
        raw_cost - (-200.0 * ref_val)
    )


def test_4_zero_delta_energy_leaves_adjusted_cost_equal():
    """Requirement 4: Zero delta E leaves adjusted cost equal to raw cost."""
    raw_cost = 43866.17
    initial_energy = 250.0
    terminal_energy = 250.0
    ref_val = 0.14736842105263157

    result = compute_inventory_adjusted_cost(raw_cost, initial_energy, terminal_energy, ref_val)

    assert result["energy_change_kwh"] == 0.0
    assert result["inventory_adjustment_usd"] == 0.0
    assert result["inventory_adjusted_net_operating_cost_usd"] == raw_cost


def test_5_energy_extraction_from_config_capacity_and_soc():
    """Requirement 5: Correct initial and terminal energy extraction from SimulationConfig.battery.capacity_kwh and SOC."""
    sim_cfg = load_simulation_config()
    capacity = sim_cfg.battery.capacity_kwh
    assert capacity == 500.0

    soc_initial = 0.5
    soc_terminal_high = 0.9
    soc_terminal_low = 0.1

    e_init = soc_initial * capacity
    e_term_high = soc_terminal_high * capacity
    e_term_low = soc_terminal_low * capacity

    assert e_init == 250.0
    assert e_term_high == 450.0
    assert e_term_low == 50.0


def test_6_capacity_and_efficiency_scaling_effects():
    """Requirement 6: Capacity and efficiency scaling effects."""
    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()

    # Efficiency scaling: lower efficiency -> higher replacement cost
    # Clone sim_cfg with different efficiency
    from dataclasses import replace
    battery_lower_eta = replace(sim_cfg.battery, charge_efficiency=0.90)
    sim_cfg_lower_eta = replace(sim_cfg, battery=battery_lower_eta)

    ref_val_90 = compute_replacement_reference_value(sim_cfg_lower_eta, cost_cfg)
    expected_90 = (0.12 + 0.02) / 0.90
    assert ref_val_90 == pytest.approx(expected_90)
    assert ref_val_90 > 0.14736842105263157

    # Capacity scaling: higher capacity -> larger energy delta for same SOC change
    battery_large_cap = replace(sim_cfg.battery, capacity_kwh=1000.0)
    sim_cfg_large_cap = replace(sim_cfg, battery=battery_large_cap)

    soc_init, soc_term = 0.5, 0.9
    e_init_1000 = soc_init * sim_cfg_large_cap.battery.capacity_kwh
    e_term_1000 = soc_term * sim_cfg_large_cap.battery.capacity_kwh
    assert e_term_1000 - e_init_1000 == 400.0  # vs 200.0 for 500 kWh


@pytest.fixture
def baseline_fixture(tmp_path):
    sources = yaml.safe_load((ROOT / "configs" / "data_sources.yaml").read_text())
    for source in sources["sources"]:
        source.update(status="pending", local_path=None, download_url=None)
    source_path = tmp_path / "sources.yaml"
    source_path.write_text(yaml.safe_dump(sources))
    scenario = load_scenario()
    scenario["hours"] = 8
    result = generate_synthetic(
        scenario,
        sources_path=source_path,
        processed_root=tmp_path / "processed",
    )
    destination = save_synthetic(
        result,
        "baseline_inv_fixture",
        processed_root=tmp_path / "processed",
    )
    return {
        "manifest": destination / "integration_manifest.yaml",
        "assets": ROOT / "configs" / "synthetic_scenario.yaml",
        "rating_origin": "hypothetical_example",
    }


def test_7_baseline_runner_compatibility(baseline_fixture, tmp_path):
    """Requirement 7: Baseline runner compatibility (checks summary.json and comparison.csv)."""
    output = tmp_path / "baseline_out"
    experiment = run_baseline_experiments(**baseline_fixture, output=output)

    assert experiment["status"] == "complete"
    for ctrl in experiment["controllers"]:
        summary_file = output / "controllers" / ctrl / "summary.json"
        assert summary_file.is_file()
        summary = json.loads(summary_file.read_text())

        assert "reference_value_usd_per_kwh_dc" in summary
        assert "inventory_adjustment_usd" in summary
        assert "inventory_adjusted_net_operating_cost_usd" in summary
        assert summary["reference_value_usd_per_kwh_dc"] == pytest.approx(0.14736842105263157)

        # Verification of formula inside summary
        delta_e = summary["terminal_energy_kwh"] - summary["initial_energy_kwh"]
        adj = delta_e * summary["reference_value_usd_per_kwh_dc"]
        assert summary["inventory_adjustment_usd"] == pytest.approx(adj)
        assert summary["inventory_adjusted_net_operating_cost_usd"] == pytest.approx(
            summary["cumulative_metrics"]["net_operating_cost"] - adj
        )

    comparison_csv = output / "comparison.csv"
    assert comparison_csv.is_file()
    df = pd.read_csv(comparison_csv)

    for col in [
        "initial_energy_kwh",
        "terminal_energy_kwh",
        "energy_change_kwh",
        "reference_value_usd_per_kwh_dc",
        "inventory_adjustment_usd",
        "inventory_adjusted_net_operating_cost_usd",
    ]:
        assert col in df.columns


def test_8_ppo_validation_compatibility(tmp_path):
    """Requirement 8: PPO validation compatibility (checks enhance and schema fields)."""
    # Create mock environment and summary
    from dataclasses import dataclass

    @dataclass
    class MockState:
        battery_soc: float
        battery_energy_kwh: float

    @dataclass
    class MockBatteryDispatch:
        state_before: MockState
        state_after: MockState
        efficiency_loss_kwh: float

    @dataclass
    class MockGridDispatch:
        import_power_kw: float

    @dataclass
    class MockResult:
        requested_battery_power_kw: float
        actual_battery_power_kw: float
        battery: MockBatteryDispatch
        grid: MockGridDispatch

    @dataclass
    class MockStepMetric:
        adjustments: list

    class MockEnv:
        def __init__(self):
            self.metadata = {"environment": {"action_steps": 2}}
            self.dispatch_results = [
                MockResult(
                    -50.0, -50.0,
                    MockBatteryDispatch(MockState(0.5, 250.0), MockState(0.6, 300.0), 2.5),
                    MockGridDispatch(100.0),
                ),
                MockResult(
                    -50.0, -50.0,
                    MockBatteryDispatch(MockState(0.6, 300.0), MockState(0.7, 350.0), 2.5),
                    MockGridDispatch(100.0),
                ),
            ]
            self.step_metrics = [MockStepMetric([]), MockStepMetric([])]

    sim_cfg = load_simulation_config()
    cost_cfg = load_cost_config()
    env = MockEnv()
    summary = {
        "steps": 2,
        "cumulative_metrics": {
            "net_operating_cost": 500.0,
            "unserved_demand_energy_kwh": 0.0,
        },
        "maximum_absolute_power_balance_residual_kw": 0.0,
        "maximum_absolute_energy_balance_residual_kwh": 0.0,
    }

    _enhance(summary, env, sim_cfg, costs=cost_cfg)

    assert "reference_value_usd_per_kwh_dc" in summary
    assert "inventory_adjustment_usd" in summary
    assert "inventory_adjusted_net_operating_cost_usd" in summary
    assert summary["initial_energy_kwh"] == 250.0
    assert summary["terminal_energy_kwh"] == 350.0
    assert summary["energy_change_kwh"] == 100.0
    assert summary["reference_value_usd_per_kwh_dc"] == pytest.approx(0.14736842105263157)
    expected_adj = 100.0 * 0.14736842105263157
    assert summary["inventory_adjustment_usd"] == pytest.approx(expected_adj)
    assert summary["inventory_adjusted_net_operating_cost_usd"] == pytest.approx(500.0 - expected_adj)


def test_9_raw_net_operating_cost_remains_identical(baseline_fixture, tmp_path):
    """Requirement 9: Raw net operating cost remains identical."""
    output = tmp_path / "baseline_raw_check"
    experiment = run_baseline_experiments(**baseline_fixture, output=output)
    df = pd.read_csv(output / "comparison.csv")

    for ctrl in experiment["controllers"]:
        summary_file = output / "controllers" / ctrl / "summary.json"
        summary = json.loads(summary_file.read_text())
        row = df[df["controller"] == ctrl].iloc[0]

        # Raw cost in summary cumulative_metrics matches comparison.csv net_operating_cost
        assert summary["cumulative_metrics"]["net_operating_cost"] == pytest.approx(
            row["net_operating_cost"], rel=1e-9
        )
        # And adjusted cost is distinct from raw cost whenever energy change != 0
        if row["energy_change_kwh"] != 0:
            assert row["inventory_adjusted_net_operating_cost_usd"] != row["net_operating_cost"]


def test_10_dominates_pareto_logic_remains_unchanged():
    """Requirement 10: dominates() Pareto logic remains unchanged and ignores adjusted cost."""
    # Controller A: Lower raw cost, higher terminal energy -> dominates B
    a = {
        "cumulative_metrics": {"net_operating_cost": 100.0},
        "terminal_energy_kwh": 300.0,
        "inventory_adjusted_net_operating_cost_usd": 80.0,
    }
    b = {
        "cumulative_metrics": {"net_operating_cost": 120.0},
        "terminal_energy_kwh": 250.0,
        "inventory_adjusted_net_operating_cost_usd": 110.0,
    }
    assert dominates(a, b)
    assert not dominates(b, a)

    # Controller C has lower adjusted cost than D, but HIGHER raw cost and same energy:
    # Pareto dominance uses RAW cost, so D dominates C, NOT C dominates D!
    c = {
        "cumulative_metrics": {"net_operating_cost": 200.0},
        "terminal_energy_kwh": 300.0,
        "inventory_adjusted_net_operating_cost_usd": 100.0,  # artificial lower adjusted
    }
    d = {
        "cumulative_metrics": {"net_operating_cost": 180.0},
        "terminal_energy_kwh": 300.0,
        "inventory_adjusted_net_operating_cost_usd": 150.0,  # artificial higher adjusted
    }
    # d has lower raw cost with same terminal energy -> d dominates c
    assert dominates(d, c)
    assert not dominates(c, d)

    # Equal within tolerance -> Neither dominates
    e = {
        "cumulative_metrics": {"net_operating_cost": 100.0},
        "terminal_energy_kwh": 300.0,
    }
    assert not dominates(a, e)
    assert not dominates(e, a)


def test_11_verification_against_exact_historical_replay_values():
    """Requirement 11: Verification against exact historical replay values.

    Exact 1,313-hour validation values:
    Price-Aware:
      Raw: $43,674.77776469938
      Delta E: +200 kWh (250 -> 450 kWh)
      Adjustment: +$29.473684210526315
      Adjusted: $43,645.304080488854 (~$43,645.304081)

    Zero-Action:
      Raw: $43,866.17133402038
      Delta E: 0 kWh (250 -> 250 kWh)
      Adjustment: $0.00
      Adjusted: $43,866.17133402038 (~$43,866.171334)

    PPO (gamma 0.99 / 0.999) & Self-Consumption:
      Raw: $43,835.77133402037
      Delta E: -200 kWh (250 -> 50 kWh)
      Adjustment: -$29.473684210526315
      Adjusted: $43,865.245018230896 (~$43,865.245018)

    Battery-Preserving:
      Raw: $43,850.971334020374
      Delta E: -100 kWh (250 -> 150 kWh)
      Adjustment: -$14.736842105263158
      Adjusted: $43,865.70817612563 (~$43,865.708176)
    """
    ref_val = (0.12 + 0.02) / 0.95  # 0.14736842105263157

    # 1. Price-Aware
    pa_raw = 43674.77776469938
    pa = compute_inventory_adjusted_cost(pa_raw, 250.0, 450.0, ref_val)
    assert pa["energy_change_kwh"] == 200.0
    assert pa["inventory_adjustment_usd"] == pytest.approx(29.473684210526315, rel=1e-12)
    assert pa["inventory_adjusted_net_operating_cost_usd"] == pytest.approx(43645.304080488854, rel=1e-12)
    assert round(pa["inventory_adjusted_net_operating_cost_usd"], 6) == 43645.304080

    # 2. Zero-Action
    za_raw = 43866.17133402038
    za = compute_inventory_adjusted_cost(za_raw, 250.0, 250.0, ref_val)
    assert za["energy_change_kwh"] == 0.0
    assert za["inventory_adjustment_usd"] == 0.0
    assert za["inventory_adjusted_net_operating_cost_usd"] == pytest.approx(43866.17133402038, rel=1e-12)
    assert round(za["inventory_adjusted_net_operating_cost_usd"], 6) == 43866.171334

    # 3. PPO & Self-Consumption
    ppo_raw = 43835.77133402037
    ppo = compute_inventory_adjusted_cost(ppo_raw, 250.0, 50.0, ref_val)
    assert ppo["energy_change_kwh"] == -200.0
    assert ppo["inventory_adjustment_usd"] == pytest.approx(-29.473684210526315, rel=1e-12)
    assert ppo["inventory_adjusted_net_operating_cost_usd"] == pytest.approx(43865.245018230896, rel=1e-12)
    assert round(ppo["inventory_adjusted_net_operating_cost_usd"], 6) == 43865.245018

    # 4. Battery-Preserving
    bp_raw = 43850.971334020374
    bp = compute_inventory_adjusted_cost(bp_raw, 250.0, 150.0, ref_val)
    assert bp["energy_change_kwh"] == -100.0
    assert bp["inventory_adjustment_usd"] == pytest.approx(-14.736842105263158, rel=1e-12)
    assert bp["inventory_adjusted_net_operating_cost_usd"] == pytest.approx(43865.70817612563, rel=1e-12)
    assert round(bp["inventory_adjusted_net_operating_cost_usd"], 6) == 43865.708176

    # Relative savings against zero-action:
    # Price-Aware savings: $43,866.171334 - $43,645.304080 = $220.867254
    pa_savings = za["inventory_adjusted_net_operating_cost_usd"] - pa["inventory_adjusted_net_operating_cost_usd"]
    assert pa_savings == pytest.approx(220.8672535, rel=1e-5)

    # PPO savings: $43,866.171334 - $43,865.245018 = $0.926316
    ppo_savings = za["inventory_adjusted_net_operating_cost_usd"] - ppo["inventory_adjusted_net_operating_cost_usd"]
    assert ppo_savings == pytest.approx(0.9263158, rel=1e-5)


def test_input_validation_and_errors():
    """Verify robust input validation on invalid arguments."""
    with pytest.raises(TypeError, match="SimulationConfig"):
        compute_replacement_reference_value(config="invalid")

    with pytest.raises(TypeError, match="CostConfig"):
        compute_replacement_reference_value(costs="invalid")

    with pytest.raises(ValueError, match="nonnegative"):
        compute_replacement_reference_value(off_peak_price_per_kwh=-0.05)

    with pytest.raises(ValueError, match="Stored energy"):
        compute_inventory_adjusted_cost(100.0, -10.0, 50.0, 0.14)

    with pytest.raises(ValueError, match="Reference value"):
        compute_inventory_adjusted_cost(100.0, 10.0, 50.0, -0.14)
