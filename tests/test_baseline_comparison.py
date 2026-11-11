"""Saved experiment comparison, metric reconciliation, and cost edge cases."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.baselines import available_controllers
from src.data.synthetic import generate_synthetic, load_scenario, save_synthetic
from src.evaluation.baseline_comparison import (
    BaselineComparisonError,
    compare_baseline_experiment,
    cost_improvement,
)
from src.evaluation.baseline_runner import run_baseline_experiments


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path):
    sources = yaml.safe_load((ROOT / "configs" / "data_sources.yaml").read_text())
    for source in sources["sources"]:
        source.update(status="pending", local_path=None, download_url=None)
    source_path = tmp_path / "sources.yaml"
    source_path.write_text(yaml.safe_dump(sources))
    scenario = load_scenario()
    scenario["hours"] = 6
    generated = generate_synthetic(
        scenario,
        sources_path=source_path,
        processed_root=tmp_path / "processed",
    )
    dataset = save_synthetic(
        generated,
        "baseline_comparison_fixture",
        processed_root=tmp_path / "processed",
    )
    path = tmp_path / "experiment"
    run_baseline_experiments(
        manifest=dataset / "integration_manifest.yaml",
        assets=ROOT / "configs" / "synthetic_scenario.yaml",
        rating_origin="hypothetical_example",
        output=path,
    )
    return path


def test_cost_improvement_handles_positive_zero_and_negative_baselines():
    assert cost_improvement(100, 80) == {
        "absolute_improvement": 20.0,
        "percentage_improvement": 20.0,
        "percentage_status": "defined",
    }
    assert cost_improvement(0, -2) == {
        "absolute_improvement": 2.0,
        "percentage_improvement": None,
        "percentage_status": "undefined_zero_baseline",
    }
    assert cost_improvement(-10, -15) == {
        "absolute_improvement": 5.0,
        "percentage_improvement": 50.0,
        "percentage_status": "defined",
    }
    assert cost_improvement(-10, 5)["percentage_improvement"] == -150.0


def test_comparison_reuses_metrics_and_separates_transaction_from_total_cost(experiment, tmp_path):
    output = tmp_path / "comparison"
    report = compare_baseline_experiment(experiment, output)
    by_name = {row["controller"]: row for row in report["controllers"]}
    assert set(by_name) == set(available_controllers())
    for name in available_controllers():
        summary = json.loads(
            (experiment / "controllers" / name / "summary.json").read_text()
        )
        values = summary["cumulative_metrics"]
        row = by_name[name]
        assert row["grid_import_cost"] == pytest.approx(values["import_cost"])
        assert row["grid_export_revenue"] == pytest.approx(values["export_revenue"])
        assert row["net_grid_transaction_cost"] == pytest.approx(
            values["import_cost"] - values["export_revenue"]
        )
        assert row["net_operating_cost"] == pytest.approx(values["net_operating_cost"])
        assert row["estimated_battery_degradation_cost"] == pytest.approx(
            values["battery_degradation_cost"]
        )
        assert row["battery_energy_throughput_kwh"] == pytest.approx(
            values["battery_throughput_kwh"]
        )
        assert row["grid_import_energy_kwh"] == pytest.approx(
            values["grid_import_energy_kwh"]
        )
        assert row["unserved_demand_energy_kwh"] == pytest.approx(
            values["unserved_demand_energy_kwh"]
        )
        assert row["steps"] == summary["steps"]
    table = pd.read_csv(output / "performance_comparison.csv")
    assert set(table["controller"]) == set(available_controllers())
    assert json.loads((output / "performance_comparison.json").read_text()) == report


def test_improvement_is_relative_to_zero_action_and_zero_cost_is_explicit(experiment, tmp_path):
    report = compare_baseline_experiment(experiment, tmp_path / "comparison")
    rows = {row["controller"]: row for row in report["controllers"]}
    zero = rows["zero_action"]
    assert zero["net_operating_cost_absolute_improvement"] == pytest.approx(0)
    if zero["net_operating_cost"] != 0:
        assert zero["net_operating_cost_percentage_improvement"] == pytest.approx(0)
    else:
        assert zero["net_operating_cost_percentage_improvement"] is None
        assert zero["net_operating_cost_percentage_status"] == "undefined_zero_baseline"
    for name, row in rows.items():
        if name != "zero_action":
            expected = zero["net_operating_cost"] - row["net_operating_cost"]
            assert row["net_operating_cost_absolute_improvement"] == pytest.approx(expected)


def test_comparison_requires_zero_action_baseline(experiment):
    experiment_summary_path = experiment / "experiment_summary.json"
    experiment_summary = json.loads(experiment_summary_path.read_text())
    experiment_summary["controllers"].remove("zero_action")
    experiment_summary["results"].pop("zero_action")
    experiment_summary_path.write_text(json.dumps(experiment_summary))
    with pytest.raises(BaselineComparisonError, match="zero_action"):
        compare_baseline_experiment(experiment)


def test_mismatched_comparison_period_is_rejected(experiment):
    summary_path = experiment / "experiment_summary.json"
    experiment_summary = json.loads(summary_path.read_text())
    name = next(item for item in experiment_summary["controllers"] if item != "zero_action")
    entry = experiment_summary["results"][name]
    hourly_path = experiment / entry["hourly_results"]
    frame = pd.read_csv(hourly_path)
    frame.loc[0, "timestamp_utc"] = "2026-02-01T00:00:00Z"
    frame.to_csv(hourly_path, index=False, float_format="%.12g")
    entry["hourly_results_sha256"] = hashlib.sha256(hourly_path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(experiment_summary))
    with pytest.raises(BaselineComparisonError, match="identical timestamp"):
        compare_baseline_experiment(experiment)


def test_existing_comparison_directory_is_not_overwritten(experiment, tmp_path):
    output = tmp_path / "comparison"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError, match="already exists"):
        compare_baseline_experiment(experiment, output)
    assert marker.read_text() == "unchanged"
