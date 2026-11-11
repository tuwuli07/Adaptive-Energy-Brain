"""Shared baseline runner output, fairness, and repeatability."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.baselines import available_controllers
from src.data.synthetic import generate_synthetic, load_scenario, save_synthetic
from src.evaluation.baseline_runner import run_baseline_experiments


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
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
        "baseline_runner_fixture",
        processed_root=tmp_path / "processed",
    )
    return {
        "manifest": destination / "integration_manifest.yaml",
        "assets": ROOT / "configs" / "synthetic_scenario.yaml",
        "rating_origin": "hypothetical_example",
    }


def run(inputs, output):
    return run_baseline_experiments(**inputs, output=output)


def test_all_registered_controllers_run_with_matching_episode_and_initial_state(inputs, tmp_path):
    output = tmp_path / "experiment"
    summary = run(inputs, output)
    assert set(summary["controllers"]) == set(available_controllers())
    assert summary["input_rows"] == 8
    assert summary["steps"] == 7
    assert summary["synthetic_inputs_present"]
    assert summary["episode_policy"]["first_row_is_history_only"]
    assert summary["baseline_configuration"]["price_aware"]
    assert summary["source_provenance"]

    initial_states = set()
    timestamps = set()
    for name in available_controllers():
        result = summary["results"][name]
        controller_dir = output / "controllers" / name
        hourly = pd.read_csv(controller_dir / "hourly_results.csv")
        controller_summary = json.loads(
            (controller_dir / "summary.json").read_text()
        )
        assert len(hourly) == summary["steps"]
        assert {
            "timestamp_utc",
            "battery_requested_kw",
            "battery_actual_kw",
            "battery_soc_before",
            "battery_soc_after",
            "net_operating_cost",
            "cumulative_net_operating_cost",
            "cumulative_renewable_utilization",
            "power_balance_residual_kw",
            "energy_balance_residual_kwh",
        } <= set(hourly.columns)
        assert hourly["timestamp_utc"].is_monotonic_increasing
        assert (hourly["violation_count"] == 0).all()
        assert (hourly["power_balance_residual_kw"].abs() <= 1e-6).all()
        assert (hourly["energy_balance_residual_kwh"].abs() <= 1e-6).all()
        assert controller_summary["steps"] == summary["steps"]
        assert controller_summary["initial_soc"] == summary["initial_soc"]
        assert controller_summary["controller"] == name
        assert result["hourly_results_sha256"] == hashlib.sha256(
            (controller_dir / "hourly_results.csv").read_bytes()
        ).hexdigest()
        initial_states.add(controller_summary["initial_soc"])
        timestamps.add(tuple(hourly["timestamp_utc"]))
    assert len(initial_states) == 1
    assert len(timestamps) == 1
    comparison = pd.read_csv(output / "comparison.csv")
    assert set(comparison["controller"]) == set(available_controllers())
    assert (comparison["steps"] == 7).all()
    assert json.loads((output / "experiment_summary.json").read_text()) == summary


def test_repeated_experiments_are_reproducible_and_outputs_are_not_overwritten(inputs, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first_summary = run(inputs, first)
    second_summary = run(inputs, second)
    assert first_summary == second_summary
    for name in available_controllers():
        first_csv = first / "controllers" / name / "hourly_results.csv"
        second_csv = second / "controllers" / name / "hourly_results.csv"
        assert first_csv.read_bytes() == second_csv.read_bytes()
    before = {
        path.relative_to(first): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError, match="already exists"):
        run(inputs, first)
    after = {
        path.relative_to(first): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_selected_controller_and_interval_are_recorded(inputs, tmp_path):
    result = run_baseline_experiments(
        **inputs,
        output=tmp_path / "selected",
        controllers=["zero_action"],
        start="2026-01-01T02:00:00Z",
        end="2026-01-01T06:00:00Z",
    )
    assert result["controllers"] == ["zero_action"]
    assert result["input_rows"] == 4
    assert result["steps"] == 3
    assert result["interval"]["start_utc"] == "2026-01-01T02:00:00Z"
    hourly = pd.read_csv(
        tmp_path / "selected" / "controllers" / "zero_action" / "hourly_results.csv"
    )
    assert len(hourly) == 3
    assert hourly["timestamp_utc"].iloc[0] == "2026-01-01T03:00:00Z"


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"manifest": None, "split_root": None}, "exactly one"),
        ({"split_root": "split", "manifest": None, "partition": None}, "partition"),
        ({"partition": "test"}, "only when loading"),
    ],
)
def test_invalid_source_selection_is_rejected(inputs, tmp_path, changes, match):
    arguments = dict(inputs)
    arguments.update(changes)
    arguments["output"] = tmp_path / "invalid"
    with pytest.raises(ValueError, match=match):
        run_baseline_experiments(**arguments)
