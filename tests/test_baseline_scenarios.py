"""Reproducible stress scenarios and baseline comparison figures."""

import hashlib
import json
from pathlib import Path

from matplotlib import image as mpl_image
import pandas as pd
import pytest
import yaml

from src.baselines import available_controllers
from src.data.synthetic import generate_synthetic, load_scenario, save_synthetic
from src.evaluation.baseline_scenarios import (
    apply_baseline_scenario,
    available_scenarios,
    run_baseline_scenarios,
)
from src.evaluation import baseline_visualization
from src.evaluation.baseline_visualization import plot_baseline_experiment
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import load_simulation_data


ROOT = Path(__file__).resolve().parents[1]


def _file_hashes(directory):
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _make_inputs(tmp_path):
    sources = yaml.safe_load((ROOT / "configs" / "data_sources.yaml").read_text())
    for source in sources["sources"]:
        source.update(status="pending", local_path=None, download_url=None)
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(yaml.safe_dump(sources))
    scenario = load_scenario()
    scenario["hours"] = 8
    generated = generate_synthetic(
        scenario,
        sources_path=sources_path,
        processed_root=tmp_path / "processed",
    )
    dataset = save_synthetic(
        generated,
        "baseline_scenario_fixture",
        processed_root=tmp_path / "processed",
    )
    return dataset / "integration_manifest.yaml", dataset


def test_scenario_runs_are_reproducible_preserve_inputs_and_generate_all_plots(
    tmp_path, monkeypatch
):
    manifest, dataset_dir = _make_inputs(tmp_path)
    before = _file_hashes(dataset_dir)
    source_data = load_simulation_data(manifest)
    base_config = load_simulation_config()

    low_a, _, low_metadata_a = apply_baseline_scenario(
        source_data, "low_renewable", base_config
    )
    low_b, _, low_metadata_b = apply_baseline_scenario(
        source_data, "low_renewable", base_config
    )
    assert low_metadata_a == low_metadata_b
    assert low_a.metadata["scenario"] == low_b.metadata["scenario"]
    renewable_index = max(
        range(len(source_data)),
        key=lambda index: (
            source_data[index].pv_available_kw + source_data[index].wind_available_kw
        ),
    )
    assert low_a[renewable_index].pv_available_kw == pytest.approx(
        source_data[renewable_index].pv_available_kw * 0.25
    )
    assert low_a[renewable_index].wind_available_kw == pytest.approx(
        source_data[renewable_index].wind_available_kw * 0.25
    )
    assert (
        source_data[renewable_index].pv_available_kw
        != low_a[renewable_index].pv_available_kw
    )
    spiked_data, _, spike_metadata = apply_baseline_scenario(
        source_data, "electricity_price_spikes", base_config
    )
    assert spike_metadata["spike_timestamps_utc"]
    spike_index = source_data.metadata["interval"]["hours"] // 2
    assert spiked_data[spike_index].import_price >= 1.0
    assert spiked_data[0].import_price == source_data[0].import_price
    high_data, _, _ = apply_baseline_scenario(
        source_data, "high_campus_demand", base_config
    )
    assert high_data[renewable_index].campus_load_kw == pytest.approx(
        source_data[renewable_index].campus_load_kw * 1.25
    )

    suite_output = tmp_path / "scenario_suite"
    index = run_baseline_scenarios(
        manifest=manifest,
        assets=ROOT / "configs" / "synthetic_scenario.yaml",
        rating_origin="hypothetical_example",
        output=suite_output,
    )
    assert tuple(item["scenario_name"] for item in index["scenario_runs"]) == available_scenarios()
    assert _file_hashes(dataset_dir) == before
    assert json.loads((suite_output / "scenario_index.json").read_text()) == index

    experiments = {
        item["scenario_name"]: suite_output / item["experiment_directory"]
        for item in index["scenario_runs"]
    }
    normal_summary = json.loads(
        (experiments["normal_operation"] / "experiment_summary.json").read_text()
    )
    for name in available_scenarios():
        experiment_path = experiments[name]
        summary = json.loads((experiment_path / "experiment_summary.json").read_text())
        assert summary["scenario"]["scenario_id"] == f"{name}_v1"
        assert summary["controllers"] == list(available_controllers())
        assert summary["steps"] == normal_summary["steps"] == 7
        frames = {
            controller: pd.read_csv(
                experiment_path / "controllers" / controller / "hourly_results.csv"
            )
            for controller in available_controllers()
        }
        assert len({tuple(frame["timestamp_utc"]) for frame in frames.values()}) == 1

    normal_frame = pd.read_csv(
        experiments["normal_operation"] / "controllers" / "zero_action" / "hourly_results.csv"
    )
    low_frame = pd.read_csv(
        experiments["low_renewable"] / "controllers" / "zero_action" / "hourly_results.csv"
    )
    assert low_frame["solar_available_kw"].to_numpy() == pytest.approx(
        normal_frame["solar_available_kw"].to_numpy() * 0.25
    )
    high_frame = pd.read_csv(
        experiments["high_campus_demand"] / "controllers" / "zero_action" / "hourly_results.csv"
    )
    assert high_frame["campus_demand_kw"].to_numpy() == pytest.approx(
        normal_frame["campus_demand_kw"].to_numpy() * 1.25
    )
    price_summary = json.loads(
        (experiments["electricity_price_spikes"] / "experiment_summary.json").read_text()
    )
    assert price_summary["scenario"]["spike_timestamps_utc"]
    constrained_summary = json.loads(
        (experiments["limited_grid_import"] / "experiment_summary.json").read_text()
    )
    assert constrained_summary["simulation_config"]["grid"]["import_limit_kw"] == pytest.approx(100)
    assert constrained_summary["simulation_config"]["unserved_load"]["handling"] == "record"
    constrained_zero = json.loads(
        (experiments["limited_grid_import"] / "controllers" / "zero_action" / "summary.json").read_text()
    )
    assert constrained_zero["cumulative_metrics"]["unserved_demand_energy_kwh"] > 0

    plots_dir = experiments["low_renewable"] / "plots"
    captured = {}
    original_write = baseline_visualization._write

    def capture_plot(figure, path):
        axes = figure.axes
        if path.name in {
            "net_operating_cost_comparison.png",
            "renewable_utilization_comparison.png",
            "battery_throughput_comparison.png",
        }:
            captured[path.name] = [patch.get_height() for patch in axes[0].patches]
        elif path.name == "grid_import_export_over_time.png":
            captured[path.name] = [
                [line.get_ydata() for line in axis.lines] for axis in axes
            ]
        elif path.name in {"battery_soc_over_time.png", "renewable_curtailment.png",
                           "unserved_demand.png"}:
            captured[path.name] = [line.get_ydata() for line in axes[0].lines]
        original_write(figure, path)

    monkeypatch.setattr(baseline_visualization, "_write", capture_plot)
    paths = plot_baseline_experiment(
        experiments["low_renewable"], output_dir=plots_dir
    )
    assert len(paths) == 7
    for path in paths:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        image = mpl_image.imread(path)
        assert image.shape[0] > 300 and image.shape[1] > 500
        assert image.std() > 0.01
    experiment_path = experiments["low_renewable"]
    summaries = {
        controller: json.loads(
            (experiment_path / "controllers" / controller / "summary.json").read_text()
        )
        for controller in available_controllers()
    }
    comparison_values = {
        controller: summary["cumulative_metrics"]
        for controller, summary in summaries.items()
    }
    assert captured["net_operating_cost_comparison.png"] == pytest.approx(
        [values["net_operating_cost"] for values in comparison_values.values()]
    )
    assert captured["battery_throughput_comparison.png"] == pytest.approx(
        [values["battery_throughput_kwh"] for values in comparison_values.values()]
    )
    expected_utilization = [
        values["renewable_utilization"]
        for values in comparison_values.values()
        if values["renewable_utilization"] is not None
    ]
    assert captured["renewable_utilization_comparison.png"] == pytest.approx(
        expected_utilization
    )
    plot_frame = pd.read_csv(
        experiment_path
        / "controllers"
        / available_controllers()[0]
        / "hourly_results.csv"
    )
    assert captured["renewable_curtailment.png"][0] == pytest.approx(
        plot_frame["curtailment_kw"].to_numpy()
    )
    assert captured["unserved_demand.png"][0] == pytest.approx(
        plot_frame["unserved_demand_kw"].to_numpy()
    )
    assert captured["grid_import_export_over_time.png"][0][0] == pytest.approx(
        plot_frame["grid_import_kw"].to_numpy()
    )
    assert captured["grid_import_export_over_time.png"][1][0] == pytest.approx(
        plot_frame["grid_export_kw"].to_numpy()
    )
    assert captured["battery_soc_over_time.png"][0] == pytest.approx(
        list(plot_frame["battery_soc_before"])
        + [plot_frame["battery_soc_after"].iloc[-1]]
    )
    with pytest.raises(FileExistsError, match="already exist"):
        plot_baseline_experiment(experiments["low_renewable"], output_dir=plots_dir)
    with pytest.raises(FileExistsError, match="already exists"):
        run_baseline_scenarios(
            manifest=manifest,
            assets=ROOT / "configs" / "synthetic_scenario.yaml",
            rating_origin="hypothetical_example",
            output=suite_output,
        )
