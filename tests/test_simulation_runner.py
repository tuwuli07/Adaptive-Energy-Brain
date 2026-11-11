"""Offline runner exports, provenance, determinism and useful CLI failures."""

import hashlib
import json
from pathlib import Path

from matplotlib import image as mpl_image
import pandas as pd
import pytest
import yaml

from scripts.run_simulation import main, run_simulation
from src.data.synthetic import generate_synthetic, load_scenario, save_synthetic
from src.simulation.visualization import plot_simulation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    sources = yaml.safe_load((ROOT / "configs/data_sources.yaml").read_text())
    for source in sources["sources"]:
        source.update(status="pending", local_path=None, download_url=None)
    source_path = tmp_path / "sources.yaml"
    source_path.write_text(yaml.safe_dump(sources))
    scenario = load_scenario()
    scenario["hours"] = 8
    result = generate_synthetic(scenario, sources_path=source_path,
                                processed_root=tmp_path / "processed")
    destination = save_synthetic(result, "runner_fixture", processed_root=tmp_path / "processed")
    return {"manifest": destination / "integration_manifest.yaml",
            "assets": ROOT / "configs/synthetic_scenario.yaml",
            "rating_origin": "hypothetical_example"}


def hashes(root):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}


def test_runner_exports_complete_episode_charts_metrics_and_provenance(inputs, tmp_path):
    before = hashes(inputs["manifest"].parent)
    output = tmp_path / "run"
    summary = run_simulation(**inputs, output=output)
    assert summary["status"] == "complete"
    assert summary["input_rows"] == 8
    assert summary["steps"] == 7
    assert "SYNTHETIC" in summary["label"]
    assert summary["physical_violation_count"] == 0
    saved = json.loads((output / "summary.json").read_text())
    assert saved == summary
    assert saved["economic_assumptions"]["parameter_origin"] == "hypothetical_example"
    assert saved["renewable_ratings"][0]["capacity_kw"] == 300
    assert saved["source_provenance"]["environment"]["history_row_in_episode_metrics"] is False
    assert saved["source_provenance"]["sources"]
    frame = pd.read_csv(output / "hourly_results.csv")
    assert len(frame) == 7
    required = {"timestamp_utc", "solar_available_kw", "wind_available_kw", "campus_demand_kw",
                "battery_soc_after", "battery_charge_kw", "battery_discharge_kw", "grid_import_kw",
                "grid_export_kw", "net_operating_cost", "curtailment_kw", "unserved_demand_kw",
                "battery_degradation_cost", "renewable_utilized_energy_kwh", "currency_code"}
    assert required <= set(frame)
    assert frame["battery_requested_kw"].eq(0).all()
    assert frame["currency_code"].eq("USD").all()
    assert frame["synthetic_inputs_present"].all()
    assert frame["timestamp_utc"].iloc[0] == "2026-01-01T01:00:00Z"
    assert frame["net_operating_cost"].sum() == pytest.approx(summary["cumulative_metrics"]["net_operating_cost"])
    assert frame["battery_soc_after"].between(.1, .9).all()
    for name in ("energy_flows.png", "battery.png", "costs.png"):
        path = output / name
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        pixels = mpl_image.imread(path)
        assert pixels.shape[0] > 300 and pixels.shape[1] > 500
        assert pixels.std() > .01
        assert summary["output_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashes(inputs["manifest"].parent) == before


def test_repeated_runs_identical_results_and_no_overwrite(inputs, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a = run_simulation(**inputs, output=first)
    b = run_simulation(**inputs, output=second)
    assert a == b
    assert (first / "hourly_results.csv").read_bytes() == (second / "hourly_results.csv").read_bytes()
    before = hashes(first)
    with pytest.raises(FileExistsError, match="already exists"):
        run_simulation(**inputs, output=first)
    assert hashes(first) == before


def test_cli_selection_and_completion_output(inputs, tmp_path, capsys):
    output = tmp_path / "selected"
    code = main(["--manifest", str(inputs["manifest"]), "--assets", str(inputs["assets"]),
                 "--rating-origin", inputs["rating_origin"], "--output", str(output),
                 "--start", "2026-01-01T02:00:00Z", "--end", "2026-01-01T05:00:00Z"])
    assert code == 0
    assert "Results saved" in capsys.readouterr().out
    assert len(pd.read_csv(output / "hourly_results.csv")) == 2


@pytest.mark.parametrize("option", ["--manifest", "--assets", "--config", "--costs"])
def test_missing_input_files_produce_useful_errors(inputs, tmp_path, capsys, option):
    args = ["--manifest", str(inputs["manifest"]), "--assets", str(inputs["assets"]),
            "--rating-origin", inputs["rating_origin"], "--output", str(tmp_path / "output")]
    if option in args:
        args[args.index(option) + 1] = str(tmp_path / "absent.yaml")
    else:
        args.extend([option, str(tmp_path / "absent.yaml")])
    assert main(args) == 1
    assert "Simulation error:" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_invalid_config_and_mismatched_asset_ratings_fail_before_export(inputs, tmp_path):
    config = yaml.safe_load((ROOT / "configs/simulation.yaml").read_text())
    config["battery"]["capacity_kwh"] = -1
    path = tmp_path / "bad_config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="capacity_kwh"):
        run_simulation(**inputs, output=tmp_path / "bad_run", config=path)
    assets = tmp_path / "bad_assets.yaml"
    assets.write_text("solar: {asset_id: WRONG, capacity_kw: 300}\nwind: {asset_id: W01, capacity_kw: 150}\n")
    with pytest.raises(ValueError, match="entity IDs"):
        run_simulation(**{**inputs, "assets": assets}, output=tmp_path / "bad_assets_run")
    assert not (tmp_path / "bad_run").exists()
    assert not (tmp_path / "bad_assets_run").exists()


def test_plotting_empty_or_existing_output_rejected(inputs, tmp_path):
    with pytest.raises(ValueError, match="empty"):
        plot_simulation(pd.DataFrame(), tmp_path, currency_code="USD", label="Test")
    output = tmp_path / "plots"
    run_simulation(**inputs, output=output)
    frame = pd.read_csv(output / "hourly_results.csv")
    before = hashes(output)
    with pytest.raises(FileExistsError):
        plot_simulation(frame, output, currency_code="USD", label="Test")
    assert hashes(output) == before
