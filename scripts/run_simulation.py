"""Replay a processed manifest with a deterministic zero-battery REQUEST.

Dispatch may adjust the request for physical feasibility; actual actions and
adjustments are exported. No optimization controller, acquisition, generation,
preprocessing, reward or training occurs. First row is observation history, so
N input rows yield N-1 dispatched hours. --start/--end select [start,end), including
that history row. Use a fresh --output directory for each run.

Example (existing explicitly synthetic bundle; all ratings hypothetical):
python scripts/run_simulation.py --manifest data/processed/synthetic_hypothetical_campus_v1/integration_manifest.yaml --assets configs/synthetic_scenario.yaml --rating-origin hypothetical_example --output results/zero_battery_run

Asset YAML: solar/wind may each be {asset_id,capacity_kw} or a list of those
mappings. Extra synthetic-scenario settings are ignored, not regenerated. Ratings
must match the processed entity registry; no ratings are inferred from data maxima.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.actions import BatteryAction
from src.simulation.config import _UniqueKeyLoader, load_simulation_config
from src.simulation.data_loader import load_simulation_data
from src.simulation.demand import CampusDemand
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config
from src.simulation.renewables import SolarPV, WindGeneration
from src.simulation.visualization import plot_simulation


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _components(data, asset_path, rating_origin):
    assets = yaml.load(Path(asset_path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(assets, dict):
        raise ValueError("Asset configuration must be a YAML mapping.")
    metadata = data.metadata

    def provenance(kind, entity):
        candidates = [source["metadata"] for source in metadata["sources"].values()
                      if source["record_type"] == kind and entity in source["entity_ids"]]
        if len(candidates) != 1:
            raise ValueError(f"Expected one unambiguous {kind} provenance for {entity}.")
        return candidates[0]

    components = []
    for kind, component in (("solar", SolarPV), ("wind", WindGeneration)):
        entries = assets.get(kind)
        entries = [entries] if isinstance(entries, dict) else entries
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Asset configuration requires explicit {kind} ratings.")
        if any(not isinstance(entry, dict) or not {"asset_id", "capacity_kw"} <= entry.keys() for entry in entries):
            raise ValueError(f"Each {kind} asset requires asset_id and capacity_kw.")
        if len({entry["asset_id"] for entry in entries}) != len(entries):
            raise ValueError(f"Duplicate {kind} asset IDs.")
        if {entry["asset_id"] for entry in entries} != set(metadata["entities"][kind]):
            raise ValueError(f"{kind} ratings must match processed dataset entity IDs.")
        for entry in entries:
            components.append(component(entry["asset_id"], entry["capacity_kw"],
                                        provenance("generation", entry["asset_id"]),
                                        capacity_origin=rating_origin))
    buildings = metadata["entities"]["buildings"]
    demand = CampusDemand({b: provenance("demand", b) for b in buildings}, building_ids=list(buildings))
    return components, demand


def run_simulation(*, manifest, assets, rating_origin, output, config=None, costs=None,
                   start=None, end=None):
    """Run a full selected episode, then exclusively create its output bundle."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a fresh directory.")
    config_path = Path(config) if config is not None else ROOT / "configs/simulation.yaml"
    cost_path = Path(costs) if costs is not None else ROOT / "configs/simulation_costs.yaml"
    simulation_config = load_simulation_config(config_path)
    cost_config = load_cost_config(cost_path)
    data = load_simulation_data(manifest, start=start, end=end)
    renewables, demand = _components(data, assets, rating_origin)
    environment = MicrogridEnvironment(data, renewables, demand, simulation_config, cost_config)
    origins = data.metadata["source_origin_summary"]
    synthetic = any(origin == "synthetic" for origin in origins.values())
    label = "SYNTHETIC INPUTS PRESENT - simulation, not campus observations" if synthetic else "Simulation replay - see source provenance"
    environment.reset()
    rows, records = [], []
    while not environment.terminated:
        step = environment.step(BatteryAction(0))
        dispatch = step.dispatch
        rows.append({
            "timestamp_utc": dispatch.timestamp_utc,
            "interval_hours": step.metrics.time_step_hours,
            "currency_code": step.metrics.currency_code,
            "synthetic_inputs_present": synthetic,
            "solar_available_kw": sum(r.available_generation_kw for r in dispatch.renewable_outputs if r.generation_type == "solar"),
            "wind_available_kw": sum(r.available_generation_kw for r in dispatch.renewable_outputs if r.generation_type == "wind"),
            "renewable_used_kw": dispatch.renewables.total_used_kw,
            "campus_demand_kw": dispatch.demand.campus_load_kw,
            "battery_soc_before": dispatch.battery.state_before.battery_soc,
            "battery_soc_after": dispatch.battery.state_after.battery_soc,
            "battery_requested_kw": dispatch.requested_battery_power_kw,
            "battery_actual_kw": dispatch.actual_battery_power_kw,
            "battery_charge_kw": dispatch.battery.charge_power_kw,
            "battery_discharge_kw": dispatch.battery.discharge_power_kw,
            "grid_import_kw": dispatch.grid.import_power_kw,
            "grid_export_kw": dispatch.grid.export_power_kw,
            "curtailment_kw": dispatch.renewables.total_curtailed_kw,
            "unserved_demand_kw": dispatch.unserved_demand_kw,
            **asdict(step.metrics.values),
            "renewable_utilization": step.metrics.values.renewable_utilization,
            "cumulative_net_operating_cost": step.cumulative_metrics.net_operating_cost,
            "power_balance_residual_kw": dispatch.balance_residual_kw,
            "energy_balance_residual_kwh": dispatch.energy_balance_residual_kwh,
            "adjustment_count": len(step.metrics.adjustments),
            "violation_count": len(step.metrics.violations),
        })
        records.append({"timestamp_utc": dispatch.timestamp_utc,
                        "adjustments": [asdict(a) for a in step.metrics.adjustments],
                        "violations": [asdict(v) for v in step.metrics.violations]})
    frame = pd.DataFrame(rows)
    summary = {
        "status": "complete", "label": label, "input_rows": len(data),
        "steps": len(rows), "controller": "constant requested battery power: 0 kW; dispatch safety adjustments remain active",
        "currency_code": cost_config.currency_code,
        "simulation_config": asdict(simulation_config), "economic_assumptions": asdict(cost_config),
        "rating_origin": rating_origin,
        "renewable_ratings": [{"asset_id": component.asset_id,
                               "generation_type": component.generation_type,
                               "capacity_kw": component.capacity_kw,
                               "parameter_origin": rating_origin} for component in renewables],
        "source_provenance": environment.metadata,
        "input_files": {name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
                        for name, path in (("manifest", manifest), ("assets", assets),
                                           ("simulation_config", config_path), ("cost_config", cost_path))},
        "cumulative_metrics": {**asdict(environment.cumulative_metrics),
                               "renewable_utilization": environment.cumulative_metrics.renewable_utilization},
        "metric_definitions": {
            "renewable_utilization": "Bus-delivered renewable kWh / available renewable kWh; includes charging/export, never recounts battery discharge. Null for zero availability.",
            "degradation": "Linear AC charge+discharge throughput cost proxy; not validated electrochemical aging.",
        },
        "constraint_records": records,
        "physical_violation_count": sum(row["violation_count"] for row in rows),
    }
    # Public return value and disk report use identical JSON-native structures.
    summary = json.loads(json.dumps(summary, allow_nan=False))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "hourly_results.csv").open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, float_format="%.12g")
    charts = plot_simulation(frame, output, currency_code=cost_config.currency_code, label=label)
    summary["output_sha256"] = {path.name: sha256(path) for path in [output / "hourly_results.csv", *charts]}
    # Written last: a partial export has no successful completion summary.
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--rating-origin", required=True,
                        choices=("hypothetical_example", "documented_specification", "mixed"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--costs", type=Path)
    parser.add_argument("--start", help="UTC start, inclusive; first row supplies history")
    parser.add_argument("--end", help="UTC end, exclusive")
    args = parser.parse_args(argv)
    try:
        summary = run_simulation(**vars(args))
        print(json.dumps({key: summary[key] for key in (
            "label", "steps", "currency_code", "cumulative_metrics", "physical_violation_count")}, indent=2))
        print(f"Results saved to {args.output.resolve()}")
        return 0
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(f"Simulation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
