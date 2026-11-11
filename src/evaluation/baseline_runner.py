"""Reproducible, same-episode evaluation for registered rule-based baselines."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from src.baselines import available_controllers, create_controller
from src.baselines.battery_preserving import load_battery_preserving_config
from src.baselines.price_aware import load_price_aware_config
from src.simulation.config import (
    SimulationConfig,
    load_simulation_config,
    validate_simulation_config,
)
from src.simulation.data_loader import (
    SimulationData,
    load_simulation_data,
    load_simulation_split,
)
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_CONFIG = ROOT / "configs" / "baselines.yaml"
DEFAULT_SIMULATION_CONFIG = ROOT / "configs" / "simulation.yaml"
DEFAULT_COST_CONFIG = ROOT / "configs" / "simulation_costs.yaml"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_native(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _input_provenance(data):
    return {
        source_id: {
            "origin": source["metadata"]["origin"],
            "synthetic": bool(source["metadata"].get("synthetic", False)),
            "input_file": source.get("input_file"),
        }
        for source_id, source in sorted(data.metadata["sources"].items())
    }


def _controller(name, *, baseline_config, cost_config, currency_code):
    if name == "price_aware":
        return create_controller(
            name,
            config_path=baseline_config,
            degradation_cost_per_kwh_throughput=(
                cost_config.battery_degradation_per_kwh_throughput
            ),
            currency_code=currency_code,
        )
    if name == "battery_preserving":
        return create_controller(name, config_path=baseline_config)
    return create_controller(name)


def _controller_parameters(controller):
    return {
        key: value
        for key, value in sorted(vars(controller).items())
        if not key.startswith("_") and isinstance(value, (str, int, float, bool, type(None)))
    }


def _run_controller(name, controller, environment, synthetic):
    controller.reset()
    initial_observation = environment.reset()
    initial_soc = initial_observation.state.battery_soc
    boundary_socs = [initial_soc]
    hourly, trace = [], []

    while not environment.terminated:
        observation = environment.observation
        action = controller.select_action(observation)
        step = environment.step(action)
        dispatch = step.dispatch
        boundary_socs.append(step.observation.state.battery_soc)
        values = asdict(step.metrics.values)
        cumulative = asdict(step.cumulative_metrics)
        hourly.append({
            "timestamp_utc": dispatch.timestamp_utc,
            "interval_hours": step.metrics.time_step_hours,
            "currency_code": step.metrics.currency_code,
            "synthetic_inputs_present": synthetic,
            "solar_available_kw": sum(
                item.available_generation_kw
                for item in dispatch.renewable_outputs
                if item.generation_type == "solar"
            ),
            "wind_available_kw": sum(
                item.available_generation_kw
                for item in dispatch.renewable_outputs
                if item.generation_type == "wind"
            ),
            "renewable_used_kw": dispatch.renewables.total_used_kw,
            "campus_demand_kw": dispatch.demand.campus_load_kw,
            "served_demand_kw": dispatch.served_demand_kw,
            "unserved_demand_kw": dispatch.unserved_demand_kw,
            "battery_soc_before": dispatch.battery.state_before.battery_soc,
            "battery_soc_after": dispatch.battery.state_after.battery_soc,
            "battery_requested_kw": dispatch.requested_battery_power_kw,
            "battery_actual_kw": dispatch.actual_battery_power_kw,
            "battery_charge_kw": dispatch.battery.charge_power_kw,
            "battery_discharge_kw": dispatch.battery.discharge_power_kw,
            "grid_import_kw": dispatch.grid.import_power_kw,
            "grid_export_kw": dispatch.grid.export_power_kw,
            "curtailment_kw": dispatch.renewables.total_curtailed_kw,
            **values,
            "renewable_utilization": step.metrics.values.renewable_utilization,
            **{
                f"cumulative_{key}": value
                for key, value in cumulative.items()
            },
            "cumulative_renewable_utilization": (
                step.cumulative_metrics.renewable_utilization
            ),
            "power_balance_residual_kw": dispatch.balance_residual_kw,
            "energy_balance_residual_kwh": dispatch.energy_balance_residual_kwh,
            "adjustment_count": len(step.metrics.adjustments),
            "violation_count": len(step.metrics.violations),
        })
        trace.append({
            "timestamp_utc": dispatch.timestamp_utc,
            "adjustments": [asdict(item) for item in step.metrics.adjustments],
            "violations": [asdict(item) for item in step.metrics.violations],
        })

    metrics = environment.cumulative_metrics
    summary = {
        "status": "complete",
        "controller": name,
        "input_rows": environment.metadata["environment"]["action_steps"] + 1,
        "steps": len(hourly),
        "currency_code": environment.step_metrics[0].currency_code if hourly else None,
        "initial_soc": initial_soc,
        "minimum_soc": min(boundary_socs),
        "maximum_soc": max(boundary_socs),
        "terminal_soc": boundary_socs[-1],
        "cumulative_metrics": {
            **asdict(metrics),
            "renewable_utilization": metrics.renewable_utilization,
        },
        "physical_violation_count": sum(row["violation_count"] for row in hourly),
        "adjustment_count": sum(row["adjustment_count"] for row in hourly),
        "maximum_absolute_power_balance_residual_kw": max(
            (abs(row["power_balance_residual_kw"]) for row in hourly), default=0.0
        ),
        "maximum_absolute_energy_balance_residual_kwh": max(
            (abs(row["energy_balance_residual_kwh"]) for row in hourly), default=0.0
        ),
        "constraint_records": trace,
    }
    return _json_native(summary), hourly


def run_baseline_experiments(
    *,
    assets,
    rating_origin,
    output,
    manifest=None,
    split_root=None,
    partition=None,
    controllers=None,
    start=None,
    end=None,
    simulation_config=None,
    costs=None,
    baseline_config=None,
    data=None,
    scenario_metadata=None,
):
    """Run named baselines with fresh simulator/controller state on one episode.

    Supply exactly one processed `manifest`, a Part 2 `split_root` and
    `partition`, or validated in-memory `data`. Each controller receives the
    same immutable observations, physical configuration, and starting SOC.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a fresh directory.")
    if sum(value is not None for value in (manifest, split_root, data)) != 1:
        raise ValueError("Supply exactly one of manifest, split_root or data.")
    if split_root is not None and partition not in {"train", "validation", "test"}:
        raise ValueError("split_root requires partition: train, validation, or test.")
    if manifest is not None and partition is not None:
        raise ValueError("partition is valid only when loading from split_root.")
    if data is not None and not isinstance(data, SimulationData):
        raise ValueError("data must be validated SimulationData.")
    if data is not None and (partition is not None or start is not None or end is not None):
        raise ValueError("partition/start/end are not valid with preselected in-memory data.")
    if rating_origin not in {
        "hypothetical_example",
        "documented_specification",
        "mixed",
    }:
        raise ValueError("rating_origin must be explicit.")

    if controllers is None:
        selected = available_controllers()
    elif isinstance(controllers, str):
        selected = (controllers,)
    else:
        selected = tuple(controllers)
    if (
        not selected
        or not all(isinstance(name, str) for name in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError("Select one or more unique registered controller names.")
    unknown = set(selected) - set(available_controllers())
    if unknown:
        raise ValueError(
            f"Unknown controller(s): {', '.join(sorted(unknown))}; "
            f"registered: {', '.join(available_controllers())}."
        )

    baseline_path = Path(baseline_config) if baseline_config is not None else DEFAULT_BASELINE_CONFIG
    simulation_path = (
        Path(simulation_config)
        if simulation_config is not None and not isinstance(simulation_config, SimulationConfig)
        else DEFAULT_SIMULATION_CONFIG
    )
    cost_path = Path(costs) if costs is not None else DEFAULT_COST_CONFIG
    price_parameters = load_price_aware_config(baseline_path)
    battery_parameters = load_battery_preserving_config(baseline_path)
    if isinstance(simulation_config, SimulationConfig):
        physical_config = validate_simulation_config(asdict(simulation_config))
    else:
        physical_config = load_simulation_config(simulation_path)
    cost_config = load_cost_config(cost_path)
    if price_parameters.currency_code != cost_config.currency_code:
        raise ValueError("Baseline and economic configuration currencies must match.")
    if physical_config.grid.currency_code != cost_config.currency_code:
        raise ValueError("Simulation and economic configuration currencies must match.")

    selection = {"start": start, "end": end}
    if manifest is not None:
        data = load_simulation_data(manifest, **selection)
    elif split_root is not None:
        data = load_simulation_split(split_root, partition, **selection)

    from scripts.run_simulation import _components

    synthetic = any(
        origin in {"synthetic", "simulated"}
        for origin in data.metadata["source_origin_summary"].values()
    )
    if isinstance(simulation_config, SimulationConfig):
        config_content = json.dumps(
            asdict(physical_config), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        simulation_file = {
            "path": "in-memory-simulation-config",
            "sha256": hashlib.sha256(config_content).hexdigest(),
        }
    else:
        simulation_file = {
            "path": str(simulation_path.resolve()),
            "sha256": _sha256(simulation_path),
        }
    input_files = {
        "assets": {
            "path": str(Path(assets).resolve()),
            "sha256": _sha256(assets),
        },
        "simulation_config": simulation_file,
        "cost_config": {
            "path": str(cost_path.resolve()),
            "sha256": _sha256(cost_path),
        },
        "baseline_config": {
            "path": str(baseline_path.resolve()),
            "sha256": _sha256(baseline_path),
        },
    }
    if manifest is not None:
        input_files["manifest"] = {
            "path": str(Path(manifest).resolve()),
            "sha256": _sha256(manifest),
        }
    elif split_root is not None:
        input_files["split_root"] = {
            "path": str(Path(split_root).resolve()),
            "partition": partition,
        }
    else:
        input_files["data_sources"] = {
            source_id: source.get("input_file")
            for source_id, source in sorted(data.metadata["sources"].items())
        }
    scenario = None
    if scenario_metadata is not None:
        scenario = _json_native(scenario_metadata)
        scenario_content = json.dumps(
            scenario, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        input_files["scenario"] = {
            "path": f"scenario:{scenario['scenario_id']}",
            "sha256": hashlib.sha256(scenario_content).hexdigest(),
        }
    evaluated_partition = partition
    if evaluated_partition is None and scenario is not None:
        evaluated_partition = scenario.get("source_partition")
    input_sources = _input_provenance(data)

    results = {}
    common_initial_soc = None
    for name in selected:
        controller = _controller(
            name,
            baseline_config=baseline_path,
            cost_config=cost_config,
            currency_code=cost_config.currency_code,
        )
        renewables, demand = _components(data, assets, rating_origin)
        environment = MicrogridEnvironment(
            data,
            renewables,
            demand,
            physical_config,
            cost_config,
        )
        summary, hourly = _run_controller(name, controller, environment, synthetic)
        if common_initial_soc is None:
            common_initial_soc = summary["initial_soc"]
        elif summary["initial_soc"] != common_initial_soc:
            raise RuntimeError("Controller episodes did not start at the same SOC.")
        results[name] = {
            "summary": summary,
            "hourly": hourly,
            "controller_parameters": _controller_parameters(controller),
            "source_provenance": environment.metadata,
        }

    steps = {item["summary"]["steps"] for item in results.values()}
    if len(steps) != 1:
        raise RuntimeError("Controller episode lengths differ for identical inputs.")

    comparison = []
    for name, result in results.items():
        summary = result["summary"]
        comparison.append({
            "controller": name,
            "steps": summary["steps"],
            "initial_soc": summary["initial_soc"],
            "terminal_soc": summary["terminal_soc"],
            "net_operating_cost": summary["cumulative_metrics"]["net_operating_cost"],
            "grid_import_energy_kwh": summary["cumulative_metrics"]["grid_import_energy_kwh"],
            "grid_export_energy_kwh": summary["cumulative_metrics"]["grid_export_energy_kwh"],
            "battery_throughput_kwh": summary["cumulative_metrics"]["battery_throughput_kwh"],
            "unserved_demand_energy_kwh": summary["cumulative_metrics"]["unserved_demand_energy_kwh"],
            "renewable_utilization": summary["cumulative_metrics"]["renewable_utilization"],
            "adjustment_count": summary["adjustment_count"],
            "physical_violation_count": summary["physical_violation_count"],
        })

    experiment = {
        "status": "complete",
        "controllers": list(selected),
        "partition": evaluated_partition,
        "input_rows": len(data),
        "steps": next(iter(steps)),
        "interval": data.metadata["interval"],
        "currency_code": cost_config.currency_code,
        "synthetic_inputs_present": synthetic,
        "rating_origin": rating_origin,
        "scenario": scenario,
        "initial_soc": common_initial_soc,
        "simulation_config": asdict(physical_config),
        "economic_assumptions": asdict(cost_config),
        "baseline_configuration": {
            "price_aware": asdict(price_parameters),
            "battery_preserving": asdict(battery_parameters),
        },
        "source_provenance": input_sources,
        "input_files": input_files,
        "episode_policy": {
            "first_row_is_history_only": True,
            "history_row_in_episode_metrics": False,
            "action_steps": len(data) - 1,
        },
        "results": {},
    }

    experiment = _json_native(experiment)
    output.mkdir(parents=True, exist_ok=False)
    controllers_dir = output / "controllers"
    controllers_dir.mkdir()
    for name, result in results.items():
        controller_dir = controllers_dir / name
        controller_dir.mkdir()
        frame = pd.DataFrame(result["hourly"])
        csv_path = controller_dir / "hourly_results.csv"
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, float_format="%.12g")
        summary = {
            **result["summary"],
            "controller_parameters": result["controller_parameters"],
            "source_provenance": result["source_provenance"],
            "shared_experiment": {
                key: value
                for key, value in experiment.items()
                if key not in {"results"}
            },
            "hourly_results_sha256": _sha256(csv_path),
        }
        summary_path = controller_dir / "summary.json"
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(_json_native(summary), stream, indent=2, allow_nan=False)
            stream.write("\n")
        experiment["results"][name] = {
            "hourly_results": str(csv_path.relative_to(output)),
            "hourly_results_sha256": _sha256(csv_path),
            "summary": str(summary_path.relative_to(output)),
            "summary_sha256": _sha256(summary_path),
        }

    comparison_path = output / "comparison.csv"
    with comparison_path.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(comparison).to_csv(stream, index=False, float_format="%.12g")
    experiment["comparison"] = {
        "path": str(comparison_path.relative_to(output)),
        "sha256": _sha256(comparison_path),
    }
    summary_path = output / "experiment_summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(_json_native(experiment), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return _json_native(experiment)


def main(argv=None):
    """Command-line entry point for baseline episode comparisons."""
    import argparse
    import sys

    from src.baselines import available_controllers

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--split-root", type=Path)
    parser.add_argument("--partition", choices=("train", "validation", "test"))
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument(
        "--rating-origin",
        required=True,
        choices=("hypothetical_example", "documented_specification", "mixed"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--controller", action="append", choices=available_controllers())
    parser.add_argument("--start", help="UTC start, inclusive; first row supplies history")
    parser.add_argument("--end", help="UTC end, exclusive")
    parser.add_argument("--simulation-config", type=Path)
    parser.add_argument("--costs", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    args = parser.parse_args(argv)
    if args.split_root is not None and args.partition is None:
        parser.error("--split-root requires --partition.")
    if args.manifest is not None and args.partition is not None:
        parser.error("--partition is valid only with --split-root.")
    try:
        summary = run_baseline_experiments(
            manifest=args.manifest,
            split_root=args.split_root,
            partition=args.partition,
            assets=args.assets,
            rating_origin=args.rating_origin,
            output=args.output,
            controllers=args.controller,
            start=args.start,
            end=args.end,
            simulation_config=args.simulation_config,
            costs=args.costs,
            baseline_config=args.baseline_config,
        )
        print(json.dumps({
            "controllers": summary["controllers"],
            "steps": summary["steps"],
            "output": str(args.output.resolve()),
        }, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(f"Baseline experiment error: {exc}", file=sys.stderr)
        return 1
