"""Approved pilot orchestration and planned-study records, without study execution."""

from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.evaluation.ppo_validation import aggregate_records, validate_policies
from src.rl.episodes import sha256
from src.rl.training import ROOT, load_training_config, source_snapshot, train_ppo, write_json
from src.simulation.config import _UniqueKeyLoader


DEFAULT_EXPERIMENT = ROOT / "configs/ppo_validation.yaml"


def load_experiment_config(path=DEFAULT_EXPERIMENT):
    value = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    fields = {"version", "stage", "learning_rates", "pilot_seed", "pilot_timesteps", "episode_hours",
              "checkpoint_interval", "validation_rows", "initial_soc", "planned_study", "comparison"}
    if not isinstance(value, dict) or set(value) != fields or value["version"] != 1 or value["stage"] != "pilot":
        raise ValueError("Expected the approved version-1 pilot experiment configuration.")
    if value["learning_rates"] != {"A": 3e-4, "B": 1e-4} or value["pilot_seed"] != 0:
        raise ValueError("Pilot requires configurations A/B and seed zero.")
    for name in ("pilot_timesteps", "episode_hours", "checkpoint_interval", "validation_rows"):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError(f"Invalid experiment integer {name}.")
    if value["pilot_timesteps"] > 4096:
        raise ValueError("Pilot execution is capped at the approved 4096 transitions per run.")
    expected_study = {"seeds": [0, 1, 2, 3, 4], "primary_checkpoint_timesteps": 32768,
                      "diagnostic_checkpoints": [8192, 16384, 32768]}
    if value["planned_study"] != expected_study or value["initial_soc"] != .5:
        raise ValueError("Planned study and initial SOC disagree with the approved protocol.")
    comparison = value["comparison"]
    if comparison != {"cost_tolerance": 1e-6, "energy_tolerance_kwh": 1e-6,
                      "rule": "paired_seed_cost_terminal_energy_pareto"}:
        raise ValueError("Use the declared numerical Pareto tolerances without a valuation coefficient.")
    return value


def training_plots(output, runs):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for run in runs:
        path = Path(run["output"]) / "logs/progress.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for axis, column in zip(axes.flat, ("train/value_loss", "train/policy_gradient_loss",
                                           "train/approx_kl", "train/std")):
            if column in frame:
                axis.plot(frame["time/total_timesteps"], frame[column], label=run["configuration"])
            axis.set(xlabel="Training transitions", ylabel=column.removeprefix("train/"))
            axis.grid(alpha=.2)
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend()
    figure.suptitle("Pilot training diagnostics, seed 0")
    figure.savefig(output / "training_diagnostics.png", dpi=150)
    plt.close(figure)


def run_pilot(*, split_root, assets, rating_origin, output, experiment_config=DEFAULT_EXPERIMENT,
              base_training_config=ROOT / "configs/ppo_training.yaml",
              simulation_config=ROOT / "configs/simulation.yaml",
              costs_config=ROOT / "configs/simulation_costs.yaml",
              baseline_config=ROOT / "configs/baselines.yaml"):
    """Run the two pilot jobs once each and retain all failures without replacement."""
    started = time.perf_counter()
    settings = load_experiment_config(experiment_config)
    base = load_training_config(base_training_config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Pilot output already exists: {output}")
    output.mkdir(parents=True)
    write_json(output / "protocol.json", settings)
    report = {"schema_version": "1.0", "stage": "pilot", "status": "running",
              "protocol": settings, "source": source_snapshot(output), "runs": [],
              "planned_study_executed": False, "research_winner": None}
    write_json(output / "pilot.json", report)
    checkpoints = []
    failure_records = []
    try:
        for name, rate in settings["learning_rates"].items():
            run_output = output / ("training_" + name)
            config = deepcopy(base)
            config.update(version=2, purpose="validation_pilot", learning_rate=rate,
                          seed=settings["pilot_seed"], total_timesteps=settings["pilot_timesteps"],
                          episode_hours=settings["episode_hours"], checkpoint_interval=settings["checkpoint_interval"])
            path = output / ("training_" + name + ".yaml")
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            record = {"configuration": name, "seed": settings["pilot_seed"], "output": str(run_output),
                      "learning_rate": rate, "status": "failed", "configuration_sha256": sha256(path)}
            wall_started = time.perf_counter()
            try:
                result = train_ppo(split_root=split_root, assets=assets, rating_origin=rating_origin,
                                   output=run_output, training_config=path,
                                   simulation_config=simulation_config, costs_config=costs_config)
                pointer = json.loads((run_output / "latest_checkpoint.json").read_text(encoding="utf-8"))
                record.update(status=result["status"], progress=result["progress"], verification=result["verification"])
                checkpoints.append({"id": "PPO_" + name, "configuration": name, "seed": settings["pilot_seed"],
                                    "path": str(run_output / pointer["path"])})
            except Exception as exc:
                record.update(exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                failure_records.append({"kind": "ppo", "configuration": name, "seed": settings["pilot_seed"],
                                        "status": "failed", "controller": "PPO_" + name,
                                        "failure_stage": "training", "diagnostics": record})
            except BaseException as exc:
                record.update(status="interrupted", exception=type(exc).__name__, traceback=traceback.format_exc())
                raise
            finally:
                record["wall_seconds"] = time.perf_counter() - wall_started
                report["runs"].append(record)
                write_json(output / "pilot.json", report)
        validation = validate_policies(
            split_root=split_root, assets=assets, rating_origin=rating_origin, output=output / "validation",
            checkpoints=checkpoints, expected_rows=settings["validation_rows"], initial_soc=settings["initial_soc"],
            expected_timesteps=settings["pilot_timesteps"], simulation_config=simulation_config,
            costs_config=costs_config, baseline_config=baseline_config)
        screening = aggregate_records(validation["controllers"] + failure_records, [settings["pilot_seed"]],
                                      settings["pilot_timesteps"], pilot=True,
                                      cost_tolerance=settings["comparison"]["cost_tolerance"],
                                      energy_tolerance=settings["comparison"]["energy_tolerance_kwh"])
        write_json(output / "screening.json", screening)
        report["screening"] = screening
        report["validation_status"] = validation["status"]
        report["validation_runtime_seconds"] = validation["runtime_seconds"]
        report["status"] = "completed" if not failure_records and validation["status"] == "completed" else "completed_with_failures"
        training_plots(output, report["runs"])
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["diagnostics"] = {"exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["artifacts"] = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*")
                               if p.is_file() and p != output / "pilot.json"}
        write_json(output / "pilot.json", report)
    return report
