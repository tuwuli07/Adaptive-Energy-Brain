"""Bounded paired gamma experiment; no research-scale execution path."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

from scripts.run_simulation import _components
from src.evaluation.ppo_validation import validate_policies
from src.rl.episodes import TrainingEpisodeEnvironment, load_training_data, sha256
from src.rl.experiments import training_plots
from src.rl.training import (ROOT, load_training_config, new_ppo, seed_everything,
                             source_snapshot, train_ppo, verify_checkpoint, write_json)
from src.simulation.config import _UniqueKeyLoader, load_simulation_config
from src.simulation.metrics import load_cost_config


def tensor_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_diagnostic_config(path):
    settings = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    expected = {"version", "stage", "gammas", "seed", "learning_rate", "total_timesteps",
                "episode_hours", "checkpoint_interval", "validation_checkpoints", "validation_rows", "initial_soc"}
    if (not isinstance(settings, dict) or set(settings) != expected or settings["version"] != 1
            or settings["stage"] != "gamma_diagnostic" or settings["seed"] != 0
            or settings["gammas"] != {"control": .99, "treatment": .999}
            or settings["learning_rate"] != .0003 or settings["initial_soc"] != .5):
        raise ValueError("Expected the approved paired gamma diagnostic protocol.")
    for name in ("total_timesteps", "episode_hours", "checkpoint_interval", "validation_rows"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"Invalid diagnostic {name}.")
    steps = settings["total_timesteps"]
    checkpoints = settings["validation_checkpoints"]
    if (steps > 8192 or not isinstance(checkpoints, list) or len(checkpoints) != 2
            or any(type(n) is not int or n < 1 for n in checkpoints)
            or checkpoints != [steps // 2, steps]
            or any(n % settings["checkpoint_interval"] for n in checkpoints)):
        raise ValueError("Diagnostic budget/checkpoints exceed or disagree with the approved design.")
    return settings


def paired_preflight(configs, inputs, *, differing_setting="gamma"):
    """Train-only provenance, initial policy, and complete planned sampler sequence."""
    if len(configs) != 2 or differing_setting not in {"gamma", "reward_scale"}:
        raise ValueError("Paired preflight requires two configs and one supported varying setting.")
    first, second = configs.values()
    if {k for k in first if first[k] != second[k]} != {differing_setting}:
        raise ValueError(f"Paired training settings must differ only in {differing_setting}.")
    data, provenance = load_training_data(inputs["split_root"])
    config = load_simulation_config(inputs["simulation_config"])
    if config.battery.initial_soc != .5:
        raise ValueError("Diagnostic requires initial SOC 0.5.")
    costs = load_cost_config(inputs["costs_config"])
    renewable, demand = _components(data, inputs["assets"], inputs["rating_origin"])
    records = []
    for settings in configs.values():
        seed_everything(settings["seed"])
        environment = TrainingEpisodeEnvironment(data, renewable, demand, config, costs,
                                                  hours=settings["episode_hours"], sampler_seed=settings["seed"])
        try:
            model = new_ppo(settings, environment)
            rng = np.random.default_rng(settings["seed"])
            windows = [{"start_index": int(rng.integers(0, len(data) - settings["episode_hours"])),
                        "child_seed": int(rng.integers(0, 2**31))}
                       for _ in range(settings["total_timesteps"] // settings["episode_hours"] + 1)]
            records.append({"initial_policy_sha256": tensor_hash(model), "windows": windows})
        finally:
            environment.close()
    if records[0] != records[1]:
        raise ValueError("Paired initial policies or sampler sequences differ.")
    return {"verified": True, "data": provenance, **records[0],
            "settings_differ_only_in": [differing_setting], "test_partition_access": False}


def run_diagnostic(*, split_root, assets, rating_origin, output,
                   experiment_config=ROOT / "configs/ppo_gamma_diagnostic.yaml",
                   base_training_config=ROOT / "configs/ppo_training.yaml",
                   simulation_config=ROOT / "configs/simulation.yaml",
                   costs_config=ROOT / "configs/simulation_costs.yaml"):
    started = time.perf_counter()
    settings = load_diagnostic_config(experiment_config)
    base = load_training_config(base_training_config)
    configs = {}
    for name, gamma in settings["gammas"].items():
        config = deepcopy(base)
        config.update(version=2, purpose="gamma_diagnostic", gamma=gamma,
                      seed=settings["seed"], learning_rate=settings["learning_rate"],
                      total_timesteps=settings["total_timesteps"], episode_hours=settings["episode_hours"],
                      checkpoint_interval=settings["checkpoint_interval"])
        if config["total_timesteps"] % config["n_steps"] or config["checkpoint_interval"] % config["n_steps"]:
            raise ValueError("Diagnostic checkpoints must follow completed PPO updates.")
        configs[name] = config
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs = dict(split_root=split_root, assets=assets, rating_origin=rating_origin,
                  simulation_config=simulation_config, costs_config=costs_config)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(base["torch_threads"])
    output.mkdir(parents=True)
    report = {"schema_version": "1.0", "stage": "gamma_diagnostic", "status": "running",
              "protocol": settings, "source": source_snapshot(output), "runs": [], "validations": [],
              "research_winner": None, "multi_seed_study_executed": False}
    write_json(output / "diagnostic.json", report)
    try:
        report["preflight"] = paired_preflight(configs, inputs)
        write_json(output / "diagnostic.json", report)
        successful = []
        for name, config in configs.items():
            path = output / (name + ".yaml")
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_output = output / name
            record = {"configuration": name, "gamma": config["gamma"], "seed": 0,
                      "output": str(run_output), "status": "running", "config_sha256": sha256(path)}
            report["runs"].append(record)
            write_json(output / "diagnostic.json", report)
            start = time.perf_counter()
            try:
                result = train_ppo(**inputs, output=run_output, training_config=path, diagnostics=True)
                from stable_baselines3 import PPO
                initial = run_output / "checkpoints/step_000000000"
                verify_checkpoint(initial)
                if tensor_hash(PPO.load(initial / "model.zip", device="cpu")) != report["preflight"]["initial_policy_sha256"]:
                    raise ValueError("Actual initial policy differs from paired preflight.")
                events = [json.loads(line) for line in (run_output / "episodes.jsonl").read_text().splitlines()]
                windows = [{k: e[k] for k in ("start_index", "child_seed")} for e in events if e["event"] == "reset"]
                if windows != report["preflight"]["windows"]:
                    raise ValueError("Actual windows differ from paired preflight.")
                record.update(status=result["status"], progress=result["progress"], verification=result["verification"],
                              actual_pairing_verified=True)
                successful.append(name)
            except Exception as exc:
                record.update(status="failed", exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            except BaseException:
                record.update(status="interrupted", traceback=traceback.format_exc())
                raise
            finally:
                record["wall_seconds"] = time.perf_counter() - start
                write_json(output / "diagnostic.json", report)
        for step in settings["validation_checkpoints"]:
            jobs = [{"id": name, "configuration": name, "seed": 0,
                     "path": str(output / name / "checkpoints" / f"step_{step:09d}")} for name in successful]
            validation = validate_policies(**inputs, output=output / f"validation_{step}", checkpoints=jobs,
                                           expected_rows=settings["validation_rows"], initial_soc=settings["initial_soc"],
                                           expected_timesteps=step)
            report["validations"].append({"timesteps": step, "status": validation["status"],
                                          "runtime_seconds": validation["runtime_seconds"]})
            write_json(output / "diagnostic.json", report)
        training_plots(output, report["runs"])
        report["status"] = ("completed" if len(successful) == 2 and
                            all(v["status"] == "completed" for v in report["validations"]) else "completed_with_failures")
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
        report["wall_seconds"] = time.perf_counter() - started
        report["artifacts"] = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*")
                               if p.is_file() and p != output / "diagnostic.json"}
        write_json(output / "diagnostic.json", report)
    return report
