"""Run the approved paired PPO reward-scale diagnostic."""

from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import torch
import yaml

from src.evaluation.ppo_validation import validate_policies
from src.rl.diagnostic_experiment import paired_preflight, tensor_hash
from src.rl.episodes import sha256
from src.rl.experiments import training_plots
from src.rl.training import (ROOT, load_training_config, new_ppo, seed_everything,
                             source_snapshot, train_ppo, verify_checkpoint, write_json)
from src.simulation.config import _UniqueKeyLoader


def load_reward_scale_config(path=ROOT / "configs/ppo_reward_scale_diagnostic.yaml"):
    settings = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    expected = {
        "version", "stage", "gamma", "reward_scales", "seed", "learning_rate",
        "total_timesteps", "episode_hours", "checkpoint_interval",
        "validation_checkpoints", "validation_rows", "initial_soc",
    }
    if (not isinstance(settings, dict) or set(settings) != expected
            or settings["version"] != 1 or settings["stage"] != "reward_scale_diagnostic"
            or settings["gamma"] != .99 or settings["reward_scales"] != {"control": 1.0, "treatment": .0025}
            or settings["seed"] != 0 or settings["learning_rate"] != .0003
            or settings["total_timesteps"] != 8192 or settings["episode_hours"] != 168
            or settings["checkpoint_interval"] != 2048
            or settings["validation_checkpoints"] != [4096, 8192]
            or settings["validation_rows"] != 1314 or settings["initial_soc"] != .5):
        raise ValueError("Expected the approved paired PPO reward-scale protocol.")
    return settings


def _validate_base_settings(base):
    approved = {
        "device": "cpu",
        "torch_threads": 1,
        "policy_layers": [64, 64],
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": .99,
        "gae_lambda": .95,
        "clip_range": .2,
        "ent_coef": 0.0,
        "vf_coef": .5,
        "max_grad_norm": .5,
        "normalize_advantage": True,
    }
    changed = [name for name, value in approved.items() if base.get(name) != value]
    if changed:
        raise ValueError(f"Base PPO settings differ from the approved protocol: {changed}")


def run_reward_scale_diagnostic(
    *,
    split_root,
    assets,
    rating_origin,
    output,
    experiment_config=ROOT / "configs/ppo_reward_scale_diagnostic.yaml",
    base_training_config=ROOT / "configs/ppo_training.yaml",
    simulation_config=ROOT / "configs/simulation.yaml",
    costs_config=ROOT / "configs/simulation_costs.yaml",
):
    started = time.perf_counter()
    protocol = load_reward_scale_config(experiment_config)
    base = load_training_config(base_training_config)
    _validate_base_settings(base)
    configs = {}
    for name, scale in protocol["reward_scales"].items():
        config = deepcopy(base)
        config.update(
            version=2,
            purpose="reward_scale_diagnostic",
            gamma=protocol["gamma"],
            seed=protocol["seed"],
            learning_rate=protocol["learning_rate"],
            total_timesteps=protocol["total_timesteps"],
            episode_hours=protocol["episode_hours"],
            checkpoint_interval=protocol["checkpoint_interval"],
            reward_scale=scale,
        )
        if (config["total_timesteps"] % config["n_steps"]
                or config["checkpoint_interval"] % config["n_steps"]):
            raise ValueError("Reward-scale checkpoints must follow completed PPO updates.")
        configs[name] = config

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs = {
        "split_root": split_root,
        "assets": assets,
        "rating_origin": rating_origin,
        "simulation_config": simulation_config,
        "costs_config": costs_config,
    }
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(base["torch_threads"])
    output.mkdir(parents=True)
    report = {
        "schema_version": "1.0",
        "stage": "reward_scale_diagnostic",
        "status": "running",
        "protocol": protocol,
        "source": source_snapshot(output),
        "runs": [],
        "validations": [],
        "research_winner": None,
        "multi_seed_study_executed": False,
    }
    write_json(output / "diagnostic.json", report)
    try:
        report["preflight"] = paired_preflight(
            configs, inputs, differing_setting="reward_scale"
        )
        write_json(output / "diagnostic.json", report)
        successful = []
        for name, config in configs.items():
            config_path = output / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_output = output / name
            record = {
                "configuration": name,
                "gamma": config["gamma"],
                "reward_scale": config["reward_scale"],
                "seed": config["seed"],
                "output": str(run_output),
                "status": "running",
                "config_sha256": sha256(config_path),
            }
            report["runs"].append(record)
            write_json(output / "diagnostic.json", report)
            run_started = time.perf_counter()
            try:
                result = train_ppo(
                    **inputs,
                    output=run_output,
                    training_config=config_path,
                    diagnostics=True,
                )
                from stable_baselines3 import PPO

                initial = run_output / "checkpoints/step_000000000"
                verify_checkpoint(initial)
                initial_model = PPO.load(initial / "model.zip", device="cpu")
                if tensor_hash(initial_model) != report["preflight"]["initial_policy_sha256"]:
                    raise ValueError("Actual initial policy differs from paired preflight.")
                events = [
                    json.loads(line)
                    for line in (run_output / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                windows = [
                    {key: event[key] for key in ("start_index", "child_seed")}
                    for event in events if event["event"] == "reset"
                ]
                if windows != report["preflight"]["windows"]:
                    raise ValueError("Actual windows differ from paired preflight.")
                record.update(
                    status=result["status"],
                    progress=result["progress"],
                    verification=result["verification"],
                    actual_pairing_verified=True,
                )
                if result["status"] == "completed":
                    successful.append(name)
            except Exception as exc:
                record.update(
                    status="failed",
                    exception=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )
            except BaseException:
                record.update(status="interrupted", traceback=traceback.format_exc())
                raise
            finally:
                record["wall_seconds"] = time.perf_counter() - run_started
                write_json(output / "diagnostic.json", report)

        if successful:
            for step in protocol["validation_checkpoints"]:
                jobs = [
                    {
                        "id": name,
                        "configuration": name,
                        "seed": protocol["seed"],
                        "path": str(
                            output / name / "checkpoints" / f"step_{step:09d}"
                        ),
                    }
                    for name in successful
                ]
                validation = validate_policies(
                    **inputs,
                    output=output / f"validation_{step}",
                    checkpoints=jobs,
                    expected_rows=protocol["validation_rows"],
                    initial_soc=protocol["initial_soc"],
                    expected_timesteps=step,
                )
                report["validations"].append(
                    {
                        "timesteps": step,
                        "status": validation["status"],
                        "runtime_seconds": validation["runtime_seconds"],
                    }
                )
                write_json(output / "diagnostic.json", report)

        training_plots(output, report["runs"])
        report["status"] = (
            "completed"
            if len(successful) == 2
            and len(report["validations"]) == len(protocol["validation_checkpoints"])
            and all(validation["status"] == "completed" for validation in report["validations"])
            else "completed_with_failures"
        )
    except BaseException as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            exception=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
        report["wall_seconds"] = time.perf_counter() - started
        report["artifacts"] = {
            path.relative_to(output).as_posix(): sha256(path)
            for path in output.rglob("*")
            if path.is_file() and path != output / "diagnostic.json"
        }
        write_json(output / "diagnostic.json", report)
    return report
