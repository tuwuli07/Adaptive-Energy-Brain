"""Run the approved Part 6.4 PBRS diagnostic experiment."""

from copy import deepcopy
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

from src.evaluation.ppo_validation import validate_policies
from src.rl.episodes import sha256
from src.rl.training import (ROOT, load_training_config, seed_everything,
                             source_snapshot, train_ppo, verify_checkpoint, write_json)
from src.simulation.config import _UniqueKeyLoader


def load_pbrs_config(path=ROOT / "configs/ppo_pbrs_diagnostic.yaml"):
    settings = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    expected = {
        "version", "stage", "reward_scale", "gamma", "seed", "learning_rate",
        "total_timesteps", "episode_hours", "checkpoint_interval",
        "validation_checkpoints", "validation_rows", "initial_soc",
        "action_space_type", "reward_shaping", "control_source",
    }
    if (not isinstance(settings, dict) or set(settings) != expected
            or settings["version"] != 1 or settings["stage"] != "pbrs_diagnostic"
            or settings["reward_scale"] != 0.0025 or settings["gamma"] != 0.999
            or settings["seed"] != 0 or settings["learning_rate"] != 0.0003
            or settings["total_timesteps"] != 16384 or settings["episode_hours"] != 168
            or settings["checkpoint_interval"] != 2048
            or settings["validation_checkpoints"] != [4096, 8192, 16384]
            or settings["validation_rows"] != 1314 or settings["initial_soc"] != 0.5
            or settings["action_space_type"] != "discrete_3"):
        raise ValueError("Expected the approved PBRS diagnostic protocol.")
    return settings


def _validate_base_settings(base):
    approved = {
        "device": "cpu",
        "torch_threads": 1,
        "policy_layers": [64, 64],
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 4,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
    }
    changed = [name for name, value in approved.items() if base.get(name) != value]
    if changed:
        raise ValueError(f"Base PPO settings differ from the approved protocol: {changed}")


def run_pbrs_diagnostic(
    *,
    split_root,
    assets,
    rating_origin,
    output,
    experiment_config=ROOT / "configs/ppo_pbrs_diagnostic.yaml",
    base_training_config=ROOT / "configs/ppo_training.yaml",
    simulation_config=ROOT / "configs/simulation.yaml",
    costs_config=ROOT / "configs/simulation_costs.yaml",
    control_experiment_root=Path("E:/AdaptiveEnergyBrainStorage/experiments/part6_4_discrete_diagnostic_20261115_treatment"),
):
    started = time.perf_counter()
    protocol = load_pbrs_config(experiment_config)
    base = load_training_config(base_training_config)
    _validate_base_settings(base)

    # Configure treatment run with discrete actions and PBRS
    treatment_config = deepcopy(base)
    treatment_config.update(
        version=2,
        purpose="pbrs_diagnostic",
        action_space_type="discrete_3",
        reward_shaping=protocol["reward_shaping"],
        gamma=protocol["gamma"],
        reward_scale=protocol["reward_scale"],
        seed=protocol["seed"],
        learning_rate=protocol["learning_rate"],
        total_timesteps=protocol["total_timesteps"],
        episode_hours=protocol["episode_hours"],
        checkpoint_interval=protocol["checkpoint_interval"],
    )

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    inputs = {
        "split_root": split_root,
        "assets": assets,
        "rating_origin": rating_origin,
        "simulation_config": simulation_config,
        "costs_config": costs_config,
    }

    # Verify control source exists
    control_root = Path(control_experiment_root) / "treatment"
    if not control_root.exists():
        raise FileNotFoundError(f"Control experiment directory not found: {control_root}")

    for step in protocol["validation_checkpoints"]:
        ckpt = control_root / "checkpoints" / f"step_{step:09d}"
        if not ckpt.exists():
            raise FileNotFoundError(f"Required control checkpoint missing: {ckpt}")
        verify_checkpoint(ckpt)

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(base["torch_threads"])
    output.mkdir(parents=True)

    report = {
        "schema_version": "1.0",
        "stage": "pbrs_diagnostic",
        "status": "running",
        "protocol": protocol,
        "control_source": str(control_root),
        "source": source_snapshot(output),
        "runs": [],
        "validations": [],
        "research_winner": None,
        "multi_seed_study_executed": False,
    }
    write_json(output / "diagnostic.json", report)

    try:
        # Preflight verification: verify control window sequence
        control_events = [
            json.loads(line)
            for line in (control_root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        control_windows = [
            {key: event[key] for key in ("start_index", "child_seed")}
            for event in control_events if event["event"] == "reset"
        ]

        report["preflight"] = {
            "control_checkpoints_verified": True,
            "expected_windows_count": len(control_windows),
            "control_source": str(control_root),
            "treatment_action_space": "Discrete(3)",
            "discrete_mapping": {"0": -150.0, "1": 0.0, "2": 150.0},
            "pbrs_enabled": True,
            "reference_value_usd_per_kwh_dc": protocol["reward_shaping"]["reference_value_usd_per_kwh_dc"],
        }
        write_json(output / "diagnostic.json", report)

        # Run treatment training
        name = "treatment"
        config_path = output / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(treatment_config, sort_keys=False), encoding="utf-8")
        run_output = output / name
        record = {
            "configuration": name,
            "action_space_type": "discrete_3",
            "reward_shaping": protocol["reward_shaping"],
            "gamma": treatment_config["gamma"],
            "reward_scale": treatment_config["reward_scale"],
            "seed": treatment_config["seed"],
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

            # Verify initial checkpoint
            initial = run_output / "checkpoints/step_000000000"
            verify_checkpoint(initial)

            # Verify window sequence matching
            events = [
                json.loads(line)
                for line in (run_output / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            treatment_windows = [
                {key: event[key] for key in ("start_index", "child_seed")}
                for event in events if event["event"] == "reset"
            ]
            if treatment_windows != control_windows:
                raise ValueError("Treatment sampled window draws differ from control window draws.")

            record.update(
                status=result["status"],
                progress=result["progress"],
                verification=result["verification"],
                window_pairing_verified=True,
            )
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

        # Frozen validation at checkpoints 4096, 8192, 16384 comparing control and treatment
        if record.get("status") == "completed":
            for step in protocol["validation_checkpoints"]:
                jobs = [
                    {
                        "id": "control_discrete",
                        "configuration": "control_discrete",
                        "seed": protocol["seed"],
                        "path": str(control_root / "checkpoints" / f"step_{step:09d}"),
                    },
                    {
                        "id": "treatment_pbrs",
                        "configuration": "treatment_pbrs",
                        "seed": protocol["seed"],
                        "path": str(run_output / "checkpoints" / f"step_{step:09d}"),
                    },
                ]
                val_output = output / f"validation_{step}"
                validation = validate_policies(
                    **inputs,
                    output=val_output,
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
                        "summary_file": str(val_output / "validation.json"),
                    }
                )
                write_json(output / "diagnostic.json", report)

        report["status"] = (
            "completed"
            if record.get("status") == "completed"
            and len(report["validations"]) == len(protocol["validation_checkpoints"])
            and all(v["status"] == "completed" for v in report["validations"])
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
