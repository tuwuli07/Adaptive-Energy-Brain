"""CPU PPO infrastructure, completed-update checkpoints, and fresh-episode recovery."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

from scripts.run_simulation import _components
from src.rl.episodes import TrainingEpisodeEnvironment, load_training_data, sha256
from src.rl.observations import FEATURE_NAMES, SCHEMA
from src.simulation.config import _UniqueKeyLoader, load_simulation_config
from src.simulation.metrics import load_cost_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/ppo_training.yaml"
METADATA_SCHEMA = ROOT / "configs/rl_experiment_metadata.schema.json"


class RewardScaleVecEnv(VecEnvWrapper):
    """Scale PPO rewards without changing the wrapped simulator or its info."""

    def __init__(self, venv, scale):
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or scale <= 0
                or scale > float(np.finfo(np.float32).max)):
            raise ValueError("reward_scale must be a finite positive float32 value.")
        self.scale = float(scale)
        super().__init__(venv)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        scaled_rewards = (
            np.asarray(rewards, dtype=np.float64) * self.scale
        ).astype(np.float32)
        if not np.isfinite(scaled_rewards).all():
            raise ValueError("Scaled PPO rewards must remain finite.")
        return observations, scaled_rewards, dones, infos


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Publish one JSON file atomically in its destination filesystem."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_metadata(metadata):
    """Validate the small versioned schema without another runtime dependency."""
    schema = read_json(METADATA_SCHEMA)
    if set(metadata) != set(schema["required"]):
        raise ValueError("Experiment metadata fields disagree with schema.")
    for name, spec in schema["properties"].items():
        value = metadata[name]
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"Invalid metadata {name}.")
        kind = spec.get("type")
        if kind == "object" and not isinstance(value, dict):
            raise ValueError(f"Metadata {name} must be an object.")
        if kind == "string" and not isinstance(value, str):
            raise ValueError(f"Metadata {name} must be a string.")
        for required in spec.get("required", []):
            if required not in value:
                raise ValueError(f"Missing metadata {name}.{required}.")
    json.dumps(metadata, allow_nan=False)


def load_training_config(path=DEFAULT_CONFIG):
    values = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    expected = {
        "version", "purpose", "seed", "episode_hours", "total_timesteps", "device",
        "torch_threads", "policy_layers", "learning_rate", "n_steps", "batch_size",
        "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
        "max_grad_norm", "normalize_advantage",
    }
    if isinstance(values, dict) and "reward_scale" in values:
        expected.add("reward_scale")
    if isinstance(values, dict) and values.get("version") == 2:
        expected.add("checkpoint_interval")
    if isinstance(values, dict) and "action_space_type" in values:
        expected.add("action_space_type")
    if isinstance(values, dict) and "reward_shaping" in values:
        expected.add("reward_shaping")
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("PPO configuration requires exactly the documented fields.")
    purposes = {
        1: {"infrastructure_smoke"},
        2: {"validation_pilot", "validation_experiment", "gamma_diagnostic", "reward_scale_diagnostic", "discounting_diagnostic", "discrete_diagnostic", "pbrs_diagnostic"},
    }
    if (type(values["version"]) is not int or values["version"] not in purposes
            or values["purpose"] not in purposes[values["version"]] or values["device"] != "cpu"
            or values["normalize_advantage"] is not True):
        raise ValueError("Unsupported PPO infrastructure contract.")
    for name in ("seed", "episode_hours", "total_timesteps", "torch_threads",
                 "n_steps", "batch_size", "n_epochs"):
        minimum = 0 if name == "seed" else 1
        if type(values[name]) is not int or values[name] < minimum:
            raise ValueError(f"Invalid integer PPO setting {name}.")
    layers = values["policy_layers"]
    if not isinstance(layers, list) or not layers or any(type(n) is not int or n < 1 for n in layers):
        raise ValueError("policy_layers must contain positive integer widths.")
    if (values["n_steps"] < 2 or values["batch_size"] < 2
            or values["n_steps"] % values["batch_size"]
            or values["total_timesteps"] % values["n_steps"]):
        raise ValueError("Batch must divide rollout and timestep budget must contain whole rollouts.")
    interval = values.get("checkpoint_interval", values["n_steps"])
    if type(interval) is not int or interval < 1 or interval % values["n_steps"]:
        raise ValueError("Checkpoint interval must contain whole rollouts.")
    for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
                 "vf_coef", "max_grad_norm"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Nonfinite/nonnumeric PPO setting {name}.")
        if value < 0 or (name in {"learning_rate", "clip_range", "max_grad_norm"} and value == 0):
            raise ValueError(f"Invalid PPO setting {name}.")
    if values["gamma"] > 1 or values["gae_lambda"] > 1 or values["clip_range"] > 1:
        raise ValueError("gamma, gae_lambda and clip_range must be at most one.")
    if "reward_scale" in values:
        scale = values["reward_scale"]
        if (isinstance(scale, bool) or not isinstance(scale, (float, int))
                or not math.isfinite(scale) or scale <= 0
                or scale > float(np.finfo(np.float32).max)):
            raise ValueError("reward_scale must be a finite positive float32 value.")
    return values


def source_snapshot(output):
    """Archive relevant code/config bytes, including uncommitted source files."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout
    revision = git("rev-parse", "HEAD").decode().strip()
    dirty = bool(git("status", "--porcelain"))
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    selected = sorted({name for name in names if name and (
        name.startswith(("src/", "scripts/", "configs/"))
        or ("/" not in name and name.startswith("requirements")))})
    hashes = {}
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in selected:
            path = ROOT / name
            if path.is_file():
                content = path.read_bytes()
                archive.writestr(name, content)
                hashes[name] = hashlib.sha256(content).hexdigest()
    write_json(output / "source_manifest.json", hashes)
    return {"git_revision": revision, "dirty_worktree": dirty,
            "source_manifest_sha256": sha256(output / "source_manifest.json"),
            "source_snapshot_sha256": sha256(output / "source_snapshot.zip")}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def rng_state(environment):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "sampler": environment.sampler_state()}


def restore_rng(state, environment):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    environment.restore_sampler(state["sampler"])


def optimizer_steps(model):
    steps = [int(value["step"].item()) for value in model.policy.optimizer.state.values()
             if "step" in value]
    return max(steps, default=0)


def publish_checkpoint(output, model, environment, completed_updates, contract):
    """Commit a bundle by atomically publishing its manifest, then latest pointer."""
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    final = checkpoints / f"step_{model.num_timesteps:09d}"
    if final.exists():
        raise FileExistsError(f"Checkpoint already exists: {final}")
    # Readers require manifest.json. Until that commit marker exists this
    # exclusively created directory is incomplete and cannot be resumed.
    stage = final
    stage.mkdir()
    model.save(stage / "model.zip")
    saved_rng = rng_state(environment)
    torch.save(saved_rng, stage / "state.pt")
    manifest = {
        "version": 1, "timesteps": model.num_timesteps,
        "completed_rollout_updates": completed_updates,
        "optimizer_steps": optimizer_steps(model), "contract": contract,
        "files": {name: sha256(stage / name) for name in ("model.zip", "state.pt")},
    }
    # Catch serialization failures before exposing a recoverable checkpoint.
    try:
        loaded = PPO.load(stage / "model.zip", device="cpu")
    finally:
        # SB3 loading reseeds global generators while constructing its policy.
        random.setstate(saved_rng["python"])
        np.random.set_state(saved_rng["numpy"])
        torch.set_rng_state(saved_rng["torch"])
    if optimizer_steps(loaded) != manifest["optimizer_steps"]:
        raise ValueError("Checkpoint optimizer state did not reload.")
    write_json(stage / "manifest.json", manifest)
    write_json(output / "latest_checkpoint.json", {
        "path": str(final.relative_to(output)), "manifest_sha256": sha256(final / "manifest.json")})
    return final


def verify_checkpoint(path):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if manifest.get("version") != 1 or set(manifest.get("files", {})) != {"model.zip", "state.pt"}:
        raise ValueError("Unsupported checkpoint bundle.")
    for name, digest in manifest["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    return manifest


def new_ppo(settings, environment):
    """Construct the shared seeded CPU PPO policy without collecting transitions."""
    return PPO("MlpPolicy", environment, device="cpu", seed=settings["seed"],
                        policy_kwargs={"net_arch": {"pi": settings["policy_layers"],
                                                    "vf": settings["policy_layers"]},
                                       "activation_fn": torch.nn.Tanh},
                        **{name: settings[name] for name in (
                            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                            "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
                            "normalize_advantage")})


def train_ppo(*, split_root, assets, rating_origin, output, training_config=DEFAULT_CONFIG,
              simulation_config=ROOT / "configs/simulation.yaml",
              costs_config=ROOT / "configs/simulation_costs.yaml", resume=None, diagnostics=False):
    """Execute whole PPO updates and continue checkpoints into fresh episodes.

    Only load locally produced, trusted checkpoint bundles. SB3 and RNG sidecars
    contain Python/PyTorch serialized state. SHA-256 checks establish integrity.
    """
    started = time.perf_counter()
    learning_seconds = 0.0
    output = Path(output).resolve()
    settings = load_training_config(training_config)
    if output.exists():
        raise FileExistsError(f"Run output already exists: {output}")
    data, provenance = load_training_data(split_root)
    config = load_simulation_config(simulation_config)
    costs = load_cost_config(costs_config)
    renewables, demand = _components(data, assets, rating_origin)
    from src.rl.environment import DiscreteActionAdapter
    action_adapter = DiscreteActionAdapter if settings.get("action_space_type") == "discrete_3" else None
    reward_adapter = None
    shaping_cfg = settings.get("reward_shaping")
    if shaping_cfg and shaping_cfg.get("enabled"):
        from src.rl.shaping import PotentialBasedRewardShapingWrapper
        gamma_shaping = float(shaping_cfg.get("gamma", settings.get("gamma", 0.999)))
        v_ref = float(shaping_cfg.get("reference_value_usd_per_kwh_dc", (0.12 + 0.02) / 0.95))
        capacity = float(shaping_cfg.get("capacity_kwh", 500.0))
        reward_adapter = lambda env: PotentialBasedRewardShapingWrapper(
            env, gamma=gamma_shaping, reference_value_usd_per_kwh_dc=v_ref, capacity_kwh=capacity
        )
    environment = TrainingEpisodeEnvironment(data, renewables, demand, config, costs,
                                              hours=settings["episode_hours"], sampler_seed=settings["seed"],
                                              action_adapter=action_adapter,
                                              reward_adapter=reward_adapter)
    output.mkdir(parents=True)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    model = None
    vector = None
    metadata = None
    telemetry = None
    try:
        torch.set_num_threads(settings["torch_threads"])
        seed_everything(settings["seed"])
        source = source_snapshot(output)
        write_json(output / "resolved_config.json", {
            "training": settings, "simulation": asdict(config), "costs": asdict(costs),
            "asset_config": yaml.safe_load(Path(assets).read_text(encoding="utf-8")),
            "adapter": yaml.safe_load((ROOT / "configs/rl_environment.yaml").read_text(encoding="utf-8")),
            "asset_config_sha256": sha256(assets), "rating_origin": rating_origin,
        })
        contract_settings = {key: value for key, value in settings.items() if key != "total_timesteps"}
        contract = {
            "settings": contract_settings, "simulation": asdict(config), "costs": asdict(costs),
            "source_manifest_sha256": source["source_manifest_sha256"],
            "asset_config_sha256": sha256(assets), "rating_origin": rating_origin,
            "data": provenance, "observation_schema": SCHEMA,
            "software": {name: importlib.metadata.version(name) for name in
                         ("stable-baselines3", "gymnasium", "numpy", "torch", "pandas", "matplotlib")},
        }
        metadata = {
            "schema_version": "1.0" if settings["version"] == 1 else "1.1",
            "phase": "6.3" if settings["version"] == 1 else "6.4", "purpose": settings["purpose"],
            "experiment_id": output.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running", "data": provenance, "configuration": settings,
            "source": source, "software": {**contract["software"], "python": sys.version,
                                           "device": "cpu", "torch_threads": settings["torch_threads"]},
            "contracts": {"observation_schema": SCHEMA, "feature_order": FEATURE_NAMES,
                          "power_reference_kw": 1000, "price_reference_per_kwh": 1,
                          "observation_bounds": "SOC/mask [0,1], calendar [-1,1], power [0,inf], signed prices by config",
                          "action": "Box(-1,1,(1,),float32), signed static direction-specific AC kW ratings",
                          "reward": "negative committed net_operating_cost; scaled only at the PPO vector interface",
                          "terminal_soc": "free_report", "shortage_handling": "error",
                          "seeds": {"python_numpy_torch_ppo": settings["seed"],
                                    "sampler": settings["seed"], "child": "recorded sampler draws"}},
            "progress": {"requested_additional_timesteps": settings["total_timesteps"],
                         "collected_timesteps": 0, "checkpoint_timesteps": 0,
                         "completed_rollout_updates": 0, "optimizer_steps": 0},
            "lineage": {}, "artifacts": {}, "verification": {}, "diagnostics": {},
        }
        validate_metadata(metadata)
        write_json(output / "experiment.json", metadata)
        def ledger(record):
            with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        environment._ledger = ledger
        vector = RewardScaleVecEnv(
            DummyVecEnv([lambda: Monitor(environment, str(output / "monitor.csv"))]),
            settings.get("reward_scale", 1.0),
        )
        completed_updates = 0
        if resume is None:
            model = new_ppo(settings, vector)
        else:
            manifest = verify_checkpoint(resume)
            if manifest["contract"] != json.loads(json.dumps(contract)):
                raise ValueError("Continuation contract differs from the saved checkpoint.")
            model = PPO.load(Path(resume) / "model.zip", env=vector, device="cpu", force_reset=True)
            if (model.num_timesteps != manifest["timesteps"]
                    or optimizer_steps(model) != manifest["optimizer_steps"]):
                raise ValueError("Saved model counters disagree with checkpoint manifest.")
            # Trusted locally generated state, including Python and NumPy RNGs.
            state = torch.load(Path(resume) / "state.pt", map_location="cpu", weights_only=False)
            restore_rng(state, environment)
            completed_updates = manifest["completed_rollout_updates"]
            metadata["lineage"] = {
                "parent_checkpoint": str(Path(resume).resolve()),
                "parent_manifest_sha256": sha256(Path(resume) / "manifest.json"),
                "fresh_episode_continuation": True, "restored_optimizer_steps": optimizer_steps(model),
            }
        model.set_logger(configure(str(output / "logs"), ["csv"]))
        if diagnostics:
            from src.rl.diagnostics import PPODiagnostics
            telemetry = PPODiagnostics(
                output / "telemetry", reward_scale=settings.get("reward_scale", 1.0)
            )
            telemetry.attach(model)
            metadata["diagnostics"]["instrumentation"] = {
                "schema_version": "1.0", "method": "callback_and_removable_torch_hooks",
                "gradients": "actual weighted-loss parameter gradients before/after global clipping",
                "advantages": "raw rollout GAE, before minibatch normalization",
                "terminal_values": "actual SB3 timeout predict_values forward output"}
        initial_parameters = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        start_timesteps = model.num_timesteps
        initial_optimizer_steps = optimizer_steps(model)
        latest = publish_checkpoint(output, model, environment, completed_updates, contract)
        metadata["progress"]["checkpoint_timesteps"] = model.num_timesteps
        metadata["progress"]["completed_rollout_updates"] = completed_updates
        metadata["progress"]["optimizer_steps"] = initial_optimizer_steps
        while model.num_timesteps - start_timesteps < settings["total_timesteps"]:
            # One public learn call returns only after this rollout's update.
            learning_started = time.perf_counter()
            try:
                model.learn(total_timesteps=settings["n_steps"], reset_num_timesteps=False, log_interval=None, callback=telemetry)
            finally:
                learning_seconds += time.perf_counter() - learning_started
            if not all(torch.isfinite(value).all() for value in model.policy.state_dict().values()):
                raise ValueError("Nonfinite policy parameters after update.")
            losses = {key: float(value) for key, value in model.logger.name_to_value.items()
                      if key.startswith("train/") and isinstance(value, (int, float, np.number))}
            for name, value in tuple(losses.items()):
                if not math.isfinite(value):
                    if name == "train/explained_variance":
                        losses[name] = None  # Undefined when target variance is zero.
                    else:
                        raise ValueError(f"Nonfinite PPO training metric: {name}")
            model.logger.record("time/total_timesteps", model.num_timesteps)
            model.logger.dump(step=model.num_timesteps)
            completed_updates += 1
            elapsed_steps = model.num_timesteps - start_timesteps
            interval = settings.get("checkpoint_interval", settings["n_steps"])
            if elapsed_steps % interval == 0 or elapsed_steps == settings["total_timesteps"]:
                latest = publish_checkpoint(output, model, environment, completed_updates, contract)
            metadata["progress"].update({
                "collected_timesteps": model.num_timesteps - start_timesteps,
                "checkpoint_timesteps": read_json(latest / "manifest.json")["timesteps"],
                "completed_rollout_updates": completed_updates,
                "optimizer_steps": optimizer_steps(model), "last_losses": losses,
            })
            write_json(output / "experiment.json", metadata)
        changed = any(not torch.equal(initial_parameters[name], value)
                      for name, value in model.policy.state_dict().items())
        if not changed or optimizer_steps(model) <= initial_optimizer_steps:
            raise ValueError("PPO update did not change parameters and advance optimizer.")
        # Fixed train window for serialization verification, without sampling.
        probe = environment._make_child(0)
        try:
            observation, _ = probe.reset(seed=settings["seed"])
            expected = model.predict(observation, deterministic=True)[0]
            loaded = PPO.load(latest / "model.zip", device="cpu")
            actual = loaded.predict(observation, deterministic=True)[0]
            np.testing.assert_array_equal(expected, actual)
            probe.step(actual)
        finally:
            probe.close()
        metadata["verification"] = {"parameters_changed": changed,
                                    "finite_parameters_and_losses": True,
                                    "checkpoint_reload": True, "deterministic_actions_equal": True,
                                    "inference_step": True}
        metadata["status"] = "completed"
    except BaseException as exc:
        if metadata is None:
            write_json(output / "failure.json", {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "exception": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
            })
        if metadata is not None:
            metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            metadata["diagnostics"] = {"exception": type(exc).__name__, "message": str(exc),
                                       "traceback": traceback.format_exc(),
                                       "committed_child_steps": environment._child._steps}
            if model is not None:
                metadata["progress"]["collected_timesteps"] = model.num_timesteps - locals().get("start_timesteps", model.num_timesteps)
        raise
    finally:
        if telemetry is not None:
            telemetry.close()
        if vector is not None:
            vector.close()
        else:
            environment.close()
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)
        if metadata is not None:
            metadata["progress"]["runtime_seconds"] = time.perf_counter() - started
            metadata["progress"]["learning_seconds"] = learning_seconds
            metadata["artifacts"] = {
                path.relative_to(output).as_posix(): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "experiment.json" and ".pending_" not in str(path)
            }
            validate_metadata(metadata)
            write_json(output / "experiment.json", metadata)
    return deepcopy(metadata)
