"""Train-only physical data loading and reproducible split-local episode sampling."""

from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.data.sequences import load_partition_sources
from src.rl.environment import MicrogridGymEnvironment
from src.simulation.data_loader import SimulationData, prepare_simulation_data


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_verified_partition(split_root, partition):
    """Open one allowed partition plus shared metadata, checking complete coverage.

    The existing convenience simulation loader verifies every partition by
    default. Select train explicitly at the underlying existing loader instead.
    """
    if partition not in {"train", "validation"}:
        raise ValueError("Only train or validation partition access is supported here.")
    label = "Training" if partition == "train" else "Validation"
    root = Path(split_root).resolve()
    manifest_path = root / "split_manifest.json"
    if manifest_path.resolve() != manifest_path:
        raise ValueError("Shared manifest must remain at the split root.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_dir = root / partition
    if train_dir.resolve() != train_dir or (root / "preprocessing.json").resolve() != root / "preprocessing.json":
        raise ValueError(f"{label} and shared preprocessing paths must stay in their declared locations.")
    for relative in manifest["checksums"]:
        parts = Path(relative).parts
        if parts and parts[0] == partition and train_dir not in (root / relative).resolve().parents:
            raise ValueError(f"{label} manifest entry escapes the selected partition directory.")
    files = sorted(train_dir.glob("*.metadata.json")) + sorted(train_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"{label} partition has no source files.")
    recorded = {}
    for path in files:
        resolved = path.resolve()
        if train_dir.resolve() not in resolved.parents:
            raise ValueError(f"{label} source escapes the selected partition directory.")
        relative = path.relative_to(root).as_posix()
        if relative not in manifest["checksums"]:
            raise ValueError(f"Missing {partition} checksum: {relative}")
        recorded[relative] = manifest["checksums"][relative]
    sources = load_partition_sources(root, partition, verify_partitions=(partition,))
    bounds = manifest["boundaries"][partition]
    for source in sources:
        split = source["metadata"].get("split", {})
        if (split.get("name") != partition or split.get("boundaries") != bounds
                or split.get("fitted_on") != "train"):
            raise ValueError("Source must identify the verified chronological selected partition.")
    data = prepare_simulation_data(sources)
    interval = data.metadata["interval"]
    if any(interval[key] != bounds[key] for key in ("start_utc", "end_exclusive_utc", "hours")):
        raise ValueError(f"{label} data coverage disagrees with split boundaries.")
    metadata = data.metadata
    metadata["training_partition" if partition == "train" else "validation_partition"] = bounds
    data = SimulationData(tuple(data), metadata)
    provenance = {
        "partition": partition, "split_root": str(root), "boundaries": bounds,
        "split_manifest_sha256": sha256(root / "split_manifest.json"),
        "preprocessing_sha256": sha256(root / "preprocessing.json"),
        ("training_files" if partition == "train" else "validation_files"): recorded, "sources": metadata["sources"],
        "rows": len(data), "preprocessing": "fixed adapter unit conversions",
    }
    return data, provenance


def load_training_data(split_root):
    """Preserve the Part 6.3 train-only entry point and provenance format."""
    return load_verified_partition(split_root, "train")


class TrainingEpisodeEnvironment(gym.Env):
    """Sample overlapping windows uniformly with replacement at each reset.

    Each child is the existing fixed-window adapter. Policy observations never
    receive the sampler, replay rows, or window ledger. Restored sampler state
    takes priority over the first framework reset seed during continuation.
    """

    metadata = {"render_modes": []}

    def __init__(self, data, renewables, demand, config, costs, *, hours=168,
                 sampler_seed=0, ledger=None, action_adapter=None, reward_adapter=None):
        if type(hours) is not int or hours < 1 or len(data) <= hours:
            raise ValueError("Training partition must contain at least episode_hours + 1 rows.")
        if "training_partition" not in data.metadata:
            raise ValueError("Training partition provenance is required.")
        if config.unserved_load.handling != "error":
            raise ValueError("Shortage-aware training requires a separately defined reliability objective.")
        self._data = data
        self._renewables, self._demand = tuple(renewables), demand
        self._config, self._costs = config, costs
        self.hours = hours
        self.rng = np.random.default_rng(sampler_seed)
        self.episode_count = 0
        self._restored = False
        self._action_adapter = action_adapter
        self._reward_adapter = reward_adapter
        self._child = self._make_child(0)
        self.observation_space = self._child.observation_space
        self.action_space = self._child.action_space
        self._active = False
        self._ledger = ledger
        self.current_window = None

    def _make_child(self, start):
        rows = tuple(self._data[start:start + self.hours + 1])
        metadata = self._data.metadata
        end = (datetime.fromisoformat(rows[-1].timestamp_utc.replace("Z", "+00:00"))
               + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        bounds = metadata["training_partition"]
        if rows[0].timestamp_utc < bounds["start_utc"] or end > bounds["end_exclusive_utc"]:
            raise ValueError("Episode crosses training partition boundaries.")
        metadata["interval"] = {"start_utc": rows[0].timestamp_utc,
                                "end_exclusive_utc": end, "hours": len(rows)}
        child = MicrogridGymEnvironment(SimulationData(rows, metadata), self._renewables,
                                        self._demand, self._config, self._costs)
        if self._action_adapter is not None:
            child = self._action_adapter(child)
        if self._reward_adapter is not None:
            child = self._reward_adapter(child)
        return child

    def sampler_state(self):
        return {"rng": deepcopy(self.rng.bit_generator.state), "episodes": self.episode_count}

    def restore_sampler(self, state):
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self.episode_count = state["episodes"]
        self._restored = True

    def reset(self, *, seed=None, options=None):
        if options is not None and (not isinstance(options, dict) or options):
            raise ValueError("Only empty training reset options are supported.")
        super().reset(seed=seed)
        if seed is not None and not self._restored:
            self.rng = np.random.default_rng(seed)
            self.episode_count = 0
        self._restored = False
        start = int(self.rng.integers(0, len(self._data) - self.hours))
        child = self._make_child(start)
        child_seed = int(self.rng.integers(0, 2**31))
        observation, info = child.reset(seed=child_seed)
        self._child.close()
        self._child = child
        self.episode_count += 1
        self.current_window = {
            "episode": self.episode_count, "start_index": start, "child_seed": child_seed,
            "history_utc": info["observed_interval_utc"],
            "scored_start_utc": info["decision_timestamp_utc"],
            "end_exclusive_utc": info["episode"]["environment"]["end_exclusive_utc"],
            "scored_hours": self.hours, "initial_soc": self._config.battery.initial_soc,
        }
        self._active = True
        if self._ledger is not None:
            self._ledger({"event": "reset", **self.current_window})
        info["training_window"] = deepcopy(self.current_window)
        return observation, info

    def step(self, action):
        if not self._active:
            raise RuntimeError("Training episode requires reset.")
        result = self._child.step(action)
        observation, reward, terminated, truncated, info = result
        if terminated or truncated:
            self._active = False
            if self._ledger is not None:
                self._ledger({"event": "completed", **self.current_window,
                              "metrics": info["cumulative_metrics"]})
        return observation, reward, terminated, truncated, info

    def close(self):
        self._child.close()
        self._active = False
