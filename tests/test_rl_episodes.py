"""Train partition isolation and reproducible synthetic window sampling."""

import builtins
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("gymnasium")

from scripts.run_simulation import _components
from src.data.splitting import save_splits, split_datasets
from src.data.synthetic import generate_synthetic, load_scenario
from src.rl.episodes import TrainingEpisodeEnvironment, load_training_data
from src.simulation.config import load_simulation_config
from src.simulation.metrics import load_cost_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def training_inputs(tmp_path):
    sources = yaml.safe_load((ROOT / "configs/data_sources.yaml").read_text(encoding="utf-8"))
    for source in sources["sources"]:
        source.update(status="pending", local_path=None, download_url=None)
    source_path = tmp_path / "sources.yaml"
    source_path.write_text(yaml.safe_dump(sources), encoding="utf-8")
    scenario = load_scenario()
    scenario["hours"] = 48
    generated = generate_synthetic(scenario, sources_path=source_path, processed_root=tmp_path / "empty")
    root = save_splits(split_datasets(generated["sources"]), "rl_training_fixture", processed_root=tmp_path)
    return {"split_root": root, "assets": ROOT / "configs/synthetic_scenario.yaml",
            "rating_origin": "hypothetical_example"}


def episode_environment(inputs, hours=8):
    data, _ = load_training_data(inputs["split_root"])
    renewable, demand = _components(data, inputs["assets"], inputs["rating_origin"])
    return TrainingEpisodeEnvironment(data, renewable, demand, load_simulation_config(),
                                      load_cost_config(), hours=hours)


def forbid_heldout_reads(monkeypatch, root):
    """Intercept both Path.open and normal open, including pandas CSV reads."""
    original_io, original_builtin = io.open, builtins.open
    accesses = []
    def guard(original):
        def open_file(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                path = Path(file).resolve()
                if root.resolve() in path.parents:
                    relative = path.relative_to(root.resolve())
                    assert relative.parts[0] not in {"validation", "test"}, relative
                    accesses.append(relative.as_posix())
            return original(file, *args, **kwargs)
        return open_file
    monkeypatch.setattr(io, "open", guard(original_io))
    monkeypatch.setattr(builtins, "open", guard(original_builtin))
    return accesses


def test_loader_opens_only_train_and_shared_metadata(training_inputs, monkeypatch):
    root = training_inputs["split_root"]
    accesses = forbid_heldout_reads(monkeypatch, root)
    data, provenance = load_training_data(root)
    assert provenance["partition"] == "train" and len(data) == 33
    assert any(name.startswith("train/") for name in accesses)
    assert "preprocessing.json" in accesses
    assert all(name.startswith("train/") for name in provenance["training_files"])


@pytest.mark.parametrize("corruption", ["bytes", "missing_hash", "partition_label"])
def test_training_provenance_corruption_rejected(training_inputs, corruption):
    root = training_inputs["split_root"]
    manifest = json.loads((root / "split_manifest.json").read_text())
    path = next((root / "train").glob("*.metadata.json"))
    relative = path.relative_to(root).as_posix()
    if corruption == "bytes":
        path.write_text("{}")
    elif corruption == "missing_hash":
        manifest["checksums"].pop(relative)
    else:
        payload = json.loads(path.read_text())
        payload["metadata"]["split"]["name"] = "test"
        path.write_text(json.dumps(payload))
        manifest["checksums"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "split_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_training_data(root)


def test_seeded_sampling_boundaries_restore_and_lifecycle(training_inputs):
    env = episode_environment(training_inputs)
    def sample(seed):
        env.reset(seed=seed)
        records = [deepcopy(env.current_window)]
        for _ in range(20):
            env.reset()
            records.append(deepcopy(env.current_window))
        return records
    first = sample(7)
    assert first == sample(7)
    assert first != sample(8)
    bounds = env._data.metadata["training_partition"]
    for record in first:
        assert 0 <= record["start_index"] <= len(env._data) - env.hours - 1
        assert record["history_utc"] >= bounds["start_utc"]
        assert record["end_exclusive_utc"] <= bounds["end_exclusive_utc"]
    state = env.sampler_state()
    env.reset()
    expected = env.current_window
    env.restore_sampler(state)
    env.reset(seed=99)  # framework seed must not erase restored sampler state
    assert env.current_window == expected
    for index in range(8):
        result = env.step(np.array([0], dtype=np.float32))
        assert result[3] == (index == 7)
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0])
    env.reset()
    env.close()


def test_last_possible_window_and_short_partition(training_inputs):
    env = episode_environment(training_inputs, hours=32)
    env.reset(seed=0)
    assert env.current_window["start_index"] == 0
    with pytest.raises(ValueError, match="episode_hours"):
        episode_environment(training_inputs, hours=33)


def test_manifest_cannot_redirect_train_verification_to_heldout(training_inputs, monkeypatch):
    root = training_inputs["split_root"]
    manifest = json.loads((root / "split_manifest.json").read_text())
    heldout = next((root / "test").glob("*.csv"))
    manifest["checksums"]["train/../test/" + heldout.name] = "invalid"
    (root / "split_manifest.json").write_text(json.dumps(manifest))
    forbid_heldout_reads(monkeypatch, root)
    with pytest.raises(ValueError, match="escapes"):
        load_training_data(root)
