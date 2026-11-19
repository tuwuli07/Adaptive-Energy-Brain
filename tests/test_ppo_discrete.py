"""Tests for the discrete PPO diagnostic experiment configuration and preflight."""

from pathlib import Path
import pytest
import yaml

from src.rl.discrete_experiment import load_discrete_config, _validate_base_settings
from src.rl.training import load_training_config, ROOT


def test_discrete_config_loading_and_validation():
    """Verify load_discrete_config strictly checks protocol settings."""
    cfg = load_discrete_config()
    assert cfg["version"] == 1
    assert cfg["stage"] == "discrete_diagnostic"
    assert cfg["reward_scale"] == 0.0025
    assert cfg["gamma"] == 0.999
    assert cfg["seed"] == 0
    assert cfg["learning_rate"] == 0.0003
    assert cfg["total_timesteps"] == 16384
    assert cfg["episode_hours"] == 168
    assert cfg["checkpoint_interval"] == 2048
    assert cfg["validation_checkpoints"] == [4096, 8192, 16384]
    assert cfg["validation_rows"] == 1314
    assert cfg["initial_soc"] == 0.5


def test_discrete_base_settings_validation():
    """Verify base PPO settings strictly adhere to approved parameters."""
    base = load_training_config(ROOT / "configs/ppo_training.yaml")
    _validate_base_settings(base)

    # Modified settings raise ValueError
    bad_base = dict(base)
    bad_base["learning_rate"] = 0.01  # not in base
    bad_base["clip_range"] = 0.5
    with pytest.raises(ValueError, match="differ"):
        _validate_base_settings(bad_base)
