"""Portable storage-path and Windows workspace configuration tests."""

import json
from pathlib import Path

from scripts.train_forecaster import _configured_path


ROOT = Path(__file__).resolve().parents[1]


def test_configured_path_preserves_relative_default_behavior(monkeypatch, tmp_path):
    monkeypatch.delenv("AEB_SPLIT_ROOT", raising=False)
    result = _configured_path(
        tmp_path,
        explicit=None,
        environment_name="AEB_SPLIT_ROOT",
        default="data/processed/splits",
    )
    assert result == (tmp_path / "data" / "processed" / "splits").resolve()


def test_configured_path_allows_external_environment_override(monkeypatch, tmp_path):
    external = tmp_path / "E drive data" / "splits"
    monkeypatch.setenv("AEB_SPLIT_ROOT", str(external))
    result = _configured_path(
        tmp_path / "repository",
        explicit=None,
        environment_name="AEB_SPLIT_ROOT",
        default="data/processed/splits",
    )
    assert result == external.resolve()


def test_experiment_and_checkpoint_paths_can_be_placed_on_external_storage(
    monkeypatch, tmp_path
):
    experiments = tmp_path / "experiments"
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setenv("AEB_EXPERIMENTS_ROOT", str(experiments))
    monkeypatch.setenv("AEB_CHECKPOINTS_ROOT", str(checkpoints))
    experiment_root = _configured_path(
        tmp_path / "repository",
        explicit=None,
        environment_name="AEB_EXPERIMENTS_ROOT",
        default="results/forecasting",
    )
    checkpoint_root = _configured_path(
        tmp_path / "repository",
        explicit=None,
        environment_name="AEB_CHECKPOINTS_ROOT",
        default=experiment_root,
    )
    experiment_id = "part5_9-run"
    assert experiment_root / experiment_id == experiments / experiment_id
    assert checkpoint_root / experiment_id == checkpoints / experiment_id


def test_explicit_cli_path_takes_precedence_over_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AEB_EXPERIMENTS_ROOT", str(tmp_path / "environment"))
    explicit = tmp_path / "explicit"
    result = _configured_path(
        tmp_path / "repository",
        explicit=explicit,
        environment_name="AEB_EXPERIMENTS_ROOT",
        default="results/forecasting",
    )
    assert result == explicit.resolve()


def test_windows_setup_script_is_scoped_and_contains_no_cleanup_commands():
    script = (ROOT / "scripts" / "setup_windows_storage.ps1").read_text(
        encoding="utf-8"
    )
    assert "SetEnvironmentVariable" not in script
    assert "setx " not in script.lower()
    assert "Remove-Item" not in script
    assert '"TMP" = Join-Path $resolvedRoot "temp"' in script
    assert '"TEMP" = Join-Path $resolvedRoot "temp"' in script
    assert "PYTHONPYCACHEPREFIX" in script
    assert "PIP_CACHE_DIR" in script
    assert "AEB_EXPERIMENTS_ROOT" in script
    assert "AEB_CHECKPOINTS_ROOT" in script


def test_local_storage_templates_have_no_machine_specific_absolute_paths():
    env_template = (ROOT / ".env.example").read_text(encoding="utf-8")
    vscode = json.loads(
        (ROOT / ".vscode" / "settings.json").read_text(encoding="utf-8")
    )
    assert "E:\\" not in env_template and "F:\\" not in env_template
    assert vscode["python.defaultInterpreterPath"] == (
        "${workspaceFolder}\\.venv\\Scripts\\python.exe"
    )
    assert vscode["python.envFile"] == "${workspaceFolder}\\.env.storage"
    assert "setup_windows_storage.ps1" in " ".join(
        vscode["terminal.integrated.profiles.windows"][
            "PowerShell (Adaptive Energy Brain storage)"
        ]["args"]
    )


def test_colab_notebook_clears_windows_storage_overrides_for_training():
    notebook = json.loads(
        (ROOT / "notebooks" / "train_gnn_transformer_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "training_env.pop(local_storage_key, None)" in code
    assert '"AEB_SPLIT_ROOT", "AEB_EXPERIMENTS_ROOT", "AEB_CHECKPOINTS_ROOT"' in code
