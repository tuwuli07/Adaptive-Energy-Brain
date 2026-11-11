"""Structural and compatibility checks for the Part 5.9 Colab notebook."""

import ast
import json
from pathlib import Path
import re

import pytest
import yaml

from scripts.train_forecaster import _runtime_hardware, _source_revision
from src.forecasting.training import ForecastTrainingError


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "train_gnn_transformer_colab.ipynb"


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_colab_notebook_has_valid_json_and_required_sections():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    expected_sections = [
        "1. Introduction and prerequisites",
        "2. GPU runtime verification",
        "3. GitHub repository setup",
        "4. Dependency installation",
        "5. Google Drive mounting",
        "6. Dataset and preprocessing artifact setup",
        "7. Configuration and dataset verification",
        "8. GPU smoke test",
        "9. Full training",
        "10. Training history visualization",
        "11. Checkpoint persistence",
        "12. Checkpoint reload verification",
        "13. Results and next steps",
    ]
    positions = [markdown.index(section) for section in expected_sections]
    assert positions == sorted(positions)


def test_all_executable_notebook_cells_are_python_syntax():
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        ast.parse(source, filename=f"{NOTEBOOK_PATH.name}:cell-{index}")


def test_notebook_reuses_cli_and_train_validation_only_data_contract():
    notebook = _notebook()
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "scripts/train_forecaster.py" in code
    assert "load_forecasting_train_validation_dataloaders" in code
    assert "load_forecasting_dataloaders(" not in code
    assert "masked_mse_loss" in code
    assert "test_partition_used" in code
    assert 'SPLIT_ROOT / "test"' in code
    assert 'startswith("test/")' in code
    assert 'path.replace("\\\\", "/")' in code
    assert "test directories plus preprocessing" in code
    assert "test_manifest_entries" in code
    assert "BUS01" in code
    assert "--device" in code and '"cuda"' in code
    assert "global_X" in code and "not passed to this model" in code


def test_notebook_defaults_match_checked_in_model_training_and_graph_configs():
    model_config = yaml.safe_load(
        (ROOT / "configs" / "forecasting_model.yaml").read_text(encoding="utf-8")
    )
    training_config = yaml.safe_load(
        (ROOT / "configs" / "forecasting_training.yaml").read_text(encoding="utf-8")
    )
    graph_config = yaml.safe_load(
        (ROOT / "configs" / "forecasting_data.yaml").read_text(encoding="utf-8")
    )
    assert model_config["data"]["lookback_hours"] == 24
    assert model_config["data"]["horizon_hours"] == 24
    assert model_config["targets"]["nodes"] == [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "PV01",
        "W01",
    ]
    assert model_config["graph"]["node_order"] == [
        node["id"] for node in graph_config["graph"]["nodes"]
    ]
    assert model_config["targets"]["unit"] == "kW"
    assert training_config["training"]["device"] == "auto"
    assert training_config["training"]["optimizer"] == "adamw"


def test_notebook_uses_configurable_private_auth_and_persistent_drive_paths():
    notebook = _notebook()
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'REPOSITORY_URL = ""' in code
    assert "REPOSITORY_REF" in code and "COMMIT_SHA" in code
    assert 'userdata.get("GITHUB_TOKEN")' in code
    assert "DRIVE_DATA_ROOT" in code
    assert "PREPROCESSING_PATH" in code
    assert "GRAPH_CONFIG_PATH" in code
    assert "TRAINING_OUTPUT_ROOT" in code
    assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}", code)


def test_runtime_metadata_records_cuda_state_and_device_information():
    hardware = _runtime_hardware("cpu")
    assert set(hardware) == {
        "cuda_available",
        "cuda_runtime",
        "device_count",
        "device_name",
    }
    assert isinstance(hardware["cuda_available"], bool)
    assert isinstance(hardware["device_count"], int)
    assert hardware["device_name"] is None


def test_source_revision_requires_a_full_commit_hash(monkeypatch):
    monkeypatch.delenv("AEB_GIT_COMMIT_SHA", raising=False)
    assert _source_revision() is None
    monkeypatch.setenv("AEB_GIT_COMMIT_SHA", "A" * 40)
    assert _source_revision() == "a" * 40
    monkeypatch.setenv("AEB_GIT_COMMIT_SHA", "b" * 64)
    assert _source_revision() == "b" * 64
    monkeypatch.setenv("AEB_GIT_COMMIT_SHA", "not-a-full-hash")
    with pytest.raises(ForecastTrainingError, match="full 40- or 64-character"):
        _source_revision()


def test_metadata_schema_accepts_optional_colab_provenance_fields():
    schema = json.loads(
        (ROOT / "configs" / "experiment_metadata.schema.json").read_text(
            encoding="utf-8"
        )
    )
    software = schema["properties"]["software"]
    assert software["additionalProperties"] is False
    assert software["properties"]["source_revision"]["pattern"]
    hardware = software["properties"]["hardware"]
    assert set(hardware["required"]) == {
        "cuda_available",
        "cuda_runtime",
        "device_count",
        "device_name",
    }
