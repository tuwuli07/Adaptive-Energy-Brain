import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.graph_builder import build_graph
from src.forecasting.baselines import (
    historical_average_forecast,
    persistence_forecast,
    seasonal_persistence_forecast,
)
from src.forecasting.dataloader import validate_forecasting_graph
from src.forecasting.evaluation import (
    EXPECTED_GRAPH_NODES,
    EXPECTED_TARGETS,
    EvaluationConfig,
    ForecastEvaluationError,
    _nonoverlapping_examples,
    _validate_full_training_history,
    _validate_training_metadata,
    calculate_forecast_metrics,
    inverse_scale_node_history,
    masked_forecast_metrics,
    run_forecast_evaluation,
    save_evaluation_artifacts,
)
from src.forecasting.models.gnn_transformer import GNNTransformerForecaster
from src.forecasting.training import graph_topology_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def _mask(sample_count=2):
    mask = np.ones((sample_count, len(EXPECTED_GRAPH_NODES), 1), dtype=bool)
    mask[:, -1, :] = False
    return mask


def test_masked_metrics_compute_physical_mae_rmse_and_mse():
    actual = np.zeros((2, 3, 8, 1), dtype=np.float32)
    predicted = np.full_like(actual, 2.0)
    predicted[:, :, -1, :] = 99999

    result = masked_forecast_metrics(predicted, actual, _mask())

    assert result == {
        "valid_observation_count": 2 * 3 * 7,
        "mae_kw": 2.0,
        "rmse_kw": 2.0,
        "mse_kw2": 4.0,
    }


def test_invalid_masked_placeholders_are_excluded_but_valid_nan_is_rejected():
    actual = np.zeros((1, 1, 8, 1), dtype=np.float32)
    predicted = np.zeros_like(actual)
    predicted[:, :, -1, :] = np.nan
    assert masked_forecast_metrics(predicted, actual, _mask(1))["mae_kw"] == 0

    predicted[:, :, 0, :] = np.nan
    with pytest.raises(ForecastEvaluationError, match="must be finite"):
        masked_forecast_metrics(predicted, actual, _mask(1))


def test_all_invalid_metrics_report_zero_count_explicitly():
    values = np.zeros((1, 2, 8, 1), dtype=np.float32)
    mask = np.zeros((1, 8, 1), dtype=bool)

    result = masked_forecast_metrics(values, values, mask)

    assert result == {
        "valid_observation_count": 0,
        "mae_kw": None,
        "rmse_kw": None,
        "mse_kw2": None,
    }


def test_per_target_and_horizon_metrics_preserve_order_and_exclude_bus():
    actual = np.zeros((2, 2, 8, 1), dtype=np.float32)
    prediction = np.zeros_like(actual)
    for step in range(2):
        prediction[:, step, :7, 0] = step + 1
    prediction[:, :, -1, :] = 100000
    predictions = {
        "gnn_transformer": prediction,
        "persistence": np.full_like(prediction, 4.0),
        "seasonal_persistence": np.zeros_like(prediction),
        "historical_average": np.full_like(prediction, 3.0),
    }
    result = calculate_forecast_metrics(
        predictions,
        actual,
        _mask(2),
        graph_node_order=EXPECTED_GRAPH_NODES,
    )

    assert list(result["per_target"]) == list(EXPECTED_TARGETS)
    assert result["valid_observation_count"] == 2 * 2 * 7
    assert result["per_target"]["B01"]["category"] == "building_electricity_demand"
    assert result["per_target"]["PV01"]["category"] == "renewable_generation"
    assert result["per_target"]["B01"]["methods"]["gnn_transformer"]["valid_observation_count"] == 4
    assert [entry["forecast_hour"] for entry in result["per_forecast_horizon"]] == [1, 2]
    assert [
        entry["methods"]["gnn_transformer"]["mae_kw"]
        for entry in result["per_forecast_horizon"]
    ] == [1.0, 2.0]
    assert result["masked_placeholder_count"] == 2 * 2
    assert result["relative_improvement"]["persistence"]["mae_kw"] > 0
    assert result["relative_improvement"]["seasonal_persistence"]["mae_kw"] is None


def test_metric_masks_require_boolean_and_do_not_hide_shape_errors():
    values = np.zeros((1, 2, 8, 1), dtype=np.float32)
    with pytest.raises(ForecastEvaluationError, match="boolean"):
        masked_forecast_metrics(values, values, np.ones((8, 1)))
    with pytest.raises(ForecastEvaluationError, match="shape"):
        masked_forecast_metrics(values[:, :, :7], values[:, :, :7], np.ones((8, 1), dtype=bool))


def test_baselines_use_only_history_and_share_aligned_output_shapes():
    history = np.arange(2 * 24 * 8, dtype=np.float32).reshape(2, 24, 8, 1)
    methods = {
        "persistence": persistence_forecast(history, 4, time_axis=1),
        "seasonal_persistence": seasonal_persistence_forecast(
            history, 4, time_axis=1, season_period=24
        ),
        "historical_average": historical_average_forecast(history, 4, time_axis=1),
    }
    assert {values.shape for values in methods.values()} == {(2, 4, 8, 1)}
    assert np.array_equal(methods["persistence"][:, 0], history[:, -1])
    assert np.array_equal(methods["seasonal_persistence"][:, 0], history[:, 0])
    assert np.array_equal(methods["historical_average"][:, 0], history.mean(axis=1))


def test_nonoverlapping_example_selection_is_chronological_and_reproducible():
    origins = tuple(
        (datetime(2026, 11, 7, tzinfo=timezone.utc) + timedelta(hours=hour))
        .isoformat()
        .replace("+00:00", "Z")
        for hour in range(48)
    )
    selected = _nonoverlapping_examples(origins, horizon_hours=24, limit=3)
    assert selected == (0, 24)


def test_evaluation_config_rejects_invalid_horizon_and_unknown_fields():
    with pytest.raises(ForecastEvaluationError, match="positive integer"):
        EvaluationConfig(forecast_horizon_hours=0).validate()
    with pytest.raises(ForecastEvaluationError, match="Unknown"):
        EvaluationConfig.from_mapping({"evaluation": {"mystery": 1}})


def test_partial_batch_smoke_checkpoint_is_rejected_before_final_evaluation():
    state = {
        "epoch": 1,
        "training_config": {"max_epochs": 1},
        "history": [
            {
                "epoch": 1,
                "training_valid_targets": 2,
                "validation_valid_targets": 1,
            }
        ],
    }
    manifest = {
        "boundaries": {
            "train": {"hours": 6132},
            "validation": {"hours": 1314},
        }
    }
    config = {"lookback_hours": 24, "horizon_hours": 24, "stride_hours": 1}

    with pytest.raises(ForecastEvaluationError, match="smoke-test checkpoints"):
        _validate_full_training_history(
            state,
            manifest=manifest,
            sequence_config=config,
        )


def test_inverse_scaling_uses_train_fitted_node_scalers_and_bus_placeholder():
    config = yaml.safe_load((ROOT / "configs/forecasting_data.yaml").read_text())
    scalers = {
        "demand": {
            "method": "standard",
            "fitted_on": "train",
            "entities": {
                name: {"building_load_kw": {"mean": 10.0, "scale": 2.0}}
                for name in EXPECTED_TARGETS[:5]
            },
        },
        "solar": {
            "method": "standard",
            "fitted_on": "train",
            "entities": {"PV01": {"available_generation_kw": {"mean": 20.0, "scale": 3.0}}},
        },
        "wind": {
            "method": "standard",
            "fitted_on": "train",
            "entities": {"W01": {"available_generation_kw": {"mean": 30.0, "scale": 4.0}}},
        },
    }
    history = np.ones((1, 2, 8, 1), dtype=np.float32)

    physical = inverse_scale_node_history(
        history,
        sequence_config=config,
        preprocessing={"scalers": scalers},
    )

    assert physical[0, 0, 0, 0] == 12
    assert physical[0, 0, 5, 0] == 23
    assert physical[0, 0, 6, 0] == 34
    assert np.all(physical[:, :, -1, :] == 0)


def test_incompatible_training_metadata_is_rejected():
    state = {"experiment_id": "exp1"}
    metadata = {
        "experiment_id": "exp1",
        "data": {
            "preprocessing_artifacts": {"sha256": "wrong"},
            "dataset_manifest_sha256": "manifest",
            "preprocessing_fit_partition": "train",
            "graph_topology_sha256": "graph",
            "target_nodes": list(EXPECTED_TARGETS),
            "target_unit": "kW",
        },
        "model": {"architecture": {"node_order": list(EXPECTED_GRAPH_NODES)}},
        "training": {
            "selection_partition": "validation",
            "test_partition_used_for_selection": False,
        },
    }
    with pytest.raises(ForecastEvaluationError, match="does not match"):
        _validate_training_metadata(
            metadata,
            expected_model_config={},
            training_state=state,
            graph_fingerprint="graph",
            preprocessing_sha256="preprocessing",
            manifest_sha256="manifest",
            training_partition_sha256="train",
            validation_partition_sha256="validation",
            checkpoint_sha256="checkpoint",
            experiment_id="exp1",
        )


def _tiny_model_config():
    config = yaml.safe_load((ROOT / "configs/forecasting_model.yaml").read_text())
    config = copy.deepcopy(config)
    config["model"]["gnn_encoder"].update(hidden_dim=8, num_layers=1, dropout=0.0)
    config["model"]["transformer_encoder"].update(
        d_model=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_seq_len=24,
    )
    config["model"]["forecast_head"]["dropout"] = 0.0
    return config


def _synthetic_pipeline_fixture(tmp_path, monkeypatch):
    sequence_config = yaml.safe_load(
        (ROOT / "configs/forecasting_data.yaml").read_text()
    )
    model_config = _tiny_model_config()
    graph_data = build_graph(sequence_config)
    graph = validate_forecasting_graph(
        graph_data,
        node_count=8,
        expected_node_order=EXPECTED_GRAPH_NODES,
    )
    test_bounds = {
        "start_utc": "2026-11-07T06:00:00Z",
        "end_exclusive_utc": "2027-01-01T00:00:00Z",
        "hours": 1314,
    }
    partition_bounds = {
        "train": {
            "start_utc": "2026-01-01T00:00:00Z",
            "end_exclusive_utc": "2026-09-13T12:00:00Z",
            "hours": 6132,
        },
        "validation": {
            "start_utc": "2026-09-13T12:00:00Z",
            "end_exclusive_utc": "2026-11-07T06:00:00Z",
            "hours": 1314,
        },
        "test": test_bounds,
    }
    sample_count, lookback, horizon = 3, 24, 24
    starts = np.arange(sample_count)
    base = np.datetime64("2026-11-07T06:00:00", "ns")
    input_times = np.stack(
        [
            base + (start + np.arange(lookback)) * np.timedelta64(1, "h")
            for start in starts
        ]
    )
    target_times = np.stack(
        [
            base + (start + lookback + np.arange(horizon)) * np.timedelta64(1, "h")
            for start in starts
        ]
    )
    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(sample_count, lookback, 8, 1)).astype(np.float32)
    inputs[:, :, -1] = 0
    targets = np.abs(rng.normal(size=(sample_count, horizon, 8, 1))).astype(np.float32)
    targets[:, :, -1] = 0
    mask = np.ones((8, 1), dtype=bool)
    mask[-1] = False
    targets_data = {
        "arrays": {
            "X": inputs,
            "global_X": np.zeros((sample_count, lookback, 3), dtype=np.float32),
            "y": targets,
            "input_timestamps": input_times,
            "target_timestamps": target_times,
            "node_input_mask": mask,
            "target_mask": mask,
        },
        "graph": graph_data,
        "metadata": {
            "partition": "test",
            "boundaries": test_bounds,
            "config": sequence_config,
            "node_order": list(EXPECTED_GRAPH_NODES),
            "target_unit": "kW",
            "source_metadata": {
                "synthetic_demand": {
                    "origin": "simulated",
                    "synthetic": True,
                    "dataset_label": "SYNTHETIC_TEST_FIXTURE",
                    "split": {
                        "name": "test",
                        "fitted_on": "train",
                        "boundaries": test_bounds,
                    },
                }
            },
        },
    }
    calls = []

    def load_test(_root, partition, *, verify_partitions):
        assert Path(_root).name == "splits"
        calls.append((partition, tuple(verify_partitions)))
        return [{"source_id": "synthetic_fixture"}]

    def prepare_test(sources, config):
        assert sources[0]["source_id"] == "synthetic_fixture"
        assert config["lookback_hours"] == 24
        return targets_data

    monkeypatch.setattr("src.forecasting.evaluation.load_partition_sources", load_test)
    monkeypatch.setattr(
        "src.forecasting.evaluation.prepare_sequences",
        prepare_test,
    )
    root = tmp_path / "splits"
    root.mkdir()
    manifest = {
        "boundaries": partition_bounds,
        "checksums": {
            "train/synthetic.csv": "1" * 64,
            "validation/synthetic.csv": "2" * 64,
            "test/synthetic.csv": "a" * 64,
        },
    }
    manifest_path = root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    preprocessing_path = root / "preprocessing.json"
    scaler_entities = {
        entity: {"building_load_kw": {"mean": 10.0, "scale": 2.0}}
        for entity in EXPECTED_TARGETS[:5]
    }
    preprocessing = {
        "version": 1,
        "fitted_on": "train",
        "boundaries": partition_bounds,
        "scalers": {
            "demand": {
                "method": "standard",
                "fitted_on": "train",
                "entities": scaler_entities,
            },
            "solar": {
                "method": "standard",
                "fitted_on": "train",
                "entities": {
                    "PV01": {
                        "available_generation_kw": {"mean": 0.0, "scale": 1.0}
                    }
                },
            },
            "wind": {
                "method": "standard",
                "fitted_on": "train",
                "entities": {
                    "W01": {
                        "available_generation_kw": {"mean": 0.0, "scale": 1.0}
                    }
                },
            },
        },
    }
    preprocessing_path.write_text(json.dumps(preprocessing), encoding="utf-8")

    model = GNNTransformerForecaster.from_config(model_config)
    experiment_id = "synthetic-evaluation-fixture"
    checkpoint_path = tmp_path / "checkpoints" / experiment_id / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    state = {
        "experiment_id": experiment_id,
        "seed": 42,
        "epoch": 1,
        "training_config": {"max_epochs": 1},
        "history": [
            {
                "epoch": 1,
                "training_valid_targets": 6085 * 24 * 7,
                "validation_valid_targets": 1267 * 24 * 7,
            }
        ],
        "selection_partition": "validation",
        "test_partition_used_for_selection": False,
        "target_unit": "kW",
        "graph_topology_sha256": graph_topology_fingerprint(graph),
        "target_order": list(EXPECTED_TARGETS),
        "preprocessing_artifacts": {
            "preprocessing": "data/processed/splits_test/preprocessing.json",
            "split_manifest": "data/processed/splits_test/split_manifest.json",
        },
    }
    model.save_checkpoint(checkpoint_path, training_state=state)
    training_metadata = {
        "experiment_id": experiment_id,
        "status": "completed",
        "data": {
            "preprocessing_artifacts": {
                "path": "data/processed/splits_test/preprocessing.json",
                "sha256": hashlib.sha256(preprocessing_path.read_bytes()).hexdigest(),
            },
            "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "training_partition_sha256": hashlib.sha256(
                json.dumps(
                    {"train/synthetic.csv": "1" * 64},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "validation_partition_sha256": hashlib.sha256(
                json.dumps(
                    {"validation/synthetic.csv": "2" * 64},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "preprocessing_fit_partition": "train",
            "graph_topology_sha256": graph_topology_fingerprint(graph),
            "target_nodes": list(EXPECTED_TARGETS),
            "target_unit": "kW",
        },
        "model": {
            "architecture": {"node_order": list(EXPECTED_GRAPH_NODES)},
            "configuration": model_config,
        },
        "training": {
            "best_epoch": 1,
            "selection_partition": "validation",
            "test_partition_used_for_selection": False,
        },
        "artifacts": {
            "best_checkpoint_sha256": hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
        },
    }
    metadata_path = tmp_path / "experiment.json"
    metadata_path.write_text(json.dumps(training_metadata), encoding="utf-8")
    return {
        "calls": calls,
        "root": root,
        "preprocessing": preprocessing_path,
        "sequence_config": sequence_config,
        "model_config": model_config,
        "checkpoint": checkpoint_path,
        "metadata": metadata_path,
    }


def test_cpu_evaluation_loads_test_only_and_handles_incomplete_final_batch(
    tmp_path, monkeypatch
):
    fixture = _synthetic_pipeline_fixture(tmp_path, monkeypatch)

    output = run_forecast_evaluation(
        checkpoint_path=fixture["checkpoint"],
        split_root=fixture["root"],
        preprocessing_path=fixture["preprocessing"],
        model_config=fixture["model_config"],
        sequence_config=fixture["sequence_config"],
        evaluation_config=EvaluationConfig(
            batch_size=2,
            device="cpu",
            forecast_horizon_hours=24,
            seasonal_period_hours=24,
        ),
        training_metadata_path=fixture["metadata"],
    )

    assert fixture["calls"] == [("test", ("test",))]
    assert output.targets.shape == (3, 24, 8, 1)
    assert output.predictions["gnn_transformer"].shape == output.targets.shape
    assert output.results["evaluation"]["sample_count"] == 3
    assert output.results["evaluation"]["metrics"]["valid_observation_count"] == 3 * 24 * 7
    assert output.results["dataset"]["synthetic"] is True
    assert output.results["checkpoint"]["training_metadata_hash_verified"] is True
    assert set(output.predictions) == {
        "gnn_transformer",
        "persistence",
        "historical_average",
        "seasonal_persistence",
    }


def test_evaluation_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        run_forecast_evaluation(
            checkpoint_path=tmp_path / "missing.pt",
            split_root=tmp_path,
            preprocessing_path=tmp_path / "preprocessing.json",
            model_config={},
            sequence_config={},
            evaluation_config=EvaluationConfig(device="cpu"),
        )


def test_evaluation_rejects_mismatched_training_metadata(tmp_path, monkeypatch):
    fixture = _synthetic_pipeline_fixture(tmp_path, monkeypatch)
    metadata = json.loads(fixture["metadata"].read_text())
    metadata["data"]["preprocessing_artifacts"]["sha256"] = "0" * 64
    fixture["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ForecastEvaluationError, match="does not match"):
        run_forecast_evaluation(
            checkpoint_path=fixture["checkpoint"],
            split_root=fixture["root"],
            preprocessing_path=fixture["preprocessing"],
            model_config=fixture["model_config"],
            sequence_config=fixture["sequence_config"],
            evaluation_config=EvaluationConfig(device="cpu"),
            training_metadata_path=fixture["metadata"],
        )


def test_structured_artifacts_and_output_overwrite_protection(tmp_path, monkeypatch):
    fixture = _synthetic_pipeline_fixture(tmp_path, monkeypatch)
    output = run_forecast_evaluation(
        checkpoint_path=fixture["checkpoint"],
        split_root=fixture["root"],
        preprocessing_path=fixture["preprocessing"],
        model_config=fixture["model_config"],
        sequence_config=fixture["sequence_config"],
        evaluation_config=EvaluationConfig(batch_size=2, device="cpu"),
        training_metadata_path=fixture["metadata"],
    )
    destination = tmp_path / "new-evaluation"

    saved = save_evaluation_artifacts(
        output, destination, representative_window_count=1
    )

    payload = json.loads((saved / "evaluation.json").read_text())
    arrays = np.load(saved / "predictions.npz", allow_pickle=False)
    assert payload["dataset"]["synthetic"] is True
    assert payload["artifacts"]["predictions_npz"] == "predictions.npz"
    assert arrays["predictions_gnn_transformer"].shape == (3, 24, 8, 1)
    assert (saved / "plots" / "actual_vs_predicted.png").is_file()
    assert (saved / "plots" / "baseline_comparison.png").is_file()
    assert (saved / "plots" / "per_target_errors.png").is_file()
    assert (saved / "plots" / "forecast_horizon_errors.png").is_file()
    assert len(list((saved / "plots" / "representative_forecasts").glob("*.png"))) == 7
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_evaluation_artifacts(
            output, destination, representative_window_count=1
        )
