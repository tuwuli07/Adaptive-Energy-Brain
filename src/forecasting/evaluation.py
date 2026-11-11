"""Leakage-aware final evaluation for trained node-level forecasters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.sequences import load_partition_sources, prepare_sequences
from src.forecasting.baselines import (
    ForecastingBaselineError,
    historical_average_forecast,
    persistence_forecast,
    seasonal_persistence_forecast,
)
from src.forecasting.dataloader import (
    ForecastGraph,
    collate_forecasting_batch,
    inverse_scale_with_training_scaler,
    move_batch_to_device,
    validate_forecasting_graph,
)
from src.forecasting.dataset import ForecastDatasetError, ForecastingDataset
from src.forecasting.metrics import ForecastingMetricError, forecast_metrics
from src.forecasting.models.gnn_transformer import (
    GNNTransformerError,
    GNNTransformerForecaster,
)
from src.forecasting.training import graph_topology_fingerprint, resolve_device


EXPECTED_GRAPH_NODES = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "PV01",
    "W01",
    "BUS01",
)
EXPECTED_TARGETS = ("B01", "B02", "B03", "B04", "B05", "PV01", "W01")
METHODS = (
    "gnn_transformer",
    "persistence",
    "seasonal_persistence",
    "historical_average",
)


class ForecastEvaluationError(ValueError):
    """Invalid checkpoint, evaluation data, or forecasting metric contract."""


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 64
    device: str = "auto"
    forecast_horizon_hours: int = 24
    seasonal_period_hours: int = 24
    representative_window_count: int = 3

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "EvaluationConfig":
        if not isinstance(config, Mapping):
            raise ForecastEvaluationError("Evaluation configuration must be a mapping.")
        values = config.get("evaluation", config)
        if not isinstance(values, Mapping):
            raise ForecastEvaluationError("evaluation settings must be a mapping.")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ForecastEvaluationError(
                f"Unknown evaluation configuration fields: {sorted(unknown)}."
            )
        try:
            result = cls(**dict(values))
        except TypeError as exc:
            raise ForecastEvaluationError(f"Invalid evaluation configuration: {exc}") from exc
        result.validate()
        return result

    def validate(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("forecast_horizon_hours", self.forecast_horizon_hours),
            ("seasonal_period_hours", self.seasonal_period_hours),
            ("representative_window_count", self.representative_window_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ForecastEvaluationError(f"{name} must be a positive integer.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ForecastEvaluationError("device must be 'auto', 'cpu', or 'cuda'.")


@dataclass(frozen=True)
class ForecastEvaluationOutput:
    """Evaluation metrics and aligned arrays used for durable artifacts."""

    results: Mapping[str, Any]
    predictions: Mapping[str, np.ndarray]
    targets: np.ndarray
    mask: np.ndarray
    issue_times_utc: tuple[str, ...]
    target_timestamps_utc: tuple[tuple[str, ...], ...]


def _expanded_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) != 4:
        raise ForecastEvaluationError(
            "Forecast arrays must use [sample, horizon, node, target] dimensions."
        )
    sample_count, _, node_count, channels = shape
    if mask.dtype != np.bool_:
        raise ForecastEvaluationError("target mask must contain boolean values.")
    if mask.shape == (node_count, 1):
        candidate = mask.reshape(1, 1, node_count, 1)
    elif mask.shape == (sample_count, node_count, 1):
        candidate = mask[:, None, :, :]
    elif mask.shape == (sample_count, node_count, channels):
        candidate = mask[:, None, :, :]
    elif mask.shape == shape:
        candidate = mask
    else:
        raise ForecastEvaluationError(
            f"Target mask shape {mask.shape} cannot be applied to {shape}."
        )
    try:
        return np.broadcast_to(candidate, shape)
    except ValueError as exc:
        raise ForecastEvaluationError(
            f"Target mask shape {mask.shape} cannot be broadcast to {shape}."
        ) from exc


def masked_forecast_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute physical-unit MAE, RMSE and MSE over valid target elements."""
    predicted = np.asarray(predictions)
    actual = np.asarray(targets)
    if predicted.shape != actual.shape or predicted.ndim != 4:
        raise ForecastEvaluationError(
            "predictions and targets must share [sample, horizon, node, target] shape."
        )
    if (
        not np.issubdtype(predicted.dtype, np.number)
        or np.issubdtype(predicted.dtype, np.complexfloating)
        or not np.issubdtype(actual.dtype, np.number)
        or np.issubdtype(actual.dtype, np.complexfloating)
    ):
        raise ForecastEvaluationError("Predictions and targets must be real numeric arrays.")
    valid = _expanded_mask(np.asarray(mask), predicted.shape)
    count = int(valid.sum())
    if count == 0:
        return {
            "valid_observation_count": 0,
            "mae_kw": None,
            "rmse_kw": None,
            "mse_kw2": None,
        }
    if not np.isfinite(predicted[valid]).all() or not np.isfinite(actual[valid]).all():
        raise ForecastEvaluationError(
            "Valid predictions and targets must be finite; invalid values are not silently dropped."
        )
    try:
        base = forecast_metrics(predicted, actual, mask=valid)
    except ForecastingMetricError as exc:
        raise ForecastEvaluationError(f"Cannot calculate masked metrics: {exc}") from exc
    errors = predicted[valid].astype(np.float64) - actual[valid].astype(np.float64)
    return {
        "valid_observation_count": count,
        "mae_kw": base["mae"],
        "rmse_kw": base["rmse"],
        "mse_kw2": float(np.mean(np.square(errors))),
    }


def _relative_improvement(
    model_value: float | None, baseline_value: float | None
) -> float | None:
    if (
        model_value is None
        or baseline_value is None
        or not math.isfinite(model_value)
        or not math.isfinite(baseline_value)
        or baseline_value <= 0
    ):
        return None
    return (baseline_value - model_value) / baseline_value


def calculate_forecast_metrics(
    predictions: Mapping[str, np.ndarray],
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    graph_node_order: Sequence[str],
    target_order: Sequence[str] = EXPECTED_TARGETS,
) -> dict[str, Any]:
    """Aggregate overall, per-target and per-horizon metrics without batch bias."""
    actual = np.asarray(targets)
    if actual.ndim != 4:
        raise ForecastEvaluationError("targets must have [sample, horizon, node, target] shape.")
    nodes = tuple(graph_node_order)
    target_names = tuple(target_order)
    if nodes != EXPECTED_GRAPH_NODES or target_names != EXPECTED_TARGETS:
        raise ForecastEvaluationError("Evaluation requires the audited node and target ordering.")
    if actual.shape[2] != len(nodes) or actual.shape[3] != 1:
        raise ForecastEvaluationError("Target dimensions do not match the eight-node contract.")
    if not predictions or "gnn_transformer" not in predictions:
        raise ForecastEvaluationError("GNN-Transformer predictions are required.")
    valid = _expanded_mask(np.asarray(mask), actual.shape)
    target_indices = tuple(nodes.index(name) for name in target_names)
    target_valid = valid[:, :, target_indices, :]
    if not target_valid.any():
        raise ForecastEvaluationError("The evaluation set contains no valid forecasting targets.")
    if np.any(valid[:, :, nodes.index("BUS01"), :]):
        raise ForecastEvaluationError("BUS01 must remain excluded from forecasting metrics.")

    normalized_predictions: dict[str, np.ndarray] = {}
    for method, raw in predictions.items():
        values = np.asarray(raw)
        if values.shape != actual.shape:
            raise ForecastEvaluationError(
                f"{method} prediction shape {values.shape} differs from targets {actual.shape}."
            )
        normalized_predictions[method] = values

    def selected_metrics(
        values: np.ndarray, actual_values: np.ndarray, mask_values: np.ndarray
    ) -> dict[str, float | int | None]:
        return masked_forecast_metrics(values, actual_values, mask_values)

    overall: dict[str, Any] = {}
    for method, values in normalized_predictions.items():
        overall[method] = selected_metrics(
            values[:, :, target_indices, :],
            actual[:, :, target_indices, :],
            target_valid,
        )

    per_target: dict[str, Any] = {}
    for name in target_names:
        index = nodes.index(name)
        node_mask = valid[:, :, index : index + 1, :]
        method_metrics = {
            method: selected_metrics(
                values[:, :, index : index + 1, :],
                actual[:, :, index : index + 1, :],
                node_mask,
            )
            for method, values in normalized_predictions.items()
        }
        baseline_comparisons = {}
        for baseline in ("persistence", "seasonal_persistence", "historical_average"):
            if baseline not in method_metrics:
                continue
            baseline_comparison = {}
            for metric in ("mae_kw", "rmse_kw"):
                baseline_comparison[metric] = _relative_improvement(
                    method_metrics["gnn_transformer"][metric],
                    method_metrics[baseline][metric],
                )
            baseline_comparisons[baseline] = baseline_comparison
        per_target[name] = {
            "category": "building_electricity_demand"
            if name.startswith("B")
            else "renewable_generation",
            "unit": "kW",
            "methods": method_metrics,
            "relative_improvement": baseline_comparisons,
        }

    per_horizon: list[dict[str, Any]] = []
    for step in range(actual.shape[1]):
        step_mask = target_valid[:, step : step + 1, :, :]
        methods = {
            method: selected_metrics(
                values[:, step : step + 1, target_indices, :],
                actual[:, step : step + 1, target_indices, :],
                step_mask,
            )
            for method, values in normalized_predictions.items()
        }
        per_horizon.append(
            {
                "forecast_hour": step + 1,
                "methods": methods,
            }
        )

    relative_improvement = {}
    for baseline in ("persistence", "seasonal_persistence", "historical_average"):
        if baseline not in overall:
            continue
        relative_improvement[baseline] = {
            metric: _relative_improvement(overall["gnn_transformer"][metric], overall[baseline][metric])
            for metric in ("mae_kw", "rmse_kw")
        }
    return {
        "overall": overall,
        "per_target": per_target,
        "per_forecast_horizon": per_horizon,
        "relative_improvement": relative_improvement,
        "valid_observation_count": int(target_valid.sum()),
        "masked_placeholder_count": int(valid.size - valid[:, :, target_indices, :].size),
        "masked_target_observation_count": int(target_valid.size - target_valid.sum()),
    }


def inverse_scale_node_history(
    scaled_history: np.ndarray,
    *,
    sequence_config: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
) -> np.ndarray:
    """Convert scaled histories to physical kW for the causal baselines."""
    history = np.asarray(scaled_history)
    if history.ndim != 4 or history.shape[2] != len(EXPECTED_GRAPH_NODES) or history.shape[3] != 1:
        raise ForecastEvaluationError("Scaled history must be [batch, lookback, 8, 1].")
    if not np.isfinite(history).all():
        raise ForecastEvaluationError("Scaled history contains non-finite observations.")
    graph = sequence_config.get("graph")
    nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
    scalers = preprocessing.get("scalers")
    if not isinstance(nodes, list) or not isinstance(scalers, Mapping):
        raise ForecastEvaluationError("Sequence graph nodes and preprocessing scalers are required.")
    node_config = {item.get("id"): item for item in nodes if isinstance(item, Mapping)}
    if tuple(node_config) != EXPECTED_GRAPH_NODES:
        raise ForecastEvaluationError("Sequence configuration node order is incompatible.")
    output = np.zeros(history.shape, dtype=np.float64)
    for node_index, node_id in enumerate(EXPECTED_GRAPH_NODES):
        node = node_config[node_id]
        if node.get("type") == "bus":
            continue
        entity_id = node.get("entity_id")
        input_column = node.get("input_column")
        if not isinstance(entity_id, str) or not isinstance(input_column, str):
            raise ForecastEvaluationError(f"Node {node_id} has no input entity/column mapping.")
        suffix = "_scaled"
        if not input_column.endswith(suffix):
            raise ForecastEvaluationError(f"Node {node_id} input is not a scaled historical feature.")
        feature = input_column[: -len(suffix)]
        candidates = [
            scaler
            for scaler in scalers.values()
            if isinstance(scaler, Mapping)
            and scaler.get("method") == "standard"
            and scaler.get("fitted_on") == "train"
            and isinstance(scaler.get("entities"), Mapping)
            and entity_id in scaler["entities"]
            and feature in scaler["entities"][entity_id]
        ]
        if len(candidates) != 1:
            raise ForecastEvaluationError(
                f"Expected exactly one train-fitted scaler for {node_id}.{feature}."
            )
        try:
            physical = inverse_scale_with_training_scaler(
                history[:, :, node_index, 0],
                candidates[0],
                entity_id=entity_id,
                feature=feature,
            )
        except (ValueError, KeyError) as exc:
            raise ForecastEvaluationError(
                f"Cannot invert the training scaler for {node_id}: {exc}"
            ) from exc
        output[:, :, node_index, 0] = physical
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ForecastEvaluationError(f"Cannot read artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ForecastEvaluationError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ForecastEvaluationError(f"{label} must contain a JSON object.")
    return value


def _read_checkpoint_payload(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ForecastEvaluationError(f"Cannot read trained checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ForecastEvaluationError("Checkpoint payload must be a mapping.")
    if payload.get("format_version") != GNNTransformerForecaster.CHECKPOINT_FORMAT_VERSION:
        raise ForecastEvaluationError("Unsupported GNN-Transformer checkpoint format.")
    return payload


def _validate_training_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_model_config: Mapping[str, Any],
    training_state: Mapping[str, Any],
    graph_fingerprint: str,
    preprocessing_sha256: str,
    manifest_sha256: str,
    training_partition_sha256: str,
    validation_partition_sha256: str,
    checkpoint_sha256: str,
    experiment_id: str,
) -> None:
    data = metadata.get("data")
    model = metadata.get("model")
    training = metadata.get("training")
    if not all(isinstance(item, Mapping) for item in (data, model, training)):
        raise ForecastEvaluationError("Training metadata lacks data/model/training mappings.")
    preprocessing_ref = data.get("preprocessing_artifacts")
    if (
        not isinstance(preprocessing_ref, Mapping)
        or preprocessing_ref.get("sha256") != preprocessing_sha256
        or data.get("dataset_manifest_sha256") != manifest_sha256
        or data.get("training_partition_sha256") != training_partition_sha256
        or data.get("validation_partition_sha256") != validation_partition_sha256
        or data.get("preprocessing_fit_partition") != "train"
        or data.get("graph_topology_sha256") != graph_fingerprint
        or tuple(data.get("target_nodes", ())) != EXPECTED_TARGETS
        or data.get("target_unit") != "kW"
    ):
        raise ForecastEvaluationError(
            "Training metadata does not match the selected split, preprocessing, graph, or targets."
        )
    architecture = model.get("architecture")
    if not isinstance(architecture, Mapping) or tuple(architecture.get("node_order", ())) != EXPECTED_GRAPH_NODES:
        raise ForecastEvaluationError("Training metadata model graph node order is incompatible.")
    if model.get("configuration") != expected_model_config:
        raise ForecastEvaluationError(
            "Training metadata model configuration differs from the selected model config."
        )
    if (
        metadata.get("experiment_id") != experiment_id
        or metadata.get("status") != "completed"
        or training_state.get("experiment_id") != experiment_id
        or not isinstance(metadata.get("artifacts"), Mapping)
        or metadata["artifacts"].get("best_checkpoint_sha256") != checkpoint_sha256
        or training.get("selection_partition") != "validation"
        or training.get("test_partition_used_for_selection") is not False
        or training.get("best_epoch") != training_state.get("epoch")
    ):
        raise ForecastEvaluationError(
            "Training metadata does not prove validation-only model selection."
        )


def _discover_training_metadata(checkpoint_path: Path, experiment_id: str) -> Path | None:
    candidates = (
        checkpoint_path.parent / "experiment.json",
        checkpoint_path.parent.parent / experiment_id / "experiment.json",
        checkpoint_path.parent.parent / "experiments" / experiment_id / "experiment.json",
        checkpoint_path.parent.parent.parent / "experiments" / experiment_id / "experiment.json",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _validate_preprocessing(
    preprocessing: Mapping[str, Any],
    *,
    dataset: ForecastingDataset,
    preprocessing_path: Path,
    split_root: Path,
) -> tuple[str, str]:
    if preprocessing.get("fitted_on") != "train":
        raise ForecastEvaluationError("Preprocessing must be fitted on the training partition.")
    preprocessing_boundaries = preprocessing.get("boundaries")
    if (
        not isinstance(preprocessing_boundaries, Mapping)
        or preprocessing_boundaries.get("test") != dataset.metadata.get("boundaries")
    ):
        raise ForecastEvaluationError(
            "Preprocessing split boundaries do not match the held-out test partition."
        )
    scalers = preprocessing.get("scalers")
    if not isinstance(scalers, Mapping) or not scalers:
        raise ForecastEvaluationError("Preprocessing artifact must include its train-fitted scalers.")
    for name, scaler in scalers.items():
        if not isinstance(scaler, Mapping) or scaler.get("fitted_on") != "train":
            raise ForecastEvaluationError(f"Scaler {name!r} is not marked as train-fitted.")
    canonical_preprocessing = split_root / "preprocessing.json"
    if not canonical_preprocessing.is_file():
        raise ForecastEvaluationError(f"Split bundle lacks {canonical_preprocessing}.")
    preprocessing_sha = _sha256_file(preprocessing_path)
    if preprocessing_sha != _sha256_file(canonical_preprocessing):
        raise ForecastEvaluationError(
            "Selected preprocessing artifact differs from the one paired with this split bundle."
        )
    manifest_path = split_root / "split_manifest.json"
    manifest = _read_json(manifest_path, "split manifest")
    manifest_boundaries = manifest.get("boundaries")
    if (
        not isinstance(manifest_boundaries, Mapping)
        or manifest_boundaries.get("test") != dataset.metadata.get("boundaries")
    ):
        raise ForecastEvaluationError("Test sequence boundaries do not match the split manifest.")
    return preprocessing_sha, _sha256_file(manifest_path)


def _validate_full_training_history(
    training_state: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    sequence_config: Mapping[str, Any],
) -> None:
    """Reject partial-batch smoke checkpoints before test inference."""
    history = training_state.get("history")
    epoch = training_state.get("epoch")
    training_config = training_state.get("training_config")
    if (
        not isinstance(history, list)
        or not history
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or not isinstance(training_config, Mapping)
        or not isinstance(training_config.get("max_epochs"), int)
        or epoch > training_config["max_epochs"]
        or not isinstance(history[-1], Mapping)
        or history[-1].get("epoch") != epoch
    ):
        raise ForecastEvaluationError(
            "Checkpoint lacks a completed, auditable Part 5.8 epoch history."
        )
    try:
        lookback = int(sequence_config["lookback_hours"])
        horizon = int(sequence_config["horizon_hours"])
        stride = int(sequence_config["stride_hours"])
        boundaries = manifest["boundaries"]
        expected_counts = {}
        for partition in ("train", "validation"):
            hours = int(boundaries[partition]["hours"])
            samples = max(0, (hours - lookback - horizon) // stride + 1)
            expected_counts[partition] = samples * horizon * len(EXPECTED_TARGETS)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ForecastEvaluationError(
            f"Cannot determine full-partition training counts from the manifest: {exc}"
        ) from exc
    for row in history:
        if (
            not isinstance(row, Mapping)
            or row.get("training_valid_targets") != expected_counts["train"]
            or row.get("validation_valid_targets") != expected_counts["validation"]
        ):
            raise ForecastEvaluationError(
                "Checkpoint history indicates partial data or an incompatible split; "
                "smoke-test checkpoints cannot be used for final evaluation."
            )


def _load_test_loader(
    split_root: Path,
    *,
    sequence_config: Mapping[str, Any],
    batch_size: int,
) -> tuple[DataLoader, ForecastGraph, ForecastingDataset]:
    try:
        sources = load_partition_sources(
            split_root, "test", verify_partitions=("test",)
        )
        prepared = prepare_sequences(sources, sequence_config)
        dataset = ForecastingDataset(prepared, expected_partition="test")
        graph = validate_forecasting_graph(
            dataset.graph,
            node_count=len(dataset.node_order),
            expected_node_order=EXPECTED_GRAPH_NODES,
        )
    except (ForecastDatasetError, ValueError, OSError, KeyError, TypeError) as exc:
        raise ForecastEvaluationError(f"Cannot prepare the held-out test split: {exc}") from exc
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_forecasting_batch,
    )
    return loader, graph, dataset


def run_forecast_evaluation(
    *,
    checkpoint_path: str | Path,
    split_root: str | Path,
    preprocessing_path: str | Path,
    model_config: Mapping[str, Any],
    sequence_config: Mapping[str, Any],
    evaluation_config: EvaluationConfig,
    training_metadata_path: str | Path | None = None,
) -> ForecastEvaluationOutput:
    """Load a completed checkpoint and evaluate only its checksum-verified test partition."""
    evaluation_config.validate()
    checkpoint = Path(checkpoint_path).resolve()
    root = Path(split_root).resolve()
    preprocessing_file = Path(preprocessing_path).resolve()
    if not isinstance(model_config, Mapping) or not isinstance(sequence_config, Mapping):
        raise ForecastEvaluationError("Model and sequence configurations must be mappings.")
    try:
        device = resolve_device(evaluation_config.device)
    except (ValueError, RuntimeError) as exc:
        raise ForecastEvaluationError(f"Cannot select evaluation device: {exc}") from exc

    payload = _read_checkpoint_payload(checkpoint)
    training_state = payload.get("training_state")
    if not isinstance(training_state, Mapping):
        raise ForecastEvaluationError(
            "Checkpoint lacks Part 5.8 training provenance; final evaluation is refused."
        )
    if (
        training_state.get("selection_partition") != "validation"
        or training_state.get("test_partition_used_for_selection") is not False
        or training_state.get("target_unit") != "kW"
    ):
        raise ForecastEvaluationError(
            "Checkpoint does not prove validation-only selection and physical-kW targets."
        )

    expected_graph_config = model_config.get("graph")
    expected_targets = model_config.get("targets")
    expected_data = model_config.get("data")
    expected_graph_nodes = (
        expected_graph_config.get("node_order")
        if isinstance(expected_graph_config, Mapping)
        else None
    )
    expected_target_nodes = (
        expected_targets.get("nodes")
        if isinstance(expected_targets, Mapping)
        else None
    )
    if (
        tuple(expected_graph_nodes or ()) != EXPECTED_GRAPH_NODES
        or tuple(expected_target_nodes or ()) != EXPECTED_TARGETS
        or not isinstance(expected_targets, Mapping)
        or expected_targets.get("unit") != "kW"
        or not isinstance(expected_data, Mapping)
    ):
        raise ForecastEvaluationError("Model config does not match the audited seven-target contract.")
    graph_spec = sequence_config.get("graph")
    graph_nodes_spec = graph_spec.get("nodes") if isinstance(graph_spec, Mapping) else None
    if not isinstance(graph_nodes_spec, list) or tuple(
        node.get("id") for node in graph_nodes_spec if isinstance(node, Mapping)
    ) != EXPECTED_GRAPH_NODES:
        raise ForecastEvaluationError("Sequence graph configuration has incompatible node ordering.")

    loader, graph, dataset = _load_test_loader(
        root,
        sequence_config=sequence_config,
        batch_size=evaluation_config.batch_size,
    )
    if tuple(dataset.node_order) != EXPECTED_GRAPH_NODES:
        raise ForecastEvaluationError("Test dataset node ordering is incompatible.")
    target_mask = np.asarray(dataset._arrays["target_mask"])
    valid_nodes = tuple(
        node
        for node, valid in zip(dataset.node_order, target_mask[:, 0].tolist())
        if valid
    )
    if valid_nodes != EXPECTED_TARGETS:
        raise ForecastEvaluationError(
            f"Test target mask is {valid_nodes}, expected exactly {EXPECTED_TARGETS}."
        )
    if dataset.metadata.get("target_unit") != "kW":
        raise ForecastEvaluationError("Test labels must remain in physical kW.")
    if graph.metadata.get("construction") != "assumed_shared_bus":
        raise ForecastEvaluationError("The expected assumed shared-bus graph provenance is missing.")
    graph_hash = graph_topology_fingerprint(graph)

    preprocessing = _read_json(preprocessing_file, "preprocessing artifact")
    preprocessing_sha, manifest_sha = _validate_preprocessing(
        preprocessing,
        dataset=dataset,
        preprocessing_path=preprocessing_file,
        split_root=root,
    )
    experiment_id = training_state.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ForecastEvaluationError("Checkpoint has no experiment identifier.")
    if (
        training_state.get("graph_topology_sha256") != graph_hash
        or tuple(training_state.get("target_order", ())) != EXPECTED_TARGETS
    ):
        raise ForecastEvaluationError("Checkpoint graph fingerprint or target ordering is incompatible.")
    preprocessing_refs = training_state.get("preprocessing_artifacts")
    if (
        not isinstance(preprocessing_refs, Mapping)
        or not isinstance(preprocessing_refs.get("preprocessing"), str)
        or not isinstance(preprocessing_refs.get("split_manifest"), str)
        or Path(preprocessing_refs["preprocessing"].replace("\\", "/")).name != "preprocessing.json"
        or Path(preprocessing_refs["split_manifest"].replace("\\", "/")).name != "split_manifest.json"
    ):
        raise ForecastEvaluationError("Checkpoint preprocessing/split references are missing or invalid.")

    supplied_metadata_path = (
        Path(training_metadata_path).resolve()
        if training_metadata_path is not None
        else _discover_training_metadata(checkpoint, experiment_id)
    )
    if supplied_metadata_path is None:
        raise ForecastEvaluationError(
            "Matching Part 5.8 experiment.json is required to verify checkpoint, split, and scaler hashes."
        )
    metadata = _read_json(supplied_metadata_path, "training experiment metadata")
    manifest = _read_json(root / "split_manifest.json", "split manifest")
    training_partition_sha = _partition_sha256(root, "train")
    validation_partition_sha = _partition_sha256(root, "validation")
    checkpoint_sha = _sha256_file(checkpoint)
    _validate_training_metadata(
        metadata,
        expected_model_config=model_config,
        training_state=training_state,
        graph_fingerprint=graph_hash,
        preprocessing_sha256=preprocessing_sha,
        manifest_sha256=manifest_sha,
        training_partition_sha256=training_partition_sha,
        validation_partition_sha256=validation_partition_sha,
        checkpoint_sha256=checkpoint_sha,
        experiment_id=experiment_id,
    )
    _validate_full_training_history(
        training_state,
        manifest=manifest,
        sequence_config=sequence_config,
    )
    metadata_verified = True

    try:
        model = GNNTransformerForecaster.load_checkpoint(checkpoint, map_location=device)
    except GNNTransformerError as exc:
        raise ForecastEvaluationError(f"Cannot reconstruct trained forecaster: {exc}") from exc
    if (
        model.graph_node_ids != EXPECTED_GRAPH_NODES
        or model.target_node_ids != EXPECTED_TARGETS
        or model.target_channels != ("power_kw",)
        or model.target_scope != "node_level"
    ):
        raise ForecastEvaluationError("Checkpoint architecture does not match the seven node targets.")
    configured_lookback = int(expected_data.get("lookback_hours", -1))
    configured_horizon = int(expected_data.get("horizon_hours", -1))
    if (
        dataset[0]["X"].shape[0] != model.lookback_hours
        or model.lookback_hours != configured_lookback
        or dataset[0]["y"].shape[0] != configured_horizon
        or model.forecast_horizon != configured_horizon
    ):
        raise ForecastEvaluationError("Checkpoint dimensions do not match saved test sequences.")
    horizon = evaluation_config.forecast_horizon_hours
    if horizon > min(model.forecast_horizon, dataset[0]["y"].shape[0]):
        raise ForecastEvaluationError(
            "Evaluation horizon cannot exceed the checkpoint/data-set forecast horizon."
        )
    if model_config.get("targets", {}).get("target_scaling") not in (None, "not_selected"):
        raise ForecastEvaluationError(
            "Target scaling is not supported unless its inverse transform is explicitly available."
        )

    model.eval()
    graph_device = graph.to(device)
    all_targets: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    prediction_batches: dict[str, list[np.ndarray]] = {
        "gnn_transformer": [],
        "persistence": [],
        "historical_average": [],
    }
    issue_times: list[str] = []
    target_times: list[tuple[str, ...]] = []
    seasonal_available = (
        dataset[0]["X"].shape[0] >= evaluation_config.seasonal_period_hours
    )
    if seasonal_available:
        prediction_batches["seasonal_persistence"] = []

    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            if tuple(batch["X"].shape[1:]) != (
                model.lookback_hours,
                len(EXPECTED_GRAPH_NODES),
                model.gnn_encoder.input_dim,
            ):
                raise ForecastEvaluationError("Test batch input dimensions do not match checkpoint.")
            if batch["y"].shape[1] < horizon:
                raise ForecastEvaluationError("Test batch target horizon is shorter than requested.")
            predicted = model(
                batch["X"],
                graph_device.edge_index,
                graph_device.edge_weight,
                target_mask=batch["target_mask"],
                node_order=graph.node_order,
            )
            if predicted.shape != batch["y"].shape:
                raise ForecastEvaluationError("Model output shape does not match saved test targets.")
            target_np = batch["y"][:, :horizon].detach().cpu().numpy()
            mask_np = batch["target_mask"].detach().cpu().numpy()
            pred_np = predicted[:, :horizon].detach().cpu().numpy()
            scaled_history = raw_batch["X"].numpy()
            physical_history = inverse_scale_node_history(
                scaled_history,
                sequence_config=sequence_config,
                preprocessing=preprocessing,
            )
            batch_horizon = horizon
            try:
                baseline_predictions = {
                    "persistence": persistence_forecast(
                        physical_history, batch_horizon, time_axis=1
                    ),
                    "historical_average": historical_average_forecast(
                        physical_history, batch_horizon, time_axis=1
                    ),
                }
                if seasonal_available:
                    baseline_predictions["seasonal_persistence"] = seasonal_persistence_forecast(
                        physical_history,
                        batch_horizon,
                        time_axis=1,
                        season_period=evaluation_config.seasonal_period_hours,
                    )
            except ForecastingBaselineError as exc:
                raise ForecastEvaluationError(f"Cannot generate a causal baseline: {exc}") from exc
            all_targets.append(target_np)
            all_masks.append(mask_np)
            prediction_batches["gnn_transformer"].append(pred_np)
            for method, values in baseline_predictions.items():
                prediction_batches[method].append(np.asarray(values, dtype=np.float32))
            for sample_inputs, sample_targets in zip(
                raw_batch["input_timestamps"], raw_batch["target_timestamps"]
            ):
                if sample_inputs[-1] >= sample_targets[0]:
                    raise ForecastEvaluationError(
                        "A test forecast origin is not strictly earlier than its first target."
                    )
                if issue_times and sample_inputs[-1] <= issue_times[-1]:
                    raise ForecastEvaluationError("Test forecasts are not in chronological order.")
                issue_times.append(sample_inputs[-1])
                target_times.append(tuple(sample_targets[:horizon]))

    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    predictions = {
        method: np.concatenate(values, axis=0)
        for method, values in prediction_batches.items()
        if values
    }
    metric_payload = calculate_forecast_metrics(
        predictions,
        targets,
        masks,
        graph_node_order=graph.node_order,
    )
    boundaries = dataset.metadata["boundaries"]
    source_metadata = dataset.metadata["source_metadata"]
    source_origins = sorted(
        {
            entry.get("origin")
            for entry in source_metadata.values()
            if isinstance(entry, Mapping) and isinstance(entry.get("origin"), str)
        }
    )
    data_synthetic = any(
        bool(entry.get("synthetic"))
        for entry in source_metadata.values()
        if isinstance(entry, Mapping)
    )
    data_label = next(
        (
            entry.get("dataset_label")
            for entry in source_metadata.values()
            if isinstance(entry, Mapping) and entry.get("dataset_label")
        ),
        None,
    )
    revision = os.environ.get("AEB_GIT_COMMIT_SHA")
    if revision is not None and not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision.strip()):
        raise ForecastEvaluationError("AEB_GIT_COMMIT_SHA must be a full Git commit hash.")
    if revision is None:
        try:
            candidate = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            revision = candidate if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", candidate) else None
        except (OSError, subprocess.SubprocessError):
            revision = None

    results: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "completed",
        "experiment_id": experiment_id,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit_sha": revision,
        "checkpoint": {
            "identifier": checkpoint.stem,
            "path": str(checkpoint),
            "sha256": _sha256_file(checkpoint),
            "training_metadata_sha256": (
                _sha256_file(supplied_metadata_path)
                if supplied_metadata_path is not None
                else None
            ),
            "training_metadata_hash_verified": metadata_verified,
        },
        "dataset": {
            "dataset_label": data_label,
            "synthetic": data_synthetic,
            "source_origins": source_origins,
            "source_ids": sorted(source_metadata),
            "split_identifier": str(root.name),
            "split_manifest_sha256": manifest_sha,
            "test_partition_sha256": _partition_sha256(root, "test"),
            "test_partition_boundaries_utc": boundaries,
            "preprocessing_fit_partition": preprocessing["fitted_on"],
            "preprocessing_sha256": preprocessing_sha,
            "graph_reference": graph.metadata.get("construction"),
            "graph_topology_sha256": graph_hash,
            "graph_topology_provenance": graph.metadata.get("construction"),
            "graph_node_order": list(graph.node_order),
            "targets": [
                {
                    "name": target,
                    "unit": "kW",
                    "category": "building_electricity_demand"
                    if target.startswith("B")
                    else "renewable_generation",
                }
                for target in EXPECTED_TARGETS
            ],
            "lookback_hours": int(dataset.metadata["config"]["lookback_hours"]),
            "forecast_horizon_hours": horizon,
            "target_data_units": "kW",
        },
        "evaluation": {
            "partition": "test",
            "partition_use": "final_evaluation_only",
            "test_used_for_training_or_selection": False,
            "sample_count": len(dataset),
            "evaluated_sample_count": int(targets.shape[0]),
            "common_sample_count": int(targets.shape[0]),
            "excluded_samples": {
                "count": 0,
                "reasons": [],
                "seasonal_persistence_unavailable": not seasonal_available,
                "seasonal_persistence_unavailable_reason": (
                    None
                    if seasonal_available
                    else (
                        f"lookback_hours={dataset.metadata['config']['lookback_hours']} "
                        f"is shorter than seasonal_period_hours="
                        f"{evaluation_config.seasonal_period_hours}"
                    )
                ),
                "method_exclusions": (
                    {}
                    if seasonal_available
                    else {
                        "seasonal_persistence": {
                            "sample_count": len(dataset),
                            "reason": (
                                "insufficient_historical_context: "
                                f"lookback_hours={dataset.metadata['config']['lookback_hours']} "
                                f"< seasonal_period_hours="
                                f"{evaluation_config.seasonal_period_hours}"
                            ),
                        }
                    }
                ),
            },
            "forecast_origin_first_utc": issue_times[0],
            "forecast_origin_last_utc": issue_times[-1],
            "target_timestamp_first_utc": target_times[0][0],
            "target_timestamp_last_utc": target_times[-1][-1],
            "timestamp_alignment": "each model/baseline row uses the same forecast origin and target timestamps",
            "rolling_window_overlap": True,
            "metrics": metric_payload,
        },
        "runtime": {
            "device": str(device),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "limitations": [
            "Results describe the available dataset provenance and are not evidence of real campus operations.",
            "The assumed shared-bus topology is not measured electrical infrastructure.",
            "The model consumes historical node power only; global_X weather features are not encoded.",
            "The held-out test partition is used only for this final evaluation and must not guide further model selection.",
        ],
    }
    if not data_synthetic:
        results["limitations"].append(
            "Dataset source provenance is not marked synthetic; verify source licensing and representativeness before operational claims."
        )
    return ForecastEvaluationOutput(
        results=results,
        predictions=predictions,
        targets=targets,
        mask=masks,
        issue_times_utc=tuple(issue_times),
        target_timestamps_utc=tuple(target_times),
    )


def _partition_sha256(split_root: Path, partition: str) -> str:
    manifest = _read_json(split_root / "split_manifest.json", "split manifest")
    checksums = manifest.get("checksums")
    if not isinstance(checksums, Mapping):
        raise ForecastEvaluationError("Split manifest has no checksum mapping.")
    selected = {
        path: digest
        for path, digest in checksums.items()
        if isinstance(path, str) and Path(path).parts and Path(path).parts[0] == partition
    }
    if not selected:
        raise ForecastEvaluationError(
            f"Split manifest has no checksums for {partition}."
        )
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _nonoverlapping_examples(
    issue_times_utc: Sequence[str], horizon_hours: int, limit: int
) -> tuple[int, ...]:
    chosen: list[int] = []
    previous: np.datetime64 | None = None
    spacing = np.timedelta64(horizon_hours, "h")
    for index, value in enumerate(issue_times_utc):
        current = np.datetime64(value.replace("Z", ""), "ns")
        if previous is None or current - previous >= spacing:
            chosen.append(index)
            previous = current
            if len(chosen) == limit:
                break
    return tuple(chosen)


def save_evaluation_artifacts(
    output: ForecastEvaluationOutput,
    output_dir: str | Path,
    *,
    representative_window_count: int,
) -> Path:
    """Write JSON, aligned prediction arrays and labeled plots without overwrites."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation results: {destination}")
    if (
        isinstance(representative_window_count, bool)
        or not isinstance(representative_window_count, int)
        or representative_window_count < 1
    ):
        raise ForecastEvaluationError("representative_window_count must be positive.")
    results = dict(output.results)
    evaluation = results.get("evaluation")
    metrics = evaluation.get("metrics") if isinstance(evaluation, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise ForecastEvaluationError("Evaluation output lacks structured metrics.")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        results_path = destination / "evaluation.json"
        results_path.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        arrays: dict[str, np.ndarray] = {
            "targets_kw": output.targets,
            "target_mask": output.mask,
            "issue_times_utc": np.asarray(output.issue_times_utc, dtype=str),
            "target_timestamps_utc": np.asarray(output.target_timestamps_utc, dtype=str),
        }
        arrays.update(
            {f"predictions_{name}": values for name, values in output.predictions.items()}
        )
        np.savez_compressed(destination / "predictions.npz", **arrays)
        _write_plots(output, destination, representative_window_count)
        results["artifacts"] = {
            "evaluation_json": results_path.name,
            "predictions_npz": "predictions.npz",
            "plots_directory": "plots",
        }
        results_path.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError, ImportError) as exc:
        raise ForecastEvaluationError(f"Cannot write evaluation artifacts: {exc}") from exc
    return destination


def _write_plots(
    output: ForecastEvaluationOutput,
    destination: Path,
    representative_window_count: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ForecastEvaluationError(
            "Matplotlib is required to save evaluation plots."
        ) from exc

    plot_root = destination / "plots"
    example_root = plot_root / "representative_forecasts"
    example_root.mkdir(parents=True)
    metrics = output.results["evaluation"]["metrics"]
    target_indices = [EXPECTED_GRAPH_NODES.index(name) for name in EXPECTED_TARGETS]
    valid = _expanded_mask(output.mask, output.targets.shape)
    actual_target = output.targets[:, :, target_indices, :]
    predicted_target = output.predictions["gnn_transformer"][:, :, target_indices, :]
    valid_target = valid[:, :, target_indices, :]

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        actual_target[valid_target],
        predicted_target[valid_target],
        s=5,
        alpha=0.25,
        label="GNN-Transformer",
    )
    bounds = [
        float(np.min(actual_target[valid_target])),
        float(np.max(actual_target[valid_target])),
    ]
    axis.plot(bounds, bounds, color="black", linestyle="--", label="Ideal")
    axis.set(title="Held-out actual vs predicted", xlabel="Actual (kW)", ylabel="Predicted (kW)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    overall = metrics["overall"]
    methods = tuple(overall)
    labels = [method.replace("_", "\n") for method in methods]
    x = np.arange(len(methods))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        x - width / 2,
        [np.nan if overall[method]["mae_kw"] is None else overall[method]["mae_kw"] for method in methods],
        width,
        label="MAE (kW)",
    )
    axis.bar(
        x + width / 2,
        [np.nan if overall[method]["rmse_kw"] is None else overall[method]["rmse_kw"] for method in methods],
        width,
        label="RMSE (kW)",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Error (kW)")
    axis.set_title("Held-out forecasting baseline comparison")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "baseline_comparison.png", dpi=150)
    plt.close(fig)

    target_metrics = metrics["per_target"]
    names = list(target_metrics)
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        x - width / 2,
        [
            np.nan
            if target_metrics[name]["methods"]["gnn_transformer"]["mae_kw"] is None
            else target_metrics[name]["methods"]["gnn_transformer"]["mae_kw"]
            for name in names
        ],
        width,
        label="MAE (kW)",
    )
    axis.bar(
        x + width / 2,
        [
            np.nan
            if target_metrics[name]["methods"]["gnn_transformer"]["rmse_kw"] is None
            else target_metrics[name]["methods"]["gnn_transformer"]["rmse_kw"]
            for name in names
        ],
        width,
        label="RMSE (kW)",
    )
    axis.set_xticks(x, names)
    axis.set_ylabel("Error (kW)")
    axis.set_title("GNN-Transformer error by forecasting target")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "per_target_errors.png", dpi=150)
    plt.close(fig)

    horizons = metrics["per_forecast_horizon"]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        [entry["forecast_hour"] for entry in horizons],
        [
            np.nan
            if entry["methods"]["gnn_transformer"]["mae_kw"] is None
            else entry["methods"]["gnn_transformer"]["mae_kw"]
            for entry in horizons
        ],
        marker="o",
        label="MAE (kW)",
    )
    axis.plot(
        [entry["forecast_hour"] for entry in horizons],
        [
            np.nan
            if entry["methods"]["gnn_transformer"]["rmse_kw"] is None
            else entry["methods"]["gnn_transformer"]["rmse_kw"]
            for entry in horizons
        ],
        marker="o",
        label="RMSE (kW)",
    )
    axis.set(
        title="GNN-Transformer error by forecast horizon",
        xlabel="Forecast hour",
        ylabel="Error (kW)",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "forecast_horizon_errors.png", dpi=150)
    plt.close(fig)

    example_indices = _nonoverlapping_examples(
        output.issue_times_utc,
        output.targets.shape[1],
        representative_window_count,
    )
    for node_name in EXPECTED_TARGETS:
        node_index = EXPECTED_GRAPH_NODES.index(node_name)
        for example_number, sample_index in enumerate(example_indices, start=1):
            timestamps = output.target_timestamps_utc[sample_index]
            actual_values = output.targets[sample_index, :, node_index, 0]
            target_validity = valid[sample_index, :, node_index, 0]
            fig, axis = plt.subplots(figsize=(10, 4))
            axis.plot(
                timestamps,
                np.where(target_validity, actual_values, np.nan),
                marker="o",
                label="Actual",
            )
            for method, values in output.predictions.items():
                axis.plot(
                    timestamps,
                    np.where(target_validity, values[sample_index, :, node_index, 0], np.nan),
                    marker=".",
                    label=method.replace("_", " "),
                )
            axis.set(
                title=f"{node_name} forecast example {example_number} (issue {output.issue_times_utc[sample_index]})",
                xlabel="Target timestamp (UTC)",
                ylabel="Power (kW)",
            )
            axis.tick_params(axis="x", labelrotation=35)
            axis.legend()
            fig.tight_layout()
            fig.savefig(example_root / f"{node_name}_window_{example_number:02d}.png", dpi=150)
            plt.close(fig)
