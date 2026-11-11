"""Masked, validation-selected training for the Part 5.7 forecaster."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import uuid
import time
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Subset

from src.forecasting.dataloader import ForecastGraph, move_batch_to_device
from src.forecasting.models.gnn_transformer import GNNTransformerError, GNNTransformerForecaster


class ForecastTrainingError(ValueError):
    """Invalid training configuration, batch, loss, or checkpoint state."""


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 50
    patience: int = 8
    min_delta: float = 0.0001
    seed: int = 42
    optimizer: str = "adamw"
    loss: str = "masked_mse"
    gradient_clip_norm: float | None = 1.0
    device: str = "auto"
    deterministic: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TrainingConfig":
        if not isinstance(config, Mapping):
            raise ForecastTrainingError("training configuration must be a mapping.")
        values = config.get("training", config)
        if not isinstance(values, Mapping):
            raise ForecastTrainingError("training settings must be a mapping.")
        expected = set(cls.__dataclass_fields__)
        unknown = set(values) - expected
        if unknown:
            raise ForecastTrainingError(
                f"Unknown training configuration fields: {sorted(unknown)}."
            )
        try:
            result = cls(**dict(values))
        except TypeError as exc:
            raise ForecastTrainingError(f"Invalid training configuration: {exc}") from exc
        result.validate()
        return result

    def validate(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("max_epochs", self.max_epochs),
            ("patience", self.patience),
            ("seed", self.seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ForecastTrainingError(f"{name} must be an integer.")
            if value < (0 if name == "seed" else 1):
                raise ForecastTrainingError(
                    f"{name} must be {'nonnegative' if name == 'seed' else 'positive'}."
                )
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("min_delta", self.min_delta),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "learning_rate" and value == 0)
            ):
                raise ForecastTrainingError(
                    f"{name} must be finite and "
                    f"{'positive' if name == 'learning_rate' else 'nonnegative'}."
                )
        if self.gradient_clip_norm is not None and (
            isinstance(self.gradient_clip_norm, bool)
            or not isinstance(self.gradient_clip_norm, (int, float))
            or not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0
        ):
            raise ForecastTrainingError(
                "gradient_clip_norm must be finite and positive or null."
            )
        if self.optimizer != "adamw":
            raise ForecastTrainingError("Only the AdamW optimizer is supported.")
        if self.loss != "masked_mse":
            raise ForecastTrainingError("Only masked_mse is supported.")
        if not isinstance(self.device, str) or self.device not in {"auto", "cpu", "cuda"}:
            raise ForecastTrainingError("device must be auto, cpu, or cuda.")
        if type(self.deterministic) is not bool:
            raise ForecastTrainingError("deterministic must be a boolean.")


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    best_loss: float | None = None
    best_epoch: int | None = None
    patience_reference_loss: float | None = None
    wait_count: int = 0
    stop_reason: str | None = None

    def update(self, epoch: int, validation_loss: float) -> bool:
        if not math.isfinite(validation_loss):
            raise ForecastTrainingError("Validation loss is non-finite.")
        best_checkpoint_improved = (
            self.best_loss is None or validation_loss < self.best_loss
        )
        if best_checkpoint_improved:
            self.best_loss = validation_loss
            self.best_epoch = epoch
        patience_improved = (
            self.patience_reference_loss is None
            or validation_loss < self.patience_reference_loss - self.min_delta
        )
        if patience_improved:
            self.patience_reference_loss = validation_loss
            self.wait_count = 0
            self.stop_reason = None
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stop_reason = "validation_loss_not_improved_for_patience_epochs"
        return best_checkpoint_improved

    @property
    def should_stop(self) -> bool:
        return self.stop_reason is not None

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "patience_reference_loss": self.patience_reference_loss,
            "wait_count": self.wait_count,
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("patience") != self.patience
            or float(state.get("min_delta")) != self.min_delta
        ):
            raise ForecastTrainingError(
                "Checkpoint early-stopping settings do not match the current configuration."
            )
        best_loss = state.get("best_loss")
        if best_loss is not None and (
            isinstance(best_loss, bool)
            or not isinstance(best_loss, (int, float))
            or not math.isfinite(best_loss)
        ):
            raise ForecastTrainingError("Checkpoint has an invalid best validation loss.")
        self.best_loss = None if best_loss is None else float(best_loss)
        self.best_epoch = state.get("best_epoch")
        reference_loss = state.get("patience_reference_loss")
        if reference_loss is not None and (
            isinstance(reference_loss, bool)
            or not isinstance(reference_loss, (int, float))
            or not math.isfinite(reference_loss)
        ):
            raise ForecastTrainingError("Checkpoint has an invalid early-stopping reference loss.")
        self.patience_reference_loss = (
            None if reference_loss is None else float(reference_loss)
        )
        self.wait_count = int(state.get("wait_count", 0))
        self.stop_reason = state.get("stop_reason")


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float
    stop_reason: str
    duration_seconds: float
    checkpoint_path: str
    resume_checkpoint_path: str
    device: str


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, CPU and available CUDA generators."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ForecastTrainingError("seed must be a nonnegative integer.")
    if type(deterministic) is not bool:
        raise ForecastTrainingError("deterministic must be a boolean.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ForecastTrainingError("CUDA was requested but is not available.")
    return torch.device(device)


def _expanded_target_mask(
    mask: torch.Tensor, target_shape: torch.Size, device: torch.device
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ForecastTrainingError("target_mask must be a boolean tensor.")
    if mask.device != device:
        raise ForecastTrainingError("target_mask and prediction tensors must share a device.")
    batch_size, _, node_count, channels = target_shape
    if mask.shape == (node_count, 1):
        candidate = mask.view(1, 1, node_count, 1)
    elif mask.shape == (batch_size, node_count, 1):
        candidate = mask.unsqueeze(1)
    elif mask.shape == (batch_size, node_count, channels):
        candidate = mask.unsqueeze(1)
    elif mask.shape == target_shape:
        candidate = mask
    else:
        raise ForecastTrainingError(
            "target_mask must be [nodes, 1], [batch, nodes, 1], "
            "[batch, nodes, channels], or match the target tensor."
        )
    try:
        return candidate.expand(target_shape)
    except RuntimeError as exc:
        raise ForecastTrainingError(
            "target_mask dimensions cannot broadcast to targets."
        ) from exc


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over only valid [batch, horizon, node, target] values."""
    if (
        not isinstance(predictions, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or predictions.ndim != 4
        or targets.shape != predictions.shape
    ):
        raise ForecastTrainingError(
            "predictions and targets must have equal [batch, horizon, nodes, targets] shapes."
        )
    if (
        not predictions.is_floating_point()
        or not targets.is_floating_point()
        or predictions.device != targets.device
    ):
        raise ForecastTrainingError("predictions and targets must be floating tensors on one device.")
    valid = _expanded_target_mask(target_mask, targets.shape, targets.device)
    if not torch.any(valid):
        raise ForecastTrainingError("A batch must contain at least one valid target.")
    selected_predictions = predictions[valid]
    selected_targets = targets[valid]
    if not torch.isfinite(selected_predictions).all() or not torch.isfinite(selected_targets).all():
        raise ForecastTrainingError("Valid predictions and targets must be finite.")
    return torch.square(selected_predictions - selected_targets).mean()


def _loss_sum_and_count(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if (
        not isinstance(predictions, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or predictions.ndim != 4
        or targets.shape != predictions.shape
    ):
        raise ForecastTrainingError("Predictions and targets must have matching 4-D shapes.")
    valid = _expanded_target_mask(target_mask, targets.shape, targets.device)
    count = int(valid.sum().item())
    if count == 0:
        raise ForecastTrainingError("A batch must contain at least one valid target.")
    selected_predictions = predictions[valid]
    selected_targets = targets[valid]
    if not torch.isfinite(selected_predictions).all() or not torch.isfinite(selected_targets).all():
        raise ForecastTrainingError("Valid predictions and targets must be finite.")
    squared_error_sum = torch.square(selected_predictions - selected_targets).sum()
    if not torch.isfinite(squared_error_sum):
        raise ForecastTrainingError("Masked MSE is non-finite.")
    return squared_error_sum, count


def _validate_batch_contract(
    model: GNNTransformerForecaster,
    graph: ForecastGraph,
    batch: Mapping[str, Any],
) -> None:
    inputs, targets, target_mask = batch.get("X"), batch.get("y"), batch.get("target_mask")
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4:
        raise ForecastTrainingError("Training batch X must be [batch, lookback, nodes, features].")
    if not isinstance(targets, torch.Tensor) or targets.ndim != 4:
        raise ForecastTrainingError("Training batch y must be [batch, horizon, nodes, targets].")
    if tuple(graph.node_order) != model.graph_node_ids:
        raise ForecastTrainingError("Model graph node order differs from the training graph.")
    if (
        inputs.shape[0] != targets.shape[0]
        or inputs.shape[2] != len(graph.node_order)
        or targets.shape[1] != model.forecast_horizon
        or targets.shape[2] != len(graph.node_order)
        or targets.shape[3] != len(model.target_channels)
    ):
        raise ForecastTrainingError(
            "Training input/target dimensions do not match model and graph configuration."
        )
    valid = _expanded_target_mask(target_mask, targets.shape, targets.device)
    target_nodes = torch.zeros(
        len(graph.node_order), dtype=torch.bool, device=targets.device
    )
    target_indices = torch.tensor(
        model.target_node_indices, dtype=torch.long, device=targets.device
    )
    target_nodes[target_indices] = True
    valid_nodes = valid.any(dim=(1, 3))
    if torch.any(valid_nodes & ~target_nodes.unsqueeze(0)):
        raise ForecastTrainingError(
            "Target mask marks a non-target/context node as a valid forecasting label."
        )


def _loader_partition(loader: DataLoader) -> str | None:
    dataset: Any = loader.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return getattr(dataset, "partition", None)


def _validate_loader_partition(loader: DataLoader, expected: str) -> None:
    actual = _loader_partition(loader)
    if actual != expected:
        raise ForecastTrainingError(
            f"{expected} loader must contain only {expected} sequences; got {actual!r}."
        )


def graph_topology_fingerprint(graph: ForecastGraph) -> str:
    payload = {
        "node_order": list(graph.node_order),
        "edge_index": graph.edge_index.detach().cpu().tolist(),
        "edge_weight": (
            None
            if graph.edge_weight is None
            else graph.edge_weight.detach().cpu().tolist()
        ),
        "metadata": dict(graph.metadata),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rng_state(train_loader: DataLoader, device: torch.device) -> dict[str, Any]:
    generator = getattr(train_loader, "generator", None)
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gaussian": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "train_loader": generator.get_state() if generator is not None else None,
    }


def _restore_rng_state(
    state: Mapping[str, Any], train_loader: DataLoader, device: torch.device
) -> None:
    try:
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state["algorithm"],
                numpy_state["keys"].cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gaussian"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state["torch_cpu"].cpu())
        if device.type == "cuda":
            if not torch.cuda.is_available() or not state.get("torch_cuda"):
                raise ForecastTrainingError(
                    "Checkpoint lacks CUDA RNG state required for CUDA resumption."
                )
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        generator = getattr(train_loader, "generator", None)
        loader_rng = state.get("train_loader")
        if generator is not None and loader_rng is not None:
            generator.set_state(loader_rng.cpu())
    except (KeyError, RuntimeError, TypeError) as exc:
        if isinstance(exc, ForecastTrainingError):
            raise
        raise ForecastTrainingError(f"Cannot restore checkpoint RNG state: {exc}") from exc


def _load_checkpoint_payload(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ForecastTrainingError(f"Cannot load training checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ForecastTrainingError("Training checkpoint payload must be a mapping.")
    if payload.get("format_version") != GNNTransformerForecaster.CHECKPOINT_FORMAT_VERSION:
        raise ForecastTrainingError("Unsupported GNN-Transformer checkpoint format.")
    return payload


def load_forecaster_for_evaluation(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> GNNTransformerForecaster:
    """Load a Part 5.7-compatible checkpoint for evaluation without optimizer state."""
    try:
        model = GNNTransformerForecaster.load_checkpoint(checkpoint_path, map_location=device)
    except GNNTransformerError as exc:
        raise ForecastTrainingError(f"Cannot restore forecasting model: {exc}") from exc
    model.eval()
    return model


def _checkpoint_training_state(
    *,
    optimizer: Optimizer,
    config: TrainingConfig,
    epoch: int,
    elapsed_seconds: float,
    experiment_id: str,
    stopper: EarlyStopping,
    history: list[dict[str, float | int]],
    graph_fingerprint: str,
    preprocessing_artifacts: Mapping[str, str],
    target_order: tuple[str, ...],
    target_unit: str,
    train_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "elapsed_seconds": elapsed_seconds,
        "experiment_id": experiment_id,
        "seed": config.seed,
        "training_config": asdict(config),
        "optimizer_state_dict": optimizer.state_dict(),
        "early_stopping_state": stopper.state_dict(),
        "history": list(history),
        "graph_topology_sha256": graph_fingerprint,
        "preprocessing_artifacts": dict(preprocessing_artifacts),
        "target_order": list(target_order),
        "target_unit": target_unit,
        "selection_partition": "validation",
        "test_partition_used_for_selection": False,
        "rng_state": _rng_state(train_loader, device),
    }


def _restore_training_state(
    *,
    path: str | Path,
    model: GNNTransformerForecaster,
    optimizer: Optimizer,
    config: TrainingConfig,
    experiment_id: str,
    stopper: EarlyStopping,
    graph_fingerprint: str,
    preprocessing_artifacts: Mapping[str, str],
    target_order: tuple[str, ...],
    train_loader: DataLoader,
    device: torch.device,
) -> tuple[int, list[dict[str, float | int]], float]:
    payload = _load_checkpoint_payload(path)
    state = payload.get("training_state")
    if not isinstance(state, Mapping):
        raise ForecastTrainingError("Checkpoint does not contain resumable training state.")
    if state.get("experiment_id") != experiment_id or state.get("seed") != config.seed:
        raise ForecastTrainingError("Resume checkpoint experiment ID or seed does not match.")
    if state.get("graph_topology_sha256") != graph_fingerprint:
        raise ForecastTrainingError("Resume checkpoint graph topology does not match.")
    if state.get("preprocessing_artifacts") != dict(preprocessing_artifacts):
        raise ForecastTrainingError("Resume checkpoint preprocessing references do not match.")
    if tuple(state.get("target_order", ())) != target_order:
        raise ForecastTrainingError("Resume checkpoint target ordering does not match.")
    saved_config = state.get("training_config")
    if not isinstance(saved_config, Mapping):
        raise ForecastTrainingError("Resume checkpoint lacks training configuration.")
    saved_config = dict(saved_config)
    current_config = asdict(config)
    saved_max_epochs = saved_config.pop("max_epochs", None)
    current_max_epochs = current_config.pop("max_epochs", None)
    if saved_config != current_config or (
        isinstance(saved_max_epochs, bool)
        or not isinstance(saved_max_epochs, int)
        or not isinstance(current_max_epochs, int)
        or current_max_epochs < saved_max_epochs
    ):
        raise ForecastTrainingError(
            "Resume training settings must match; max_epochs may only stay equal or increase."
        )
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ForecastTrainingError(f"Resume checkpoint state is incompatible: {exc}") from exc
    stopper.load_state_dict(state["early_stopping_state"])
    _restore_rng_state(state["rng_state"], train_loader, device)
    epoch = state.get("epoch")
    history = state.get("history")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ForecastTrainingError("Resume checkpoint has an invalid epoch.")
    if not isinstance(history, list):
        raise ForecastTrainingError("Resume checkpoint has invalid epoch history.")
    elapsed = state.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ForecastTrainingError("Resume checkpoint has invalid accumulated duration.")
    return epoch + 1, history, float(elapsed)


def fit_forecaster(
    model: GNNTransformerForecaster,
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    graph: ForecastGraph,
    config: TrainingConfig,
    experiment_id: str,
    checkpoint_path: str | Path,
    resume_checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    preprocessing_artifacts: Mapping[str, str] | None = None,
    target_unit: str = "kW",
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    progress_callback: Callable[[int, float, float, bool], None] | None = None,
) -> TrainingResult:
    """Train with valid-element weighted losses and validation-only selection."""
    config.validate()
    if not experiment_id or not isinstance(experiment_id, str):
        raise ForecastTrainingError("experiment_id must be nonempty text.")
    if target_unit != "kW":
        raise ForecastTrainingError("The audited forecasting target unit is kW.")
    if model.target_scope != "node_level":
        raise ForecastTrainingError(
            "Part 5.8 training requires the audited node-level target contract."
        )
    for label, limit in (
        ("max_train_batches", max_train_batches),
        ("max_validation_batches", max_validation_batches),
    ):
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ForecastTrainingError(f"{label} must be a positive integer or null.")
    if progress_callback is not None and not callable(progress_callback):
        raise ForecastTrainingError("progress_callback must be callable or null.")
    _validate_loader_partition(train_loader, "train")
    _validate_loader_partition(validation_loader, "validation")
    if train_loader.batch_size is not None and train_loader.batch_size < 1:
        raise ForecastTrainingError("Training loader batch size is invalid.")

    device = resolve_device(config.device)
    model.to(device)
    graph = graph.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    stopper = EarlyStopping(config.patience, float(config.min_delta))
    best_path = Path(checkpoint_path)
    resume_path = (
        Path(resume_checkpoint_path)
        if resume_checkpoint_path is not None
        else best_path.with_name(f"{best_path.stem}.last{best_path.suffix or '.pt'}")
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    if resume_from is None and (best_path.exists() or resume_path.exists()):
        raise FileExistsError(
            "Checkpoint path already exists; choose a new experiment or explicitly resume."
        )
    if resume_from is not None and not Path(resume_from).is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_from}")

    graph_hash = graph_topology_fingerprint(graph)
    preprocessing = dict(preprocessing_artifacts or {})
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in preprocessing.items()):
        raise ForecastTrainingError("Preprocessing references must map text names to text paths.")
    target_order = tuple(model.target_node_ids)
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    prior_duration = 0.0
    if resume_from is not None:
        if not best_path.is_file():
            raise ForecastTrainingError(
                "The best-validation checkpoint is required alongside a resume checkpoint."
            )
        best_state = _load_checkpoint_payload(best_path).get("training_state")
        if (
            not isinstance(best_state, Mapping)
            or best_state.get("experiment_id") != experiment_id
            or best_state.get("graph_topology_sha256") != graph_hash
            or tuple(best_state.get("target_order", ())) != target_order
        ):
            raise ForecastTrainingError(
                "Existing best checkpoint does not belong to this experiment, graph, and target order."
            )
        start_epoch, history, prior_duration = _restore_training_state(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            config=config,
            experiment_id=experiment_id,
            stopper=stopper,
            graph_fingerprint=graph_hash,
            preprocessing_artifacts=preprocessing,
            target_order=target_order,
            train_loader=train_loader,
            device=device,
        )
    else:
        seed_everything(config.seed, deterministic=config.deterministic)
    if train_loader.batch_size is not None and train_loader.batch_size != config.batch_size:
        raise ForecastTrainingError(
            "Training loader batch size must match training configuration."
        )
    if (
        validation_loader.batch_size is not None
        and validation_loader.batch_size != config.batch_size
    ):
        raise ForecastTrainingError(
            "Validation loader batch size must match training configuration."
        )
    if best_path.resolve() == resume_path.resolve():
        raise ForecastTrainingError("Best and resumable checkpoint paths must be different.")

    started = time.perf_counter()
    best_written = best_path.exists()
    resume_written = resume_path.exists()
    stop_reason = "max_epochs_reached"

    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train()
        train_squared_sum = 0.0
        train_count = 0
        train_batches = (
            itertools.islice(train_loader, max_train_batches)
            if max_train_batches is not None
            else train_loader
        )
        for raw_batch in train_batches:
            batch = move_batch_to_device(raw_batch, device)
            _validate_batch_contract(model, graph, batch)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                batch["X"],
                graph.edge_index,
                graph.edge_weight,
                target_mask=batch["target_mask"],
                node_order=graph.node_order,
            )
            batch_sum, batch_count = _loss_sum_and_count(
                predictions, batch["y"], batch["target_mask"]
            )
            batch_loss = batch_sum / batch_count
            if not torch.isfinite(batch_loss):
                raise ForecastTrainingError("Training loss is non-finite.")
            batch_loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise ForecastTrainingError("Training gradients are missing or non-finite.")
            if config.gradient_clip_norm is not None:
                try:
                    clip_grad_norm_(
                        model.parameters(),
                        float(config.gradient_clip_norm),
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    raise ForecastTrainingError(
                        f"Gradient norm is non-finite: {exc}"
                    ) from exc
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ForecastTrainingError("Optimizer produced non-finite model parameters.")
            train_squared_sum += float(batch_sum.detach().item())
            train_count += batch_count
        if train_count == 0:
            raise ForecastTrainingError("Training epoch contained no valid target values.")

        model.eval()
        validation_squared_sum = 0.0
        validation_count = 0
        with torch.inference_mode():
            validation_batches = (
                itertools.islice(validation_loader, max_validation_batches)
                if max_validation_batches is not None
                else validation_loader
            )
            for raw_batch in validation_batches:
                batch = move_batch_to_device(raw_batch, device)
                _validate_batch_contract(model, graph, batch)
                predictions = model(
                    batch["X"],
                    graph.edge_index,
                    graph.edge_weight,
                    target_mask=batch["target_mask"],
                    node_order=graph.node_order,
                )
                batch_sum, batch_count = _loss_sum_and_count(
                    predictions, batch["y"], batch["target_mask"]
                )
                validation_squared_sum += float(batch_sum.item())
                validation_count += batch_count
        if validation_count == 0:
            raise ForecastTrainingError("Validation epoch contained no valid target values.")
        train_loss = train_squared_sum / train_count
        validation_loss = validation_squared_sum / validation_count
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise ForecastTrainingError("Epoch loss is non-finite.")
        history.append(
            {
                "epoch": epoch,
                "training_loss": train_loss,
                "validation_loss": validation_loss,
                "training_valid_targets": train_count,
                "validation_valid_targets": validation_count,
            }
        )
        improved = stopper.update(epoch, validation_loss)
        training_state = _checkpoint_training_state(
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            elapsed_seconds=prior_duration + time.perf_counter() - started,
            experiment_id=experiment_id,
            stopper=stopper,
            history=history,
            graph_fingerprint=graph_hash,
            preprocessing_artifacts=preprocessing,
            target_order=target_order,
            target_unit=target_unit,
            train_loader=train_loader,
            device=device,
        )
        if improved:
            model.save_checkpoint(
                best_path,
                overwrite=best_written,
                training_state=training_state,
            )
            best_written = True
        model.save_checkpoint(
            resume_path,
            overwrite=resume_written,
            training_state=training_state,
        )
        resume_written = True
        if progress_callback is not None:
            progress_callback(epoch, train_loss, validation_loss, improved)
        if stopper.should_stop:
            stop_reason = stopper.stop_reason or "early_stopping"
            break
    if stopper.best_epoch is None or stopper.best_loss is None:
        raise ForecastTrainingError("No best validation checkpoint was produced.")
    best_model = load_forecaster_for_evaluation(best_path, device=device)
    model.load_state_dict(best_model.state_dict(), strict=True)
    return TrainingResult(
        history=history,
        best_epoch=stopper.best_epoch,
        best_validation_loss=stopper.best_loss,
        stop_reason=stop_reason,
        duration_seconds=prior_duration + time.perf_counter() - started,
        checkpoint_path=str(best_path),
        resume_checkpoint_path=str(resume_path),
        device=str(device),
    )


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write structured JSON atomically without replacing an unfinished file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
