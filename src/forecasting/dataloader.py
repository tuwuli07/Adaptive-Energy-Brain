"""Deterministic PyTorch loaders and validated graph data for forecasting."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.graph_builder import load_forecasting_config
from src.data.sequences import load_partition_sources, prepare_sequences
from src.forecasting.dataset import ForecastingDataset


class ForecastDataLoaderError(ValueError):
    """Invalid graph configuration, partitions, or loader options."""


@dataclass(frozen=True)
class ForecastGraph:
    """Standard-PyTorch graph tensors with the Part 2 node order and provenance."""

    edge_index: torch.Tensor
    node_attributes: torch.Tensor
    node_order: tuple[str, ...]
    edge_weight: torch.Tensor | None
    metadata: Mapping[str, Any]

    @property
    def disconnected_nodes(self) -> tuple[str, ...]:
        connected = torch.zeros(len(self.node_order), dtype=torch.bool)
        if self.edge_index.numel():
            connected[self.edge_index.detach().cpu().reshape(-1)] = True
        return tuple(
            node for node, is_connected in zip(self.node_order, connected.tolist())
            if not is_connected
        )

    def to(self, device: torch.device | str) -> "ForecastGraph":
        return ForecastGraph(
            edge_index=self.edge_index.to(device),
            node_attributes=self.node_attributes.to(device),
            node_order=self.node_order,
            edge_weight=None if self.edge_weight is None else self.edge_weight.to(device),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ForecastingDataLoaders:
    """Train, validation, and test loaders plus one shared validated graph."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    graph: ForecastGraph
    node_order: tuple[str, ...]

    def __getitem__(self, partition: str) -> DataLoader:
        if partition not in {"train", "validation", "test"}:
            raise KeyError(f"Unknown forecasting partition {partition!r}.")
        return getattr(self, partition)


@dataclass(frozen=True)
class ForecastingTrainValidationLoaders:
    """Train and validation loaders plus their shared graph, excluding test data."""

    train: DataLoader
    validation: DataLoader
    graph: ForecastGraph
    node_order: tuple[str, ...]


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ForecastDataLoaderError(f"{name} must be on CPU when validating graph metadata.")
        value = value.detach().numpy()
    return np.asarray(value)


def validate_forecasting_graph(
    graph: Mapping[str, Any],
    *,
    node_count: int,
    expected_node_order: tuple[str, ...] | list[str] | None = None,
) -> ForecastGraph:
    """Validate graph dimensions/indices without fabricating topology or weights."""
    if not isinstance(graph, Mapping):
        raise ForecastDataLoaderError("Graph data must be a mapping.")
    metadata = graph.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ForecastDataLoaderError("Graph provenance metadata is required.")
    node_order = metadata.get("node_order")
    if (
        not isinstance(node_order, list)
        or len(node_order) != node_count
        or any(not isinstance(node, str) or not node for node in node_order)
        or len(set(node_order)) != len(node_order)
    ):
        raise ForecastDataLoaderError("Graph node_order must uniquely match the dataset node count.")
    order = tuple(node_order)
    if expected_node_order is not None and order != tuple(expected_node_order):
        raise ForecastDataLoaderError("Graph node order does not match the dataset node order.")

    edge_index = _to_numpy(graph.get("edge_index"), "edge_index")
    if edge_index.dtype != np.int64 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ForecastDataLoaderError("edge_index must be int64 with shape [2, edges].")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= node_count):
        raise ForecastDataLoaderError("edge_index contains a node index outside the dataset.")

    attributes = _to_numpy(graph.get("node_attributes"), "node_attributes")
    if (
        attributes.dtype != np.float32
        or attributes.ndim != 2
        or attributes.shape[0] != node_count
        or attributes.shape[1] != 6
        or not np.isfinite(attributes).all()
    ):
        raise ForecastDataLoaderError(
            "node_attributes must be finite float32 [nodes, 6] values."
        )

    weight_key = "edge_weight" if "edge_weight" in graph else "edge_weights"
    edge_weight = None
    if weight_key in graph:
        weights = _to_numpy(graph[weight_key], weight_key)
        if (
            weights.dtype not in (np.float32, np.float64)
            or weights.shape != (edge_index.shape[1],)
            or not np.isfinite(weights).all()
        ):
            raise ForecastDataLoaderError(
                "Optional edge weights must be finite floating values with one value per edge."
            )
        edge_weight = torch.as_tensor(weights, dtype=torch.float32)

    copied_metadata = dict(metadata)
    if not copied_metadata.get("construction") or not copied_metadata.get("edge_weight_semantics"):
        raise ForecastDataLoaderError("Graph construction and edge-weight semantics must be documented.")
    return ForecastGraph(
        edge_index=torch.as_tensor(edge_index.copy(), dtype=torch.int64),
        node_attributes=torch.as_tensor(attributes.copy(), dtype=torch.float32),
        node_order=order,
        edge_weight=edge_weight,
        metadata=copied_metadata,
    )


def collate_forecasting_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack model tensors while preserving each sample's UTC timestamps."""
    if not samples:
        raise ForecastDataLoaderError("Cannot collate an empty forecasting batch.")
    tensor_keys = (
        "X",
        "global_X",
        "y",
        "node_input_mask",
        "target_mask",
    )
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    batch["input_timestamps"] = [sample["input_timestamps"] for sample in samples]
    batch["target_timestamps"] = [sample["target_timestamps"] for sample in samples]
    batch["metadata"] = [sample["metadata"] for sample in samples]
    return batch


def move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device | str
) -> dict[str, Any]:
    """Move tensor-valued batch fields to CPU/CUDA, retaining timestamp metadata."""
    target_device = torch.device(device)

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(target_device, non_blocking=target_device.type == "cuda")
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    return move(dict(batch))


def _validate_partition_graphs(
    datasets: Mapping[str, ForecastingDataset],
) -> ForecastGraph:
    first = datasets["train"]
    expected_order = first.node_order
    canonical = validate_forecasting_graph(
        first.graph, node_count=len(expected_order), expected_node_order=expected_order
    )
    canonical_edges = canonical.edge_index.numpy()
    canonical_attributes = canonical.node_attributes.numpy()
    for partition, dataset in datasets.items():
        if dataset.node_order != expected_order:
            raise ForecastDataLoaderError(f"{partition} node order differs from training.")
        current = validate_forecasting_graph(
            dataset.graph,
            node_count=len(expected_order),
            expected_node_order=expected_order,
        )
        if not np.array_equal(current.edge_index.numpy(), canonical_edges):
            raise ForecastDataLoaderError(f"{partition} graph edges differ from training.")
        if not np.array_equal(current.node_attributes.numpy(), canonical_attributes):
            raise ForecastDataLoaderError(f"{partition} node attributes differ from training.")
        if current.metadata != canonical.metadata:
            raise ForecastDataLoaderError(f"{partition} graph provenance differs from training.")
        if (current.edge_weight is None) != (canonical.edge_weight is None):
            raise ForecastDataLoaderError(f"{partition} edge-weight availability differs from training.")
        if current.edge_weight is not None and not torch.equal(current.edge_weight, canonical.edge_weight):
            raise ForecastDataLoaderError(f"{partition} edge weights differ from training.")
    return canonical


def build_forecasting_dataloaders(
    partitions: Mapping[str, Mapping[str, Any]],
    *,
    batch_size: int = 64,
    shuffle_train: bool = True,
    seed: int | None = 0,
    device: torch.device | str = "cpu",
    num_workers: int = 0,
) -> ForecastingDataLoaders:
    """Build deterministic evaluation and optionally shuffled training loaders.

    ``partitions`` must map train, validation, and test to independent
    ``prepare_sequences`` results. Their windows are already confined to each
    chronological split; no history is borrowed across split boundaries.
    """
    if set(partitions) != {"train", "validation", "test"}:
        raise ForecastDataLoaderError(
            "Provide exactly the train, validation, and test sequence partitions."
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ForecastDataLoaderError("batch_size must be a positive integer.")
    if type(shuffle_train) is not bool:
        raise ForecastDataLoaderError("shuffle_train must be a boolean.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ForecastDataLoaderError("seed must be an integer or None.")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ForecastDataLoaderError("num_workers must be a nonnegative integer.")
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ForecastDataLoaderError("device must be CPU or CUDA.")

    datasets = {
        partition: ForecastingDataset(
            partitions[partition], expected_partition=partition
        )
        for partition in ("train", "validation", "test")
    }
    graph = _validate_partition_graphs(datasets)
    loaders = {}
    for partition, dataset in datasets.items():
        generator = None
        if partition == "train" and seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        loaders[partition] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=partition == "train" and shuffle_train,
            generator=generator,
            num_workers=num_workers,
            pin_memory=target_device.type == "cuda",
            drop_last=False,
            collate_fn=collate_forecasting_batch,
        )
    return ForecastingDataLoaders(
        train=loaders["train"],
        validation=loaders["validation"],
        test=loaders["test"],
        graph=graph,
        node_order=graph.node_order,
    )


def build_forecasting_train_validation_dataloaders(
    partitions: Mapping[str, Mapping[str, Any]],
    *,
    batch_size: int = 64,
    seed: int | None = 0,
    device: torch.device | str = "cpu",
    num_workers: int = 0,
) -> ForecastingTrainValidationLoaders:
    """Build training/validation loaders without requiring or accessing test data."""
    if set(partitions) != {"train", "validation"}:
        raise ForecastDataLoaderError(
            "Provide exactly the train and validation sequence partitions."
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ForecastDataLoaderError("batch_size must be a positive integer.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ForecastDataLoaderError("seed must be an integer or None.")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ForecastDataLoaderError("num_workers must be a nonnegative integer.")
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ForecastDataLoaderError("device must be CPU or CUDA.")

    datasets = {
        partition: ForecastingDataset(
            partitions[partition], expected_partition=partition
        )
        for partition in ("train", "validation")
    }
    graph = _validate_partition_graphs(datasets)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=target_device.type == "cuda",
        drop_last=False,
        collate_fn=collate_forecasting_batch,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=target_device.type == "cuda",
        drop_last=False,
        collate_fn=collate_forecasting_batch,
    )
    return ForecastingTrainValidationLoaders(
        train=train_loader,
        validation=validation_loader,
        graph=graph,
        node_order=graph.node_order,
    )


def load_forecasting_train_validation_dataloaders(
    split_root: str | Path,
    *,
    batch_size: int = 64,
    seed: int | None = 0,
    device: torch.device | str = "cpu",
    num_workers: int = 0,
    sequence_config: Mapping[str, Any] | None = None,
) -> ForecastingTrainValidationLoaders:
    """Prepare only train/validation sequences using saved train-fitted scalers.

    Test files are neither loaded nor hashed by this path.
    """
    config = (
        dict(sequence_config)
        if sequence_config is not None
        else load_forecasting_config()
    )
    partitions = {}
    try:
        for partition in ("train", "validation"):
            sources = load_partition_sources(
                split_root,
                partition,
                verify_partitions=("train", "validation"),
            )
            partitions[partition] = prepare_sequences(sources, config)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise ForecastDataLoaderError(
            f"Cannot prepare train/validation sequences from the verified split bundle: {exc}"
        ) from exc
    return build_forecasting_train_validation_dataloaders(
        partitions,
        batch_size=batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
    )


def load_forecasting_dataloaders(
    split_root: str | Path,
    *,
    batch_size: int = 64,
    shuffle_train: bool = True,
    seed: int | None = 0,
    device: torch.device | str = "cpu",
    num_workers: int = 0,
    sequence_config: Mapping[str, Any] | None = None,
) -> ForecastingDataLoaders:
    """Verify a Part 2 split bundle and build loaders without refitting scalers."""
    config = (
        dict(sequence_config)
        if sequence_config is not None
        else load_forecasting_config()
    )
    partitions = {}
    try:
        for partition in ("train", "validation", "test"):
            sources = load_partition_sources(split_root, partition)
            partitions[partition] = prepare_sequences(sources, config)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise ForecastDataLoaderError(
            f"Cannot prepare forecasting sequences from the verified split bundle: {exc}"
        ) from exc
    return build_forecasting_dataloaders(
        partitions,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        seed=seed,
        device=device,
        num_workers=num_workers,
    )


def inverse_scale_with_training_scaler(
    values: np.ndarray,
    scaler: Mapping[str, Any],
    *,
    entity_id: str,
    feature: str,
) -> np.ndarray:
    """Invert an existing Part 2 train-fitted standard scaler for one entity.

    The current target arrays are already in physical kW and must not be
    transformed. This helper is for scaled historical features or a future
    model output only when it was explicitly scaled with the supplied artifact.
    """
    if not isinstance(scaler, Mapping) or (
        scaler.get("method") != "standard" or scaler.get("fitted_on") != "train"
    ):
        raise ForecastDataLoaderError("Expected an existing train-fitted standard scaler.")
    try:
        parameters = scaler["entities"][entity_id][feature]
        mean = float(parameters["mean"])
        scale = float(parameters["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ForecastDataLoaderError(
            f"No fitted scaler parameters for {entity_id}.{feature}."
        ) from exc
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise ForecastDataLoaderError("Scaler mean and positive scale must be finite.")
    array = np.asarray(values)
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not np.isfinite(array).all()
    ):
        raise ForecastDataLoaderError("Values to inverse-scale must be finite numeric values.")
    return array.astype(np.float64, copy=False) * scale + mean
