"""Train the Part 5.7 GNN-Transformer using train and validation partitions only.

Example:
    python scripts/train_forecaster.py --device cpu --epochs 2 \
        --batch-size 2 --max-train-batches 2 --max-validation-batches 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting.dataloader import load_forecasting_train_validation_dataloaders
from src.forecasting.models import GNNTransformerForecaster
from src.forecasting.training import (
    ForecastTrainingError,
    TrainingConfig,
    fit_forecaster,
    graph_topology_fingerprint,
    seed_everything,
    utc_now_iso,
    write_json,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ForecastTrainingError(f"Cannot load YAML configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ForecastTrainingError(f"Configuration must be a YAML mapping: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _configured_path(
    root: Path,
    *,
    explicit: str | Path | None,
    environment_name: str,
    default: str | Path,
) -> Path:
    value = explicit
    if value is None:
        value = os.environ.get(environment_name) or default
    return _resolve(root, value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_hash_from_manifest(split_root: Path, partition: str) -> str:
    manifest_path = split_root / "split_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksums = manifest["checksums"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ForecastTrainingError(
            f"Cannot read the verified split manifest: {exc}"
        ) from exc
    selected = {
        path: digest
        for path, digest in checksums.items()
        if Path(path).parts and Path(path).parts[0] == partition
    }
    if not selected:
        raise ForecastTrainingError(
            f"Split manifest contains no checksum entries for {partition}."
        )
    serialized = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_revision() -> str | None:
    value = os.environ.get("AEB_GIT_COMMIT_SHA")
    if value is None:
        return None
    revision = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ForecastTrainingError(
            "AEB_GIT_COMMIT_SHA must contain a full 40- or 64-character hexadecimal commit hash."
        )
    return revision


def _runtime_hardware(device: str) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    hardware: dict[str, Any] = {
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": None,
    }
    if device.startswith("cuda:") or device == "cuda":
        if not cuda_available:
            raise ForecastTrainingError(
                "CUDA was selected but is unavailable while collecting experiment metadata."
            )
        hardware["device_name"] = torch.cuda.get_device_name(torch.device(device))
    return hardware


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs/forecasting_training.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/forecasting_model.yaml")
    parser.add_argument("--split-root", type=Path, default=None)
    parser.add_argument("--sequence-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-id")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--min-delta", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--resume", type=Path)
    return parser


def _make_metadata(
    *,
    experiment_id: str,
    model_config: dict[str, Any],
    training_config: TrainingConfig,
    result: Any,
    loaders: Any,
    split_root: Path,
    model_config_path: Path,
    sequence_config_path: Path,
    preprocessing_path: Path,
    checkpoint_path: Path,
    resume_checkpoint_path: Path,
    history_path: Path,
    root: Path,
    started_at: str,
    source_revision: str | None,
) -> dict[str, Any]:
    source_metadata = loaders.train.dataset.metadata["source_metadata"]
    source_ids = sorted(source_metadata)
    origins = sorted(
        {
            entry["origin"]
            for entry in source_metadata.values()
            if isinstance(entry, dict) and entry.get("origin") in {"measured", "simulated", "derived"}
        }
    )
    if not source_ids or not origins:
        raise ForecastTrainingError("Dataset source IDs/origins are missing from provenance.")
    feature_columns = sorted(
        set(model_config["features"]["node_history"]["columns_by_node_type"].values())
    )
    if not feature_columns:
        raise ForecastTrainingError("Model configuration has no node-history feature columns.")
    split_manifest = split_root / "split_manifest.json"
    graph_hash = graph_topology_fingerprint(loaders.graph)
    data = {
        "dataset_manifest_sha256": _sha256_file(split_manifest),
        "source_ids": source_ids,
        "source_origins": origins,
        "training_partition_sha256": _partition_hash_from_manifest(split_root, "train"),
        "validation_partition_sha256": _partition_hash_from_manifest(split_root, "validation"),
        "feature_columns": feature_columns,
        "target_column": ",".join(model_config["targets"]["channels"]),
        "lookback_hours": int(model_config["data"]["lookback_hours"]),
        "horizon_hours": int(model_config["data"]["horizon_hours"]),
        "preprocessing_fit_partition": "train",
        "graph_topology_sha256": graph_hash,
        "graph_reference": model_config["graph"]["config"],
        "target_nodes": list(model_config["targets"]["nodes"]),
        "target_channels": list(model_config["targets"]["channels"]),
        "target_unit": model_config["targets"]["unit"],
        "preprocessing_artifacts": {
            "path": _portable_path(preprocessing_path, root),
            "sha256": _sha256_file(preprocessing_path),
        },
        "split_root": _portable_path(split_root, root),
    }
    training = {
        "epochs": len(result.history),
        "batch_size": training_config.batch_size,
        "optimizer": training_config.optimizer,
        "loss": training_config.loss,
        "selection_partition": "validation",
        "test_partition_used_for_selection": False,
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "patience": training_config.patience,
        "min_delta": training_config.min_delta,
        "gradient_clip_norm": training_config.gradient_clip_norm,
        "max_epochs": training_config.max_epochs,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "duration_seconds": result.duration_seconds,
        "stop_reason": result.stop_reason,
        "checkpoint_path": _portable_path(checkpoint_path, root),
        "resume_checkpoint_path": _portable_path(resume_checkpoint_path, root),
        "history_path": _portable_path(history_path, root),
    }
    software = {
        "python_version": platform.python_version(),
        "framework": "PyTorch",
        "framework_version": torch.__version__,
        "device": result.device,
        "deterministic": training_config.deterministic,
        "package_versions": {
            "numpy": np.__version__,
            "PyYAML": yaml.__version__,
        },
        "hardware": _runtime_hardware(result.device),
    }
    if source_revision is not None:
        software["source_revision"] = source_revision
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "phase": "Part_5.8_training_validation",
        "status": "completed",
        "created_at_utc": started_at,
        "seed": training_config.seed,
        "data": data,
        "model": {
            "name": "GNN-Transformer",
            "architecture": {
                "type": "gnn_transformer",
                "graph_topology_sha256": graph_hash,
                "node_order": list(loaders.node_order),
            },
            "configuration": model_config,
        },
        "training": training,
        "software": software,
        "metrics": {"best_validation_masked_mse_kw2": result.best_validation_loss},
        "artifacts": {
            "best_checkpoint_sha256": _sha256_file(checkpoint_path),
            "resume_checkpoint_sha256": _sha256_file(resume_checkpoint_path),
            "training_history_sha256": _sha256_file(history_path),
            "preprocessing_sha256": _sha256_file(preprocessing_path),
            "model_config_sha256": _sha256_file(model_config_path),
            "sequence_config_sha256": _sha256_file(sequence_config_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_revision = _source_revision()
    training_config_path = _resolve(ROOT, args.training_config)
    model_config_path = _resolve(ROOT, args.model_config)
    training_yaml = _read_yaml(training_config_path)
    model_config = _read_yaml(model_config_path)
    training_config = TrainingConfig.from_mapping(training_yaml)
    overrides = {
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "seed": args.seed,
        "device": args.device,
    }
    training_config = replace(
        training_config,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    training_config.validate()
    paths = training_yaml.get("paths")
    if not isinstance(paths, dict):
        raise ForecastTrainingError("Training configuration requires a paths mapping.")
    split_root = _configured_path(
        ROOT,
        explicit=args.split_root,
        environment_name="AEB_SPLIT_ROOT",
        default=model_config["data"]["split_bundle"],
    )
    sequence_config_path = _resolve(
        ROOT, args.sequence_config or model_config["data"]["sequence_config"]
    )
    sequence_config = _read_yaml(sequence_config_path)
    results_root = _configured_path(
        ROOT,
        explicit=args.output_dir,
        environment_name="AEB_EXPERIMENTS_ROOT",
        default=paths["results_dir"],
    )
    checkpoint_root_value = os.environ.get("AEB_CHECKPOINTS_ROOT")
    checkpoint_root = (
        _resolve(ROOT, checkpoint_root_value)
        if checkpoint_root_value
        else None
    )
    resume = args.resume.resolve() if args.resume is not None else None
    if resume is not None:
        experiment_id = args.experiment_id or resume.parent.name
        has_configured_storage = bool(
            args.output_dir
            or os.environ.get("AEB_EXPERIMENTS_ROOT")
            or checkpoint_root
        )
        experiment_dir = (
            results_root / experiment_id
            if has_configured_storage
            else resume.parent
        )
        checkpoint_dir = resume.parent
    else:
        experiment_id = args.experiment_id or (
            "part5_8-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        if not experiment_id or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for char in experiment_id):
            raise ForecastTrainingError("experiment_id contains unsupported characters.")
        experiment_dir = results_root / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_dir = (
            checkpoint_root / experiment_id
            if checkpoint_root is not None
            else experiment_dir
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / paths["best_checkpoint_name"]
    resume_path = (
        resume
        if resume is not None
        else checkpoint_dir / paths["resume_checkpoint_name"]
    )
    history_path = experiment_dir / paths["history_filename"]
    metadata_path = experiment_dir / paths["metadata_filename"]
    preprocessing_path = split_root / "preprocessing.json"
    if not preprocessing_path.is_file():
        raise ForecastTrainingError(f"Preprocessing artifact not found: {preprocessing_path}")
    started_at = utc_now_iso()
    if resume is not None and history_path.is_file():
        try:
            previous_history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ForecastTrainingError(f"Cannot read existing training history: {exc}") from exc
        if not isinstance(previous_history, dict) or previous_history.get("experiment_id") != experiment_id:
            raise ForecastTrainingError("Existing history belongs to another experiment.")
        if isinstance(previous_history.get("started_at_utc"), str):
            started_at = previous_history["started_at_utc"]

    seed_everything(training_config.seed, deterministic=training_config.deterministic)
    loaders = load_forecasting_train_validation_dataloaders(
        split_root,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        device=training_config.device if training_config.device != "auto" else "cpu",
        sequence_config=sequence_config,
    )
    model = GNNTransformerForecaster.from_config(model_config)
    if model.graph_node_ids != loaders.node_order:
        raise ForecastTrainingError("Configured model node order differs from loaded train/validation graph.")
    for partition, loader in (("train", loaders.train), ("validation", loaders.validation)):
        sample = loader.dataset[0]
        if (
            sample["X"].shape[0] != model.lookback_hours
            or sample["X"].shape[-1] != model.gnn_encoder.input_dim
            or sample["y"].shape[0] != model.forecast_horizon
        ):
            raise ForecastTrainingError(
                f"Configured model dimensions do not match {partition} input/target tensors."
            )
        if loader.dataset.metadata.get("target_unit") != model_config["targets"]["unit"]:
            raise ForecastTrainingError(
                f"Configured target unit does not match {partition} dataset metadata."
            )
        mask_nodes = {
            node
            for node, valid in zip(loaders.node_order, sample["target_mask"][:, 0].tolist())
            if valid
        }
        if mask_nodes != set(model.target_node_ids):
            raise ForecastTrainingError(
                f"Configured target-node order/set differs from {partition} target mask."
            )
        if sample["y"].shape[-1] != len(model.target_channels):
            raise ForecastTrainingError(
                f"Configured target channels do not match {partition} target tensor."
            )
    def report(epoch: int, train_loss: float, validation_loss: float, improved: bool) -> None:
        marker = " best" if improved else ""
        print(
            f"epoch={epoch} train_masked_mse={train_loss:.6f} "
            f"validation_masked_mse={validation_loss:.6f}{marker}",
            flush=True,
        )

    result = fit_forecaster(
        model,
        train_loader=loaders.train,
        validation_loader=loaders.validation,
        graph=loaders.graph,
        config=training_config,
        experiment_id=experiment_id,
        checkpoint_path=best_path,
        resume_checkpoint_path=resume_path,
        resume_from=resume,
        preprocessing_artifacts={
            "preprocessing": _portable_path(preprocessing_path, ROOT),
            "split_manifest": _portable_path(split_root / "split_manifest.json", ROOT),
        },
        target_unit=model_config["targets"]["unit"],
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        progress_callback=report,
    )
    history_payload = {
        "experiment_id": experiment_id,
        "started_at_utc": started_at,
        "status": "completed",
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "stop_reason": result.stop_reason,
        "epochs": result.history,
        "duration_seconds": result.duration_seconds,
        "device": result.device,
        "test_partition_used": False,
    }
    write_json(history_path, history_payload)
    metadata = _make_metadata(
        experiment_id=experiment_id,
        model_config=model_config,
        training_config=training_config,
        result=result,
        loaders=loaders,
        split_root=split_root,
        model_config_path=model_config_path,
        sequence_config_path=sequence_config_path,
        preprocessing_path=preprocessing_path,
        checkpoint_path=best_path,
        resume_checkpoint_path=resume_path,
        history_path=history_path,
        root=ROOT,
        started_at=started_at,
        source_revision=source_revision,
    )
    write_json(metadata_path, metadata)
    print(f"best_epoch={result.best_epoch} best_validation_loss={result.best_validation_loss:.6f}")
    print(f"checkpoint={best_path}")
    print(f"resume_checkpoint={resume_path}")
    print(f"history={history_path}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
