"""Leakage-safe NumPy graph sequences using already-split observation sources.

Run: python -m src.data.sequences --splits data/processed/splits_NAME --output NAME
No fitting, learning, forecast-vintage mixing or cross-partition history occurs.
PyTorch can later use torch.from_numpy; PyG can consume int64 edge_index and
per-time-step node arrays. Masks must be respected for non-target context nodes.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data.graph_builder import build_graph, load_forecasting_config
from src.data.splitting import ENTITY, PARTITIONS, verify_split_bundle


ROOT = Path(__file__).resolve().parents[2]


class SequenceDataError(ValueError):
    """Misaligned observations, leakage or incompatible sequence configuration."""


def _times(values):
    if not all(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:00:00(?:Z|\+00:00)", value) for value in values):
        raise SequenceDataError("Sequence timestamps must be explicit hourly UTC strings.")
    try:
        result = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    except ValueError as exc:
        raise SequenceDataError(f"Invalid sequence timestamps: {exc}") from exc
    if result.has_duplicates or not result.is_monotonic_increasing or not (result[1:] - result[:-1] == pd.Timedelta(hours=1)).all():
        raise SequenceDataError("Sequences require a continuous unique hourly timeline.")
    return result


def prepare_sequences(sources, config=None):
    """Prepare one labeled partition with fixed graph-node ordering.

    X: [sample, lookback, node, 1], y: [sample, horizon, node, 1] in kW,
    global_X: [sample, lookback, global_feature]. node_input_mask/target_mask
    have shape [node, 1] and broadcast across sample/time axes. Structural bus
    zero placeholders are masked; missing observations are never zero-filled.
    Timestamps are datetime64[ns] UTC by convention (recorded in metadata).
    """
    config = deepcopy(config) if config is not None else load_forecasting_config()
    for field in ("lookback_hours", "horizon_hours", "stride_hours"):
        if isinstance(config.get(field), bool) or not isinstance(config.get(field), int) or config[field] <= 0:
            raise SequenceDataError(f"{field} must be a positive integer.")
    if config.get("node_input_features") != ["historical_power_scaled"] or config.get("node_targets") != ["power_kw"]:
        raise SequenceDataError("This initial layout supports one historical scaled-power feature and one kW target.")
    if not sources:
        raise SequenceDataError("No partition observations supplied.")
    graph = build_graph(config)
    nodes = graph["metadata"]["nodes"]
    labels, bounds, seen_ids = set(), [], set()
    for source in sources:
        if source["source_id"] in seen_ids:
            raise SequenceDataError("Duplicate source IDs are ambiguous.")
        seen_ids.add(source["source_id"])
        split = source["metadata"].get("split", {})
        if split.get("name") not in PARTITIONS or split.get("fitted_on") != "train":
            raise SequenceDataError("Only labeled partitions with train-fitted preprocessing are allowed.")
        labels.add(split["name"])
        bounds.append(split.get("boundaries"))
    if len(labels) != 1 or any(bound != bounds[0] for bound in bounds) or not isinstance(bounds[0], dict):
        raise SequenceDataError("Do not mix temporal splits or different split boundaries.")
    split_bounds, reference = bounds[0], None
    provenance = {}

    def select(kind, entity, column, *, unit, generation_type=None):
        nonlocal reference
        if kind not in ENTITY:
            raise SequenceDataError("Only historical observations are supported; future residual/forecast inputs are excluded.")
        candidates = [source for source in sources if source["record_type"] == kind
                      and ENTITY[kind] in source["data"]
                      and (source["data"][ENTITY[kind]] == entity).any()]
        if len(candidates) != 1:
            raise SequenceDataError(f"Expected exactly one {kind} source for entity {entity!r}.")
        source = candidates[0]
        frame = source["data"]
        group = frame[frame[ENTITY[kind]] == entity].sort_values("timestamp_utc")
        if generation_type is not None and ("generation_type" not in group or not (group["generation_type"] == generation_type).all()):
            raise SequenceDataError(f"Renewable node {entity} has inconsistent generation_type.")
        times = _times(group["timestamp_utc"].tolist())
        if not len(times):
            raise SequenceDataError("An observation entity has no rows.")
        try:
            start, end = pd.Timestamp(split_bounds["start_utc"]), pd.Timestamp(split_bounds["end_exclusive_utc"])
        except (KeyError, ValueError) as exc:
            raise SequenceDataError("Invalid split boundaries.") from exc
        if start.tzinfo is None or end.tzinfo is None or times[0] < start or times[-1] >= end:
            raise SequenceDataError("Observation timestamps fall outside their temporal split.")
        if times[0] != start or times[-1] + pd.Timedelta(hours=1) != end:
            raise SequenceDataError("Entity does not span the complete declared partition.")
        if reference is None:
            reference = times
        elif not reference.equals(times):
            raise SequenceDataError("Graph nodes and global covariates must use the same timeline.")
        if column not in group:
            raise SequenceDataError(f"Missing observed column {column} for {entity}.")
        declared = {**source["units"], **source["metadata"]["split"].get("scaled_units", {})}
        if declared.get(column) != unit:
            raise SequenceDataError(f"{entity}.{column}: expected unit {unit!r}.")
        if column.endswith("_scaled") and column not in source["metadata"]["split"].get("scaled_units", {}):
            raise SequenceDataError("Scaled features require declared train-fitted preprocessing.")
        try:
            values = pd.to_numeric(group[column], errors="raise").to_numpy(dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise SequenceDataError(f"Non-numeric sequence values in {entity}.{column}.") from exc
        if not np.isfinite(values).all():
            raise SequenceDataError(f"Missing/non-finite observations in {entity}.{column}; no implicit fills.")
        provenance[source["source_id"]] = deepcopy(source["metadata"])
        return values

    histories, targets = [], []
    node_mask = np.zeros((len(nodes), 1), dtype=bool)
    for index, node in enumerate(nodes):
        if node["type"] == "bus":
            histories.append(None)
            targets.append(None)
            continue
        expected = {"building": ("demand", "building_load_kw"), "solar": ("generation", "available_generation_kw"), "wind": ("generation", "available_generation_kw")}[node["type"]]
        if (node.get("record_type"), node.get("target_column")) != expected:
            raise SequenceDataError("Node targets must be observed building demand or available renewable power.")
        if node.get("input_column") != expected[1] + "_scaled":
            raise SequenceDataError("Node history must be the train-scaled historical observation, not future information.")
        generation_type = node["type"] if node["type"] in {"solar", "wind"} else None
        histories.append(select(node["record_type"], node["entity_id"], node["input_column"], unit="1", generation_type=generation_type))
        targets.append(select(node["record_type"], node["entity_id"], node["target_column"], unit="kW", generation_type=generation_type))
        if (targets[-1] < 0).any():
            raise SequenceDataError("Demand/available generation targets cannot be negative.")
        node_mask[index] = True
    if reference is None:
        raise SequenceDataError("Graph must contain at least one forecast target node.")
    globals_list = []
    allowed_weather = {"irradiance_w_m2_scaled", "air_temperature_c_scaled", "wind_speed_m_s_scaled", "relative_humidity_fraction_scaled"}
    for feature in config.get("global_features", []):
        if feature.get("record_type") != "weather" or feature.get("column") not in allowed_weather:
            raise SequenceDataError("Global inputs are restricted to historical scaled weather observations.")
        globals_list.append(select("weather", feature["entity_id"], feature["column"], unit="1"))
    total, lookback, horizon, stride = len(reference), config["lookback_hours"], config["horizon_hours"], config["stride_hours"]
    if total < lookback + horizon:
        raise SequenceDataError("Partition is shorter than lookback + horizon; no valid windows.")
    history = np.stack([np.zeros(total, dtype=np.float32) if values is None else values for values in histories], axis=1)[..., None]
    target = np.stack([np.zeros(total, dtype=np.float32) if values is None else values for values in targets], axis=1)[..., None]
    global_history = np.stack(globals_list, axis=1) if globals_list else np.empty((total, 0), dtype=np.float32)
    starts = range(lookback, total - horizon + 1, stride)
    times = reference.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    arrays = {
        "X": np.stack([history[index - lookback:index] for index in starts]),
        "y": np.stack([target[index:index + horizon] for index in starts]),
        "global_X": np.stack([global_history[index - lookback:index] for index in starts]),
        "input_timestamps": np.stack([times[index - lookback:index] for index in starts]),
        "target_timestamps": np.stack([times[index:index + horizon] for index in starts]),
        "node_input_mask": node_mask, "target_mask": node_mask.copy(),
    }
    if not (arrays["input_timestamps"][:, -1] < arrays["target_timestamps"][:, 0]).all():
        raise SequenceDataError("Internal error: target information overlaps input history.")
    return {"arrays": arrays, "graph": graph, "metadata": {
        "partition": next(iter(labels)), "boundaries": split_bounds, "config": config,
        "node_order": graph["metadata"]["node_order"], "target_unit": "kW",
        "timestamp_convention": "datetime64[ns] UTC; source timestamps preserved by instant",
        "source_metadata": provenance, "sample_count": len(arrays["X"]),
        "input_features": config["node_input_features"], "targets": config["node_targets"],
        "mask_policy": config.get("mask_policy"),
    }}


def load_partition_sources(
    split_root, partition, *, verify_partitions=PARTITIONS
):
    """Read one partition and verify selected split hashes without fitting."""
    if partition not in PARTITIONS:
        raise SequenceDataError("Unknown partition name.")
    root = Path(split_root)
    verify_split_bundle(root, partitions=verify_partitions)
    result = []
    for path in sorted((root / partition).glob("*.metadata.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        csv_path = path.with_name(path.name.replace(".metadata.json", ".csv"))
        payload["data"] = pd.read_csv(csv_path, dtype={column: str for column in ENTITY.values()})
        result.append(payload)
    if not result:
        raise SequenceDataError("No partition files found.")
    return result


def save_forecasting_bundle(split_root, name, config=None, *, processed_root=ROOT / "data/processed"):
    """Save NPZ tensors and JSON provenance once, without overwriting splits.

    No optimizer/model is used. Test tensors remain a separate held-out artifact.
    JSON/NPZ outputs avoid pickle; np.load(..., allow_pickle=False) is sufficient.
    """
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise SequenceDataError("Output name must be a safe basename.")
    config = deepcopy(config) if config is not None else load_forecasting_config()
    root = Path(processed_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"forecasting_{name}"
    try:
        destination.mkdir()
    except OSError as exc:
        raise SequenceDataError(f"Refusing existing/inaccessible output bundle: {destination}") from exc
    created, hashes = [], {}
    try:
        for partition in PARTITIONS:
            result = prepare_sequences(load_partition_sources(split_root, partition), config)
            path = destination / f"{partition}.npz"
            created.append(path)
            with path.open("xb") as stream:
                np.savez_compressed(stream, **result["arrays"])
            metadata_path = destination / f"{partition}.metadata.json"
            created.append(metadata_path)
            with metadata_path.open("x", encoding="utf-8") as stream:
                json.dump(result["metadata"], stream, indent=2, allow_nan=False)
            if partition == "train":
                graph_path = destination / "graph.npz"
                created.append(graph_path)
                with graph_path.open("xb") as stream:
                    np.savez_compressed(stream, edge_index=result["graph"]["edge_index"],
                                        node_attributes=result["graph"]["node_attributes"],
                                        node_ids=np.asarray(result["graph"]["metadata"]["node_order"]))
                graph_meta = destination / "graph.metadata.json"
                created.append(graph_meta)
                with graph_meta.open("x", encoding="utf-8") as stream:
                    json.dump(result["graph"]["metadata"], stream, indent=2, allow_nan=False)
        for path in created:
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = destination / "manifest.json"
        created.append(manifest)
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump({"split_source": str(Path(split_root).resolve()), "checksums": hashes,
                       "node_order": result["metadata"]["node_order"],
                       "test_policy": "Final evaluation only; no model development or fitting performed."}, stream, indent=2)
        return destination
    except Exception as exc:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        destination.rmdir()
        raise SequenceDataError(f"Forecasting bundle not saved: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_forecasting_config(args.config) if args.config else load_forecasting_config()
        print(save_forecasting_bundle(args.splits, args.output, config))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Forecasting data error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
