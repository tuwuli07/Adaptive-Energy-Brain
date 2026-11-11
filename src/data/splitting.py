"""Chronological, entity-aligned splits and train-only standard scaling.

python -m src.data.splitting --manifest PATH --output NAME
Outputs live in data/processed/splits_NAME, with separate train/validation/test
directories, metadata, preprocessing.json and a checksum manifest. All original
columns remain; *_scaled features are dimensionless. A saved scaler can be
loaded with json.load and reused via apply_scaler without fitting anything.
"""

import argparse
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from src.data.schema import SchemaValidationError, validate_dataset


ROOT = Path(__file__).resolve().parents[2]
ENTITY = {"demand": "building_id", "generation": "asset_id", "weather": "site_id", "price": "grid_id"}
PARTITIONS = ("train", "validation", "test")


class SplittingError(ValueError):
    """Unsafe split, preprocessing leakage or incompatible inference input."""


def load_split_config(path=ROOT / "configs/splitting.yaml"):
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SplittingError(f"Cannot load splitting configuration: {exc}") from exc
    return config


def _times(values):
    result = []
    for value in values:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:00:00(?:Z|\+00:00)", value):
            raise SplittingError("Timestamps must be hourly ISO UTC strings, Z or +00:00.")
        try:
            result.append(pd.Timestamp(value))
        except ValueError as exc:
            raise SplittingError(f"Invalid UTC timestamp {value!r}.") from exc
    return pd.DatetimeIndex(result).tz_convert("UTC")


def _iso(stamp):
    return stamp.isoformat().replace("+00:00", "Z")


def fit_scaler(training_frame, features, entity_column, *, suffix="_scaled"):
    """Fit population mean/std on an explicitly supplied training frame only."""
    if not features or len(features) != len(set(features)):
        raise SplittingError("Scaling features must be a nonempty unique list.")
    if entity_column not in training_frame or any(feature not in training_frame for feature in features):
        raise SplittingError("Training data is missing entity or configured feature columns.")
    groups = {}
    for entity, group in training_frame.groupby(entity_column, sort=True):
        parameters = {}
        for feature in features:
            try:
                values = pd.to_numeric(group[feature], errors="raise").to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise SplittingError(f"Non-numeric training feature {feature}.") from exc
            if not len(values) or not np.isfinite(values).all():
                raise SplittingError(f"Training feature {feature} contains missing/non-finite values.")
            mean, std = float(values.mean()), float(values.std(ddof=0))
            if not math.isfinite(mean) or not math.isfinite(std):
                raise SplittingError("Training feature magnitude overflows standard scaling.")
            parameters[feature] = {"mean": mean, "scale": std if std > 0 else 1.0,
                                   "constant": std == 0, "count": len(values)}
        groups[str(entity)] = parameters
    if not groups:
        raise SplittingError("Cannot fit an empty training partition.")
    return {"method": "standard", "variance_ddof": 0, "entity_column": entity_column,
            "features": list(features), "suffix": suffix, "entities": groups,
            "fitted_on": "train"}


def apply_scaler(frame, parameters):
    """Apply saved train-fitted parameters to a copy, without refitting/clipping."""
    if parameters.get("method") != "standard" or parameters.get("fitted_on") != "train":
        raise SplittingError("Expected train-fitted standard-scaling parameters.")
    data = frame.copy(deep=True)
    entity_column = parameters["entity_column"]
    if entity_column not in data:
        raise SplittingError(f"Inference data is missing {entity_column}.")
    for feature in parameters["features"]:
        output = feature + parameters["suffix"]
        if feature not in data or output in data:
            raise SplittingError(f"Missing feature or scaled-column collision: {feature}.")
        data[output] = np.nan
    for entity, group in data.groupby(entity_column, sort=False):
        if str(entity) not in parameters["entities"]:
            raise SplittingError(f"Entity {entity!r} has no training-fitted scaler.")
        for feature, fitted in parameters["entities"][str(entity)].items():
            values = pd.to_numeric(group[feature], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise SplittingError(f"{feature} contains missing/non-finite values; no imputation is implicit.")
            if not math.isfinite(fitted["scale"]) or fitted["scale"] <= 0 or not math.isfinite(fitted["mean"]):
                raise SplittingError("Invalid stored scaler parameters.")
            transformed = (values - fitted["mean"]) / fitted["scale"]
            if not np.isfinite(transformed).all():
                raise SplittingError("Transformed values overflow; no clipping is performed.")
            data.loc[group.index, feature + parameters["suffix"]] = transformed
    return data


def split_datasets(sources, config=None):
    """Split compatible canonical source payloads using one shared time grid.

    Returns partition source lists plus preprocessing parameters and boundaries.
    Ratios apply to distinct hours, never row counts. All entities must span the
    same complete hourly grid. Known future-endpoint cleaning is refused; hidden
    upstream transformations cannot be detected and remain a provenance duty.
    """
    config = deepcopy(config) if config is not None else load_split_config()
    try:
        ratios = config["fractions"]
        scaling = config["scaling"]
        if config["version"] != 1 or set(ratios) != set(PARTITIONS):
            raise SplittingError("Expected version 1 and train/validation/test fractions.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1 for value in ratios.values()) or not math.isclose(sum(ratios.values()), 1, abs_tol=1e-12):
            raise SplittingError("Positive split fractions must sum to one.")
        if scaling["method"] != "standard" or scaling["scope"] != "source_and_entity" or scaling["variance_ddof"] != 0 or scaling["constant_feature_scale"] != 1:
            raise SplittingError("Supported scaling is per-source/entity standard scaling, ddof=0, constant scale=1.")
        if not re.fullmatch(r"_[A-Za-z][A-Za-z0-9_]*", scaling["output_suffix"]):
            raise SplittingError("Scaled columns require an explicit safe suffix.")
    except (KeyError, TypeError) as exc:
        raise SplittingError(f"Invalid splitting configuration: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise SplittingError("No input datasets supplied.")
    prepared, reference, ids = [], None, set()
    for source in sources:
        source_id, kind = source["source_id"], source["record_type"]
        if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", source_id) or source_id in ids or kind not in ENTITY:
            raise SplittingError("Sources require unique safe IDs and supported observation record types.")
        ids.add(source_id)
        cleaning = source["metadata"].get("cleaning", {}).get("policy", {}).get("missing", {}).get("columns", {})
        operations = source.get("report", {}).get("operations", [])
        if any(rule.get("method") == "interpolate" for rule in cleaning.values()) or any(operation.get("uses_future_endpoint") for operation in operations):
            raise SplittingError("Pre-split future-endpoint interpolation is unsafe; split before fitting/filling.")
        frame = source["data"].copy(deep=True).reset_index(drop=True)
        try:
            validate_dataset(frame.to_dict("records"), kind, units=source["units"], metadata=source["metadata"])
        except SchemaValidationError as exc:
            raise SplittingError(f"{source_id}: canonical input validation failed: {exc}") from exc
        frame["timestamp_utc"] = [_iso(stamp) for stamp in _times(frame["timestamp_utc"])]
        entity_column = ENTITY[kind]
        for _, group in frame.groupby(entity_column, sort=True):
            times = _times(group["timestamp_utc"]).sort_values()
            if times.has_duplicates or not (times[1:] - times[:-1] == pd.Timedelta(hours=1)).all():
                raise SplittingError(f"{source_id}: duplicate or missing entity intervals; repair integration first.")
            if reference is None:
                reference = times
            elif not reference.equals(times):
                raise SplittingError("All source entities must share the identical hourly timeline.")
        prepared.append((source, frame.sort_values(["timestamp_utc", entity_column]).reset_index(drop=True)))
    size = len(reference)
    train_count = int(Decimal(size) * Decimal(str(ratios["train"])))
    validation_count = int(Decimal(size) * Decimal(str(ratios["validation"])))
    counts = [train_count, validation_count, size - train_count - validation_count]
    if min(counts) < 1:
        raise SplittingError("Not enough distinct timestamps for three nonempty partitions.")
    offsets = [0, counts[0], counts[0] + counts[1], size]
    boundaries = {}
    for index, partition in enumerate(PARTITIONS):
        times = reference[offsets[index]:offsets[index + 1]]
        boundaries[partition] = {"start_utc": _iso(times[0]), "end_exclusive_utc": _iso(times[-1] + pd.Timedelta(hours=1)), "hours": len(times)}
    result = {"partitions": {partition: [] for partition in PARTITIONS},
              "preprocessing": {"version": 1, "fitted_on": "train", "boundaries": boundaries,
                                "config": config, "scalers": {}, "source_metadata": {}}}
    for source, frame in prepared:
        source_id, kind = source["source_id"], source["record_type"]
        features = scaling["features"].get(kind)
        # Only observed physical columns in the canonical input are eligible.
        numeric = set(source["units"]) - {"interval_hours"}
        if not isinstance(features, list) or not features or not set(features) <= numeric:
            raise SplittingError(f"{source_id}: configure explicit available physical features.")
        train = frame[(frame["timestamp_utc"] >= boundaries["train"]["start_utc"]) & (frame["timestamp_utc"] < boundaries["train"]["end_exclusive_utc"])]
        scaler = fit_scaler(train, features, ENTITY[kind], suffix=scaling["output_suffix"])
        scaler["training_boundaries"] = boundaries["train"]
        result["preprocessing"]["scalers"][source_id] = scaler
        result["preprocessing"]["source_metadata"][source_id] = deepcopy(source["metadata"])
        for partition in PARTITIONS:
            bounds = boundaries[partition]
            subset = frame[(frame["timestamp_utc"] >= bounds["start_utc"]) & (frame["timestamp_utc"] < bounds["end_exclusive_utc"])].reset_index(drop=True)
            payload = {key: deepcopy(value) for key, value in source.items() if key != "data"}
            payload["data"] = apply_scaler(subset, scaler)
            payload["metadata"]["split"] = {"name": partition, "boundaries": bounds, "fitted_on": "train",
                                            "scaled_units": {feature + scaler["suffix"]: "1" for feature in features}}
            result["partitions"][partition].append(payload)
    return result


def build_windows(partition_source, *, features, target, lookback, horizon):
    """Build within-partition historical inputs and strictly later targets.

    No previous partition context is borrowed. Each window's complete target
    horizon stays in its partition; boundary-crossing windows are never created.
    Targets remain original values unless a scaled column is explicitly chosen.
    """
    split = partition_source["metadata"].get("split")
    if not split or split["name"] not in PARTITIONS:
        raise SplittingError("Windows require a labeled partition, not an unsplit dataset.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (lookback, horizon)):
        raise SplittingError("lookback and horizon must be positive integers.")
    frame = partition_source["data"]
    if not features or not set(features + [target]) <= set(frame):
        raise SplittingError("Window features/target are missing.")
    bounds = split["boundaries"]
    if not ((frame["timestamp_utc"] >= bounds["start_utc"]) & (frame["timestamp_utc"] < bounds["end_exclusive_utc"])).all():
        raise SplittingError("Partition contains out-of-boundary rows.")
    windows = []
    for entity, group in frame.groupby(ENTITY[partition_source["record_type"]], sort=True):
        ordered = group.sort_values("timestamp_utc")
        times = _times(ordered["timestamp_utc"])
        if times.has_duplicates or not (times[1:] - times[:-1] == pd.Timedelta(hours=1)).all():
            raise SplittingError("Window construction requires continuous unique hourly observations.")
        for index in range(lookback, len(ordered) - horizon + 1):
            inputs, targets = ordered.iloc[index - lookback:index], ordered.iloc[index:index + horizon]
            windows.append({"entity_id": entity, "inputs": inputs[features].to_numpy(dtype=float),
                            "targets": targets[target].to_numpy(dtype=float),
                            "input_timestamps": inputs["timestamp_utc"].tolist(),
                            "target_timestamps": targets["timestamp_utc"].tolist()})
    return windows


def save_splits(result, name, *, processed_root=ROOT / "data/processed"):
    """Save separate partitions once; checksum all files, including held-out test."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise SplittingError("Output name must be a safe basename.")
    root = Path(processed_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"splits_{name}"
    try:
        destination.mkdir()
    except OSError as exc:
        raise SplittingError(f"Refusing existing/inaccessible split bundle: {destination}") from exc
    created, directories = [], []
    try:
        hashes = {}

        def write_json(path, payload):
            created.append(path)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")

        for partition in PARTITIONS:
            directory = destination / partition
            directory.mkdir()
            directories.append(directory)
            for source in result["partitions"][partition]:
                path = directory / f"{source['source_id']}.csv"
                created.append(path)
                source["data"].to_csv(path, index=False, mode="x")
                write_json(path.with_suffix(".metadata.json"), {key: value for key, value in source.items() if key != "data"})
        write_json(destination / "preprocessing.json", result["preprocessing"])
        for path in created:
            hashes[str(path.relative_to(destination)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(destination / "split_manifest.json", {"boundaries": result["preprocessing"]["boundaries"],
                   "checksums": hashes, "test_policy": "Held out for final evaluation; never refit or tune on test."})
        return destination
    except Exception as exc:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        for directory in reversed(directories):
            directory.rmdir()
        destination.rmdir()
        raise SplittingError(f"Cannot save split bundle: {exc}") from exc


def verify_split_bundle(path, *, partitions=PARTITIONS):
    """Verify selected split artifacts without loading or hashing other splits."""
    selected = tuple(partitions)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(partition not in PARTITIONS for partition in selected)
    ):
        raise SplittingError("partitions must be a nonempty unique subset of train/validation/test.")
    root = Path(path).resolve()
    try:
        manifest = json.loads((root / "split_manifest.json").read_text())
        for relative, digest in manifest["checksums"].items():
            path_parts = Path(relative).parts
            if relative != "preprocessing.json" and (
                not path_parts or path_parts[0] not in selected
            ):
                continue
            target = (root / relative).resolve()
            if root not in target.parents or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise SplittingError(f"Split artifact missing/changed: {relative}.")
        return True
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, SplittingError):
            raise
        raise SplittingError(f"Cannot verify split bundle: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/splitting.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
        sources = []
        for item in manifest["datasets"]:
            source = deepcopy(item)
            path = (args.manifest.resolve().parent / source.pop("path")).resolve()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                stream.seek(0)
                source["data"] = pd.read_csv(stream, dtype={column: str for column in ENTITY.values()})
            source["metadata"]["split_input"] = {"path": str(path), "sha256": digest}
            sources.append(source)
        result = split_datasets(sources, load_split_config(args.config))
        destination = save_splits(result, args.output)
        verify_split_bundle(destination)
        print(json.dumps(result["preprocessing"]["boundaries"], indent=2))
        print(f"Saved split bundle: {destination}")
        return 0
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"Splitting error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
