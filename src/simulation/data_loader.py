"""Read-only hourly physical observations for a future simulator, not an agent.

load_simulation_data(manifest_path, start=..., end=...) reads a Part 2 datasets
manifest. load_simulation_split(split_root, partition, ...) reuses the existing
split loader/hash verification. prepare_simulation_data accepts source payloads
in memory. Selection is [start, end), strictly hourly UTC with full coverage.

Canonical names/validators and compatible unit conversions come from src/data.
No filling, resampling, model inference or dispatch occurs. Realized replay rows
are internal simulator inputs and must not be exposed as future RL observations.
Missing optional export prices stay None; an engine must disable export or
require an explicitly configured tariff before executing export transactions.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

import pandas as pd
import yaml

from src.data.cleaning import CleaningError, unit_conversion
from src.data.schema import SchemaValidationError, load_schema, validate_dataset
from src.data.sequences import load_partition_sources


class SimulationDataError(ValueError):
    """Missing, ambiguous, incompatible or discontinuous simulator input."""


@dataclass(frozen=True)
class HourlyObservation:
    timestamp_utc: str
    pv_available_kw: float
    wind_available_kw: float
    campus_load_kw: float
    import_price: float
    export_price: float | None
    building_loads_kw: Mapping[str, float]
    pv_asset_powers_kw: Mapping[str, float]
    wind_asset_powers_kw: Mapping[str, float]


class SimulationData:
    """Chronologically ordered immutable observations with defensive metadata copies."""

    def __init__(self, observations, metadata):
        self._observations = tuple(observations)
        self._metadata = deepcopy(metadata)

    def __len__(self):
        return len(self._observations)

    def __iter__(self):
        return iter(self._observations)

    def __getitem__(self, index):
        return self._observations[index]

    @property
    def metadata(self):
        return deepcopy(self._metadata)


def _boundary(value, label):
    if not isinstance(value, (str, datetime, pd.Timestamp)):
        raise SimulationDataError(f"{label}: expected an explicit UTC timestamp.")
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("explicit UTC time zone required")
        if stamp != stamp.floor("h"):
            raise ValueError("hour boundary required")
        return stamp.tz_convert("UTC")
    except (ValueError, TypeError) as exc:
        raise SimulationDataError(f"{label}: invalid hourly UTC timestamp {value!r}: {exc}") from exc


def _registry(provided, sources, category):
    if provided is None:
        candidates = []
        for source in sources:
            scenario = source["metadata"].get("source_metadata", {}).get("scenario", {})
            if category == "buildings" and isinstance(scenario.get("buildings"), list):
                candidates.append(tuple(item["id"] for item in scenario["buildings"]))
            elif category in {"solar", "wind"} and "asset_id" in scenario.get(category, {}):
                candidates.append((scenario[category]["asset_id"],))
        if not candidates or any(candidate != candidates[0] for candidate in candidates):
            raise SimulationDataError(f"Provide an explicit {category} entity registry; source metadata is absent or inconsistent.")
        provided = candidates[0]
    if isinstance(provided, str) or not isinstance(provided, (list, tuple)) or not provided:
        raise SimulationDataError(f"{category}: provide a nonempty list/tuple of entity IDs.")
    if not all(isinstance(item, str) and item.strip() for item in provided) or len(set(provided)) != len(provided):
        raise SimulationDataError(f"{category}: entity IDs must be unique nonempty text.")
    return tuple(provided)


def _normalise_source(source, schema):
    kind = source["record_type"]
    frame = source["data"]
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.has_duplicates:
        raise SimulationDataError(f"{source['source_id']}: expected a nonempty DataFrame with unique columns.")
    if source.get("measurement_kind") != "interval_mean":
        raise SimulationDataError("Physical observations require measurement_kind: interval_mean; instantaneous samples need Part 2 integration.")
    metadata = deepcopy(source["metadata"])
    if metadata.get("source_id") != source["source_id"]:
        raise SimulationDataError("Top-level source_id disagrees with provenance metadata.")
    if not isinstance(source.get("units"), dict):
        raise SimulationDataError("Source units must be an explicit mapping.")
    units = deepcopy(source["units"])
    scaled = metadata.get("split", {}).get("scaled_units", {})
    # Learning-only columns are excluded from the physics view only if their
    # dimensionless units were explicitly declared by Part 2 split metadata.
    excluded = [column for column in frame if column in scaled and column.endswith("_scaled")]
    view = frame.drop(columns=excluded).copy(deep=True)
    allowed = schema["record_types"][kind]
    allowed_names = set(allowed["required"] + allowed["optional"])
    unknown = set(view.columns) - allowed_names
    if unknown:
        raise SimulationDataError(f"{source['source_id']}: unknown physical columns {sorted(unknown)}; campus aggregates/scaled values must not be mixed with node records.")
    conversions = []
    for column in view:
        spec = schema["columns"].get(column, {})
        if spec.get("type") != "number":
            continue
        if column not in units:
            raise SimulationDataError(f"{source['source_id']}.{column}: missing declared unit.")
        target = spec["unit"]
        if "{currency}" in target:
            code = metadata.get("currency_code")
            if not isinstance(code, str) or not re.fullmatch(r"[A-Z]{3}", code):
                raise SimulationDataError("Price inputs require explicit metadata.currency_code.")
            target = target.replace("{currency}", code)
        try:
            scale, offset = unit_conversion(units[column], target)
            if view[column].map(lambda value: isinstance(value, bool)).any():
                raise ValueError("booleans are not measurements")
            values = pd.to_numeric(view[column], errors="raise")
            view[column] = values * scale + offset
        except (CleaningError, TypeError, ValueError) as exc:
            raise SimulationDataError(f"{source['source_id']}.{column}: {exc}") from exc
        if units[column] != target:
            conversions.append({"column": column, "from_unit": units[column], "to_unit": target,
                                "scale": scale, "offset": offset})
        units[column] = target
    records = [{column: None if pd.isna(value) else value for column, value in row.items()}
               for row in view.to_dict("records")]
    try:
        validate_dataset(records, kind, units=units, metadata=metadata,
                         allow_missing_export_price=kind == "price")
    except SchemaValidationError as exc:
        raise SimulationDataError(f"{source['source_id']}: {exc}") from exc
    mandatory = {"demand": ["building_load_kw"], "generation": ["available_generation_kw"], "price": ["import_price"]}[kind]
    for column in mandatory:
        if view[column].isna().any():
            raise SimulationDataError(f"{source['source_id']}.{column}: missing mandatory observations.")
    view["timestamp_utc"] = pd.to_datetime(view["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    view["_source_id"] = source["source_id"]
    entity_column = {"demand": "building_id", "generation": "asset_id", "price": "grid_id"}[kind]
    return view, {"metadata": metadata, "record_type": kind,
                  "entity_ids": sorted(view[entity_column].unique().tolist()),
                  "units": units, "unit_conversions": conversions,
                  "excluded_learning_columns": excluded,
                  "input_file": deepcopy(source.get("input_file"))}


def prepare_simulation_data(sources, *, start=None, end=None, building_ids=None,
                            solar_asset_ids=None, wind_asset_ids=None, grid_id=None):
    """Extract a complete hourly interval from validated Part 2 source payloads.

    Registry lists are explicit or recovered from consistent recorded synthetic
    scenario metadata. Never infer buildings by blindly summing all numeric rows.
    Unregistered building/renewable IDs fail; grid_id may select one price node
    with other grid IDs recorded as excluded. Missing prices for export remain
    None. Current price-schema nonnegative tariff rules are retained.
    """
    if not isinstance(sources, list) or not sources:
        raise SimulationDataError("No source datasets supplied.")
    schema, frames, lineage, seen = load_schema(), {}, {}, set()
    ignored = []
    try:
        for source in sources:
            kind = source["record_type"]
            if kind == "weather":
                ignored.append(source["source_id"])
                continue
            if kind not in {"demand", "generation", "price"}:
                raise SimulationDataError(f"Unsupported physics input record_type {kind!r}.")
            if source["source_id"] in seen:
                raise SimulationDataError("Duplicate source IDs are ambiguous.")
            seen.add(source["source_id"])
            frame, details = _normalise_source(source, schema)
            frames.setdefault(kind, []).append(frame)
            lineage[source["source_id"]] = details
        if set(frames) != {"demand", "generation", "price"}:
            raise SimulationDataError("Mandatory demand, solar/wind generation and grid price sources are required.")
        tables = {kind: pd.concat(items, ignore_index=True) for kind, items in frames.items()}
        buildings = _registry(building_ids, sources, "buildings")
        solar = _registry(solar_asset_ids, sources, "solar")
        wind = _registry(wind_asset_ids, sources, "wind")
        if set(solar) & set(wind):
            raise SimulationDataError("Solar and wind asset registries must not overlap.")
        for kind, key in (("demand", "building_id"), ("generation", "asset_id"), ("price", "grid_id")):
            if tables[kind].duplicated(["timestamp_utc", key]).any():
                raise SimulationDataError(f"Duplicate timestamp/{key} across {kind} sources; refusing double-counting.")
        all_splits = [item["metadata"].get("split") for item in lineage.values()]
        if any(split is not None for split in all_splits):
            if any(split is None for split in all_splits) or any(
                split["name"] != all_splits[0]["name"] or split["boundaries"] != all_splits[0]["boundaries"]
                for split in all_splits
            ):
                raise SimulationDataError("Sources must not mix split names, boundaries, or split and unsplit data.")
        begin = _boundary(start, "start") if start is not None else _boundary(min(table["timestamp_utc"].min() for table in tables.values()), "start")
        finish = _boundary(end, "end") if end is not None else _boundary(max(table["timestamp_utc"].max() for table in tables.values()), "end") + pd.Timedelta(hours=1)
        if finish <= begin:
            raise SimulationDataError("end must follow start; selection uses [start, end).")
        if all_splits[0] is not None:
            bounds = all_splits[0]["boundaries"]
            if begin < _boundary(bounds["start_utc"], "split start") or finish > _boundary(bounds["end_exclusive_utc"], "split end"):
                raise SimulationDataError("Selected simulation interval crosses its temporal split.")
        begin_text, finish_text = begin.isoformat().replace("+00:00", "Z"), finish.isoformat().replace("+00:00", "Z")
        expected = pd.date_range(begin, finish, freq="h", inclusive="left").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        selected = {kind: table[(table["timestamp_utc"] >= begin_text) & (table["timestamp_utc"] < finish_text)].copy()
                    for kind, table in tables.items()}
        outside_interval = {kind: len(tables[kind]) - len(selected[kind]) for kind in tables}
        if set(selected["demand"]["building_id"]) - set(buildings):
            raise SimulationDataError("Unregistered building/aggregate demand rows would cause double-counting.")
        if set(selected["generation"]["asset_id"]) - set(solar + wind):
            raise SimulationDataError("Unregistered renewable assets or aggregate generation rows.")
        available_grids = sorted(selected["price"]["grid_id"].unique())
        if grid_id is None:
            if len(available_grids) != 1:
                raise SimulationDataError("Select one explicit grid_id; price node is missing or ambiguous.")
            grid_id = available_grids[0]
        if not isinstance(grid_id, str) or not grid_id.strip():
            raise SimulationDataError("grid_id must be nonempty text.")
        excluded_grid_rows = int((selected["price"]["grid_id"] != grid_id).sum())
        selected["price"] = selected["price"][selected["price"]["grid_id"] == grid_id]

        def coverage(kind, key, entities):
            actual = pd.MultiIndex.from_frame(selected[kind][["timestamp_utc", key]])
            required = pd.MultiIndex.from_product([expected, entities], names=["timestamp_utc", key])
            missing = required.difference(actual)
            if len(missing):
                raise SimulationDataError(f"Missing/discontinuous {kind} interval: {len(missing)} required records absent; examples {list(missing[:3])}.")

        coverage("demand", "building_id", buildings)
        coverage("generation", "asset_id", solar + wind)
        coverage("price", "grid_id", (grid_id,))
        generation = selected["generation"]
        for assets, kind in ((solar, "solar"), (wind, "wind")):
            if not (generation.loc[generation["asset_id"].isin(assets), "generation_type"] == kind).all():
                raise SimulationDataError(f"Asset registry disagrees with generation_type {kind}.")
        demand_table = selected["demand"].pivot(index="timestamp_utc", columns="building_id", values="building_load_kw")
        generation_table = generation.pivot(index="timestamp_utc", columns="asset_id", values="available_generation_kw")
        price_table = selected["price"].set_index("timestamp_utc")
        currency_codes = {lineage[source_id]["metadata"].get("currency_code") for source_id in selected["price"]["_source_id"]}
        if len(currency_codes) != 1:
            raise SimulationDataError("Grid price sources use mixed currencies; FX conversion is not performed.")
        observations = []
        for timestamp in expected:
            building_values = {entity: float(demand_table.at[timestamp, entity]) for entity in buildings}
            pv_values = {entity: float(generation_table.at[timestamp, entity]) for entity in solar}
            wind_values = {entity: float(generation_table.at[timestamp, entity]) for entity in wind}
            export = price_table.at[timestamp, "export_price"] if "export_price" in price_table else None
            observations.append(HourlyObservation(
                timestamp_utc=timestamp, pv_available_kw=math.fsum(pv_values.values()),
                wind_available_kw=math.fsum(wind_values.values()), campus_load_kw=math.fsum(building_values.values()),
                import_price=float(price_table.at[timestamp, "import_price"]),
                export_price=None if export is None or pd.isna(export) else float(export),
                building_loads_kw=MappingProxyType(building_values), pv_asset_powers_kw=MappingProxyType(pv_values),
                wind_asset_powers_kw=MappingProxyType(wind_values)))
        summaries = {}
        for source_id, details in lineage.items():
            metadata = details["metadata"]
            if metadata.get("synthetic") or metadata["origin"] == "simulated":
                summaries[source_id] = "synthetic"
            elif metadata["origin"] == "derived":
                summaries[source_id] = "modeled" if metadata.get("derived_kind") == "model" else "derived_" + metadata["derived_kind"]
            else:
                summaries[source_id] = "measured"
        return SimulationData(observations, {
            "interval": {"start_utc": begin_text, "end_exclusive_utc": finish_text, "hours": len(observations)},
            "entities": {"buildings": buildings, "solar": solar, "wind": wind, "grid": grid_id},
            "currency_code": next(iter(currency_codes)), "units": {"power": "kW", "prices": f"{next(iter(currency_codes))}/kWh"},
            "sources": lineage, "source_origin_summary": summaries,
            "aggregation": "Each registered building/asset counted once per timestamp; campus totals are derived sums, not measured aggregates.",
            "ignored_weather_sources": ignored, "excluded_grid_ids": [entity for entity in available_grids if entity != grid_id],
            "excluded_rows_outside_interval": outside_interval,
            "excluded_other_grid_rows": excluded_grid_rows,
            "unavailable_export_price_hours": sum(observation.export_price is None for observation in observations),
            "export_policy": "Unknown export price remains None; engine must disable export or require an explicit tariff.",
        })
    except SimulationDataError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise SimulationDataError(f"Invalid simulation source structure: {exc}") from exc


def load_simulation_data(manifest_path, **selection):
    """Load a Part 2 datasets/path manifest read-only, recording file checksums.

    Adjacent generation_report.json checksums are verified when available;
    otherwise hashes are acquisition snapshots, not authenticity assertions.
    The manifest is not configs/data_sources.yaml (that is acquisition config).
    """
    path = Path(manifest_path)
    try:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), list):
            raise SimulationDataError("Processed manifest requires a datasets list.")
        checksum_file = path.parent / "generation_report.json"
        checksums = json.loads(checksum_file.read_text())["file_checksums"] if checksum_file.exists() else {}
        sources = []
        for item in manifest["datasets"]:
            source = deepcopy(item)
            csv_path = Path(source.pop("path"))
            if not csv_path.is_absolute():
                csv_path = path.resolve().parent / csv_path
            with csv_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                expected = source.pop("sha256", checksums.get(csv_path.name))
                if expected is not None and digest != expected:
                    raise SimulationDataError(f"Input checksum mismatch: {csv_path}")
                stream.seek(0)
                source["data"] = pd.read_csv(stream, dtype={column: str for column in ("building_id", "asset_id", "site_id", "grid_id")})
            source["input_file"] = {"path": str(csv_path.resolve()), "sha256": digest, "verified_against_recorded_hash": expected is not None}
            sources.append(source)
        return prepare_simulation_data(sources, **selection)
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SimulationDataError):
            raise
        raise SimulationDataError(f"Cannot load processed datasets from {path}: {exc}") from exc


def load_simulation_split(split_root, partition, **selection):
    """Reuse Part 2 split-file loading/hash checks; never combine temporal splits."""
    try:
        return prepare_simulation_data(load_partition_sources(split_root, partition), **selection)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        if isinstance(exc, SimulationDataError):
            raise
        raise SimulationDataError(f"Cannot load simulation split {partition}: {exc}") from exc
