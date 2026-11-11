"""Hourly UTC integration of cleaned demand, generation, weather and prices.

Each input is an interval-start observation with explicit interval_hours and
canonical units. Power/weather/prices must be interval means, not instantaneous
samples. Energy is an interval total. Means are duration-weighted; energy sums
are allocated proportionally only when splitting is explicitly approved.
Hourly average price is not an energy-weighted bill and cannot reproduce
subhourly cost when prices and power covary.

The full outer entity panel uses (timestamp_utc, record_type, entity_id). Shared
assets are not copied onto building rows. It includes every hour and observed or
explicitly expected entity. Missing/partial hours are flagged, not interpolated.
Forecast vintages and battery boundary states are deliberately not interval
observations and must be integrated separately in a later workflow.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from src.data.schema import load_schema


PROCESSED_ROOT = Path(__file__).resolve().parents[2] / "data/processed"
ENTITY_COLUMNS = {"demand": "building_id", "generation": "asset_id",
                  "weather": "site_id", "price": "grid_id"}
KEY = ["timestamp_utc", "record_type", "entity_id"]


class IntegrationError(ValueError):
    """Input semantics, alignment, joining or output validation failed."""


def _utc(value, source_timezone=None):
    if not isinstance(value, (str, datetime, pd.Timestamp)):
        raise IntegrationError(f"Invalid timestamp {value!r}: expected ISO datetime.")
    if isinstance(value, str) and not re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", value):
        raise IntegrationError(f"Invalid/non-ISO timestamp {value!r}.")
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("missing timestamp")
        if stamp.tzinfo is None:
            if not source_timezone:
                raise ValueError("naive timestamp requires explicit source_timezone")
            stamp = stamp.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
        return stamp.tz_convert("UTC")
    except Exception as exc:
        raise IntegrationError(f"Cannot normalize timestamp {value!r}: {exc}") from exc


def _iso(stamp):
    return stamp.isoformat().replace("+00:00", "Z")


def integrate_datasets(datasets, *, start=None, end=None, expected_entities=None):
    """Return {'data': DataFrame, 'report': JSON-safe dict} without writing files.

    datasets: list of mappings containing source_id, record_type, data (a cleaned
    DataFrame), units and metadata. Optional source_timezone localizes naive
    timestamps. measurement_kind must be 'interval_mean'; energy fields are
    always interval totals. energy_allocation='uniform' explicitly permits
    splitting interval energy at hourly/window boundaries. mean_allocation=
    'constant' separately permits splitting interval means, an explicit
    piecewise-constant approximation rather than recovered hourly measurements.
    Without these policies, coarse/straddling observations cannot be split.
    Input metadata and
    any cleaning report are preserved. start/end are UTC hour boundaries, end
    exclusive; omitted bounds use the union of input coverage. expected_entities
    maps record types to required IDs, allowing wholly absent entities to flag.
    """
    if not isinstance(datasets, list) or not datasets:
        raise IntegrationError("No cleaned datasets supplied; nothing will be generated.")
    schema = load_schema()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": "1h UTC", "issues": [], "sources": {}, "provenance": [],
        "aggregation_rules": {
            "power": "duration-weighted mean of interval-average kW",
            "energy": "sum interval kWh; splitting requires explicit uniform allocation",
            "weather": "duration-weighted mean; no directional/categorical averaging",
            "prices": "duration-weighted mean in one explicit currency; negative values retained",
            "coverage": "require a full hour per field; no partial-hour extrapolation",
            "joins": "full outer timestamp/entity panel; shared assets never repeated per building",
        },
    }
    prepared, entities, fields = [], set(), {}
    currencies = set()
    ids = set()
    for source in datasets:
        source_id = source.get("source_id")
        kind = source.get("record_type")
        if not isinstance(source_id, str) or not source_id.strip() or source_id in ids:
            raise IntegrationError("Each input requires a unique nonempty source_id.")
        ids.add(source_id)
        if kind not in ENTITY_COLUMNS:
            raise IntegrationError(f"{source_id}: unsupported observation record_type {kind!r}.")
        if source.get("measurement_kind") != "interval_mean":
            raise IntegrationError(f"{source_id}: explicitly declare measurement_kind: interval_mean; instantaneous samples are unsupported.")
        allocation = source.get("energy_allocation")
        if allocation not in {None, "uniform"}:
            raise IntegrationError(f"{source_id}: energy_allocation must be null or uniform.")
        mean_allocation = source.get("mean_allocation")
        if mean_allocation not in {None, "constant"}:
            raise IntegrationError(f"{source_id}: mean_allocation must be null or constant.")
        frame = source.get("data")
        if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.has_duplicates:
            raise IntegrationError(f"{source_id}: expected nonempty cleaned DataFrame with unique columns.")
        metadata = deepcopy(source.get("metadata"))
        if not isinstance(metadata, dict) or not all(name in metadata for name in ("source_id", "origin", "source_metadata")):
            raise IntegrationError(f"{source_id}: preserve source_id, origin and source_metadata.")
        if metadata["source_id"] != source_id or metadata["origin"] not in {"measured", "simulated", "derived"} or not isinstance(metadata["source_metadata"], dict):
            raise IntegrationError(f"{source_id}: invalid or conflicting provenance metadata.")
        definition = schema["record_types"][kind]
        missing = set(definition["required"]) - set(frame.columns)
        allowed = set(definition["required"] + definition["optional"])
        if missing or set(frame.columns) - allowed:
            raise IntegrationError(f"{source_id}: missing columns {sorted(missing)}; unsupported columns {sorted(set(frame.columns) - allowed)}.")
        numeric = [column for column in frame if schema["columns"][column]["type"] == "number" and column != "interval_hours"]
        declared = source.get("units", {})
        if set(declared) != set(numeric + ["interval_hours"]):
            raise IntegrationError(f"{source_id}: units must cover exactly the numeric input columns.")
        if declared["interval_hours"] != "h":
            raise IntegrationError(f"{source_id}: interval_hours must use h.")
        for column in numeric:
            spec = schema["columns"][column]
            expected_unit = spec["unit"]
            if "{currency}" in expected_unit:
                code = metadata.get("currency_code")
                if not isinstance(code, str) or not re.fullmatch(r"[A-Z]{3}", code):
                    raise IntegrationError(f"{source_id}: prices require explicit metadata.currency_code.")
                currencies.add(code)
                expected_unit = expected_unit.replace("{currency}", code)
            if declared[column] != expected_unit:
                raise IntegrationError(f"{source_id}.{column}: expected {expected_unit}, received {declared[column]}; clean units first.")
            fields.setdefault(kind, {})[column] = expected_unit
        frame = frame.copy(deep=True).reset_index(drop=True)
        time_values = [_utc(value, source.get("source_timezone")) for value in frame["timestamp_utc"]]
        frame["timestamp_utc"] = time_values
        entity_column = ENTITY_COLUMNS[kind]
        if not frame[entity_column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise IntegrationError(f"{source_id}: {entity_column} requires nonempty identifiers.")
        if kind == "generation":
            if not frame["generation_type"].isin(["solar", "wind"]).all():
                raise IntegrationError(f"{source_id}: generation_type must be solar or wind.")
            if (frame.groupby(entity_column)["generation_type"].nunique() > 1).any():
                raise IntegrationError(f"{source_id}: an asset cannot change generation_type.")
        if frame.duplicated(["timestamp_utc", entity_column]).any():
            raise IntegrationError(f"{source_id}: duplicate timestamp/entity records; resolve duplicates explicitly.")
        for column in numeric + ["interval_hours"]:
            if frame[column].map(lambda value: isinstance(value, bool)).any():
                raise IntegrationError(f"{source_id}.{column}: booleans are not numeric observations.")
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except (ValueError, TypeError) as exc:
                raise IntegrationError(f"{source_id}.{column}: invalid numeric observation.") from exc
            values = frame[column].dropna()
            if not values.map(math.isfinite).all():
                raise IntegrationError(f"{source_id}.{column}: non-finite observation.")
            if column == "interval_hours":
                if frame[column].isna().any() or (values <= 0).any():
                    raise IntegrationError(f"{source_id}: interval_hours must be positive and complete.")
            elif column not in {"import_price", "export_price"}:
                spec = schema["columns"][column]
                if (("min" in spec and (values < spec["min"]).any())
                        or ("max" in spec and (values > spec["max"]).any())
                        or ("min_exclusive" in spec and (values <= spec["min_exclusive"]).any())):
                    raise IntegrationError(f"{source_id}.{column}: physically invalid value; inspect cleaning report.")
        frame["_end"] = frame["timestamp_utc"] + pd.to_timedelta(frame["interval_hours"], unit="h")
        for power, energy in (("building_load_kw", "demand_energy_kwh"),
                              ("available_generation_kw", "available_generation_energy_kwh")):
            if power in frame and energy in frame:
                for _, observation in frame.dropna(subset=[power, energy]).iterrows():
                    if not math.isclose(observation[energy], observation[power] * observation["interval_hours"], rel_tol=1e-6, abs_tol=1e-9):
                        raise IntegrationError(f"{source_id}: inconsistent {power}/{energy}; power times hours must equal energy.")
        if kind == "generation" and "generation_capacity_kw" in frame:
            if (frame["available_generation_kw"] > frame["generation_capacity_kw"]).any():
                raise IntegrationError(f"{source_id}: generation exceeds declared capacity.")
            if (frame.groupby(entity_column)["generation_capacity_kw"].nunique() > 1).any():
                raise IntegrationError(f"{source_id}: nameplate capacity is static; do not average changing asset parameters.")
        for entity, group in frame.groupby(entity_column, sort=False):
            entities.add((kind, entity))
            ordered = group.sort_values("timestamp_utc")
            previous_end = None
            for row in ordered.to_dict("records"):
                if previous_end is not None and row["timestamp_utc"] < previous_end:
                    raise IntegrationError(f"{source_id}/{entity}: overlapping input intervals are ambiguous.")
                previous_end = row["_end"]
        report["sources"][source_id] = {
            "metadata": metadata, "input_rows": len(frame), "units": deepcopy(declared),
            "measurement_kind": "interval_mean", "energy_allocation": allocation,
            "mean_allocation": mean_allocation,
            "source_timezone": source.get("source_timezone"),
            "cleaning_report": deepcopy(source.get("report")),
        }
        prepared.append((source_id, kind, frame, numeric, allocation, mean_allocation))
    if len(currencies) > 1:
        raise IntegrationError("Mixed currencies cannot be joined without an explicit FX workflow.")
    expected_entities = expected_entities or {}
    if not isinstance(expected_entities, dict):
        raise IntegrationError("expected_entities must map observation types to entity ID lists.")
    for kind, values in expected_entities.items():
        if kind not in ENTITY_COLUMNS or not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
            raise IntegrationError("Invalid expected_entities configuration.")
        for entity in values:
            if (kind, entity) not in entities:
                report["issues"].append({"kind": "entire_entity_missing", "record_type": kind, "entity_id": entity})
            entities.add((kind, entity))
    begin = _utc(start) if start is not None else min(item[2]["timestamp_utc"].min() for item in prepared).floor("h")
    finish = _utc(end) if end is not None else max(item[2]["_end"].max() for item in prepared).ceil("h")
    if begin != begin.floor("h") or finish != finish.floor("h") or finish <= begin:
        raise IntegrationError("Window must use increasing UTC hour boundaries; end is exclusive.")
    report.update(start_utc=_iso(begin), end_utc=_iso(finish), currency_code=next(iter(currencies), None))
    accumulators, constants = {}, {}
    for source_id, kind, frame, numeric, allocation, mean_allocation in prepared:
        entity_column = ENTITY_COLUMNS[kind]
        for index, row in frame.iterrows():
            first, last = max(row["timestamp_utc"], begin), min(row["_end"], finish)
            if row["timestamp_utc"] < begin or row["_end"] > finish:
                report["issues"].append({"kind": "outside_requested_window", "source_id": source_id, "input_row": int(index)})
            if first >= last:
                continue
            entity = row[entity_column]
            if kind == "generation":
                constant_key = (kind, entity)
                value = row["generation_type"]
                if constant_key in constants and constants[constant_key] != value:
                    raise IntegrationError(f"Conflicting generation type for {entity}.")
                constants[constant_key] = value
            for hour in pd.date_range(first.floor("h"), last.ceil("h"), freq="h", inclusive="left"):
                overlap = (min(last, hour + pd.Timedelta(hours=1)) - max(first, hour)).total_seconds() / 3600
                if overlap <= 0:
                    continue
                for column in numeric:
                    value = row[column]
                    if pd.isna(value):
                        continue
                    unit = fields[kind][column]
                    energy = unit == "kWh"
                    if energy and not math.isclose(overlap, row["interval_hours"], abs_tol=1e-9) and allocation != "uniform":
                        raise IntegrationError(f"{source_id}.{column}: splitting interval energy requires energy_allocation: uniform.")
                    if not energy and column != "generation_capacity_kw" and not math.isclose(overlap, row["interval_hours"], abs_tol=1e-9) and mean_allocation != "constant":
                        raise IntegrationError(f"{source_id}.{column}: splitting interval means requires mean_allocation: constant; finer observations are otherwise needed.")
                    key = (hour, kind, entity, column)
                    parts = accumulators.setdefault(key, [])
                    segment_start, segment_end = max(first, hour), min(last, hour + pd.Timedelta(hours=1))
                    if any(segment_start < old[1] and segment_end > old[0] for old in parts):
                        raise IntegrationError(f"Conflicting sources overlap for {kind}/{entity}/{column} at {_iso(hour)}; no implicit priority.")
                    contribution = value * overlap / row["interval_hours"] if energy else value * overlap
                    parts.append((segment_start, segment_end, contribution, overlap, source_id))
                    report["provenance"].append({
                        "timestamp_utc": _iso(hour), "record_type": kind, "entity_id": entity,
                        "column": column, "source_id": source_id, "input_row": int(index),
                        "overlap_hours": overlap,
                    })
    rows = []
    for hour in pd.date_range(begin, finish, freq="h", inclusive="left"):
        for kind, entity in sorted(entities):
            row = {"timestamp_utc": _iso(hour), "record_type": kind, "entity_id": entity,
                   ENTITY_COLUMNS[kind]: entity, "interval_hours": 1.0}
            if (kind, entity) in constants:
                row["generation_type"] = constants[(kind, entity)]
            complete = True
            present = False
            for column in fields.get(kind, {}):
                parts = accumulators.get((hour, kind, entity, column), [])
                covered = sum(part[3] for part in parts)
                present = present or bool(parts)
                if not math.isclose(covered, 1.0, abs_tol=1e-9):
                    row[column] = None
                    complete = False
                    report["issues"].append({"kind": "incomplete_hour", "timestamp_utc": _iso(hour),
                                             "record_type": kind, "entity_id": entity, "column": column,
                                             "coverage_hours": covered})
                else:
                    row[column] = sum(part[2] for part in parts) if fields[kind][column] == "kWh" else sum(part[2] for part in parts) / covered
            if kind not in fields or not present:
                complete = False
                report["issues"].append({"kind": "missing_entity_interval", "timestamp_utc": _iso(hour),
                                         "record_type": kind, "entity_id": entity})
            row["integration_status"] = "complete" if complete else "incomplete"
            rows.append(row)
    data = pd.DataFrame(rows)
    if data.duplicated(KEY).any():
        raise IntegrationError("Internal error: duplicate hourly entity keys.")
    report.update(output_rows=len(data), expected_hours=int((finish - begin).total_seconds() / 3600),
                  entity_count=len(entities), units=fields,
                  complete=bool((data["integration_status"] == "complete").all()),
                  validated=True)
    return {"data": data, "report": report}


def save_integrated_dataset(result, name, *, processed_root=PROCESSED_ROOT,
                            allow_incomplete=False):
    """Save a new CSV/report bundle under data/processed without overwriting.

    Structural validation is mandatory. Flagged gaps require allow_incomplete;
    this does not fill them. Writes never target raw data. The bundle is not
    crash-atomic, but handled write failures clean up files created by this call.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise IntegrationError("Output name must be a safe directory basename.")
    data, report = result["data"], deepcopy(result["report"])
    if not report.get("validated") or data.empty or data.duplicated(KEY).any():
        raise IntegrationError("Refusing structurally invalid integration output.")
    parsed_times = [_utc(value) for value in data["timestamp_utc"]]
    if any(stamp != stamp.floor("h") for stamp in parsed_times):
        raise IntegrationError("Output timestamps are not aligned to UTC hour boundaries.")
    if not data["integration_status"].isin(["complete", "incomplete"]).all():
        raise IntegrationError("Invalid integration status.")
    if report.get("complete") != bool((data["integration_status"] == "complete").all()):
        raise IntegrationError("Output completeness and report disagree.")
    if len(data) != report["expected_hours"] * report["entity_count"]:
        raise IntegrationError("Output does not contain the expected hourly entity panel.")
    if not report.get("complete") and not allow_incomplete:
        raise IntegrationError("Incomplete joins/coverage; inspect report or explicitly allow_incomplete.")
    root = Path(processed_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    created = []
    try:
        destination.mkdir()  # Exclusive: never reuse an existing dataset bundle.
    except OSError as exc:
        raise IntegrationError(f"Cannot create new output bundle {destination}: {exc}") from exc
    try:
        csv_path = destination / "dataset.csv"
        created.append(csv_path)
        data.to_csv(csv_path, index=False, mode="x")
        report["dataset_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        report["saved_with_incomplete_approval"] = bool(allow_incomplete and not report["complete"])
        report_path = destination / "integration_report.json"
        created.append(report_path)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception as exc:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        destination.rmdir()
        raise IntegrationError(f"Cannot save integration bundle: {exc}") from exc
    return destination
