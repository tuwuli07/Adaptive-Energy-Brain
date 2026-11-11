"""Validate canonical energy records without mutating or converting input data.

Use validate_dataset(records, record_type, units=..., metadata=...). Numeric
units are mandatory, including optional numeric columns when present. Required
values cannot be missing; optional values may be absent or None. Non-finite
numbers and unknown columns are rejected. Source-specific information belongs
in metadata['source_metadata'] and is preserved in the returned deep copy.
"""

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta
import math
from numbers import Real
from pathlib import Path
import re

import yaml


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs/data_schema.yaml"


class SchemaValidationError(ValueError):
    """Canonical records, metadata or unit declarations are invalid."""


def load_schema(path=DEFAULT_SCHEMA_PATH):
    """Load the versioned canonical schema from YAML."""
    try:
        schema = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SchemaValidationError(f"Cannot read data schema: {exc}") from exc
    if not isinstance(schema, dict) or schema.get("schema_version") != 1:
        raise SchemaValidationError("Unsupported schema: expected schema_version: 1.")
    if not isinstance(schema.get("columns"), dict) or not isinstance(schema.get("record_types"), dict):
        raise SchemaValidationError("Schema requires columns and record_types mappings.")
    for name, definition in schema["record_types"].items():
        if not isinstance(definition, dict) or any(
            not isinstance(definition.get(field), list)
            for field in ("required", "optional", "key")
        ):
            raise SchemaValidationError(f"Invalid record type definition: {name}.")
        fields = definition["required"] + definition["optional"]
        if (
            len(fields) != len(set(fields))
            or not set(fields) <= schema["columns"].keys()
            or not set(definition["key"]) <= set(definition["required"])
        ):
            raise SchemaValidationError(f"Invalid columns or keys in record type {name}.")
    return schema


def _utc_timestamp(value, context):
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", value
    ):
        raise SchemaValidationError(f"{context}: expected explicit UTC ISO 8601 timestamp (Z or +00:00).")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError(f"{context}: invalid timestamp {value!r}.") from exc
    if parsed.utcoffset() != timedelta(0) or any((parsed.minute, parsed.second, parsed.microsecond)):
        raise SchemaValidationError(f"{context}: timestamp must be UTC on an hourly boundary.")
    return parsed


def _validate_value(value, definition, context):
    kind = definition["type"]
    if kind == "timestamp":
        return _utc_timestamp(value, context)
    if kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise SchemaValidationError(f"{context}: expected nonempty text.")
        if "enum" in definition and value not in definition["enum"]:
            raise SchemaValidationError(f"{context}: expected one of {definition['enum']}.")
        return value
    if kind != "number":
        raise SchemaValidationError(f"{context}: unsupported schema type {kind!r}.")
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise SchemaValidationError(f"{context}: expected a finite number, not {value!r}.")
    for bound, failed, symbol in (
        ("min", lambda limit: value < limit, ">="),
        ("max", lambda limit: value > limit, "<="),
        ("min_exclusive", lambda limit: value <= limit, ">"),
    ):
        if bound in definition and failed(definition[bound]):
            raise SchemaValidationError(f"{context}: value must be {symbol} {definition[bound]}.")
    if "const" in definition and value != definition["const"]:
        raise SchemaValidationError(f"{context}: expected {definition['const']} for the hourly schema.")
    return value


def _relationships(row, context):
    def present(*names):
        return all(row.get(name) is not None for name in names)

    def require(condition, message):
        if not condition:
            raise SchemaValidationError(f"{context}: {message}")

    for power, energy in (
        ("building_load_kw", "demand_energy_kwh"),
        ("available_generation_kw", "available_generation_energy_kwh"),
    ):
        if present(power, energy):
            require(math.isclose(row[energy], row[power] * row["interval_hours"],
                                 rel_tol=1e-6, abs_tol=1e-9),
                    f"{energy} must equal {power} * interval_hours; kW and kWh are distinct.")
    if present("available_generation_kw", "generation_capacity_kw"):
        require(row["available_generation_kw"] <= row["generation_capacity_kw"],
                "available_generation_kw exceeds generation_capacity_kw.")
    if present("import_price", "export_price"):
        require(row["export_price"] <= row["import_price"],
                "export_price must not exceed import_price in the baseline scenario.")
    if present("forecast_issue_time_utc"):
        require(row["forecast_issue_time_utc"] <= row["timestamp_utc"],
                "forecast_issue_time_utc cannot follow target timestamp_utc.")
    if present("lower_power_kw"):
        require(row["lower_power_kw"] <= row["predicted_power_kw"],
                "lower_power_kw exceeds predicted_power_kw.")
    if present("upper_power_kw"):
        require(row["upper_power_kw"] >= row["predicted_power_kw"],
                "upper_power_kw is below predicted_power_kw.")
    if present("battery_capacity_kwh"):
        require(math.isclose(row["battery_energy_kwh"],
                             row["battery_soc"] * row["battery_capacity_kwh"],
                             rel_tol=1e-6, abs_tol=1e-9),
                "battery_energy_kwh must equal battery_soc * battery_capacity_kwh.")
    if any(present(name) for name in ("battery_charge_kw", "battery_discharge_kw")):
        require(present("interval_hours"), "battery flows require interval_hours.")
    if present("battery_charge_kw", "battery_discharge_kw"):
        require(not (row["battery_charge_kw"] > 0 and row["battery_discharge_kw"] > 0),
                "battery charging and discharging cannot occur simultaneously.")


def validate_dataset(records, record_type, *, units, metadata,
                     currency_code=None, schema_path=DEFAULT_SCHEMA_PATH,
                     allow_missing_export_price=False):
    """Return a validated deep-copied envelope or raise SchemaValidationError.

    records is a nonempty iterable of mappings (e.g. DataFrame.to_dict('records')).
    No files are read except the schema; no datasets are written. Currency codes
    are explicit three-uppercase-letter identifiers, not an ISO registry lookup.
    Currency may be configured in YAML, metadata or the call, but declarations
    must agree. Source metadata is opaque and retained without interpretation.
    allow_missing_export_price is an explicit simulator-adapter opt-in for price
    records only: absent/null export prices remain unavailable, never zero.
    The default canonical price contract still requires export_price.
    """
    schema = load_schema(schema_path)
    if record_type not in schema["record_types"]:
        raise SchemaValidationError(f"Unknown record type {record_type!r}; choose {list(schema['record_types'])}.")
    if not isinstance(metadata, Mapping):
        raise SchemaValidationError("metadata must be a mapping.")
    missing = set(schema["metadata_required"]) - metadata.keys()
    if missing:
        raise SchemaValidationError(f"Missing required metadata: {sorted(missing)}.")
    if not isinstance(metadata["source_id"], str) or not metadata["source_id"].strip():
        raise SchemaValidationError("metadata.source_id must be nonempty text.")
    if metadata["origin"] not in schema["origin_values"]:
        raise SchemaValidationError(f"metadata.origin must be one of {schema['origin_values']}.")
    if not isinstance(metadata["source_metadata"], Mapping):
        raise SchemaValidationError("metadata.source_metadata must be a mapping.")
    if record_type in {"forecast", "simulation_state"} and metadata["origin"] != "derived":
        raise SchemaValidationError(f"{record_type} records require origin: derived.")
    if metadata["origin"] == "derived" and metadata.get("derived_kind") not in {
        "calculated", "simulator", "model",
    }:
        raise SchemaValidationError("Derived metadata requires derived_kind: calculated, simulator or model.")
    codes = [code for code in (currency_code, metadata.get("currency_code"), schema["currency_code"])
             if code is not None]
    if any(not isinstance(code, str) or not re.fullmatch(r"[A-Z]{3}", code) for code in codes):
        raise SchemaValidationError("currency_code must be an explicit three-uppercase-letter code.")
    if len(set(codes)) > 1:
        raise SchemaValidationError("Conflicting currency_code declarations; currency conversion is not performed.")
    currency = codes[0] if codes else None
    if not isinstance(units, Mapping):
        raise SchemaValidationError("units must map numeric column names to canonical units.")
    if isinstance(records, (str, bytes, Mapping)):
        raise SchemaValidationError("records must be an iterable of row mappings.")
    try:
        rows = list(records)
    except TypeError as exc:
        raise SchemaValidationError("records must be an iterable of row mappings.") from exc
    if not rows:
        raise SchemaValidationError("Dataset must contain at least one record.")
    definition = deepcopy(schema["record_types"][record_type])
    if type(allow_missing_export_price) is not bool:
        raise SchemaValidationError("allow_missing_export_price must be a boolean.")
    if allow_missing_export_price:
        if record_type != "price":
            raise SchemaValidationError("Optional export-price validation applies only to price records.")
        definition["required"].remove("export_price")
        definition["optional"].append("export_price")
    allowed = set(definition["required"] + definition["optional"])
    numeric_columns = set()
    seen = set()
    for index, row in enumerate(rows):
        context = f"{record_type} row {index}"
        if not isinstance(row, Mapping):
            raise SchemaValidationError(f"{context}: expected a mapping.")
        missing = set(definition["required"]) - row.keys()
        if missing:
            raise SchemaValidationError(f"{context}: missing required columns {sorted(missing)}.")
        unknown = set(row) - allowed
        if unknown:
            raise SchemaValidationError(f"{context}: unknown columns {sorted(unknown, key=str)}; preserve source fields in metadata.source_metadata.")
        parsed = {}
        for column, value in row.items():
            column_definition = schema["columns"][column]
            if column_definition["type"] == "number":
                numeric_columns.add(column)
            if value is None:
                if column in definition["required"]:
                    raise SchemaValidationError(f"{context}.{column}: required value is missing/null.")
                continue
            parsed[column] = _validate_value(value, column_definition, f"{context}.{column}")
        if definition.get("at_least_one") and not any(
            column in parsed for column in definition["at_least_one"]
        ):
            raise SchemaValidationError(f"{context}: at least one non-null weather observation is required.")
        key = tuple(parsed[column] for column in definition["key"])
        if key in seen:
            raise SchemaValidationError(f"{context}: duplicate record key {key}.")
        seen.add(key)
        _relationships(parsed, context)
    if set(units) != numeric_columns:
        raise SchemaValidationError(
            f"Unit declarations must match numeric columns; missing {sorted(numeric_columns - units.keys())}, "
            f"unexpected {sorted(units.keys() - numeric_columns, key=str)}."
        )
    for column in numeric_columns:
        expected = schema["columns"][column]["unit"]
        if "{currency}" in expected:
            if currency is None:
                raise SchemaValidationError("Price records require an explicit currency_code; no default currency is assumed.")
            expected = expected.replace("{currency}", currency)
        if units[column] != expected:
            raise SchemaValidationError(
                f"{column}: incompatible unit {units[column]!r}; expected {expected!r}. "
                "Implicit conversion is disabled, including kW/kWh and currencies."
            )
    result_metadata = deepcopy(dict(metadata))
    if currency is not None:
        result_metadata["currency_code"] = currency
    return {
        "schema_version": schema["schema_version"], "record_type": record_type,
        "records": deepcopy(rows), "units": deepcopy(dict(units)),
        "metadata": result_metadata,
    }


def validate_record(record, record_type, *, units, metadata, **kwargs):
    """Validate a single row using the same rules as validate_dataset."""
    return validate_dataset([record], record_type, units=units,
                            metadata=metadata, **kwargs)
