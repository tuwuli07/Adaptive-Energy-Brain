"""Auditable, conservative CSV cleaning; never writes to an input dataset.

load_and_clean_csv() and clean_dataframe() return data, units, metadata and a
JSON-serializable report. Frames retain original positional row indices.
Unsupported units/configuration raise CleaningError; its report preserves any
audit entries already collected. Flagged observations remain unless an explicit
missing-value or exact-duplicate policy authorizes a change.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from src.data.schema import SchemaValidationError, load_schema, validate_dataset


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/cleaning.yaml"


class CleaningError(ValueError):
    """Cleaning cannot safely continue; report contains the operations so far."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


def load_cleaning_config(path=DEFAULT_CONFIG_PATH):
    """Load explicit cleaning policies; no source data is loaded here."""
    try:
        result = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CleaningError(f"Cannot read cleaning configuration: {exc}") from exc
    if not isinstance(result, dict) or result.get("version") != 1:
        raise CleaningError("Cleaning configuration requires version: 1.")
    return result


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def unit_conversion(source, target):
    """Return an explicit affine (scale, offset) for supported compatible units."""
    if source == target:
        return 1, 0
    conversions = {
        ("W", "kW"): (0.001, 0), ("MW", "kW"): (1000, 0),
        ("Wh", "kWh"): (0.001, 0), ("MWh", "kWh"): (1000, 0),
        ("kW/m^2", "W/m^2"): (1000, 0),
        ("K", "degC"): (1, -273.15),
        ("degF", "degC"): (5 / 9, -32 * 5 / 9),
        ("degrees C", "degC"): (1, 0),
        ("km/h", "m/s"): (1 / 3.6, 0),
        ("%", "1"): (0.01, 0), ("min", "h"): (1 / 60, 0),
    }
    if (source, target) in conversions:
        return conversions[(source, target)]
    if re.fullmatch(r"[A-Z]{3}/kWh", target) and source == target.replace("/kWh", "/MWh"):
        return 0.001, 0
    raise CleaningError(
        f"Unsupported unit conversion {source!r} -> {target!r}; "
        "power/energy and currency conversions are never inferred."
    )


def clean_dataframe(frame, record_type, *, units, metadata, config=None,
                    profile=None, currency_code=None):
    """Clean a copy using declared source units keyed by original column names.

    Metadata must carry source_id, origin and source_metadata for final canonical
    validation. Unknown source columns are retained and reported, not discarded.
    No constants such as building IDs or asset capacities are invented.
    """
    policy = deepcopy(config) if config is not None else load_cleaning_config()
    schema = load_schema()
    if record_type not in schema["record_types"]:
        raise CleaningError(f"Unknown record type: {record_type!r}.")
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
        raise CleaningError("Expected a DataFrame with unique column names.")
    if not isinstance(policy, dict) or policy.get("version") != 1:
        raise CleaningError("Cleaning configuration requires version: 1.")
    if not isinstance(units, dict) or not isinstance(metadata, dict):
        raise CleaningError("units and metadata must be mappings.")
    if profile is not None:
        if profile not in policy.get("profiles", {}):
            raise CleaningError(f"Unknown cleaning profile: {profile!r}.")
        override = policy["profiles"][profile]
        if set(override) - {"column_mapping", "source_timezone", "timestamp_format"}:
            raise CleaningError("Profile only supports mappings and timestamp conventions.")
        policy.update(override)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_type": record_type, "profile": profile, "input_rows": len(frame),
        "operations": [], "issues": [], "policy": deepcopy(policy),
    }

    def issue(kind, rows, column=None, severity="warning", **details):
        report["issues"].append({
            "kind": kind, "severity": severity, "column": column,
            "rows": [int(row) for row in rows], "details": _plain(details),
        })

    def operation(kind, column=None, **details):
        report["operations"].append({"operation": kind, "column": column, **_plain(details)})

    def assign(column, index, value, kind, **details):
        previous = data.at[index, column]
        data.at[index, column] = value
        operation(kind, column, row=int(index), before=previous, after=value, **details)

    try:
        mapping = policy.get("column_mapping", {})
        if not isinstance(mapping, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
        ):
            raise CleaningError("column_mapping must map source headers to canonical names.")
        renamed = [mapping.get(column, column) for column in frame.columns]
        if len(set(renamed)) != len(renamed):
            raise CleaningError("Column mapping collision; multiple columns map to one target.")
        data = frame.copy(deep=True).reset_index(drop=True).astype(object)
        data.columns = renamed
        output_units = {}
        for column, unit in units.items():
            target = mapping.get(column, column)
            if target in output_units:
                raise CleaningError(f"Unit mapping collision for {target}.")
            output_units[target] = unit
        for old, new in zip(frame.columns, renamed):
            if old != new:
                operation("rename_column", new, source_column=old)
        definition = schema["record_types"][record_type]
        allowed = set(definition["required"] + definition["optional"])
        for column in set(data.columns) - allowed:
            issue("unknown_column_retained", [], column)
        for column in set(definition["required"]) - set(data.columns):
            issue("missing_required_column", [], column, severity="error")

        codes = [code for code in (currency_code, metadata.get("currency_code"), schema["currency_code"])
                 if code is not None]
        if any(not isinstance(code, str) or not re.fullmatch(r"[A-Z]{3}", code) for code in codes):
            raise CleaningError("currency_code must be a three-uppercase-letter code.")
        if len(set(codes)) > 1:
            raise CleaningError("Conflicting currency codes; no FX conversion is performed.")
        currency = codes[0] if codes else None
        original_missing = {}
        for column in data.columns:
            blank = data[column].map(lambda value: isinstance(value, str) and not value.strip())
            for index in data.index[blank]:
                assign(column, index, None, "blank_to_missing")
            original_missing[column] = data[column].isna()
            if original_missing[column].any():
                issue("missing_values", data.index[original_missing[column]], column)
            column_schema = schema["columns"].get(column)
            if column_schema is None:
                continue
            if column_schema["type"] == "timestamp":
                for index in data.index[~original_missing[column]]:
                    value = data.at[index, column]
                    try:
                        timestamp_format = policy.get("timestamp_format")
                        if isinstance(value, str) and not timestamp_format and not re.match(
                            r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", value
                        ):
                            raise ValueError("Non-ISO timestamps require timestamp_format.")
                        if not isinstance(value, (str, datetime, pd.Timestamp)):
                            raise ValueError("Numeric/unsupported timestamp encoding requires explicit upstream decoding.")
                        stamp = pd.Timestamp(pd.to_datetime(value, format=timestamp_format))
                        if pd.isna(stamp):
                            raise ValueError("Timestamp parsed to NaT.")
                        if stamp.tzinfo is None:
                            if not policy.get("source_timezone"):
                                raise ValueError("Naive timestamp requires explicit source_timezone.")
                            stamp = stamp.tz_localize(policy["source_timezone"], ambiguous="raise", nonexistent="raise")
                        stamp = stamp.tz_convert("UTC")
                        assign(column, index, stamp.isoformat().replace("+00:00", "Z"), "normalize_timestamp_utc")
                        if any((stamp.minute, stamp.second, stamp.microsecond, stamp.nanosecond)):
                            issue("non_hourly_timestamp", [index], column, severity="error")
                    except Exception as exc:
                        # Includes pandas/zoneinfo-specific ambiguous/nonexistent local-time errors.
                        issue("invalid_timestamp", [index], column, severity="error", value=value, reason=str(exc))
                        assign(column, index, None, "invalid_timestamp_to_missing")
            elif column_schema["type"] == "number":
                if column not in output_units:
                    raise CleaningError(f"Missing explicit input unit for {column}.")
                target_unit = column_schema["unit"]
                if "{currency}" in target_unit:
                    if currency is None:
                        raise CleaningError("Price cleaning requires explicit currency_code.")
                    target_unit = target_unit.replace("{currency}", currency)
                scale, offset = unit_conversion(output_units[column], target_unit)
                for index in data.index[~original_missing[column]]:
                    value = data.at[index, column]
                    try:
                        if isinstance(value, (bool, np.bool_)):
                            raise ValueError("Boolean is not a measurement.")
                        numeric = float(value)
                        if not math.isfinite(numeric):
                            raise ValueError("Non-finite measurement.")
                    except (ValueError, TypeError, OverflowError) as exc:
                        issue("invalid_numeric", [index], column, severity="error", value=value, reason=str(exc))
                        assign(column, index, None, "invalid_numeric_to_missing")
                        continue
                    if not isinstance(value, (float, int, np.number)):
                        assign(column, index, numeric, "parse_numeric")
                    if output_units[column] != target_unit:
                        assign(column, index, numeric * scale + offset, "convert_unit",
                               from_unit=output_units[column], to_unit=target_unit,
                               scale=scale, offset=offset)
                operation("declare_canonical_unit", column, from_unit=output_units[column],
                          to_unit=target_unit, scale=scale, offset=offset)
                output_units[column] = target_unit

        keys = definition["key"]
        duplicate_policy = policy.get("duplicates", "keep")
        if duplicate_policy not in {"keep", "error", "drop_exact"}:
            raise CleaningError("duplicates must be keep, error or drop_exact.")
        complete_keys = all(column in data for column in keys)
        if complete_keys:
            eligible = data[keys].notna().all(axis=1)
            duplicate_rows = data.loc[eligible].duplicated(keys, keep=False)
            rows = duplicate_rows.index[duplicate_rows]
            if len(rows):
                issue("duplicate_keys", rows, severity="error", keys=keys)
                if duplicate_policy == "error":
                    raise CleaningError("Duplicate record keys found; see report.")
            if duplicate_policy == "drop_exact":
                remove = data.index[data.duplicated(keep="first")]
                for index in remove:
                    operation("drop_exact_duplicate", row=int(index), before=data.loc[index].to_dict())
                data = data.drop(index=remove)

        interval = policy.get("expected_interval_hours", 1)
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
            raise CleaningError("expected_interval_hours must be positive and finite.")
        group_columns = [column for column in keys if column != "timestamp_utc"]
        groups = []
        if complete_keys:
            group_input = data[data[keys].notna().all(axis=1)]
            groups = [group for _, group in group_input.groupby(group_columns, sort=False, dropna=False)] if group_columns else [group_input]
            for group in groups:
                ordered = group.sort_values("timestamp_utc")
                times = pd.to_datetime(ordered["timestamp_utc"], utc=True)
                for position in range(1, len(ordered)):
                    elapsed = (times.iloc[position] - times.iloc[position - 1]).total_seconds() / 3600
                    if elapsed > interval:
                        issue("missing_time_intervals", [ordered.index[position - 1], ordered.index[position]],
                              "timestamp_utc", elapsed_hours=elapsed, expected_hours=interval)

        missing_policy = policy.get("missing", {})
        default_method = missing_policy.get("default", "keep")
        if default_method not in {"keep", "error", "drop"}:
            raise CleaningError("Default missing policy must be keep, error or drop.")
        rules = missing_policy.get("columns", {})
        for column in rules:
            if column not in data:
                raise CleaningError(f"Missing-value rule references absent column {column}.")
        drop_rows = set()
        for column in data.columns:
            rule = rules.get(column, {"method": default_method})
            if not isinstance(rule, dict) or rule.get("method") not in {
                "keep", "error", "drop", "constant", "forward_fill", "interpolate",
            }:
                raise CleaningError(f"Invalid missing-value policy for {column}.")
            method = rule["method"]
            candidates = data.index[data[column].isna()]
            if method == "error" and len(candidates):
                raise CleaningError(f"Missing values in {column}; see report.")
            if method == "drop":
                drop_rows.update(candidates)
            if method in {"keep", "error", "drop"}:
                continue
            if schema["columns"].get(column, {}).get("type") != "number":
                raise CleaningError(f"Only numeric fields support filling: {column}.")
            # Do not replace parse errors with invented measurements.
            candidates = [index for index in candidates if original_missing[column].at[index]]
            if method == "constant":
                value = rule.get("value")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not rule.get("reason"):
                    raise CleaningError(f"Constant fill for {column} needs a finite value and documented reason.")
                for index in candidates:
                    assign(column, index, value, "fill_constant", reason=rule["reason"])
                continue
            limit = rule.get("max_gap_hours")
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
                raise CleaningError(f"{method} requires explicit positive max_gap_hours for {column}.")
            if not complete_keys:
                raise CleaningError("Time-based filling requires complete timestamp/entity columns.")
            for group in groups:
                ordered = group.sort_values("timestamp_utc")
                if ordered["timestamp_utc"].duplicated().any():
                    issue("fill_skipped_duplicate_times", ordered.index, column)
                    continue
                indices = list(ordered.index)
                times = pd.to_datetime(ordered["timestamp_utc"], utc=True)
                position = 0
                while position < len(indices):
                    if indices[position] not in candidates:
                        position += 1
                        continue
                    start = position
                    while position < len(indices) and indices[position] in candidates:
                        position += 1
                    end = position
                    # Fill a whole bounded run or none of it; never partially
                    # interpolate the beginning of a long missing interval.
                    run = indices[start:end]
                    left = start - 1
                    right = end
                    boundary = right if method == "interpolate" else end - 1
                    valid = left >= 0 and (method != "interpolate" or right < len(indices))
                    if valid:
                        section = times.iloc[left:boundary + 1]
                        valid = all(math.isclose(value.total_seconds() / 3600, interval)
                                    for value in section.diff().dropna())
                        valid = valid and len(run) * interval <= limit
                        valid = valid and pd.notna(data.at[indices[left], column])
                        if method == "interpolate":
                            valid = valid and pd.notna(data.at[indices[right], column])
                        spec = schema["columns"][column]
                        endpoints = [data.at[indices[left], column]]
                        if method == "interpolate":
                            endpoints.append(data.at[indices[right], column])
                        for endpoint in endpoints:
                            valid = valid and pd.notna(endpoint) and math.isfinite(endpoint)
                            if column not in {"import_price", "export_price"} and pd.notna(endpoint):
                                valid = valid and ("min" not in spec or endpoint >= spec["min"])
                                valid = valid and ("max" not in spec or endpoint <= spec["max"])
                                valid = valid and ("min_exclusive" not in spec or endpoint > spec["min_exclusive"])
                    if not valid:
                        issue("fill_skipped_gap_or_boundary", run, column, method=method, max_gap_hours=limit)
                        continue
                    for offset_in_run, index in enumerate(run, start=1):
                        value = data.at[indices[left], column]
                        if method == "interpolate":
                            right_value = data.at[indices[right], column]
                            value += (right_value - value) * offset_in_run / (len(run) + 1)
                        assign(column, index, value, f"fill_{method}",
                               max_gap_hours=limit, uses_future_endpoint=method == "interpolate")
        for index in sorted(drop_rows):
            operation("drop_missing_row", row=int(index), before=data.loc[index].to_dict())
        data = data.drop(index=list(drop_rows))

        # Physical limits are separate from baseline tariff conventions.
        for column in data.columns:
            spec = schema["columns"].get(column, {})
            if spec.get("type") != "number":
                continue
            values = pd.to_numeric(data[column], errors="coerce")
            invalid = values.notna() & ~np.isfinite(values)
            if column not in {"import_price", "export_price"}:
                if "min" in spec:
                    invalid |= values < spec["min"]
                if "max" in spec:
                    invalid |= values > spec["max"]
                if "min_exclusive" in spec:
                    invalid |= values <= spec["min_exclusive"]
            if invalid.any():
                issue("physically_invalid", data.index[invalid], column, severity="error")
            if "const" in spec and (values.notna() & (values != spec["const"])).any():
                issue("schema_constant_mismatch", data.index[values.notna() & (values != spec["const"])], column)
            remaining = data.index[data[column].isna()]
            if len(remaining):
                issue("unresolved_missing", remaining, column)

        if record_type == "price" and {"import_price", "export_price"} <= set(data.columns):
            buy = pd.to_numeric(data["import_price"], errors="coerce")
            sell = pd.to_numeric(data["export_price"], errors="coerce")
            incompatible = (buy < 0) | (sell < 0) | (sell > buy)
            if incompatible.any():
                issue("baseline_tariff_incompatible_preserved", data.index[incompatible],
                      reason="Negative prices and export/import spreads are market data, not physical errors.")

        outliers = policy.get("outliers", {})
        if outliers.get("method") not in {"iqr", "disabled"}:
            raise CleaningError("Outlier method must be iqr or disabled.")
        multiplier = outliers.get("multiplier", 3)
        minimum = outliers.get("min_observations", 8)
        if not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier) or multiplier <= 0 or not isinstance(minimum, int) or minimum < 4:
            raise CleaningError("Outlier settings need a positive multiplier and at least four observations.")
        if outliers["method"] == "iqr":
            for group in groups:
                group = data.loc[data.index.intersection(group.index)]
                for column in group.columns:
                    if schema["columns"].get(column, {}).get("type") != "number":
                        continue
                    values = pd.to_numeric(group[column], errors="coerce")
                    values = values[np.isfinite(values)]
                    if len(values) < minimum:
                        continue
                    lower, upper = values.quantile([0.25, 0.75])
                    spread = upper - lower
                    flagged = values[(values < lower - multiplier * spread) | (values > upper + multiplier * spread)]
                    if len(flagged):
                        issue("suspicious_outlier_preserved", flagged.index, column,
                              lower_bound=lower - multiplier * spread,
                              upper_bound=upper + multiplier * spread)

        for column in data.columns:
            remaining = data.index[data[column].isna()]
            if len(remaining) and schema["columns"].get(column, {}).get("type") != "number":
                issue("unresolved_missing", remaining, column)
        output_metadata = deepcopy(metadata)
        output_metadata.setdefault("source_metadata", {})
        output_metadata["cleaning"] = {"column_mapping": mapping, "input_units": units,
                                       "policy": deepcopy(policy), "profile": profile}
        if currency is not None:
            output_metadata["currency_code"] = currency
        records = [{column: _plain(value) for column, value in row.items()}
                   for row in data.to_dict("records")]
        try:
            validate_dataset(records, record_type, units=output_units, metadata=output_metadata)
            report["canonical_validation"] = {"status": "passed"}
        except SchemaValidationError as exc:
            report["canonical_validation"] = {"status": "failed", "reason": str(exc)}
        report["output_rows"] = len(data)
        report["remaining_missing_by_column"] = {column: int(data[column].isna().sum()) for column in data}
        operation("complete", output_rows=len(data), canonical_validation=report["canonical_validation"])
        return {"data": data, "units": output_units,
                "metadata": output_metadata, "report": report}
    except CleaningError as exc:
        exc.report = report
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise CleaningError(f"Cleaning failed: {exc}", report) from exc


def load_and_clean_csv(path, record_type, *, units, metadata, config=None,
                       profile=None, currency_code=None):
    """Load CSV read-only, record its byte checksum and return in-memory outputs.

    Only empty cells are considered missing at read time. Strings such as NA
    remain literal until numeric/timestamp validation flags them; no zero fills.
    """
    path = Path(path)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(64 * 1024), b""):
                digest.update(chunk)
            size = stream.tell()
            stream.seek(0)
            frame = pd.read_csv(stream, dtype=object, keep_default_na=False, na_values=[""])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise CleaningError(f"Cannot load raw CSV {path}: {exc}") from exc
    result = clean_dataframe(frame, record_type, units=units, metadata=metadata,
                             config=config, profile=profile, currency_code=currency_code)
    source = {"path": str(path.resolve()), "sha256": digest.hexdigest(), "file_size_bytes": size}
    result["metadata"]["cleaning"]["raw_file"] = source
    result["report"]["operations"].insert(0, {"operation": "load_csv_read_only", **source})
    return result
