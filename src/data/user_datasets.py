"""Isolated, bounded CSV intake for descriptive analysis, never model training.

Power samples are interval-start averages, held until the next timestamp. The
last row has no inferred duration. Missing observations exclude their intervals.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
import threading
import uuid

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_DATASETS = 20
FIELDS = {
    "timestamp": {"label": "Timestamp", "units": ["ISO8601"], "aliases": ["timestamp", "timestamp_utc", "datetime", "date_time"]},
    "demand_kw": {"label": "Electricity demand", "units": ["kW", "W", "MW"], "aliases": ["demand_kw", "load_kw", "demand", "consumption_kw", "campus_demand_kw"]},
    "solar_kw": {"label": "Solar generation", "units": ["kW", "W", "MW"], "aliases": ["solar_kw", "pv_generation", "solar_available_kw"]},
    "wind_kw": {"label": "Wind generation", "units": ["kW", "W", "MW"], "aliases": ["wind_kw", "wind_generation", "wind_available_kw"]},
    "grid_import_kw": {"label": "Grid import", "units": ["kW", "W", "MW"], "aliases": ["grid_import_kw"]},
    "grid_export_kw": {"label": "Grid export", "units": ["kW", "W", "MW"], "aliases": ["grid_export_kw"]},
    "price": {"label": "Electricity price", "units": ["USD/kWh"], "aliases": ["price", "import_price", "electricity_price_usd_per_kwh"]},
    "battery_soc": {"label": "Battery charge", "units": ["fraction", "%"], "aliases": ["battery_soc", "soc", "battery_soc_after"]},
}
POWER_FIELDS = [key for key in FIELDS if key.endswith("_kw")]


class DatasetError(ValueError):
    def __init__(self, message, status=400, details=None):
        super().__init__(message)
        self.status = status
        self.details = details or {}


def parse_csv(raw, filename):
    if not isinstance(filename, str) or not filename.lower().endswith(".csv"):
        raise DatasetError("Choose a CSV file.")
    if not raw or not raw.strip():
        raise DatasetError("The CSV is empty.")
    if len(raw) > MAX_BYTES:
        raise DatasetError("The file exceeds the 10 MiB limit.", 413)
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError("Save the CSV as UTF-8 text.") from exc
    if any(ord(c) < 32 and c not in "\r\n\t" for c in content):
        raise DatasetError("The file contains unsupported binary or control characters.")
    try:
        reader = csv.reader(io.StringIO(content, newline=""), strict=True)
        columns = next(reader)
        if not 2 <= len(columns) <= 64 or any(not c.strip() or len(c) > 100 for c in columns) or len(set(columns)) != len(columns):
            raise DatasetError("Use 2–64 unique, non-empty column headings (100 characters maximum).")
        rows = []
        for line, row in enumerate(reader, 2):
            if len(row) != len(columns):
                raise DatasetError(f"CSV row {line} has the wrong number of columns.")
            if any(len(cell) > 2000 for cell in row):
                raise DatasetError(f"CSV row {line} contains an oversized value.")
            rows.append(dict(zip(columns, row)))
            if len(rows) > MAX_ROWS:
                raise DatasetError("The CSV exceeds 100,000 records.", 413)
        if not rows:
            raise DatasetError("The CSV has headings but no records.")
        return columns, rows
    except (csv.Error, StopIteration) as exc:
        raise DatasetError("The CSV is malformed. Check its commas and quotation marks.") from exc


def analyze(rows, mapping, assume_utc=False, confirm_conversions=False):
    if not isinstance(mapping, dict) or any(key not in FIELDS for key in mapping):
        raise DatasetError("Unrecognized column mapping.")
    columns = set(rows[0])
    selected = {key: spec for key, spec in mapping.items() if isinstance(spec, dict) and spec.get("column")}
    if "timestamp" not in selected or not any(key in selected for key in POWER_FIELDS):
        raise DatasetError("Map a timestamp and at least one power measurement.")
    used = [spec.get("column") for spec in selected.values()]
    if len(used) != len(set(used)):
        raise DatasetError("Map each CSV column to only one measurement.")
    converted = []
    for key, spec in selected.items():
        if spec.get("column") not in columns or spec.get("unit") not in FIELDS[key]["units"]:
            raise DatasetError(f"Choose a valid column and supported unit for {FIELDS[key]['label']}. Energy in kWh cannot be treated as power in kW.")
        if spec["unit"] in ("W", "MW", "%"):
            converted.append(key)
    if converted and not confirm_conversions:
        raise DatasetError("Confirm the selected unit conversions before applying the mapping.")
    errors, records = [], []
    missing = {key: 0 for key in selected if key != "timestamp"}
    for index, row in enumerate(rows, 2):
        record = {}
        for key, spec in selected.items():
            text = row[spec["column"]].strip()
            try:
                if key == "timestamp":
                    if not re.match(r"^\d{4}-\d\d-\d\d[T ]\d\d:\d\d", text):
                        raise ValueError("use an ISO 8601 date and time")
                    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
                    if stamp.tzinfo is None:
                        if not assume_utc:
                            raise ValueError("include a timezone offset or confirm UTC")
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    record[key] = stamp.astimezone(timezone.utc).isoformat()
                elif not text:
                    record[key] = None
                    missing[key] += 1
                else:
                    value = float(text)
                    if not math.isfinite(value):
                        raise ValueError("use a finite number")
                    unit = spec["unit"]
                    value *= {"W": .001, "MW": 1000, "%": .01}.get(unit, 1)
                    if not math.isfinite(value) or abs(value) > 1e12:
                        raise ValueError("value exceeds the supported numeric range")
                    if key in POWER_FIELDS and value < 0:
                        raise ValueError("power must be non-negative; map import and export separately")
                    if key == "battery_soc" and not 0 <= value <= 1:
                        raise ValueError("battery charge must be between 0 and 100%")
                    record[key] = value
            except (ValueError, OverflowError) as exc:
                if len(errors) < 30:
                    errors.append({"row": index, "field": key, "message": str(exc)})
        records.append(record)
    if errors:
        raise DatasetError("Some values need attention. Correct the CSV and upload it again, or adjust the mapping.", details={"errors": errors, "missing_values": missing})
    records.sort(key=lambda r: r["timestamp"])
    stamps = [datetime.fromisoformat(r["timestamp"]) for r in records]
    duplicates = len(stamps) - len(set(stamps))
    if duplicates:
        raise DatasetError("Duplicate timestamps would double-count energy. Remove duplicates before analysis.", details={"duplicate_timestamps": duplicates})
    if len(stamps) < 2:
        raise DatasetError("At least two unique timestamps are required for energy analysis.")
    deltas = [(b - a).total_seconds() / 3600 for a, b in zip(stamps, stamps[1:])]
    duration = sum(deltas)
    irregular = any(abs(d - deltas[0]) > 1e-8 for d in deltas)
    energy, covered, peaks, averages = {}, {}, {}, {}
    for key in POWER_FIELDS:
        if key not in selected:
            continue
        pairs = [(r[key], dt) for r, dt in zip(records, deltas) if r[key] is not None]
        covered[key] = sum(dt for _, dt in pairs)
        energy[key] = sum(v * dt for v, dt in pairs) if pairs else None
        values = [r[key] for r in records if r[key] is not None]
        peaks[key] = max(values) if values else None
        averages[key] = energy[key] / covered[key] if covered[key] else None
    warnings = []
    if irregular:
        warnings.append("Irregular sampling: power is held to the next timestamp, including gaps. Check long gaps before interpreting totals.")
    if any(missing.values()):
        warnings.append("Missing observations are not filled. Totals cover only intervals with a recorded starting value.")
    warnings.append("Energy uses interval-start average power until the next timestamp. The final row has no inferred duration.")
    # An output-to-demand ratio is not self-consumption. Require complete demand
    # and both renewable measurements to avoid silently inventing absent sources.
    renewable = share = None
    required = ("demand_kw", "solar_kw", "wind_kw")
    if all(k in energy and energy[k] is not None and covered[k] == duration for k in required):
        renewable = energy["solar_kw"] + energy["wind_kw"]
        share = renewable / energy["demand_kw"] * 100 if energy["demand_kw"] > 0 else None
    quality = "Ready with warnings" if irregular or any(missing.values()) else "Ready"
    compatibility = {
        "analysis": "ready", "forecasting": "requires_adaptation", "optimization": "unsupported",
        "checks": [
            {"name": "Input features", "status": "incomplete", "detail": "Requires seven node-level power histories and irradiance, temperature, and wind speed; aggregate CSV measurements do not establish these."},
            {"name": "Graph topology & node count", "status": "unverified", "detail": "Requires ordered B01–B05, PV01, W01, BUS01 nodes and 14 directed shared-bus edges."},
            {"name": "Sampling frequency", "status": "matches" if not irregular and abs(deltas[0] - 1) < 1e-8 else "incompatible", "detail": "Existing model requires contiguous hourly UTC history."},
            {"name": "History window", "status": "sufficient_length" if duration >= 23 else "insufficient", "detail": "24 consecutive hourly records are required, with complete features."},
            {"name": "Forecast horizon", "status": "fixed", "detail": "The trained model predicts the next 24 hours."},
            {"name": "Units", "status": "normalized", "detail": "Mapped power converted explicitly to kW; this does not establish model compatibility."},
            {"name": "Preprocessing", "status": "unverified", "detail": "Must match the original training-fitted scalers, masks, node attributes, and feature ordering."},
            {"name": "Model checkpoint", "status": "unverified", "detail": "No uploaded CSV is sent to a checkpoint. A verified inference adapter is required."},
        ],
        "explanation": "Ready for descriptive energy analysis. Forecasting requires additional compatible inputs and verified preprocessing. Optimization and reference study results do not apply to this upload.",
    }
    return {"records": records, "summary": {"record_count": len(records), "start": records[0]["timestamp"], "end": records[-1]["timestamp"], "duration_hours": duration, "sampling_minutes": statistics.median(deltas) * 60, "maximum_interval_hours": max(deltas), "irregular": irregular, "measurements": list(missing), "missing_values": missing, "duplicate_timestamps": 0, "quality": quality, "warnings": warnings, "energy_kwh": energy, "covered_hours": covered, "peak_kw": peaks, "average_kw": averages, "renewable_generation_kwh": renewable, "renewable_to_demand_pct": share}, "compatibility": compatibility}


class DatasetStore:
    """Private local storage adapter. Configure a mounted volume for persistence.

    No path is derived from an uploaded filename. A production multi-user
    service must replace/scope this adapter with authenticated tenant storage.
    """
    def __init__(self, root=None):
        configured = root or os.environ.get("ENERGY_DATASET_STORAGE")
        self.persistent = bool(configured)
        self.root = Path(configured or tempfile.mkdtemp(prefix="energy-uploads-" )).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def info(self):
        return {"mode": "configured_volume" if self.persistent else "temporary", "message": "Stored on the configured volume. Durability depends on your deployment's volume and backup policy." if self.persistent else "Temporary uploads: unavailable after server restart. Keep your original CSV.", "max_bytes": MAX_BYTES, "max_rows": MAX_ROWS, "max_datasets": MAX_DATASETS, "fields": FIELDS}

    def _dir(self, dataset_id):
        if not isinstance(dataset_id, str) or not re.fullmatch(r"[a-f0-9]{32}", dataset_id):
            raise DatasetError("Dataset not found.", 404)
        path = self.root / dataset_id
        if not path.is_dir() or path.is_symlink():
            raise DatasetError("Dataset not found.", 404)
        return path

    def _write(self, path, data):
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
        temp.replace(path)

    def upload(self, raw, filename):
        columns, rows = parse_csv(raw, filename)
        # Only exact names indicating units get a unit suggestion. Ambiguous
        # aliases require the user to select their actual source unit.
        mapping = {}
        for key, spec in FIELDS.items():
            match = next((c for c in columns if c.strip().lower() in spec["aliases"]), "")
            mapping[key] = {"column": match, "unit": "ISO8601" if key == "timestamp" else "kW" if match.lower().endswith("_kw") else ""}
        name = filename.replace("\\", "/").split("/")[-1][:180]
        meta = {"id": uuid.uuid4().hex, "name": name, "created_at": datetime.now(timezone.utc).isoformat(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "columns": columns, "record_count": len(rows), "preview": rows[:8], "mapping": mapping, "status": "needs_mapping", "origin": "user_upload_unverified", "license": "user_supplied_not_verified"}
        with self.lock:
            if len(self.list()) >= MAX_DATASETS:
                raise DatasetError("This storage area has reached its 20-dataset limit.", 409)
            directory = self.root / meta["id"]
            directory.mkdir()
            (directory / "original.csv").write_bytes(raw)
            self._write(directory / "metadata.json", meta)
        return meta

    def list(self):
        with self.lock:
            result = []
            for path in self.root.glob("*/metadata.json"):
                if re.fullmatch(r"[a-f0-9]{32}", path.parent.name) and not path.parent.is_symlink():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    result.append({k: data[k] for k in ("id", "name", "status", "record_count", "created_at")})
            return sorted(result, key=lambda d: d["created_at"], reverse=True)

    def get(self, dataset_id):
        with self.lock:
            return json.loads((self._dir(dataset_id) / "metadata.json").read_text(encoding="utf-8"))

    def apply_mapping(self, dataset_id, payload):
        if not isinstance(payload, dict):
            raise DatasetError("A mapping object is required.")
        with self.lock:
            directory = self._dir(dataset_id)
            meta = self.get(dataset_id)
            _, rows = parse_csv((directory / "original.csv").read_bytes(), meta["name"])
            result = analyze(rows, payload.get("mapping"), payload.get("assume_utc") is True, payload.get("confirm_conversions") is True)
            meta.update({"mapping": payload["mapping"], "assume_utc": payload.get("assume_utc") is True, "conversions_confirmed": payload.get("confirm_conversions") is True, "status": "ready", "summary": result["summary"], "compatibility": result["compatibility"]})
            self._write(directory / "analysis.json", result)
            self._write(directory / "metadata.json", meta)
            return meta

    def analysis(self, dataset_id):
        with self.lock:
            path = self._dir(dataset_id) / "analysis.json"
            if not path.exists():
                raise DatasetError("Confirm the column mapping before activating this dataset.", 409)
            return json.loads(path.read_text(encoding="utf-8"))
