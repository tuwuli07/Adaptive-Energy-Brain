"""Shared scalar and provenance checks for lightweight physical components."""

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import math
from numbers import Real
import re


class ComponentValidationError(ValueError):
    """A physical component received invalid values, limits or provenance."""


def identifier(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ComponentValidationError(f"{name}: expected a nonempty identifier without surrounding whitespace.")
    return value


def power_kw(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ComponentValidationError(f"{name}: expected a finite numeric kW value.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ComponentValidationError(f"{name}: expected finite kW.") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        bound = "positive" if positive else "nonnegative"
        raise ComponentValidationError(f"{name}: expected finite {bound} kW.")
    return result


def total_power(values, name):
    try:
        return power_kw(math.fsum(values), name)
    except OverflowError as exc:
        raise ComponentValidationError(f"{name}: aggregate power overflows finite kW.") from exc


def timestamp_utc(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:00:00(?:Z|\+00:00)", value):
        raise ComponentValidationError("timestamp_utc: expected an explicit hourly UTC ISO timestamp.")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ComponentValidationError(f"timestamp_utc: invalid timestamp {value!r}.") from exc
    return stamp.isoformat().replace("+00:00", "Z")


def source_provenance(value):
    """Preserve the Part 2 measured/simulated/derived metadata convention."""
    if not isinstance(value, Mapping):
        raise ComponentValidationError("provenance: expected a mapping.")
    identifier(value.get("source_id"), "provenance.source_id")
    if value.get("origin") not in ("measured", "simulated", "derived"):
        raise ComponentValidationError("provenance.origin: expected measured, simulated or derived.")
    if not isinstance(value.get("source_metadata"), Mapping):
        raise ComponentValidationError("provenance.source_metadata: expected a mapping.")
    if value["origin"] == "derived" and value.get("derived_kind") not in ("model", "calculated", "simulator"):
        raise ComponentValidationError("Derived provenance requires derived_kind: model, calculated or simulator.")
    if "synthetic" in value and type(value["synthetic"]) is not bool:
        raise ComponentValidationError("provenance.synthetic must be a boolean.")
    if value["origin"] == "measured" and value.get("synthetic") is True:
        raise ComponentValidationError("Measured provenance cannot simultaneously claim synthetic measurements.")
    return deepcopy(dict(value))


def input_origin(provenance):
    if provenance.get("synthetic") or provenance["origin"] == "simulated":
        return "synthetic"
    if provenance["origin"] == "derived":
        return "modeled" if provenance["derived_kind"] == "model" else "derived_" + provenance["derived_kind"]
    return "measured"


def parameter_origin(value):
    if value not in ("hypothetical_example", "documented_specification", "mixed"):
        raise ComponentValidationError("Parameter origin must be hypothetical_example, documented_specification or mixed.")
    return value
