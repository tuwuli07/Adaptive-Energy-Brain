"""Campus AC demand from existing readings; no load synthesis or control.

Use an explicit building registry to preserve and sum building loads exactly
once. If only a campus total exists, construct without building_ids and call
evaluate_total; it does not invent building-level allocations. Provenance can be
one canonical metadata mapping or an exact per-building mapping of metadata.
Optional limits must have a declared parameter origin; no maxima are inferred.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from src.simulation._validation import (
    ComponentValidationError,
    identifier,
    input_origin,
    parameter_origin,
    power_kw,
    source_provenance,
    timestamp_utc as _timestamp,
    total_power,
)


@dataclass(frozen=True)
class DemandOutput:
    timestamp_utc: str
    campus_load_kw: float
    building_loads_kw: Mapping[str, float]
    _metadata: dict = field(repr=False, compare=False)

    @property
    def metadata(self):
        return deepcopy(self._metadata)


class CampusDemand:
    def __init__(self, provenance, *, building_ids=None, building_limits_kw=None,
                 campus_limit_kw=None, limits_origin=None):
        if building_ids is not None:
            if isinstance(building_ids, str) or not isinstance(building_ids, (list, tuple)) or not building_ids:
                raise ComponentValidationError("building_ids must be a nonempty list/tuple, or None for aggregate-only demand.")
            self._building_ids = tuple(identifier(item, "building_id") for item in building_ids)
            if len(set(self._building_ids)) != len(self._building_ids):
                raise ComponentValidationError("Duplicate building IDs would double-count demand.")
        else:
            self._building_ids = None
        if isinstance(provenance, Mapping) and "source_id" in provenance:
            common = source_provenance(provenance)
            self._provenance = {item: deepcopy(common) for item in self._building_ids} if self._building_ids else common
        elif self._building_ids and isinstance(provenance, Mapping) and set(provenance) == set(self._building_ids):
            self._provenance = {item: source_provenance(provenance[item]) for item in self._building_ids}
        else:
            raise ComponentValidationError("Provide canonical provenance or exact per-building provenance.")
        if building_limits_kw is not None and not isinstance(building_limits_kw, Mapping):
            raise ComponentValidationError("building_limits_kw must be a mapping.")
        if building_limits_kw and (not self._building_ids or not set(building_limits_kw) <= set(self._building_ids)):
            raise ComponentValidationError("Demand limits reference unregistered buildings.")
        self._limits = {item: power_kw(value, f"{item}.limit_kw") for item, value in (building_limits_kw or {}).items()}
        self._campus_limit = None if campus_limit_kw is None else power_kw(campus_limit_kw, "campus_limit_kw")
        self._limits_origin = parameter_origin(limits_origin) if self._limits or self._campus_limit is not None else None

    @property
    def building_ids(self):
        return self._building_ids

    def _output(self, stamp, total, loads):
        if self._campus_limit is not None and total > self._campus_limit:
            raise ComponentValidationError("campus_load_kw exceeds the configured campus limit.")
        origins = {item: input_origin(value) for item, value in self._provenance.items()} if self._building_ids else {"campus": input_origin(self._provenance)}
        metadata = {"source_id": "simulation_campus_demand", "origin": "derived", "derived_kind": "simulator",
                    "source_metadata": {"input_provenance": deepcopy(self._provenance), "input_origins": origins,
                        "aggregation": "sum of distinct registered buildings" if self._building_ids else "provided campus aggregate; no invented building allocation",
                        "limits": {"buildings_kw": deepcopy(self._limits), "campus_kw": self._campus_limit, "origin": self._limits_origin},
                        "unit": "kW"}}
        return DemandOutput(stamp, total, MappingProxyType(dict(loads)), metadata)

    def evaluate(self, building_loads_kw, *, timestamp_utc):
        if self._building_ids is None:
            raise ComponentValidationError("Configure building_ids before aggregating building records.")
        if not isinstance(building_loads_kw, Mapping) or set(building_loads_kw) != set(self._building_ids):
            raise ComponentValidationError("Building load keys must match the registry exactly; missing, extra or campus-total rows are not allowed.")
        loads = {item: power_kw(building_loads_kw[item], f"{item}.building_load_kw") for item in self._building_ids}
        for item, limit in self._limits.items():
            if loads[item] > limit:
                raise ComponentValidationError(f"{item}: building_load_kw exceeds configured limit {limit}.")
        return self._output(_timestamp(timestamp_utc), total_power(loads.values(), "campus_load_kw"), loads)

    def evaluate_total(self, campus_load_kw, *, timestamp_utc):
        if self._building_ids is not None:
            raise ComponentValidationError("Do not mix aggregate-only and building-level demand modes.")
        return self._output(_timestamp(timestamp_utc), power_kw(campus_load_kw, "campus_load_kw"), {})

    def read(self, observation):
        loads = observation.building_loads_kw
        if self._building_ids is None:
            if loads:
                raise ComponentValidationError("Building information is available; configure a registry to preserve it.")
            return self.evaluate_total(observation.campus_load_kw, timestamp_utc=observation.timestamp_utc)
        result = self.evaluate(loads, timestamp_utc=observation.timestamp_utc)
        declared = power_kw(observation.campus_load_kw, "observation.campus_load_kw")
        if not math.isclose(result.campus_load_kw, declared, rel_tol=1e-12, abs_tol=1e-9):
            raise ComponentValidationError("Observation campus_load_kw disagrees with its building-level sum.")
        return result
