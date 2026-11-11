"""Data-driven AC renewable components; no weather-to-power model or dispatch.

SolarPV and WindGeneration share evaluate(available_generation_kw,
timestamp_utc=..., curtailment_kw=0) and read(HourlyObservation, ...). They use
existing per-asset values, never recalculate generation from irradiance/speed.
Missing power must be resolved upstream; no automatic synthetic fallback exists.
Limits are explicit kW with labeled parameter origin. Invalid requests are
rejected, not clipped: a future dispatch controller owns feasibility correction.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math

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
class RenewableOutput:
    timestamp_utc: str
    asset_id: str
    generation_type: str
    available_generation_kw: float
    curtailed_kw: float
    used_generation_kw: float
    _metadata: dict = field(repr=False, compare=False)

    @property
    def metadata(self):
        return deepcopy(self._metadata)


@dataclass(frozen=True)
class RenewableTotals:
    timestamp_utc: str
    total_available_kw: float
    total_curtailed_kw: float
    total_used_kw: float
    by_asset: tuple[RenewableOutput, ...]

    @property
    def metadata(self):
        return {"origin": "derived", "derived_kind": "simulator", "unit": "kW",
                "aggregation": "Each distinct asset counted once at one timestamp",
                "assets": {item.asset_id: item.metadata for item in self.by_asset}}


class RenewableGenerator:
    """Reusable measured-power interface for one registered renewable asset."""

    def __init__(self, asset_id, generation_type, capacity_kw, provenance, *, capacity_origin):
        self._asset_id = identifier(asset_id, "asset_id")
        if generation_type not in ("solar", "wind"):
            raise ComponentValidationError("generation_type must be solar or wind.")
        self._generation_type = generation_type
        self._capacity_kw = power_kw(capacity_kw, "capacity_kw", positive=True)
        self._capacity_origin = parameter_origin(capacity_origin)
        self._provenance = source_provenance(provenance)

    @property
    def asset_id(self):
        return self._asset_id

    @property
    def capacity_kw(self):
        return self._capacity_kw

    @property
    def generation_type(self):
        return self._generation_type

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def evaluate(self, available_generation_kw, *, timestamp_utc, curtailment_kw=0):
        stamp = _timestamp(timestamp_utc)
        available = power_kw(available_generation_kw, "available_generation_kw")
        curtailed = power_kw(curtailment_kw, "curtailment_kw")
        if available > self.capacity_kw:
            raise ComponentValidationError(f"{self.asset_id}: available generation exceeds capacity_kw {self.capacity_kw}.")
        if curtailed > available:
            raise ComponentValidationError(f"{self.asset_id}: curtailment_kw cannot exceed available generation.")
        metadata = {"source_id": f"simulation_{self.asset_id}", "origin": "derived", "derived_kind": "simulator",
                    "source_metadata": {"input_provenance": self.provenance,
                        "available_power_origin": input_origin(self._provenance),
                        "rating": {"capacity_kw": self.capacity_kw, "origin": self._capacity_origin},
                        "operation": "used_generation_kw = available_generation_kw - curtailment_kw",
                        "curtailment_origin": "explicit simulation action", "unit": "kW"}}
        return RenewableOutput(stamp, self.asset_id, self.generation_type,
                               available, curtailed, available - curtailed, metadata)

    def read(self, observation, *, curtailment_kw=0):
        """Read this asset's value, never an aggregate masquerading as an asset."""
        field_name = "pv_asset_powers_kw" if self.generation_type == "solar" else "wind_asset_powers_kw"
        values = getattr(observation, field_name, None)
        if not isinstance(values, Mapping) or self.asset_id not in values:
            raise ComponentValidationError(f"Missing observed {self.generation_type} power for asset {self.asset_id}; no weather conversion or zero fill is implicit.")
        return self.evaluate(values[self.asset_id], timestamp_utc=observation.timestamp_utc,
                             curtailment_kw=curtailment_kw)


class SolarPV(RenewableGenerator):
    def __init__(self, asset_id, capacity_kw, provenance, *, capacity_origin):
        super().__init__(asset_id, "solar", capacity_kw, provenance, capacity_origin=capacity_origin)


class WindGeneration(RenewableGenerator):
    def __init__(self, asset_id, capacity_kw, provenance, *, capacity_origin):
        super().__init__(asset_id, "wind", capacity_kw, provenance, capacity_origin=capacity_origin)


def combine_renewables(outputs):
    """Combine distinct assets at one interval without double-counting generation."""
    outputs = tuple(outputs)
    if not outputs or not all(isinstance(item, RenewableOutput) for item in outputs):
        raise ComponentValidationError("Provide at least one RenewableOutput.")
    if len({item.asset_id for item in outputs}) != len(outputs):
        raise ComponentValidationError("Duplicate renewable assets would double-count generation.")
    times = {_timestamp(item.timestamp_utc) for item in outputs}
    if len(times) != 1:
        raise ComponentValidationError("Renewables must share one hourly timestamp.")
    for item in outputs:
        if item.generation_type not in ("solar", "wind"):
            raise ComponentValidationError("Renewable outputs must identify solar or wind generation.")
        available = power_kw(item.available_generation_kw, "available_generation_kw")
        curtailed = power_kw(item.curtailed_kw, "curtailed_kw")
        used = power_kw(item.used_generation_kw, "used_generation_kw")
        if curtailed > available or not math.isclose(used, available - curtailed, rel_tol=1e-12, abs_tol=1e-9):
            raise ComponentValidationError("Inconsistent renewable output: used must equal available minus curtailed.")
    return RenewableTotals(next(iter(times)),
                           total_power((item.available_generation_kw for item in outputs), "total_available_kw"),
                           total_power((item.curtailed_kw for item in outputs), "total_curtailed_kw"),
                           total_power((item.used_generation_kw for item in outputs), "total_used_kw"), outputs)
