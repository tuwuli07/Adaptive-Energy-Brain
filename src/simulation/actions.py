"""Physical battery requests, independent of any agent or normalized action space.

Positive kW discharges to the bus; negative kW charges. Invalid values fail;
finite oversized requests remain intact for dispatch to constrain and report.
"""

from dataclasses import dataclass
import math
from numbers import Real


def finite_number(value, name):
    """Reject booleans, nonnumeric values and nonfinite commands/features."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name}: expected a finite number.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name}: expected a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected a finite number.")
    return result


@dataclass(frozen=True)
class BatteryAction:
    """Requested AC-side power; actual execution comes from DispatchResult."""

    requested_power_kw: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "requested_power_kw",
                           finite_number(self.requested_power_kw, "requested_power_kw"))

    @classmethod
    def from_directions(cls, *, charge_kw=0.0, discharge_kw=0.0):
        charge = finite_number(charge_kw, "charge_kw")
        discharge = finite_number(discharge_kw, "discharge_kw")
        if charge < 0 or discharge < 0 or (charge > 0 and discharge > 0):
            raise ValueError("Use nonnegative power in only one battery direction.")
        return cls(discharge - charge)

    def dispatch_kwargs(self):
        """Use engine.step(observation, **action.dispatch_kwargs())."""
        return {"requested_battery_power_kw": self.requested_power_kw}


@dataclass(frozen=True)
class ExecutedBatteryAction:
    """Separate original request from the feasible power actually executed."""

    requested_power_kw: float
    actual_power_kw: float
    adjustment_codes: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("requested_power_kw", "actual_power_kw"):
            object.__setattr__(self, name, finite_number(getattr(self, name), name))
        codes = tuple(self.adjustment_codes)
        if any(not isinstance(code, str) or not code for code in codes):
            raise ValueError("adjustment_codes must contain nonempty strings.")
        object.__setattr__(self, "adjustment_codes", codes)

    @classmethod
    def from_dispatch(cls, result):
        return cls(result.requested_battery_power_kw, result.actual_battery_power_kw,
                   tuple(item.code for item in result.adjustments))
