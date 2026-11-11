"""Main-grid exchange and tariff accounting, independent of energy dispatch.

POSITIVE signed kW imports from the grid; NEGATIVE exports to it (different
from the battery sign). Separate nonnegative import_kw/export_kw requests are
also accepted, but simultaneous positive directions or mixed styles fail.
Finite requests saturate at configured connection limits and return the unmet
request explicitly. A future dispatch/engine must resolve any difference and
check campus energy balance BEFORE commit; this component never sheds load.

Energy = actual AC kW * dt hours. Import cost = imported kWh * import price;
export revenue = exported kWh * export price; net cost = cost - revenue.
Prices use configured currency/kWh; mismatched currencies/units are rejected.
Provided malformed/non-finite prices always fail, even in an idle direction.
Negative prices require allow_negative_prices; then signed costs/revenues are
retained. This model does not impose export_price <= import_price. The Part 2
loader still has its stricter initial tariff schema; no data values are changed.

Missing export price applies only to actual positive export: error rejects;
disable_export reduces exchange to zero; zero_compensation explicitly books
zero revenue. Every application is recorded. No price is required for a zero
flow or an unused direction. No demand, renewable or battery policy is here.
preview/commit/step mirror the battery's atomic, stale-checked transaction API.
"""

from copy import deepcopy
from copy import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import math
from numbers import Real
import re

from src.simulation._validation import ComponentValidationError, source_provenance
from src.simulation.config import (
    DEFAULT_CONFIG_PATH,
    SimulationConfig,
    SimulationConfigError,
    load_simulation_config,
    validate_simulation_config,
)


class GridError(ComponentValidationError):
    """Invalid exchange, tariff, timestamp or accounting; no transaction committed."""


@dataclass(frozen=True)
class GridState:
    currency_code: str
    imported_energy_kwh: float = 0.0
    exported_energy_kwh: float = 0.0
    import_cost: float = 0.0
    export_revenue: float = 0.0
    net_cost: float = 0.0
    elapsed_hours: float = 0.0
    step_count: int = 0
    next_timestamp_utc: str | None = None


@dataclass(frozen=True)
class GridTransaction:
    asset_id: str
    currency_code: str
    price_unit: str
    timestamp_utc: str | None
    time_step_hours: float
    requested_power_kw: float
    actual_power_kw: float
    import_power_kw: float
    export_power_kw: float
    imported_energy_kwh: float
    exported_energy_kwh: float
    import_price: float | None
    provided_export_price: float | None
    effective_export_price: float | None
    import_cost: float
    export_revenue: float
    net_cost: float
    power_limited: bool
    limiting_reasons: tuple[str, ...]
    export_price_assumed_zero: bool
    state_before: GridState
    state_after: GridState
    _revision: int = field(repr=False)
    _metadata: dict = field(repr=False, compare=False)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def unfulfilled_power_kw(self):
        """Signed request difference; not a claim of actual unserved campus load."""
        return self.requested_power_kw - self.actual_power_kw

    @property
    def step_cost(self):
        return self.net_cost


def _number(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GridError(f"{name}: expected a finite number.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise GridError(f"{name}: must be finite.") from exc
    if not math.isfinite(result):
        raise GridError(f"{name}: must be finite.")
    if (nonnegative and result < 0) or (positive and result <= 0):
        raise GridError(f"{name}: must be {'positive' if positive else 'nonnegative'}.")
    return result


def _timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", value):
        raise GridError("timestamp_utc: expected explicit UTC ISO datetime with seconds.")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat().replace("+00:00", "Z")
    except ValueError as exc:
        raise GridError(f"Invalid timestamp_utc: {value!r}.") from exc


class Grid:
    def __init__(self, config=None, *, start_timestamp_utc=None):
        try:
            config = load_simulation_config() if config is None else config
            if not isinstance(config, SimulationConfig):
                raise GridError("config must be a validated SimulationConfig.")
            self._config = validate_simulation_config(asdict(config))
        except SimulationConfigError as exc:
            raise GridError(f"Invalid grid configuration: {exc}") from exc
        self._parameters = self._config.grid
        self._initial_timestamp = None if start_timestamp_utc is None else _timestamp(start_timestamp_utc)
        self._revision = 0
        self.reset()

    @classmethod
    def from_config(cls, path=DEFAULT_CONFIG_PATH, *, start_timestamp_utc=None):
        try:
            return cls(load_simulation_config(path), start_timestamp_utc=start_timestamp_utc)
        except SimulationConfigError as exc:
            raise GridError(f"Cannot initialize grid: {exc}") from exc

    @property
    def state(self):
        return self._state

    def fork(self):
        """Independent working ledger; immutable past transactions are shared."""
        result = copy(self)
        result._transactions = self._transactions.copy()
        return result

    @property
    def transactions(self):
        return tuple(self._transactions)

    def reset(self, *, start_timestamp_utc=None):
        stamp = self._initial_timestamp if start_timestamp_utc is None else _timestamp(start_timestamp_utc)
        state = GridState(self._parameters.currency_code, next_timestamp_utc=stamp)
        self._state, self._transactions = state, []
        self._revision += 1
        return state

    def _price(self, value, name):
        if value is None:
            return None
        result = _number(value, name)
        if result < 0 and not self._parameters.allow_negative_prices:
            raise GridError(f"{name}: negative prices require grid.allow_negative_prices: true.")
        return result

    def preview(self, requested_power_kw=None, *, import_kw=None, export_kw=None,
                import_price=None, export_price=None, currency_code=None,
                price_unit=None, time_step_hours=None, timestamp_utc=None,
                price_provenance=None):
        """Calculate a transaction without advancing counters, clock or ledger."""
        if requested_power_kw is not None and (import_kw is not None or export_kw is not None):
            raise GridError("Choose either signed requested_power_kw or separate import_kw/export_kw.")
        if import_kw is not None or export_kw is not None:
            buy_request = _number(0 if import_kw is None else import_kw, "import_kw", nonnegative=True)
            sell_request = _number(0 if export_kw is None else export_kw, "export_kw", nonnegative=True)
            if buy_request > 0 and sell_request > 0:
                raise GridError("Simultaneous import and export is not allowed.")
            request = buy_request - sell_request
        else:
            request = _number(0 if requested_power_kw is None else requested_power_kw, "requested_power_kw")
        p, before = self._parameters, self.state
        code = p.currency_code if currency_code is None else currency_code
        if code != p.currency_code:
            raise GridError(f"currency_code must match configured {p.currency_code}; no FX conversion is performed.")
        expected_unit = f"{code}/kWh"
        if price_unit is not None and price_unit != expected_unit:
            raise GridError(f"price_unit must be {expected_unit}; convert explicitly upstream.")
        dt = _number(self._config.time_step_hours if time_step_hours is None else time_step_hours,
                     "time_step_hours", positive=True)
        buy_price, sell_price = self._price(import_price, "import_price"), self._price(export_price, "export_price")
        if price_provenance is not None:
            try:
                price_provenance = source_provenance(price_provenance)
            except ComponentValidationError as exc:
                raise GridError(f"Invalid price provenance: {exc}") from exc
        stamp = before.next_timestamp_utc if timestamp_utc is None else _timestamp(timestamp_utc)
        if before.next_timestamp_utc is not None and stamp != before.next_timestamp_utc:
            raise GridError("timestamp_utc does not follow the committed grid interval.")
        if before.step_count > 0 and before.next_timestamp_utc is None and stamp is not None:
            raise GridError("Cannot add an absolute clock mid-episode; reset with start_timestamp_utc.")
        reasons = []
        buy, sell = min(max(request, 0.0), p.import_limit_kw), min(max(-request, 0.0), p.export_limit_kw)
        if request > p.import_limit_kw:
            reasons.append("import_power_limit")
        if -request > p.export_limit_kw:
            reasons.append("export_power_limit")
        effective_sell, assumed_zero = sell_price, False
        if sell > 0 and sell_price is None:
            if p.missing_export_price_handling == "error":
                raise GridError("export_price is required for positive export under the configured error policy.")
            if p.missing_export_price_handling == "disable_export":
                sell = 0.0
                reasons.append("missing_export_price_disables_export")
            else:
                effective_sell, assumed_zero = 0.0, True
        if buy > 0 and buy_price is None:
            raise GridError("import_price is required for positive import.")
        imported, exported = buy * dt, sell * dt
        cost = imported * buy_price if buy > 0 else 0.0
        revenue = exported * effective_sell if sell > 0 else 0.0
        net = cost - revenue
        for name, value in (("imported_energy_kwh", imported), ("exported_energy_kwh", exported),
                            ("import_cost", cost), ("export_revenue", revenue), ("net_cost", net)):
            _number(value, name)
        values = (before.imported_energy_kwh + imported, before.exported_energy_kwh + exported,
                  before.import_cost + cost, before.export_revenue + revenue,
                  before.net_cost + net, before.elapsed_hours + dt)
        for value in values:
            _number(value, "Cumulative grid accounting")
        next_stamp = None
        if stamp is not None:
            try:
                next_stamp = (datetime.fromisoformat(stamp.replace("Z", "+00:00")) + timedelta(hours=dt)).isoformat().replace("+00:00", "Z")
                if next_stamp == stamp:
                    raise GridError("time_step_hours is below datetime resolution.")
            except (ValueError, OverflowError) as exc:
                raise GridError(f"Grid timestamp cannot advance by the requested duration: {exc}") from exc
        after = GridState(code, *values, before.step_count + 1, next_stamp)
        actual = buy - sell
        metadata = {"origin": "derived", "derived_kind": "simulator",
                    "source_metadata": {"grid_parameters": asdict(p),
                        "parameter_origin": self._config.parameter_origin, "source_reference": self._config.source_reference,
                        "price_provenance": deepcopy(price_provenance),
                        "price_source_note": "Caller-supplied interval prices; no real utility tariff is inferred.",
                        "missing_export_price_handling": p.missing_export_price_handling,
                        "export_price_assumed_zero": assumed_zero, "power_unit": "kW", "energy_unit": "kWh",
                        "price_unit": expected_unit, "cost_currency": code,
                        "sign_convention": "positive import; negative export",
                        "limited_flow_note": "Requested/actual differences must be resolved by a future dispatch layer; no load shedding performed."}}
        return GridTransaction(p.asset_id, code, expected_unit, stamp, dt, request, actual,
                               buy, sell, imported, exported, buy_price, sell_price, effective_sell,
                               cost, revenue, net, actual != request, tuple(reasons), assumed_zero,
                               before, after, self._revision, metadata)

    def commit(self, transaction):
        if not isinstance(transaction, GridTransaction) or transaction.state_before != self.state or transaction._revision != self._revision:
            raise GridError("Stale or invalid grid transaction; state unchanged.")
        try:
            provenance = transaction.metadata["source_metadata"]["price_provenance"]
        except (KeyError, TypeError) as exc:
            raise GridError("Altered or malformed grid transaction metadata; state unchanged.") from exc
        expected = self.preview(transaction.requested_power_kw,
                                import_price=transaction.import_price,
                                export_price=transaction.provided_export_price,
                                currency_code=transaction.currency_code, price_unit=transaction.price_unit,
                                time_step_hours=transaction.time_step_hours, timestamp_utc=transaction.timestamp_utc,
                                price_provenance=provenance)
        if expected != transaction or expected.metadata != transaction.metadata:
            raise GridError("Altered or incompatible grid transaction; state unchanged.")
        self._transactions.append(transaction)
        self._state = transaction.state_after
        self._revision += 1
        return transaction

    def step(self, requested_power_kw=None, **kwargs):
        return self.commit(self.preview(requested_power_kw, **kwargs))

    def import_power(self, power_kw, **kwargs):
        return self.step(import_kw=power_kw, **kwargs)

    def export_power(self, power_kw, **kwargs):
        return self.step(export_kw=power_kw, **kwargs)
