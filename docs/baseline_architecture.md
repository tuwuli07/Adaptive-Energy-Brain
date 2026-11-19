# Part 4.1: Baseline controller architecture

## Status and scope

This document specifies four controller strategies. The zero-action,
self-consumption, price-aware, and battery-preserving controllers are
implemented. The implementations do not change simulator behavior or claim
experimental results. Controllers remain independent of the physical simulator:
each receives one causal observation and returns a battery request. The
simulator remains responsible for resolving that request against physical
constraints.

The existing interfaces are:

- [`AgentObservation`](../src/simulation/state.py) and its `features()` method
  for controller input.
- [`BatteryAction`](../src/simulation/actions.py) for signed battery requests.
- [`MicrogridEnvironment`](../src/simulation/environment.py) for episode
  lifecycle and [`DispatchResult`](../src/simulation/dispatch.py) for executed
  flows and adjustment diagnostics.
- [`StepMetrics` and `MetricValues`](../src/simulation/metrics.py) for
  per-step and cumulative accounting.

The shared controller interface, name-based factory, and four rule-based
strategies are implemented in
[`src/baselines/`](../src/baselines/). The comparative evaluation harness in
[`src/evaluation/`](../src/evaluation/) provides same-episode
experiment runs, metric comparisons, deterministic stress scenarios, and
headless plots. Controllers implement
`reset()` and `select_action(observation) -> BatteryAction`. They must not
receive the environment, replay data, dispatcher, or metrics tracker.

## Shared controller contract

| Item | Contract |
|---|---|
| Input | One `AgentObservation` for the current decision boundary. |
| Output | One `BatteryAction` containing a finite signed AC power request in kW. |
| Sign | Positive requests discharge to the bus; negative requests charge. Zero requests no discretionary battery action. |
| Physical limits | Feasible `charge_limit_kw` and `discharge_limit_kw` are observable. The controller may use them according to its rule, but dispatch is authoritative and may adjust any request for current-interval feasibility; adjustments are recorded. |
| Renewable curtailment | Not part of `BatteryAction`. The controller cannot ask for curtailment through the environment action interface; dispatch can curtail only under its configured physical-feasibility rules. |
| Timing | Act only at the observation's `decision_timestamp_utc`. `features()` describes the previous completed hour and the battery state at the current decision boundary. The request applies to the next replay row. |
| Forecasts | These Part 4 baselines do not require or consume forecasts. Optional forecasts are explicit `ForecastFeature` values attached to `AgentObservation`; the Part 5 adapter validates issue-time, data-cutoff, target-time and ordering contracts. Any future comparison must provide equivalent forecast inputs to all compared controllers. |
| Independence | A controller may retain only its own declared internal state and static parameters. It must not read future rows, realized dispatch outcomes before they are returned, or another controller's state. |

`AgentObservation.features()` currently exposes an allowlisted mapping:

| Feature | Meaning at decision time |
|---|---|
| `solar_generation_kw`, `wind_generation_kw`, `campus_demand_kw` | Values from the previous completed hourly row; not the upcoming interval's realizations. |
| `battery_soc`, `battery_energy_kwh` | Battery state at the current boundary, before the next action. |
| `grid_import_price`, `grid_export_price` | Lagged prices from the previous completed row; export price may be `None`. These do not imply advance notice of the next price. |
| `battery_capacity_kwh`, `soc_min`, `soc_max`, `charge_limit_kw`, `discharge_limit_kw`, `charge_efficiency`, `discharge_efficiency`, `time_step_hours` | Current configured battery limits, feasible power bounds, and physical conversion parameters. |
| `grid_import_limit_kw`, `grid_export_limit_kw` | Configured connection limits; the latter is zero when missing export-price handling disables export. |

The observation also carries its timestamps, configuration, and optionally
explicit forecasts. Baseline implementations should use `features()` and only
the metadata needed to interpret an allowed forecast. They must not treat
`state.timestamp_utc` or a forecast target timestamp as access to realized data.
The current environment creates observations from the previous row and does not
provide future realized demand, renewable output, or tariff values.

## Baseline strategies

All rules below use only the shared input contract, return `BatteryAction`, and
leave final feasibility to `DispatchEngine`. Use finite nonnegative magnitudes
and bound requests by the feasible power features. The actions are requests,
not guarantees about executed flows.

### 1. Zero-action controller

| Input | Output |
|---|---|
| `AgentObservation` (ignored) | `BatteryAction(0.0)` |

Requests no discretionary charging or discharging. This is the simplest
reference, not an optimized controller. Dispatch may still adjust battery power
when required for feasibility; report those adjustments rather than describing
the executed policy as always idle.

### 2. Renewable self-consumption controller

Use the lagged net-power proxy
`solar_generation_kw + wind_generation_kw - campus_demand_kw`. The configurable
`power_deadband_kw` is a finite nonnegative threshold; default it to `0.0`.

| Condition on observed proxy | Requested action |
|---|---|
| Proxy greater than the deadband | Request charging by the observed surplus; return the negative proxy as `BatteryAction` power. |
| Proxy less than the negative deadband | Request discharging by the observed deficit magnitude; return the positive proxy as `BatteryAction` power. |
| Absolute proxy is at most the deadband | `BatteryAction(0.0)`. |

This rule attempts to shift recently observed renewable surplus or demand
deficit into the next interval. It does not see or predict the next interval's
realized values. It cannot directly maximize renewable self-consumption,
control curtailment, or request export; battery limits, SOC, grid constraints,
and dispatch determine actual flows. The requested power is not clipped by the
controller; the simulator constrains it and records any adjustment.

### 3. Price-aware controller

Use only the lagged `grid_import_price` and thresholds fixed before final
evaluation. The current hypothetical defaults are configured in
[`configs/baselines.yaml`](../configs/baselines.yaml); record their values and
provenance with the experiment. Thresholds may be selected using training and
validation data, but must be frozen before test. The current implementation
does not estimate quantiles automatically.

| Observed lagged import price | Requested action |
|---|---|
| At or below the lower threshold | Request `-charge_limit_kw` only when the configured threshold spread clears efficiency and degradation costs; otherwise idle. |
| At or above the upper threshold | Request positive discharge up to the feasible limit and energy above the configured SOC reserve, only when the price clears the modeled cost test; otherwise idle. |
| Between thresholds | `BatteryAction(0.0)`. |

Require a strictly lower lower-threshold than upper-threshold; handle tied or
constant training prices by explicitly disabling the price signal (zero
request), not by inventing a spread. This rule does not use a future tariff or
forecast. It intentionally tests a lagged-price heuristic, not perfect
foresight or optimal arbitrage. It may charge from the grid during a low-price
signal and may discharge at a high-price signal; record resulting degradation,
import/export cost, and SOC.

The Part 2 canonical price schema rejects negative prices, and the default
simulation configuration does not enable them. Negative-price experiments
require an explicit supported input/configuration change and separate tests;
do not silently clip prices or interpret a missing export price as zero.

The price-aware controller is configured through
[`configs/baselines.yaml`](../configs/baselines.yaml). It compares the
round-trip-efficiency-adjusted high-price revenue with low-price replacement
energy cost plus the existing linear AC-throughput degradation proxy in
`configs/simulation_costs.yaml`. It charges only when the configured threshold
spread clears that cost test; high-price discharge must also clear the test.
For round-trip efficiency `r = charge_efficiency * discharge_efficiency`, the
test is `high_price * r > low_price + degradation_cost * (1 + r)`, where the
degradation coefficient applies to AC charge-plus-discharge throughput. The
minimum SOC reserve applies to discharge and is bounded below by the
simulator's own minimum SOC. Negative prices and non-USD thresholds are not
supported by the default source/configuration bundle. A custom threshold set
must use the same currency as both the observation and the configured
degradation cost.

The simulator supports grid-assisted charging through negative battery power:
charging is included in bus demand, with any remaining deficit supplied by grid
imports subject to grid and battery limits. However, the controller sees only
the last completed row's price. That lagged price is used to request an action
for the next interval; the upcoming realized price is not exposed or forecast.
This is a causal heuristic, not price-forecasted arbitrage.

The battery-preserving controller uses the same baseline configuration file.
It charges only when the lagged observation indicates renewable surplus, stops
at `preferred_soc_max`, and caps discharge so the requested transition does not
cross `preferred_soc_min`. `power_deadband_kw` and separate maximum charge and
discharge powers reduce avoidable cycling. A configured preferred band must fit
within the physical SOC bounds in the observation. Because this controller
does not request grid charging, it cannot restore the preferred band when
renewable surplus is unavailable. This is a conservative control heuristic,
not a validated battery-aging model or a guarantee of extended battery life.

### 4. Battery-preserving controller

Use the same lagged net-power proxy and surplus/deficit direction as the
renewable self-consumption controller, but set an explicit reserve SOC
`soc_reserve` satisfying `soc_min <= soc_reserve <= soc_max`. Freeze this
parameter before test evaluation; a recommended neutral default is the
configured initial SOC.

For a surplus, request charging up to the smaller of the observed surplus and
`charge_limit_kw`. For a deficit, additionally cap discharge to the AC energy
available above reserve:

```text
reserve_energy_kwh = soc_reserve * battery_capacity_kwh
reserve_limited_discharge_kw =
    max(0, battery_energy_kwh - reserve_energy_kwh)
    * discharge_efficiency / time_step_hours
requested_discharge_kw =
    min(deficit_kw, discharge_limit_kw, reserve_limited_discharge_kw)
```

Return zero when there is no observed deficit or no discharge headroom above
reserve. This is a reserve-aware self-consumption heuristic, not a guarantee of
future backup supply; current dispatch may still enforce other constraints.
Disclose the reserve value and report time at reserve and terminal SOC.

## Assumptions and limitations

The battery-preserving controller is configured in the `battery_preserving`
section of [`configs/baselines.yaml`](../configs/baselines.yaml). It charges
only from observed renewable surplus, stops requesting charge at the preferred
upper SOC, and limits discharge to the amount above the preferred lower SOC.
Its deadband and AC power caps are explicit parameters. As grid-assisted
charging is deliberately not requested, the controller cannot restore its
preferred band when no renewable surplus is observed; the band is a decision
target, not a guaranteed invariant. These heuristics do not establish or
guarantee improved battery lifespan.

- The simulator is an hourly, idealized single-bus energy-accounting model.
  These controllers do not add network physics, demand response, forecasting,
  reinforcement learning, optimization, or risk controls.
- Synthetic data and hypothetical asset/economic parameters are useful for
  deterministic software experiments only. They are not campus measurements or
  validated hardware specifications.
- All baseline rules receive delayed observations under the current environment.
  The resulting one-hour decision lag is part of the strategy comparison.
- Price signals are lagged, and the current data pipeline has the limitations
  described above. Performance on a synthetic tariff is not evidence of
  profitable or deployable energy arbitrage.
- No baseline creates its own reward or changes the simulator's cost
  definitions. Report actual dispatch and simulator metrics, including
  adjustments and unserved demand.
