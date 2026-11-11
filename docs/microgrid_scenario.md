# Provisional university campus microgrid scenario

## Status and scope

This is a proposed simulation design, not a description of a real campus.
No campus measurements, datasets, asset ratings, or trained models have been
obtained. This document defines requirements only; it implements no simulator.
See [data_requirements.md](data_requirements.md) for the variable contract.

## Initial topology

Start with **five simulated buildings**, identified as `B01` through `B05`.
Five is a manageable initial research scope with multiple demand series; it is
not an observed campus size. Building identities and usage types remain
unspecified until data selection. Do not infer building-level measurements by
silently dividing a campus total into five parts. Any such allocation must be
documented as simulated and cannot validate building-level forecasting.

All buildings connect to one idealized campus AC bus. Shared assets connect to
that bus: one aggregate PV plant (`PV01`), one aggregate wind plant (`W01`), one
battery (`BAT01`), and one grid connection (`GRID01`). Asset ratings are unknown;
they must be configured before simulation. Aggregate plants may represent
multiple devices, but no individual-device behavior is assumed.

| Entity | Relationship | Initial control role |
|---|---|---|
| B01-B05 | Each has one separate demand series and one bus connection | Fixed, non-dispatchable demand |
| PV01 | Shared campus supply; no building ownership allocation | Available production may be curtailed |
| W01 | Shared campus supply; no building ownership allocation | Available production may be curtailed |
| BAT01 | Shared campus storage | Charge or discharge within limits |
| GRID01 | Single campus point of common coupling | Import/export within configured limits |

Do not repeat shared PV, wind, or battery values once per building and then sum
them. Campus demand is the sum of the five building demands, without adding an
already aggregated campus demand again.

For a future GNN, an initial graph may contain five building nodes and one bus
node, with building-to-bus edges. Asset-to-bus membership is explicit metadata;
assets can later become separate nodes if justified. These edges describe the
assumed connection structure, not measured electrical distances, feeder
impedances, or statistically learned similarity. No graph is implemented yet.

## Time and energy conventions

- Decisions occur every hour: `time_step_hours = 1 h`.
- Each input row describes the interval beginning at `timestamp_utc`, covering
  `[timestamp_utc, timestamp_utc + 1 hour)`. Store timestamps in UTC with an
  explicit offset. A separately configured campus time zone supports calendars.
- Power is interval-average AC power in kW; energy is in kWh. For this design,
  interval energy equals interval-average power multiplied by one hour.
- Battery energy and state of charge are boundary states: an N-interval episode
  has N exogenous input rows and N+1 battery states.
- The environment may use realized hourly inputs to advance physics. At the
  beginning of an hour, a controller must only see information available then:
  previous observations, current battery state, known tariffs, and forecasts.
  Future realized demand and renewable output are not controller observations.

## Initial operating assumptions

The grid is available throughout each baseline episode. Grid limits must allow
a feasible fallback, such as supplying all demand when renewables are zero and
the battery is empty. Import and export limits are independent parameters;
export can be disabled by setting its limit to zero. There is no assumed net
metering agreement or real tariff. Initial tariffs are explicitly synthetic
scenario inputs unless a documented source is selected.

Ignore feeder losses, reactive power, voltage, frequency, and thermal constraints
for the first energy-balance simulator. Fixed building demand has no HVAC,
occupancy, demand response, EV, or heating submodel. Renewables can be curtailed;
their available power is distinct from power actually used. The initial battery
uses fixed charge/discharge efficiencies, no self-discharge, and no capacity
fade. Degradation cost, outages, and flexible loads are optional extensions.

Charge and discharge are nonnegative separate flows but must not occur
simultaneously. Apply the same restriction to grid import and export. Initially
require `import_price >= export_price >= 0` in a single declared currency to
avoid unsupported tariff behavior and artificial arbitrage.

Conceptual feasibility constraints (specifications, not implemented logic):

```text
campus_load = sum(building_load for B01 ... B05)
pv_used = pv_available - pv_curtailed
wind_used = wind_available - wind_curtailed
grid_import + pv_used + wind_used + battery_discharge
    = campus_load + battery_charge
next_energy = current_energy
    + charge_efficiency * battery_charge * time_step_hours
    - battery_discharge * time_step_hours / discharge_efficiency
soc = battery_energy / battery_capacity
```

All flows are AC-side except stored battery energy. Enforce SOC, power, grid,
and curtailment bounds. Initial SOC must lie inside bounds. If a scenario becomes
infeasible, report it explicitly; do not silently invent grid supply or shed
demand. Outage scenarios would require an explicit unmet-load variable and
penalty, outside the baseline minimum. A terminal SOC rule must be recorded to
prevent comparisons being distorted by draining storage at the episode end.

## Research use and boundaries

Forecasting targets are each building's demand and available PV/wind output.
Weather is optional when direct renewable power histories are available; it is
required for weather-to-power conversion if those histories are absent.
Multi-building synchronized histories are needed to evaluate spatial forecasting.

Future RL policies will use causal observations and choose storage and renewable
dispatch subject to feasibility rules. The grid closes the residual balance.
Reward initially represents negative operating cost; further objectives and
policy specialization remain to be defined. Meta-learning requires multiple
documented scenario regimes and comparable policy evaluation, not assumed
labels of an optimal policy.

Future risk experiments require forecast-error scenarios and a defined loss
(initially episode operating cost). CVaR confidence and the DRO ambiguity set
must be stated before reporting risk results. Finite synthetic scenarios test
software behavior, not robustness to a real campus distribution.

## Open decisions and missing information

- Campus location, time zone, season coverage, and calendar/holiday information.
- Availability, license, provenance, resolution, and units of source datasets.
- Whether five independent building histories can be obtained; otherwise the
  initial five-building scenario must remain explicitly synthetic.
- PV/wind ratings and conversion models if direct generation data is unavailable.
- Battery capacity, power limits, efficiencies, SOC limits, and initial SOC.
- Grid capacities, currency, tariffs, and export permission.
- Forecast history length/horizon, episode length, terminal SOC treatment, policy
  objectives, risk confidence, and DRO distance/radius calibration.
- Whether additional electrical realism or outage studies are needed later.

No numerical asset sizes, campus facts, tariffs, or weather values are asserted
here. The next data-selection step must resolve these items or record explicit
simulation assumptions before a simulator is described as usable.
