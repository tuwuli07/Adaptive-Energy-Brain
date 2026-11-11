# Energy simulation architecture

## Status, scope and existing contracts

This document preserves the original Part 3.1 design. Simulation components and
the episode environment were subsequently implemented in Parts 3.2-3.11. See
[part3_completion_report.md](part3_completion_report.md) for verified behavior,
current interfaces and deviations/limitations; proposals below are not a claim
that every original design option is implemented. Gymnasium and AI models remain
outside the implemented scope.

The design builds on [microgrid_scenario.md](microgrid_scenario.md),
[data_requirements.md](data_requirements.md),
[part2_completion_report.md](part2_completion_report.md), and
[simulation_requirements.md](simulation_requirements.md). The Part 2 completion
report verifies synthetic data only. All real source categories remain pending.
Earlier scenario documents recorded unknown ratings; the later
`configs/synthetic_scenario.yaml` supplies explicitly hypothetical ratings for
software development, not measured campus parameters.

The initial physical model is an idealized, lossless campus AC bus connecting
five fixed-demand buildings, one shared PV plant, one shared wind plant, one
battery and one grid connection. It is an hourly energy-accounting model, not
load flow, voltage/frequency control, protection coordination or transient
simulation. The bus connection is a modeling assumption.

## Data reuse and system boundary

Use physical-unit observations from validated Part 2 source CSVs or a complete
integrated entity panel. Do not feed scaled values, graph tensors or future
forecast targets into physical equations. Preserve source hashes, scenario ID,
origin labels, units and entity mapping in every episode manifest.

| Part 2 interface | Proposed simulator use |
|---|---|
| `src/data/schema.py`: `validate_dataset`, `validate_record` | Validate supported canonical input tables and battery-state records |
| `src/data/integration.py` and its quality report | Consume hourly UTC outputs; refuse incomplete required entities/fields |
| `src/data/splitting.py`: `verify_split_bundle`, `apply_scaler` | Verify immutable split inputs; use existing train-fitted parameters only for optional agent observations |
| `src/data/sequences.py` and forecasting metadata | Provide future forecaster history/ordering contracts, not simulator truth |
| `configs/synthetic_scenario.yaml` | Reuse labeled synthetic asset and grid parameters through an explicit configuration adapter |

No simulator module downloads, cleans, imputes, resamples, rescales physical
inputs, generates replacement profiles or refits preprocessing. Data errors go
back to Part 2. Reading and indexing validated observations is the adapter's
responsibility; duplicating preprocessing is not.

Canonical generation records use `asset_id`, `generation_type` and
`available_generation_kw`. Select PV01 and W01 explicitly and expose them as
`pv_available_kw` and `wind_available_kw` in the internal interval contract.
These are named views of the canonical field, not additional measurements.
Sum `building_load_kw` once per registered building to obtain `campus_load_kw`.
Never sum a campus aggregate again or repeat shared generation per building.

The existing `simulation_state` schema validates battery boundary records only.
Dispatch, grid and metric names below reuse the Part 2 requirements vocabulary
but are not all present in `configs/data_schema.yaml`. They belong to a proposed
separate transition-trace contract. Define/version its machine-readable schema
in a later implementation step; do not send extra trace fields to the existing
validator and claim they have already passed it.

## Components and proposed module responsibilities

All file locations in this table are proposals under the existing
`src/simulation/` package. No files or subpackages are created by this design.

| Component / proposed module | Inputs | Responsibility and outputs |
|---|---|---|
| Contracts / `contracts.py` | Validated configuration and entity registry | Immutable `IntervalInputs`, `ActionRequest`, `DispatchResult`, `SimulationState`, `Transition` and error types with explicit units |
| Episode data adapter / `data_adapter.py` | Part 2 files, checksums, selected split and episode window | Validate provenance/coverage and expose one immutable interval; separate internal realized data from causal observation history |
| Solar PV / `renewables.py` | PV01 available AC power and its rating | Validate bounds; expose available, curtailed and used PV power; no second weather-to-power calculation |
| Wind / `renewables.py` | W01 available AC power and its rating | Same responsibility for wind, with separate asset accounting |
| Campus demand / `demand.py` | Building demand vector and stable building IDs | Preserve per-building values; calculate total fixed demand; no demand response or fabricated load |
| Battery / `battery.py` | Boundary energy, capacity, efficiencies, SOC/power limits and proposed AC power | Calculate feasible charge/discharge limits, next energy/SOC and storage losses; no policy selection |
| Grid / `grid.py` | Residual signed exchange, import/export capacities and tariff | Enforce connection limits and compute non-simultaneous import/export plus energy cost; no infinite balancing supply |
| Energy dispatch controller / `dispatch.py` | Realized interval, battery state and action request | Deterministic feasibility resolution described below; return actual flows, requested/actual differences and adjustment reasons |
| Episode engine / `engine.py` | Validated episode, initial state and requests | Order transitions, verify invariants, commit atomically, advance time and return immutable results |
| State and metrics / `metrics.py` | Committed states/transitions | Accumulate costs, energy, curtailment, losses and constraint diagnostics; distinguish physical metrics from optional reward shaping |
| Observation builder / `observations.py` | Current boundary state, allowed history, announced tariff and optional issued forecasts | Produce causal observations and masks; never expose a future replay row to a policy |
| Simple request policy / `controllers.py` | The same causal observation a future agent receives | Optional initial idle-battery/no-voluntary-curtailment request; no optimization or privileged current-hour data |
| Future Gymnasium adapter / `gym_adapter.py` | Core engine, action/observation configuration | Translate array/dictionary representations, seeding and episode flags; physics stays in the core |

## Core interface contracts

Signatures describe intended responsibilities, not callable code available now.

| Proposed operation | Contract |
|---|---|
| `EpisodeData.read_interval(timestamp_utc)` | Internal-only access to realized demand, available renewables and prices for one interval; no mutation or preprocessing |
| `Policy.propose(observation)` | Return `ActionRequest`; policy receives no `EpisodeData` handle or mutable engine state |
| `Battery.feasible_limits(state, dt_hours)` | Return nonnegative maximum AC charge/discharge powers in kW |
| `DispatchController.resolve(inputs, state, request, config)` | Return a feasible `DispatchResult` or raise an explicit infeasibility error; does not commit state |
| `Battery.transition(state, actual_flows, dt_hours)` | Return candidate next boundary state and losses; no mutation |
| `Engine.reset(episode_spec, seed)` | Verify complete episode coverage, initialize state/metrics and return the first causal observation |
| `Engine.advance(request)` | Resolve, verify, then commit one transition; return next observation, costs, metrics and episode status |
| `Metrics.update(transition)` | Accumulate only a committed transition; failed attempts never increment costs or elapsed hours |

The controller request is independent of the safety/dispatch controller. A future
DRL policy, meta-selected policy or DRO planner replaces `Policy.propose`, not
the physical equations. Forecasting supplies optional predictions to observation
construction. It never replaces realized replay values in physical accounting.
Future risk optimization may propose requests or constraints through a versioned
interface; it cannot bypass hardware feasibility or conservation.

## State, actions and sign conventions

Time is UTC. An interval starting at t covers [t, t + 1 hour). Power is
interval-average AC kW; stored energy is kWh; SOC is a fraction. N transitions
require N input intervals and N+1 boundary states.

The initial request has three physical components:

| ActionRequest field | Unit and sign | Effect |
|---|---|---|
| `requested_battery_power_kw` | kW; positive discharge, negative charge | Suggest storage exchange; zero requests idle storage |
| `requested_pv_curtailment_kw` | Nonnegative kW | Suggest withholding some available PV output |
| `requested_wind_curtailment_kw` | Nonnegative kW | Suggest withholding some available wind output |

Request bounds use rated battery power and renewable capacities. Non-finite,
malformed or statically out-of-range actions are rejected with no state change.
Dynamically unavailable battery headroom and renewable power are handled by the
documented feasibility projection, not by exceeding the hardware limits.

The core does not accept independent grid import/export commands: the grid
closes the bus balance after storage and curtailment are resolved. Charging
increases required import or consumes renewable surplus; discharging reduces
import or increases export. Grid charging and exporting stored energy are
proposed research permissions, not assumed real-world contractual rights.

An optional future normalized adapter may map a battery command in [-1, 1]
piecewise to the charge/discharge ratings and curtailment fractions in [0, 1]
to the respective plant ratings. It must record that mapping and validate its
input ranges; the core continues using kW. No normalized-to-physical conversion
or action space is implemented now.

## Battery and bus equations

Let dt = 1 h, E be current stored kWh, C be capacity in kWh, and eta_c/eta_d
be efficiencies in (0, 1]. Let c and d be actual nonnegative AC charge and
discharge kW; only one may be positive.

```text
E_min = C * soc_min
E_max = C * soc_max
c_max = min(charge_limit_kw, (E_max - E) / (eta_c * dt))
d_max = min(discharge_limit_kw, eta_d * (E - E_min) / dt)

u = d - c
c = max(-u, 0)
d = max(u, 0)
E_next = E + eta_c * c * dt - d * dt / eta_d
soc_next = E_next / C
battery_loss_kwh = (1 - eta_c) * c * dt + (1 / eta_d - 1) * d * dt

pv_used_kw = pv_available_kw - pv_curtailed_kw
wind_used_kw = wind_available_kw - wind_curtailed_kw
R = campus_load_kw - pv_used_kw - wind_used_kw
g = R - u
grid_import_kw = max(g, 0)
grid_export_kw = max(-g, 0)

grid_import_kw + pv_used_kw + wind_used_kw + d
    = campus_load_kw + c + grid_export_kw
balance_residual_kw = left_hand_supply - right_hand_demand
```

Stored energy is on the battery's internal side. This efficiency placement is
intentional: delivering d * dt AC kWh removes d * dt / eta_d stored kWh.
Do not multiply by efficiency on both sides or treat a kW flow as stored energy.
Feasible limits are computed only after validating that E is within bounds;
negative headroom is a state error, not an allowed negative charge limit.

## Deterministic feasibility resolution

This is a proposed safety rule, not a cost-optimal dispatch algorithm. Log every
adjustment and its reason. Given fixed curtailments, battery/grid feasibility
is a one-dimensional interval:

```text
u_lower = max(-c_max, R - grid_import_limit_kw)
u_upper = min( d_max, R + grid_export_limit_kw)
```

1. Validate the request against static bounds. Limit requested curtailment to
   currently available power; keep the requested values separately in the trace.
2. Calculate R and the feasible interval. If it is nonempty, retain those
   curtailments and choose the nearest feasible u to the requested battery power.
3. If excessive deficit makes `R - grid_import_limit_kw > d_max`, restore enough
   voluntarily curtailed renewable power to remove that deficit, up to all
   curtailment. Restore PV/wind proportionally to their current curtailed powers;
   do not restore output beyond availability. Recompute the interval. If deficit
   remains when all generation is restored, fail: supply/storage/grid capacity
   is insufficient. Do not invent supply or silently shed load.
4. If excessive surplus makes `R + grid_export_limit_kw < -c_max`, increase total
   curtailment by `-c_max - grid_export_limit_kw - R` kW. Allocate it proportionally
   to currently used PV/wind output and cap each reduction by that output. This
   case necessarily has positive used renewable power under nonnegative load;
   otherwise report an inconsistency instead of dividing by zero. Recompute.
5. Require a nonempty final interval. Project the requested u onto it, derive c,
   d, import/export and renewable used powers, and calculate candidate storage
   state. The rule prioritizes retaining requested curtailments when feasible,
   then minimizing battery-command adjustment. It may force charge/discharge
   for feasibility even when the request was idle; this is visible in the trace.
6. Check bounds, energy update and bus balance before any commit. Tolerance-sized
   rounding corrections must be logged and must not conceal a material energy
   error. Unresolved violations fail the transition atomically.

This projection uses internal realized hourly inputs after the policy request.
It approximates a lower-level balancing service within an hourly interval; it
does not claim that an agent knew the whole hour's future average beforehand.
No within-hour ramp or transient feasibility is established by this model.

## Hourly sequence and information timing

1. **Reset/preflight:** select a single temporal split and immutable episode
   window; verify hashes, IDs, units, coverage and static feasibility. Set E_0
   from explicitly configured initial SOC and clear metrics. Optional observation
   warmup must come from earlier rows inside the same split, never a later row.
2. **At boundary t:** construct the observation from E_t, allowed observations
   through t-1, calendar information and any tariff explicitly announced by t.
   Optionally attach forecasts issued by t. No realized t-hour demand/weather/
   generation is provided to the request policy.
3. **Request:** a baseline rule or future agent returns the three action fields.
4. **Resolve interval t:** the engine reads the realized t-hour inputs internally;
   the dispatch controller computes feasible actual flows and adjustment reasons.
5. **Validate candidate:** compute E_(t+1), SOC, losses, grid costs and balance;
   validate all constraints. On error, keep E_t, time and cumulative metrics intact.
6. **Commit:** append one transition with source references and one new boundary
   state, update metrics, and advance the simulation clock by one hour.
7. **Return:** build the t+1 observation, now including t-hour realized history and
   dispatch results. If the episode ended, return a final state/observation without
   requiring or reading an extra exogenous t+1 input row.

## Outputs, costs and metrics

The immutable boundary state contains `timestamp_utc`, battery `asset_id`,
`battery_energy_kwh`, `battery_soc`, and optionally `battery_capacity_kwh`.
The trace stores interval timestamp, building vector, requested/actual actions,
available/curtailed/used renewable powers, grid powers, E before/after, residual,
cost, adjustment flags and source identifiers. Float64 arithmetic is proposed
for physical accounting; float32 is an adapter/learning representation only.

| Metric | Definition | Unit |
|---|---|---|
| `step_cost` | `(import_price * grid_import_kw - export_price * grid_export_kw) * dt` | Declared currency |
| `episode_cost` | Sum of committed step costs; report terminal adjustment separately | Currency |
| Demand/import/export/available/used/curtailed energy | Sum each corresponding interval-average kW * dt | kWh |
| Battery throughput | Sum `(c + d) * dt`, explicitly AC-side | kWh |
| Storage losses | Sum `battery_loss_kwh` | kWh |
| Renewable utilization | Used renewable energy / available energy; undefined if denominator is zero | Dimensionless fraction or missing with reason |
| Maximum balance error | Max absolute committed `balance_residual_kw` | kW |
| Action adjustment count | Number of intervals whose actual request differs beyond tolerance | Count |
| SOC extrema / terminal SOC | Min/max/final boundary SOC | Fraction |

Used renewable energy includes supply sent to load, storage or export. It is
not automatically renewable self-consumption. Do not label grid-charged battery
discharge as renewable without a future explicit energy-origin tracking method.
Storage losses already affect grid billing through the energy equations; do not
charge an extra loss cost by accident. Degradation/emissions/risk penalties are
optional future objective terms with explicit units and coefficients.

For the initial adapter, propose reward = negative step cost in the configured
currency. Keep physical metrics separate from any later normalized/shaped reward.
No action-adjustment penalty is assumed by default. No reward or benchmark score
is produced for an invalid, uncommitted transition.

## Future agent observations and Gymnasium compatibility

| Observation block | Allowed information at boundary t | Availability policy |
|---|---|---|
| Storage and constraints | Current E/SOC, known ratings/limits/efficiencies | Boundary state and configured constants |
| Calendar/time remaining | UTC/local calendar and configured episode position | Deterministic, available without reading future observations |
| Building/renewable history | Completed intervals ending by t; stable building/asset order | Preserve masks for unavailable startup history; do not silently fill measurements |
| Weather history | Completed observations only | Optional; no realized future weather |
| Tariff | Current price only if announced by t | Baseline announcement is an explicit assumption; no future price spikes exposed by default |
| Prior dispatch and cost | Last committed result and cumulative metrics | No uncommitted current-hour outcome |
| Optional forecasts | Predictions with issue time <= t and declared valid times | Empty/masked until an actual forecasting provider exists; no oracle targets |

The eight-node forecasting graph remains `B01-B05, PV01, W01, BUS01`; it is not
a power-flow solver. Battery/grid information can be separate observation blocks;
do not silently reorder stored graph tensors or pretend that the masked bus
target is a measured zero. The physical registry also includes BAT01 and GRID01.

Proposed adapter API: `reset(seed=..., options=...)` returns observation/info;
`step(action)` returns observation, reward, terminated, truncated, info. Define
action and observation spaces and seed reset sampling explicitly. These return
contracts follow the [official Gymnasium Env API](https://gymnasium.farama.org/api/env/).
No Gymnasium dependency is installed or environment checker executed here.

For the proposed continuing-task baseline, a planned replay/episode time limit
sets `truncated=True`, with `terminated=False`; callers must reset afterward.
Invalid inputs and infeasible transitions raise errors rather than fabricate an
absorbing physical failure or reward. A future finite-horizon task or explicit
absorbing-failure model must document a different termination contract first.
Return independent observation/info objects. Privileged diagnostics in info are
not agent features; training/evaluation pipelines must not promote them to inputs.

## Configuration decisions and proposed defaults

Existing hypothetical values are reused, not newly inferred:

- Five buildings B01-B05; PV01 = 300 kW; W01 = 150 kW.
- BAT01 = 500 kWh, 150 kW charge/discharge limits, efficiencies 0.95,
  SOC range 0.1-0.9, initial SOC 0.5.
- Grid import/export limits = 2,000/500 kW; synthetic maximum building demand
  totals 1,350 kW. Even simultaneous 150 kW charging plus that load is below the
  import limit without renewables. This is a capacity check, not a simulation run.
- USD is the synthetic tariff currency. Asia/Dhaka is a scheduling convention,
  not evidence of a campus location, real tariff or grid export permission.

New **design proposals requiring confirmation before implementation/experiments**:
168-hour development episodes; hourly decisions; always-available grid; no load
shedding; grid charging and storage export permitted; deterministic safety
projection/proportional curtailment allocation; initial idle request baseline;
cost-only reward; and current-hour tariff announcement.

Propose `terminal_soc_rule = free_report` for initial functional tests: record
final SOC, stored-energy change and operating cost without silently adding a
salvage credit. This is unsuitable for fair policy ranking if policies drain
storage differently. Before comparative research, confirm a terminal valuation
with an explicit currency/kWh coefficient or design a feasible terminal-target
constraint. Neither coefficient nor hard target is invented here.

Numerical tolerance proposals, not physical assumptions: absolute power/energy
checks of 1e-6 kW / 1e-6 kWh and SOC checks of 1e-9. Confirm scales and rounding
policy in later tests. Reset must reject incomplete configuration rather than
quietly choosing unresolved research settings.
