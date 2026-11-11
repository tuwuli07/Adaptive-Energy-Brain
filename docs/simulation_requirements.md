# Simulation requirements and acceptance criteria

## Status and references

This preserves the original Part 3.1 implementation contract for the architecture
in [simulation_architecture.md](simulation_architecture.md). Simulation code was
subsequently implemented in Parts 3.2-3.11. Requirements below are historical
proposals, not evidence of test results or a claim that every design option is
implemented. See [part3_completion_report.md](part3_completion_report.md) for the
final verification, actual behavior and remaining limitations.

Reuse [Part 2 canonical requirements](data_requirements.md),
`configs/data_schema.yaml`, validated split metadata, and the explicitly
hypothetical `configs/synthetic_scenario.yaml`. Earlier names `pv_available_kw`
and `wind_available_kw` are internal aliases of canonical generation rows;
they do not replace `available_generation_kw` in stored Part 2 tables.

## Required inputs and reset checks

| ID | Required input or check | Unit / representation | Acceptance condition |
|---|---|---|---|
| IN-01 | Hourly `timestamp_utc` and `interval_hours` | Explicit UTC ISO timestamp; h | Unique aligned one-hour intervals; no DST resampling in simulator |
| IN-02 | Per-building `building_id`, `building_load_kw` | Identifier; kW | Exactly one record per registered building/hour; finite nonnegative demand |
| IN-03 | Per-renewable `asset_id`, `generation_type`, `available_generation_kw` | Identifier; solar/wind label; kW | Exactly one PV01 and W01 record/hour; finite power from zero through configured rating |
| IN-04 | Grid ID, `import_price`, `export_price`, declared currency | Identifier; currency/kWh | Complete tariffs, one currency; baseline `import_price >= export_price >= 0` |
| IN-05 | Battery C, charge/discharge limits and efficiencies | kWh; kW; ratios | Positive finite ratings; efficiencies in (0, 1] |
| IN-06 | SOC min/max/initial | Fractions, dimensionless | `0 <= min < max <= 1`; initial inside bounds; E_0 = initial SOC * C |
| IN-07 | Grid import/export limits and baseline availability | kW; available flag | Finite nonnegative limits; grid available for every baseline interval |
| IN-08 | Episode start, length, selected split and terminal rule | UTC timestamp; positive step count; labels | Entire episode and any warmup stay inside one split; no unexpected replay exhaustion |
| IN-09 | Source manifest, hashes, units, provenance and schema version | Metadata | Hash checks pass; source origin retained; synthetic sources remain explicitly synthetic |
| IN-10 | Entity registry and shared bus membership | Identifiers and declared relationships | No duplicate/missing IDs, double-counted shared assets or inferred real topology |
| IN-11 | Numerical tolerances | kW, kWh and SOC fraction | Explicit finite positive values; proposed values are labeled in architecture |
| IN-12 | Baseline feasibility | kW | Check actual interval demand against grid plus available supply/storage; propose conservative import fallback sufficient for maximum registered demand |

Weather is optional if direct renewable available-power series are provided.
Weather is not converted again inside the simulator. Optional forecasts must
carry issue/valid timestamps and source/model versions; actual future targets
are never passed off as predictions. Training-only scalers are reused only for
learning observations, not the physical demand, battery or grid equations.

Missing physical inputs, unit mismatches, unresolved IDs, duplicate rows, failed
hashes and incomplete joins are reset errors. There is no implicit zero filling,
interpolation, unit conversion, extrapolation or fallback synthetic generation.
Wh/kWh and kW are not interchangeable even when the step happens to be one hour.

Negative market prices are not a data-cleaning error. Part 2 preserves them but
its current baseline canonical price schema rejects them. Baseline reset must
report unsupported tariff assumptions instead of deleting/changing those values.
Supporting negative prices later requires an explicit schema/scenario revision
and tests; no such revision is performed here.

## Physical transition requirements

| ID | Requirement | Required behavior |
|---|---|---|
| PHY-01 | Energy balance | Enforce grid import + used renewables + discharge = total load + charge + grid export each committed interval |
| PHY-02 | Battery bounds | Keep both E and SOC within configured bounds; enforce AC-side power and efficiency-correct headroom limits |
| PHY-03 | Mutually exclusive storage flows | Derive charge/discharge from one signed actual command; never both positive |
| PHY-04 | Mutually exclusive grid flows | Derive import/export from one signed residual; never both positive |
| PHY-05 | Renewable bounds | `0 <= curtailed <= available <= capacity`; used = available - curtailed |
| PHY-06 | Grid limits | Respect import/export bounds after all action adjustments; no hidden unlimited grid |
| PHY-07 | Fixed demand | Do not change or shed building demand to make equations balance |
| PHY-08 | Deterministic safety resolution | Use the battery feasible interval and proportional curtailment adjustments defined in architecture; record the priority rule |
| PHY-09 | Valid state transition | Compute candidate energy, losses and metrics first; commit state/time/cost together only after checks pass |
| PHY-10 | Time semantics | Store N transitions and N+1 boundary states; do not require an N+1-th exogenous input interval |
| PHY-11 | No hidden physical model | No feeder losses, reactive power, ramps, self-discharge, degradation, outages or flexible loads unless explicitly introduced later |
| PHY-12 | Reproducibility | Same validated inputs, configuration, initial state and requests produce the same transitions within declared numeric tolerance |

The dispatched action is a feasibility result, not proof that an AI policy is
safe, optimal or robust. Grid/renewable/storage models expose constraints; the
dispatch controller resolves requests; the engine checks conservation. Do not
spread independent clipping rules across modules that can invalidate one
another after the final balance check.

## Proposed output contracts

| Object / fields | Unit | Meaning and use |
|---|---|---|
| Boundary state: `timestamp_utc`, `asset_id`, `battery_energy_kwh`, `battery_soc`, optional `battery_capacity_kwh` | UTC; ID; kWh; fraction; kWh | Reuse canonical `simulation_state`; the final boundary has no required outgoing flows |
| Requested and actual battery power | Signed kW | Keep the request and realized result separately |
| `battery_charge_kw`, `battery_discharge_kw` | Nonnegative AC kW | Actual interval storage exchange |
| `pv_available_kw`, `wind_available_kw`, `pv_curtailed_kw`, `wind_curtailed_kw`, `pv_used_kw`, `wind_used_kw` | AC kW | Internal/trace views; stored Part 2 generation schema remains canonical |
| Building demand vector and `campus_load_kw` | AC kW | Preserve each building; total computed once |
| `grid_import_kw`, `grid_export_kw` | AC kW | Actual residual exchange after safety resolution |
| `balance_residual_kw` | kW | Supply minus demand, verified before commit |
| `battery_loss_kwh` | kWh | Efficiency losses for the completed transition |
| `step_cost`, accumulated operating cost, optional terminal adjustment | Declared currency | Keep each cost term distinct; no implied terminal credit |
| `reward` | Currency initially | Negative committed step cost; optional later shaping must be separate and versioned |
| `action_adjusted`, adjustment reasons, failure codes | Boolean / text labels | Make dynamic constraint interventions auditable |
| Episode manifest | Metadata | Inputs/hashes, split, IDs/order, schema/config versions, origins, policy ID, seed, terminal treatment and numerical tolerances |

The transition trace and metric persistence schemas are proposed work for later
parts. Existing `schema.py` is not a whole-simulator validator. Output origin is
`derived` with `derived_kind: simulator`; retain the upstream synthetic/measured
lineage instead of relabeling synthetic inputs as campus measurements.
Keep episode outputs in a new versioned results location, never in `data/raw/`
or over immutable processed/split files. No output directories are created here.

## Failure, episode and observation semantics

| Condition | Proposed response | State/metrics change? |
|---|---|---|
| Malformed/non-finite/statically invalid action | Raise an action validation error | None |
| Valid request exceeds current SOC headroom or available renewable power | Adjust with documented feasibility rule; log request/actual difference | Commit only the checked feasible result |
| Export surplus with insufficient storage headroom | Increase renewable curtailment, then recompute/check balance | Commit only if feasible |
| Deficit caused by voluntary curtailment | Restore available output as needed before declaring shortage | Commit only if feasible |
| True supply shortfall, grid bound violation or unresolved balance error | Raise an infeasible-transition error; include diagnostic reason | None; no hidden unmet-load record or reward |
| Invalid boundary SOC/energy or corrupt dataset row | Raise state/data error | None |
| Planned episode limit or replay-window end | Return final boundary observation; proposed continuing-task adapter sets truncated true | Final valid transition already committed |
| Step called before reset or after episode end | Raise lifecycle error | None |

Do not confuse an invalid software/data transition with a modeled absorbing
failure. A future outage/reliability extension must define unmet-load kW,
penalties, transitions and termination rules explicitly. Current baseline
infeasibility is an error requiring investigation, not an invented training event.

At reset, verify enough same-split observations for any requested warmup plus
the whole episode. If a cold start is allowed instead, fixed-shape history must
carry explicit availability masks; placeholders are not observed zeros. The
final observation uses completed history and boundary state; unavailable next
tariff/forecast fields are masked without reading a future replay row.

Future RL observations may include present battery state, static constraints,
calendar/time position, past building/renewable/weather measurements, prior
dispatch metrics, announced current tariffs and genuinely issued forecasts.
By default exclude current-hour realized exogenous averages, all future realized
inputs, future forecast errors, test targets, unpublished future price spikes
and any privileged failure diagnostics. Future agents are centralized campus
controllers; multiple buildings do not imply a multi-agent environment.

Gymnasium integration is a future adapter around the core, with the reset/step
contracts and continuing-task truncation convention documented in the
architecture. Observation/action representations and spaces must be specified
before implementation; no Gymnasium compatibility test has been run yet.

## Acceptance checks for later implementation

These are required future test cases, not tests executed in Part 3.1:

1. Load valid hourly inputs and reject missing entities, duplicate timestamps,
   scaled physics inputs, bad hashes, invalid ratings and cross-split episodes.
2. Show that summing building demand and shared generation does not duplicate
   the campus aggregate or multiply a shared asset by the building count.
3. Verify import/export residual calculations for deficit, surplus and exact
   balance, including export-disabled operation and zero renewables.
4. Verify battery idle/charge/discharge, SOC saturation, efficiency placement,
   power bounds, loss accounting and non-simultaneous flows.
5. Exercise both branches of safety curtailment adjustment and the nearest
   feasible battery projection; log each applied correction.
6. Reject true infeasibility atomically: time, E, transition count and cost must
   be unchanged; no load is silently removed.
7. Verify cost units with kW * h * currency/kWh, export revenue sign, metric
   aggregation and explicit terminal treatment.
8. Confirm the same seed/episode/requests reproduce states and trace metadata;
   reset seeds episode selection, not hidden regeneration of exogenous data.
9. Change future replay rows while holding history fixed: the current policy
   observation and request must stay unchanged. Applied physical dispatch may
   differ once the altered interval is realized internally.
10. Check that forecasts are gated by issue time, absent providers remain
    masked, and actual Part 2 prediction targets cannot enter observations.
11. Verify N transitions/N+1 states, terminal observation without an extra data
    row, appropriate adapter flags, and no action after episode end.
12. When a Gymnasium wrapper is implemented, check its spaces, seed behavior and
    independent returned observations using the then-installed official checker.

Arithmetic illustration using hypothetical ratings, **not an executed test**:
with E = 250 kWh, efficiencies 0.95, load 1,000 kW, PV used 300 kW and wind used
100 kW, a feasible 150 kW discharge needs 450 kW grid import and produces
E_next = 250 - 150 / 0.95 = approximately 92.105263 kWh. A 150 kW charge instead
needs 750 kW import and produces E_next = 250 + 0.95 * 150 = 392.5 kWh. Both
respect the assumed 50-450 kWh stored-energy bounds. These examples check the
intended signs and dimensions, not real campus performance.

## Decisions requiring confirmation

| Decision | Existing assumption or proposed choice | Why confirmation matters |
|---|---|---|
| Asset and campus configuration | Reuse five buildings, 300/150 kW renewables and 500 kWh battery from synthetic configuration | No real-campus calibration or observed ratings exist |
| Network fidelity | Ideal single bus; ignore electrical dynamics/losses | Does not establish feeder or protection feasibility |
| Episode policy | Propose 168-hour development episodes, same-split warmup, deterministic evaluation windows | Not yet a simulator config; prevents hidden temporal leakage |
| Initial and terminal storage | Initial SOC 0.5 from synthetic config; propose free_report for functional tests only | Fair policy comparison needs confirmed terminal valuation/constraint |
| Grid permissions | Always available, charging from grid and battery export allowed | Real export/storage contracts remain unknown |
| Action handling | Three kW requests; feasibility projection and proportional curtailment | Defines which applied actions the learning agent actually controls |
| Tariff availability | Announced current-hour tariff, no future unpublished spikes | Existing price schema has no publication timestamp; availability is a scenario assumption |
| Objective | Negative step operating cost initially, no wear/risk/action penalties | Coefficients and terminal economics must be explicit before research comparisons |
| Numeric checks | Proposed 1e-6 kW/kWh and 1e-9 SOC tolerances | Must be validated against selected arithmetic precision and scale |
| Additional studies | No outages, load shedding, EV/HVAC, degradation or carbon model initially | Each needs new requirements/data and explicit failure semantics |

Implementation can start after confirming these design choices or recording
replacements in a future simulator configuration. Part 3.1 does not install
dependencies, change schemas, implement physics, train models or establish that
the proposed acceptance tests pass.
