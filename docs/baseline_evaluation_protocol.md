# Part 4.1: Baseline evaluation protocol

## Status

This document defines the protocol for baseline evaluation. Four controllers
and a shared runner are implemented in `src/baselines/` and
`src/evaluation/baseline_runner.py`; `scripts/run_baselines.py` executes the
comparison. This does not implement a terminal-SOC rule or claim experimental
results. The existing `scripts/run_simulation.py` remains a zero-request
single-policy replay utility.

The shared runner accepts either a processed dataset manifest or one existing
split partition, loads the same physical configuration and input rows for every
selected controller, resets fresh controller/environment instances, and writes
controller-specific hourly CSV and JSON summaries plus a comparison CSV. It
refuses an existing output directory. Example:

```powershell
python scripts/run_baselines.py `
  --manifest data/processed/synthetic_hypothetical_campus_v1/integration_manifest.yaml `
  --assets configs/synthetic_scenario.yaml `
  --rating-origin hypothetical_example `
  --output results/baseline_comparison
```

For partition-aware evaluation, use `--split-root ... --partition test` instead
of `--manifest`. A split episode is loaded from exactly one partition. The
runner's default selects all registered controllers; repeat `--controller NAME`
to select a subset.

After an experiment, generate a validated comparison against zero-action:

```powershell
python scripts/compare_baselines.py `
  --experiment results/baseline_comparison `
  --output results/baseline_comparison_report
```

The comparison command verifies saved checksums, metric reconciliation against
the runner's simulator-recorded cumulative values, and identical controller
timestamp sequences, initial SOC, currency, physical configuration, costs, and
input metadata. It fails if the zero-action result is absent. It reports grid
transaction cost separately from modeled total operating cost, calculates
absolute savings and signed percentage improvement (percentage undefined when
the zero-action cost is zero), and blocks a cost-superiority claim when unmet
demand or physical constraint violations are worse than zero-action. It writes
to a new directory and will not overwrite existing results.

## Data partitions and held-out test

Use the existing chronological, entity-aligned split described in
[`configs/splitting.yaml`](../configs/splitting.yaml) and implemented by
[`src/data/splitting.py`](../src/data/splitting.py):

- Train: 70% of unique hourly timestamps.
- Validation: 15%.
- Test: 15%, reserved for final evaluation.

The configured split allocation floors the train and validation timestamp
counts and assigns the remainder to test. All sources/entities must share the
same complete hourly grid. Preserve the split metadata and input hashes.

1. Use training data for any price-threshold estimation. Use validation only
   for selecting a documented threshold convention or other predeclared
   controller parameter.
2. Freeze controller parameters and the evaluation protocol before test.
   Do not use test results to select controllers, thresholds, reserves,
   preprocessing, or episode windows.
3. Keep preprocessing fitted only on training rows. These baselines consume
   physical-unit observation features; do not feed scaled values into simulator
   physics or use test-fitted transformations.
4. Build evaluation episodes within one partition only. Do not construct a
   window or controller history that crosses a train/validation/test boundary.
   The environment's first row in each episode is history, not a scored
   transition.
5. If any test data are inspected for debugging or design decisions, disclose
   that the test is no longer an untouched final evaluation set.

## Fair comparison requirements

For every comparison, run each controller with the same:

- Immutable dataset partition, source/entity selection, input files, hashes,
  physical units, timestamp range, and episode boundaries.
- One-row history context and exact scored transition timestamps. With `N`
  hourly rows supplied to `MicrogridEnvironment`, the first row is history and
  the environment executes `N-1` transitions.
- Initial battery SOC and battery/grid/renewable configuration, including
  efficiencies, limits, curtailment and unserved-load handling.
- Economic assumptions, currency, metric implementation, and episode length.
- Observation timing, forecast availability (if introduced), and controller
  interface. Do not give one strategy extra data or a shorter decision lag.

Create a fresh environment for each controller and reset it to the same initial
configuration. Record all parameters and input/configuration hashes. Use the
same physical input rows in the same order. No controller may access the
environment, data loader, future rows, dispatch results before its decision, or
another controller's state. Current realized inputs may be used internally by
the simulator for physical dispatch and settlement, not exposed to the policy.

The simulator is deterministic for fixed data, configuration, initial state,
and action sequence. If a future controller uses randomness, define and record
its seed policy; stochastic policies should receive comparable evaluation
conditions and report variability across predeclared seeds. The current rule
baselines need no random seed.

## Episode and causal timing

Each hourly row represents `[timestamp_utc, timestamp_utc + 1 hour)`. At a
decision boundary, the controller sees the previous completed row and the
current battery state. Its action is applied to the next row. Therefore the
first row of each supplied episode initializes history and is excluded from
dispatch and metrics; an episode of `N` rows produces `N-1` actions. The final
observation is returned at the terminal boundary without reading an additional
row.

The existing `AgentObservation` feature allowlist is the controller's permitted
input. In particular, solar, wind, demand, and prices are lagged observations,
not upcoming interval realizations or advance tariff knowledge. Future realized
energy data must never be passed as features, controller arguments, or hidden
metadata. Explicit forecasts are permitted only if their issue and data-cutoff
times precede the decision and their targets follow it; report forecast source,
model, timestamps, and availability policy, and supply equivalent forecasts to
all compared strategies.

## Outcomes and reporting

For every controller and episode, retain the simulator's per-step and cumulative
metrics, dispatch records, and source provenance. At minimum report:

- Net operating cost and its import, export, battery-throughput/degradation,
  and unserved-demand terms.
- Grid import/export energy, served and unserved demand energy, renewable
  available/utilized/curtailed energy, and renewable utilization (including
  its undefined case when availability is zero).
- Initial, minimum, maximum, and terminal battery SOC; charge/discharge energy
  and throughput.
- Physical violation and dispatch-adjustment counts, and maximum absolute
  power/energy balance residuals.
- Dataset origin (measured, simulated, or derived), input/configuration hashes,
  partition, episode timestamps, controller parameters, and any terminal-SOC
  treatment.

Compare controllers on matching episodes and report per-episode values as well
as aggregate summaries. Do not report only an average that hides infeasible
episodes, unserved demand, SOC changes, or dispatch intervention.

## Terminal state and unresolved design conflict

The current environment does not enforce a terminal SOC target or apply a
terminal energy value. It reports interval operating costs and final state.
Consequently, a controller that ends with less stored energy may appear cheaper
than one that preserves energy, even though their endpoints are not economically
equivalent.

Until a terminal treatment is specified and implemented, report raw simulator
cost and terminal SOC separately, and do not claim a cost-only ranking is a
fair measure of total episode value. Before policy selection or final claims,
pre-register a common terminal rule (for example, enforce a shared terminal
SOC target or apply a documented terminal value) and validate its implementation.
Do not invent a terminal value from the held-out test data.

## Data, scenario, and price limitations

- `configs/data_sources.yaml` currently marks source entries pending. The
  available synthetic scenario and its asset ratings/prices are explicitly
  hypothetical, not measured campus data. Label all such evaluations
  synthetic and do not generalize results to a real campus.
- The default scenario uses a hypothetical USD tariff. Currency and prices must
  match across data, simulator configuration, and economic assumptions. No
  currency conversion or real tariff agreement is inferred.
- The current Part 2 canonical price schema rejects negative prices, and the
  simulator requires explicit configuration to allow them. The default does
  not enable negative prices. A study involving negative prices needs a
  documented schema/configuration extension and tests; never silently clip or
  rewrite the source values.
- The price-aware baseline sees the previous completed row's import price, not
  the upcoming price. It is a lagged-price heuristic. It must not use a future
  tariff schedule unless that schedule is explicitly supplied as information
  available at the decision boundary and the same availability is applied to
  every controller.
- Grid export-price may be missing under some supported configuration policies.
  Record the selected missing-price policy; do not silently substitute zero.
- This single-bus simulator omits feeder/network losses, voltage/frequency
  dynamics, outages, and other operational constraints. Passing this protocol
  is not evidence of operational safety or deployment readiness.

## Controlled stress scenarios and visualization

The deterministic scenario runner applies each selected condition to a fresh
in-memory copy of the same validated episode and runs every selected controller
with the same modified inputs and initial physical configuration:

- `normal_operation`: unchanged input observations.
- `low_renewable`: scales each solar and wind asset's available power to 25%.
- `high_campus_demand`: scales each building's load to 125%.
- `electricity_price_spikes`: multiplies import prices by four, with a minimum
  spike value of 1.0 currency/kWh, in a deterministic short window centered
  within the selected episode. The exact timestamps are saved in scenario
  metadata.
- `limited_grid_import`: reduces the import limit to the lesser of 10% of its
  configured value or 100 kW, and records unmet load so infeasible stress
  episodes can be evaluated consistently rather than aborted.

These are controlled stress assumptions, not measured disruptions or forecasts.
They have stable versioned IDs, are recorded in each experiment summary, and do
not modify the original manifest or data files. For price spikes, settlement
uses the scenario's realized simulated tariff while controllers continue to
receive only the simulator's causal, lagged price observation.

Run all five scenarios (repeat `--scenario NAME` to select a subset):

```powershell
python scripts/run_baseline_scenarios.py `
  --split-root data/processed/splits_hypothetical_campus_v1 `
  --partition validation `
  --assets configs/synthetic_scenario.yaml `
  --rating-origin hypothetical_example `
  --output results/baseline_stress_suite
```

Alternatively, use `--manifest` instead of `--split-root` and `--partition`.
Keep the held-out test partition untouched until final evaluation.

Each scenario gets its own experiment folder and reproducible scenario ID. The
scenario index records the source manifest or split-manifest hash, controllers, episode length,
and experiment-summary hash. Existing output folders are never overwritten.

Generate comparison CSV/JSON and the seven required plots from an existing,
checksum-validated baseline experiment:

```powershell
python scripts/compare_baselines.py `
  --experiment results/baseline_stress_suite/low_renewable_v1 `
  --output results/low_renewable_comparison `
  --plots
```

Plots cover net operating cost, renewable utilization, battery SOC over time,
grid import/export over time, battery throughput, renewable curtailment, and
unserved demand. They use simulator-recorded outputs, include units and
scenario/data provenance in titles, and refuse to replace existing plot files.
Zero renewable availability is shown as undefined rather than as zero
utilization. Compare terminal SOC and reliability alongside cost; stress-scenario
figures alone do not establish controller superiority or real-world performance.

## Readiness and unresolved limitations

The simulator's causal observation, signed `BatteryAction`, deterministic
dispatch, and metric interfaces support these baseline comparisons without
changing physical equations or action signs. The controllers and multi-policy
evaluation harness are implemented for the current rule-based baselines.

Terminal SOC is not valued or constrained, so report endpoint SOC and avoid
cost-only conclusions. Preserve held-out test data and use it only for final
evaluation after controller/scenario parameters are frozen. No scenario or
comparison output should be represented as measured campus performance unless
its input provenance substantiates that claim.
