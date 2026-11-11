# Part 4 Completion Report: Rule-Based Baselines

## Status

Part 4 baseline controllers and the shared evaluation workflow are implemented
and verified against the existing Part 3 simulator. No Part 5 forecasting model
was started. The held-out test partition was not used for the representative
experiments in this report.

## Implemented components

- Shared `BaselineController` contract with `reset()` and validated
  `select_action(AgentObservation) -> BatteryAction`.
- Name-based controller registry and factory.
- Four registered deterministic strategies: `zero_action`,
  `self_consumption`, `price_aware`, and `battery_preserving`.
- Same-episode baseline experiment runner. It stores per-step requested and
  executed battery actions, dispatch adjustments, physical violations, hourly
  metrics, cumulative metrics, input/configuration provenance, and hashes.
- Comparison workflow that verifies experiment checksums and metric totals,
  compares against `zero_action`, distinguishes grid transaction cost from
  total modeled operating cost, and flags worse unserved demand or violations.
- Seven baseline comparison figures and five versioned deterministic stress
  scenarios.
- Updated baseline architecture and evaluation protocol documentation.

## Reused Part 3 components

No separate physical simulator or duplicate energy-balance model was added.
Baseline experiments reuse:

- `AgentObservation` and its causal feature allowlist.
- `BatteryAction` and its signed AC power convention.
- `MicrogridEnvironment` and `DispatchEngine` for shared simulation,
  constraints, energy balancing, and execution adjustments.
- `MetricValues`, `MetricsTracker`, and `CostConfig` for cost and energy
  accounting.
- The existing data manifest and split loaders, including checksum and
  chronological validation.
- The existing headless Matplotlib conventions in the simulation plotting
  module.

Action sign is consistent throughout: positive kW requests discharge; negative
kW requests charge. Controllers receive no simulator, replay table, metrics,
or dispatch result. The environment exposes the previous completed row at the
decision boundary; the current realized dispatch row remains private.

## Experimental setup and provenance

Representative runs used the existing synthetic hypothetical campus split:

- Partition: **validation** only.
- Episode: 24 input rows from `2026-09-13T12:00:00Z` through
  `2026-09-14T12:00:00Z`; the first row is history, leaving 23 scored hourly
  transitions from `2026-09-13T13:00:00Z` to `2026-09-14T11:00:00Z`.
- Input sources: synthetic campus demand, solar, wind, and grid prices.
- Currency: hypothetical USD; all scenario inputs are synthetic/modelled, not
  measurements from an operating campus.
- Physical assumptions: 500 kWh battery, initial SOC 0.5, SOC bounds 0.1–0.9,
  150 kW charge/discharge limits, 0.95 charge/discharge efficiencies, 2000 kW
  import limit and 500 kW export limit for normal operation.
- Grid, battery, tariff, and cost assumptions retain their configured
  hypothetical provenance. Terminal SOC is not valued or equalized.
- Every controller received the same scenario rows, time boundaries, initial
  conditions, and simulator configuration. A fresh environment and controller
  episode state were used per controller.

The split loader verified source checksums before simulation; the source split
checksums were checked again after the runs. No source dataset was changed.
Scenario-specific values were constructed in memory and saved with versioned
scenario IDs and the source split-manifest hash.

## Controller configurations

Values below are the checked-in hypothetical defaults:

| Controller | Configured behavior |
|---|---|
| `zero_action` | Requests 0 kW battery power. |
| `self_consumption` | Uses lagged renewable generation minus lagged demand; default power deadband is 0 kW. Dispatch constrains requests. |
| `price_aware` | Lagged import-price thresholds: 0.12 and 0.18 USD/kWh; minimum SOC reserve 0.20. Charging/discharging must clear the round-trip efficiency and configured throughput-degradation cost test. |
| `battery_preserving` | Preferred SOC band 0.30–0.80; 5 kW deadband; 50 kW charge and discharge request caps; charges only from observed renewable surplus. |

These are rule parameters, not fitted or optimized settings. The price-aware
rule sees the previous completed row's price, not advance knowledge of a price
spike. The battery degradation value is a linear throughput proxy, not a
validated battery-life model.

## Evaluation methodology and stress scenarios

All comparisons were produced by the same runner and Part 3 metric implementation.
The comparator checked the per-controller episode timestamps, shared metadata,
configuration, currency, initial SOC, checksums, and cumulative metric
reconciliation against hourly records.

The five scenario IDs and modifications were:

| Scenario | Controlled change |
|---|---|
| `normal_operation_v1` | No input or physical-limit modification. |
| `low_renewable_v1` | Scaled each solar and wind asset's available power to 25%. |
| `high_campus_demand_v1` | Scaled each building's load to 125%. |
| `electricity_price_spikes_v1` | Applied a deterministic 4x import-price multiplier with a 1.0 USD/kWh minimum to the recorded central episode interval. Controllers still see only causal lagged prices. |
| `limited_grid_import_v1` | Reduced import capacity to 100 kW and selected the simulator's explicit `record` unmet-load policy so the infeasible stress condition could be measured rather than aborting. |

The limited-grid case intentionally produces unmet demand. Its lower grid bill
must not be treated as improved performance. The comparison gate withholds a
cost-superiority claim when unmet energy or physical constraint violations are
worse than `zero_action`.

## Actual representative results

The following values come from the saved 23-step synthetic validation
experiments. Net cost is the simulator's modeled operating cost in hypothetical
USD (grid import cost minus export revenue plus modeled degradation and
unserved-demand costs). The configured unserved-demand price is zero, so
unserved energy remains separately important. Renewable utilization is the
simulator metric; the validation episode reports 1.0 in all these runs.
Terminal SOC is shown because it differs by controller and is not economically
equalized.

| Scenario | Controller | Net operating cost (USD) | Grid import (kWh) | Unserved (kWh) | Terminal SOC | Violations | Adjustments |
|---|---|---:|---:|---:|---:|---:|---:|
| Normal | `zero_action` | 841.29 | 6549.04 | 0.00 | 0.50 | 0 | 0 |
| Normal | `self_consumption` | 810.89 | 6359.04 | 0.00 | 0.10 | 0 | 23 |
| Normal | `price_aware` | 870.07 | 6774.96 | 0.00 | 0.90 | 0 | 0 |
| Normal | `battery_preserving` | 826.09 | 6454.04 | 0.00 | 0.30 | 0 | 0 |
| Low renewable | `zero_action` | 1027.99 | 8067.55 | 0.00 | 0.50 | 0 | 0 |
| Low renewable | `self_consumption` | 997.59 | 7877.55 | 0.00 | 0.10 | 0 | 23 |
| Low renewable | `price_aware` | 1056.77 | 8293.47 | 0.00 | 0.90 | 0 | 0 |
| Low renewable | `battery_preserving` | 1012.79 | 7972.55 | 0.00 | 0.30 | 0 | 0 |
| High demand | `zero_action` | 1113.85 | 8692.47 | 0.00 | 0.50 | 0 | 0 |
| High demand | `self_consumption` | 1083.45 | 8502.47 | 0.00 | 0.10 | 0 | 23 |
| High demand | `price_aware` | 1142.63 | 8918.39 | 0.00 | 0.90 | 0 | 0 |
| High demand | `battery_preserving` | 1098.65 | 8597.47 | 0.00 | 0.30 | 0 | 0 |
| Price spikes | `zero_action` | 1553.51 | 6549.04 | 0.00 | 0.50 | 0 | 0 |
| Price spikes | `self_consumption` | 1523.11 | 6359.04 | 0.00 | 0.10 | 0 | 23 |
| Price spikes | `price_aware` | 1336.62 | 6810.88 | 0.00 | 0.90 | 0 | 0 |
| Price spikes | `battery_preserving` | 1538.31 | 6454.04 | 0.00 | 0.30 | 0 | 0 |
| Limited import | `zero_action` | 303.80 | 2300.00 | 4059.04 | 0.10 | 0 | 27 |
| Limited import | `self_consumption` | 296.54 | 2259.67 | 4099.37 | 0.10 | 0 | 47 |
| Limited import | `price_aware` | 297.89 | 2267.17 | 4091.87 | 0.10 | 0 | 44 |
| Limited import | `battery_preserving` | 303.80 | 2300.00 | 4059.04 | 0.10 | 0 | 27 |

These are short synthetic validation demonstrations, not statistically
meaningful results or evidence of real-campus performance. In normal, low
renewable, and high-demand cases, lower costs coincide with depleted terminal
SOC for `self_consumption`; the price-aware controller ends at the upper SOC
bound. They are therefore not equivalent-endpoint cost comparisons. In the
limited-import case, `self_consumption` and `price_aware` have both lower
operating cost and higher unmet energy than `zero_action`; their cost reductions
are not reliability improvements. The dispatch recorded interventions but no
physical constraint violations in these experiments.

## Generated figures and artifacts

The run suite is available under
[`../results/part4_10_baseline_handoff_validation_24h/`](../results/part4_10_baseline_handoff_validation_24h/).
Each scenario folder contains controller summaries, hourly CSV files,
experiment metadata, comparison CSV/JSON, and these seven figures in
`comparison_and_plots/`:

1. `net_operating_cost_comparison.png`
2. `renewable_utilization_comparison.png`
3. `battery_soc_over_time.png`
4. `grid_import_export_over_time.png`
5. `battery_throughput_comparison.png`
6. `renewable_curtailment.png`
7. `unserved_demand.png`

Five scenarios × seven figures produced **35 PNGs**. Plot values were verified
against the saved hourly and cumulative experiment outputs by automated tests.
The images are titled with the scenario and synthetic provenance; the plots are
not smoothed or extrapolated.

## Verification

- Full available pytest suite: **606 passed** on the final implementation.
- Baseline-specific, scenario, comparison, and simulator integration tests
  cover the four registered controllers, action validation/sign, reset and
  determinism, identical episode lengths/initial SOC, dispatch constraints,
  metric reconciliation, scenario reproducibility, input preservation, plot
  outputs, and plotted values.
- `scripts/run_baseline_scenarios.py --help` and
  `scripts/compare_baselines.py --help` both completed successfully.
- Representative run: five scenarios × four controllers × 23 transitions,
  followed by five artifact-validated comparisons and all 35 plots.
- The existing chronological split loader verified the selected validation
  data. The held-out test partition was neither loaded nor used.

## Known limitations and remaining issues

- All results here use simulated, hypothetical source data. Measured campus
  datasets, verified tariffs, and calibrated equipment specifications are not
  available.
- The price-aware policy is lagged-price threshold logic; it is not forecasting
  or optimal arbitrage. Negative prices remain outside the current canonical
  price-input configuration.
- Terminal SOC has no terminal value or shared target; endpoint differences
  make cost-only comparisons incomplete.
- Battery degradation uses the configured linear AC-throughput proxy, not
  validated electrochemical aging.
- The limited-import stress scenario deliberately allows unmet demand to be
  recorded, while its inherited unserved-demand cost is zero. Reliability must
  be reported separately from cost.
- The current simulator is a single-bus hourly replay. These experiments do not
  establish feeder-level behavior, sub-hourly control, operational safety, or
  deployment readiness.
- The short validation window is a smoke/representative experiment only. Freeze
  all settings before performing a final evaluation on the held-out test
  partition.

## Readiness for Part 5: GNN-Transformer Forecasting

Part 4 provides a causal observation boundary, chronological train/validation/
test split discipline, synthetic graph topology and sequence-preparation
utilities, and saved provenance to build on. Existing Part 2 tests verify graph
ordering, sequence dimensions, strictly future targets, held-out window
boundaries, and resistance to future-target leakage.

The forecasting package itself currently contains no implemented GNN,
Transformer, training loop, or model evaluation. Part 5 can proceed as a
separate implementation phase: use training-only fitting, validation-only
selection, leave the test split untouched until final evaluation, record model
and forecast issue/data-cutoff timestamps, and expose forecasts only through
the simulator's validated `ForecastFeature` interface. No Part 5 model or
forecast-driven controller was introduced here.
