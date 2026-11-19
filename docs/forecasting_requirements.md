# Part 5.1 Forecasting Requirements

## Status and scope

This document records the Part 5.1 requirements derived from the then-available
Part 2 artifacts. Its statements about an unimplemented model and readiness
gaps are a point-in-time baseline, not the current project status. Parts
5.1–5.10 are now complete; see the [Part 5 completion report](part5_completion_report.md)
for experiment evidence and the current integration boundary. Data findings
and limitations are detailed in [the forecasting data audit](forecasting_data_audit.md).

## Prediction task

1. Forecast hourly **building electricity demand** for B01–B05 and **available
   solar and wind generation power** for PV01 and W01.
2. Produce node-level predictions for seven target nodes. A campus demand sum
   may be computed from the five building predictions for a separately
   specified evaluation/use case; it is not an existing dataset label and must
   not replace the per-building targets.
3. Use a configurable default of 24 hourly observations as historical context
   and predict the following 24 hourly target intervals. These settings match
   `configs/forecasting_data.yaml` and the saved tensors. The horizon is not an
   observed data guarantee beyond the available one-year synthetic sample.
4. Return predictions aligned with target timestamps, ordered nodes, and a
   valid-target mask. A masked node must not contribute to any loss or metric.

## Input and output contract

| Tensor / metadata | Contract |
|---|---|
| `X` | Historical node power features, `[batch, lookback, 8, 1]`, `float32`, standardized using existing source/entity training-only scalers. |
| `global_X` | Historical site weather, `[batch, lookback, 3]`, `float32`, containing scaled irradiance (W/m² source), air temperature (°C source), and wind speed (m/s source). The tensor values are dimensionless after scaling. |
| `edge_index` | Directed graph indices `[2, 14]`, `int64`; use only with the explicit assumed-topology qualification. |
| `node_attributes` | Static configured attributes `[8, 6]`, `float32`; type indicators, hypothetical capacity/1000 kW, and capacity-known flag. |
| `node_input_mask`, `target_mask` | `[8, 1]` boolean masks; seven target nodes valid, BUS01 invalid. |
| `y` / predictions | `[batch, horizon, 8, 1]`; targets/predictions are in physical kW. BUS01 is a structural zero placeholder and is not a zero-valued target. |
| timestamps | Inputs and targets carry hourly UTC timestamps. The final input time must be earlier than the first target time; target horizons stay inside their source partition. |

The graph node order is exactly `B01, B02, B03, B04, B05, PV01, W01,
BUS01`. Do not infer different order from sorting or file enumeration.

## Feature availability and causality

- Inputs may contain historical observed node power and historical scaled
  weather only, as represented in the saved bundle. Keep input issue/decision
  time, data cutoff, and target times explicit.
- No known-future weather forecasts, forecast vintages, or declared
  known-future tariff covariates exist in the audited dataset. Future realized
  weather, generation, demand, or prices may be labels/evaluation data only;
  they must not enter model inputs.
- Timestamps are available for calendar derivation, but calendar covariates are
  not currently part of the tensor contract. Add them only with explicit
  feature definitions and leakage tests.
- Preserve `ForecastFeature` requirements when making forecasts available to
  Part 3: model issue time and input data cutoff must be no later than the
  decision; each forecast target must follow the decision. Forecasts are
  modeled predictions, never copied future replay observations.

## Data preparation and experimental protocol

- Consume the checksum-verified Part 2 split bundle and saved per-source/entity
  scalers. Do not refit, clip, impute, interpolate, or normalize using
  validation/test data.
- Keep windows and complete target horizons within a single chronological
  partition. The 70/15/15 boundaries and UTC hours are recorded in
  [the audit](forecasting_data_audit.md).
- Select architectures, checkpoints, and hyperparameters using train and
  validation only. Keep test sealed until final evaluation; disclose any
  accidental test inspection and do not continue to describe it as untouched.
- Record data and split hashes, source provenance/origin, features and units,
  graph provenance, split boundaries, scaler identity, sequence lengths, seed,
  model/training settings, software versions, metrics, and artifact hashes
  using the experiment metadata contract in
  `configs/experiment_metadata.schema.json`.
- Do not overwrite existing source, split, or experiment artifacts. Keep
  synthetic/hypothetical results clearly labeled; they do not establish
  real-campus or deployment performance.

## Model and evaluation requirements recorded for Part 5.2

- Use the graph only as the documented modeling assumption in the data audit;
  do not call its edges measured topology, physical connection weights, or
  electrical parameters. If a physical graph is required, obtain and document
  validated topology before claiming physical relevance.
- Support the actual input dimensions and target mask. The training objective
  must ignore BUS01 and any future missing-target masks; do not train against
  structural placeholders.
- Define target scaling, if any, using training targets only. Invert it before
  reporting metrics or interfacing with physical-unit code. Report errors in
  kW and disaggregate by building, renewable asset, and target type so large
  loads cannot hide generation errors.
- Compare against an explicitly implemented and documented forecasting
  baseline (for example, persistence) on the same partitions/windows. Part 4
  rule-based battery controllers are not forecasting baselines.
- Keep model training separate from the Part 3 physical simulator and Part 4
  controller implementations. A forecast may later be adapted through
  `ForecastFeature`; do not change simulator physics or rewrite the completed
  controllers.
- Do not add LLM calls to the training loop or implement RAG as part of Part 5.

## Known readiness gaps at the Part 5.1 audit

Available examples are synthetic only, and the assumed shared-bus graph is not
validated physical topology. No known-future covariate set, measured campus
data, forecast model, forecast metric baseline, or model-specific target
normalization exists yet. The initial Part 5.2 design must resolve
architecture/training choices and the loss/metric policy without weakening
the data and leakage constraints above.
