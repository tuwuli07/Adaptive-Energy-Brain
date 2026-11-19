# Part 5 Completion Report — GNN-Transformer Forecasting

## Status and evidence boundary

Parts 5.1–5.10 are complete as a research pipeline: data requirements and
audit, forecasting baselines, loaders, graph and temporal encoders, integrated
model, training, Colab GPU workflow, and held-out evaluation. Part 5.11 adds
the integration-readiness adapter and this closure report.

The training and evaluation results below are transcribed from the Colab
completion details supplied for this task. The checkpoints, experiment
metadata, and evaluation output remain in Google Drive; they were not copied
into or independently inspected in this local checkout. Do not treat this
report as a substitute for those artifacts. The dataset and graph are
synthetic/hypothetical, and no result demonstrates measured campus behavior,
deployment readiness, or superiority to all forecasting methods.

## Implementation and architecture

The forecasting pipeline reuses Part 2's chronological splits, train-fitted
input preprocessing, graph construction, and timestamped sequences:

1. `src/forecasting/dataset.py` and `dataloader.py` preserve partition
   boundaries, graph node order, target masks, and timestamps.
2. `GNNTransformerForecaster` consumes historical node-power inputs and graph
   metadata, then emits direct multi-step node predictions. The input and
   output tensors use `[batch, time, node, channel]` layouts.
3. `training.py` and `scripts/train_forecaster.py` train only on train data,
   select checkpoints using validation loss, mask BUS01, and persist resumable
   checkpoints and JSON run metadata.
4. `evaluation.py` and `scripts/evaluate_forecaster.py` verify matching
   checkpoint/configuration/preprocessing/topology provenance and evaluate the
   held-out split against forecasting baselines on aligned timestamps and
   valid target observations.

The current task has no stored system-level label. The targets are seven
node-level power series in kW: B01–B05 building demand, PV01 solar available
generation, and W01 wind available generation. BUS01 is a context node with a
false target mask. The eight-node graph has 14 directed edges and is an assumed
shared-bus topology, not measured electrical infrastructure. Historical
weather features are prepared in `global_X` but are not consumed by the
current model; no known-future covariates are used.

## Part 5.9 GPU training

The supplied Colab run record reports:

| Item | Reported result |
|---|---|
| Dataset | Synthetic hypothetical campus microgrid |
| Device | NVIDIA Tesla T4 |
| Git revision | `5d1292a251194aad13b9bd7757717fce3c40e9f8` |
| Experiment ID | `part5_9-colab-20261111T172326Z` |
| Lookback / horizon | 24 / 24 hourly steps |
| Graph / output targets | 8 nodes, 14 edges, 7 targets |
| Best epoch | 35 |
| Best validation MSE | approximately 368.871 kW² |
| Training duration | approximately 282.85 seconds |
| Checkpoints | Best and resumable checkpoints reported saved |
| Experiment metadata | Reported saved with the run |

The local training implementation uses train-only fitting and validation-only
checkpoint selection. The held-out test partition is not part of Part 5.9.
Although the GPU result is reported complete, its external checkpoint and
metadata bytes are not locally available for independent checksum or reload
verification in this workspace.

## Part 5.10 held-out evaluation

The supplied Colab report says evaluation used 1,267 test windows and 212,856
valid target observations. It reports no test-set use in training or checkpoint
selection. All methods were compared in physical kW on common forecast origins,
target timestamps, and the same seven-target mask.

| Forecasting method | Test MAE (kW) | Test RMSE (kW) |
|---|---:|---:|
| GNN-Transformer | 12.6947 | 17.2082 |
| Persistence | 29.6110 | 51.5081 |
| Historical average | 24.9920 | 37.4353 |
| Seasonal persistence | **9.3719** | **16.2025** |

Seasonal persistence performed best overall. The GNN-Transformer improved on
ordinary persistence and historical average in the supplied aggregate results,
but did not outperform the seasonal baseline. These figures are the reported
aggregate results; per-target and per-horizon detail must be read from the
external evaluation artifact before making more granular claims.

## Forecast-to-controller integration contract

The simulator already has the intended receiving boundary:
`AgentObservation.forecasts` contains typed `ForecastFeature` values, and
`AgentObservation` checks that forecasts are issued and cut off no later than
the decision time and target strictly later. `ForecastFeature` represents one
nonnegative scalar forecast in kW. `BaselineController.select_action` still
receives one causal observation and returns `BatteryAction`; the simulator
remains responsible for physical feasibility and dispatch.

The previously missing piece was conversion from ordered node forecasts into
those simulator features. `src/forecasting/integration.py` adds the
dependency-light boundary helper `adapt_node_forecasts`:

- Input is one forecast origin in `[horizon, 8, 1]` kW (or a singleton
  `[1, horizon, 8, 1]` batch), the authoritative order
  `B01, B02, B03, B04, B05, PV01, W01, BUS01`, and a boolean target mask with
  the seven forecast targets true and BUS01 false.
- It rejects missing/incompatible provenance, non-kW units, nonfinite valid
  values, malformed ordering/masks, and non-hourly or misaligned target
  timestamps. The default negative-prediction policy is strict rejection.
  An explicitly selected `project_to_zero` policy projects negative valid
  node outputs to zero only on a private copy at this controller-integration
  boundary; it is recorded in `ForecastProvenance` and never changes model
  predictions, training artifacts, or held-out evaluation results.
- Each target hour becomes three existing `ForecastFeature`s: campus demand
  is the sum of B01–B05, solar is PV01, and wind is W01. BUS01 is never
  aggregated, even if its placeholder contains an arbitrary value.
- Forecast issue time must be no later than the observation decision; the
  historical data cutoff must be no later than issue time; the first forecast
  target is the next hourly interval. `AgentObservation` performs its existing
  final causality and duplicate checks.
- `ForecastProvenance` carries experiment/method identifiers, the explicit
  negative-value policy, and graph and preprocessing references beside the
  resulting observation. Each existing `ForecastFeature.model_id` is
  `method_id@experiment_id`, so different methods in the same experiment
  remain distinguishable while retaining the run identity. Baselines may
  explicitly mark unused graph/preprocessing references as `not_applicable`.

The helper returns a new compatible observation plus provenance sidecar. A
future episode runner should retain that sidecar in its run manifest and pass
only the observation to the controller. The adapter does not prove that the
caller obtained predictions without future realized data; it validates timing
metadata and tensor contracts, so forecast lineage must also be enforced by
the caller's data-loading boundary. No environment, dispatch, simulator, or
existing baseline code is modified, and this is not a forecast-serving system.

## Reproducibility and limitations

Reported external artifacts are located at:

- Training: `MyDrive/adaptive-energy-brain/results/forecasting/part5_9-colab-20261111T172326Z/`
- Evaluation: `MyDrive/adaptive-energy-brain/results/evaluations/part5_9-colab-20261111T172326Z/`

These are Google Drive paths, not Windows paths. This report does not claim
that these files exist in the local repository. Preserve the exact source
revision, training/evaluation JSON, split manifest, preprocessing file, graph
configuration, and checkpoints together when reproducing or integrating the
run. The test set remains sealed for final reporting; it must not be used for
further selection or tuning.

Limitations and negative findings:

- Only one synthetic hypothetical year/site was used; no measured campus
  dataset, external validation site, or verified physical topology is
  available.
- The assumed shared-bus graph and capacities are not measured engineering
  data.
- Seasonal persistence beat the learned model overall in the reported test
  evaluation. The learned model is not a default winner.
- Only node history is encoded; global weather history and declared
  known-future covariates are absent from the model input contract.
- Aggregate metrics do not establish controller value. Forecast accuracy
  improvements may not improve energy-management outcomes.
- The new adapter is offline and caller-driven. It does not load checkpoints,
  schedule inference, refresh forecasts during an episode, or validate the
  physical truthfulness of model provenance.
- Results and duration are as reported by the Colab run; local artifact-level
  verification is unavailable.

## Verification

Offline verification used the repository's existing Python environment and
did not require Colab, Drive, external services, or model retraining:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_forecasting_integration.py tests\test_simulation_state.py tests\test_baseline_interface.py tests\test_simulation_environment.py tests\test_baseline_runner.py tests\test_forecasting_evaluation.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

The focused adapter/simulator/baseline/evaluation regressions passed **126
tests**; the complete offline suite passed **828 tests** in **161.17 seconds**.
The exact commands are recorded in the current
[development handoff](development_handoff.md).

## DRL research handoff

The next phase should specify requirements and evaluation before selecting or
implementing an algorithm:

1. Reuse `MicrogridEnvironment`, `AgentObservation`, and `BatteryAction`.
   Controllers must never receive the environment, replay table, future rows,
   dispatch internals, or realized next-step outcomes.
2. Define state features and forecast use explicitly. The current observation
   exposes the previous completed hour and pre-action battery state. Forecasts
   must be issued by the decision boundary and target only future intervals.
   Compare at least a no-forecast policy condition with any forecast-informed
   condition; do not assume the GNN-Transformer is preferable to seasonal
   persistence.
3. Define reward terms, units, signs, scaling, and constraint penalties before
   training. Keep reward separate from reported physical/accounting metrics so
   reward shaping cannot conceal unserved energy, cost, curtailment, battery
   wear, or violations.
4. Respect battery SOC, efficiency, power, and grid constraints. Actions are
   signed AC-side kW requests; dispatch is authoritative and records
   adjustments. Do not bypass physical constraint handling with a policy-only
   feasibility assumption.
5. Compare policies on identical episode windows and scenarios against all
   four rule-based controllers: zero action, self-consumption, price-aware,
   and battery-preserving. Record seeds, scenario/data hashes, policy
   configuration, reward, physical metrics, and dispatch diagnostics.
6. Preserve chronological train/validation/test boundaries and freeze policy
   and reward decisions before final held-out evaluation. Design multiple
   seeds and uncertainty analysis before making generalization claims.
7. Treat meta-learning, DRO/CVaR, and optional orchestration as subsequent,
   separately justified research. No DRL algorithm or dependency is selected
   by this Part 5 completion task.
