# Part 6.4 PPO pilot and validation protocol

## Approved execution scope

The approved pilot trains configurations A and B once each with seed 0 and
4,096 transitions. Their learning rates are 0.0003 and 0.0001. All other PPO
settings match the approved CPU configuration, including separate [64,64] Tanh
actor/value networks, rollout 128, batch 64, four epochs, gamma 0.99, GAE lambda
0.95, entropy coefficient 0, and one CPU thread. Episodes contain 168 scored
hours plus one history row, sampled uniformly with replacement from train.

```powershell
.\.venv\Scripts\python.exe scripts/run_ppo_experiments.py --split-root data/processed/splits_hypothetical_campus_v1 --output E:\AdaptiveEnergyBrainStorage\experiments\part6_4_pilot_20261113
```

The command uses `configs/ppo_validation.yaml`. Its executable stage is the
two-run pilot and each run is capped at 4,096 transitions. The planned five-seed,
two-configuration study is stored as protocol metadata and awaits approval.
The planned primary checkpoint is fixed at 32,768 transitions. Diagnostic
milestones are 8,192, 16,384, and 32,768. Pilot records carry a separate stage
and purpose, and their screening never chooses a research winner.

## Partition and information boundaries

The shared verified-partition loader accepts train or validation and explicitly
restricts hash verification and CSV loading to the requested partition. Shared
manifest/preprocessing files are checked for integrity. Selected source files
must have recorded hashes, correct split labels, complete hourly coverage, and
paths contained within their declared partition. Test access is rejected by
this entry point. `load_training_data` retains its original train-only API and
provenance keys. TrainingEpisodeEnvironment still requires training provenance.

The pilot finishes both training calls before validation begins. The validation
entry point can also run separately through `scripts/validate_ppo.py`, using a
JSON list of checkpoint id, configuration, seed, and path records. Frozen policy
inference runs with deterministic actions and evaluation mode. A semantic hash
of policy and optimizer tensors/scalars must match before and after evaluation.
Validation preserves Python, NumPy, and PyTorch RNG state on exit.

FrozenPPOController uses the existing ObservationEncoder and the shared
`map_normalized_action` function from the Gymnasium adapter. The unchanged
BaselineController API passes only AgentObservation into policy selection.
The shared physical replay helper resets each controller and core environment,
records committed costs/flows, and verifies identical scored timestamps.
Future replay rows and current-hour realized tariffs are outside policy inputs.

## Fixed validation episode

The existing split manifest declares 1,314 validation rows, beginning
2026-09-13T12:00:00Z and ending exclusively 2026-11-07T06:00:00Z. The first row
is completed history. The first scored interval starts 2026-09-13T13:00:00Z,
giving exactly 1,313 scored hours and a final boundary at the declared end.
Every policy begins at SOC 0.5, equivalent to 250 stored kWh with the configured
500 kWh battery. All controllers share input hashes, equipment, accounting,
tariffs, initial conditions, and scored timestamps. The report checks checkpoint
equipment, costs, assets, seed, schema, split-manifest hash, and timestep budget.

A full continuous replay is the primary validation episode. It preserves all
validation coverage and avoids repeated initial-storage endowments from short
resets. It is longer than the training episodes, which is an explicit horizon
generalization condition. Weekly diagnostics can be derived from its trace
while retaining continuous battery state. The evaluation uses existing cost
accounting and reports terminal storage separately.

## Cost and retained-energy comparison

The primary coordinates are net operating cost and final stored battery energy.
Report initial and final SOC/energy, stored-energy change, import/export energy,
throughput, degradation cost, curtailment, losses, shortages, action adjustments,
balance residuals, and completion. Renewable utilization includes export and
charging, following existing accounting definitions.

Pareto dominance requires no higher cost and no lower terminal energy, with at
least one strict improvement. Numerical comparison tolerances are 0.000001
currency units and 0.000001 kWh. These tolerances reconcile floating-point
accounting and assign no economic value to stored energy. Reporting includes a
descriptive frontier across completed eligible controllers and a separate PPO
configuration screening record. A cost/energy tradeoff retains both candidates.
Pilot screening always records selected_configuration=null and research_winner=null.

For the planned study, seeds are paired across configurations. Selection requires
the fixed 32,768-transition checkpoint and all expected seeds. Dominance must hold
across all paired seeds, with a strict improvement in at least one coordinate
for at least one seed. Failed and missing seeds remain explicit in the record
and make that configuration ineligible. Successful-subset statistics remain
descriptive. Report mean, sample standard deviation, range, and individual
paired differences. A single pilot seed has undefined sample standard deviation,
represented as null. Hourly rows and overlapping training windows are correlated.

The joint coordinates support comparison of cost and inventory without choosing
a terminal-energy price. A scalar adjusted-cost ranking still requires an
approved terminal methodology. The current shortage-cost coefficient remains
zero, and shortage-aware training requires an approved reliability objective.

## Checkpoints, metadata, and failure records

Version-1 smoke training configurations retain their 128-step publication cadence
and metadata schema 1.0. Version-2 configurations add checkpoint_interval and
explicit validation_pilot or validation_experiment purposes, using metadata 1.1.
Pilot publication occurs at step zero, every 2,048 transitions, and completion.
Whole-rollout updates continue between publications. Existing model/state bundle
formats and manifest commit markers remain readable. Historical checkpoint
continuation retains its source/config/data/software matching contract.

Each pilot run records its resolved YAML, source/config snapshot, train hashes,
seeds, progress, checkpoint hashes, logs, completion status, runtime, and failures.
The validation record stores separate validation provenance, baseline-config
hash, common timestamps, checkpoint identity, physical metrics, behavior, and
frozen-state verification. Failed controller evaluations retain exceptions,
tracebacks, committed-step counts, and partial accounting. The matrix attempts
each approved training job once and records failed jobs without replacement.
An interruption aborts the pilot while retaining completed records and bundles.

Top-level artifacts include protocol.json, pilot.json, screening.json, per-run
training folders, and validation/validation.json. The comparison.csv file has
one row per controller and each completed controller has hourly.csv. Diagnostic
plots show cost versus terminal energy, continuous SOC/action/cost trajectories,
and training loss/KL/standard-deviation curves. Artifact hashes connect the
reports to their inputs and checkpoints. Source snapshots include uncommitted
code/configuration bytes alongside the Git revision and dirty-worktree flag.

## Verification and pilot evidence

Synthetic tests cover partition read guards, exact baseline replay equivalence,
PPO action equivalence with Gymnasium, deterministic and frozen inference,
window accounting, terminal energy, publication cadence, failed-run retention,
seed aggregation, Pareto tradeoffs, and the planned fixed-checkpoint rule.
Focused regressions and the complete offline suite precede the approved pilot.
Full experiment jobs stay outside the automated suite. Exact final commands,
test results, pilot measurements, artifact paths, and revised compute estimates
are recorded in the development handoff after execution.

### Completed approved pilot (2026-11-13)

Both seed-0 runs completed 4,096 transitions and published 0/2,048/4,096 bundles.
A/B training wall times were 85.222/73.667 seconds; validation replay times were
15.877/17.863 seconds. Costs were 43,839.8993/43,844.5483 USD, both ending at
SOC 0.1 (50 kWh, a 200 kWh drawdown). Both policies continuously requested
discharge, reaching the SOC floor at hours 5/14. There were no shortages or
physical violations. All four baseline replays completed under identical conditions.
Price aware was the descriptive cost/energy frontier for this replay; no research
winner is selected. The five-seed study has not run.

Focused tests: 184 passed (122.80 s). Final offline suite: 899 passed (296.62 s).
The one warning concerns the intentionally unbounded Gymnasium observation space.
Full metrics, exact commands, limitations, and runtime estimates are in
[the development handoff](development_handoff.md#current-task-part-64-completed-pilot).
Pilot artifacts are at `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_pilot_20261113`.
A separate `part6_4_pilot_review_20261113` directory contains clearer plot layouts,
first-48-hour behavior diagnostics, and review provenance; original artifacts remain
immutable. Estimated ten-run study time including 30 validation passes is 1.6-2.0
hours, with a recommended 2-3 hour allocation. Further execution requires approval.
