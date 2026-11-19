# Part 6.1: DRL evaluation protocol

## Status and evidence boundary

The Part 6.1 evaluation specification was approved on 2026-11-12. Part 6.3
adds a bounded training-partition smoke demonstration. See
[DRL requirements](drl_requirements.md) for implemented adapter and training
contracts and retained planning history. This protocol extends the principles of
[the baseline evaluation protocol](baseline_evaluation_protocol.md) without
changing existing controllers, runners, simulator dispatch, or artifacts.
Results will concern a synthetic hypothetical campus, not real operations.
Reported Part 5 results remain external/user-reported evidence; this milestone
does not independently verify their artifacts. Seasonal persistence reportedly
outperformed the GNN-Transformer overall; neither establishes control value.

## Part 6.2 implementation boundary

The fixed-window no-forecast Gymnasium adapter is implemented, with the approved
15-entry float32 representation, fixed power/price references 1000/1, static-
rating action mapping, committed operating-cost reward, and replay-exhaustion
truncation. It does not select windows, fit scalers, tune a policy, or alter
baseline behavior. Dataset loaders and later runners remain responsible for
split isolation and artifact provenance. Checks use synthetic fixtures, not the
held-out project partition. Exact commands/results are recorded in the handoff.

Core failures remain atomic/retryable. A wrapper error after physical commit
raises an explicit no-rollback error and requires reset; it is not a failed
physical transition with zero cost. Any later runner must classify this run as
incomplete and retain the committed accounting, never silently retry the same
action as if nothing happened. Infinite space bounds are intentional; actual
observations are checked for finite float32 representation without clipping.

Terminal SOC valuation/target treatment remains unresolved before comparative
claims. The zero shortage-cost coefficient still requires an approved reliability
objective before shortage-aware training. No such research decision, training,
forecast ablation, or performance claim was made in Part 6.2.

## Part 6.4 pilot protocol

The approved pilot runs configurations A/B at learning rates 0.0003/0.0001,
seed 0, and 4,096 training transitions each. All other settings are paired.
Training uses only verified train data, 168-hour windows, and the fixed causal
adapter. Validation follows completed training and uses frozen deterministic
inference on the entire chronological validation partition. Each policy and
all four baselines receive the same 1,313 scored timestamps, one initial history
row, configured equipment/costs, and SOC 0.5. Source/config hashes and checkpoint
contracts identify every result. A semantic policy/optimizer fingerprint checks
that inference leaves learned state intact.

Report net operating cost and terminal stored energy jointly, including SOC and
energy change. Numerical tolerances are 0.000001 currency units for cost and 0.000001 kWh
for stored energy. No terminal-energy valuation is added to
accounting or reward. Every failed training/evaluation job remains recorded and
is attempted once. Pilot screening records no selected configuration or research
winner, even when one coordinate pair descriptively dominates another.

The planned study remains five seeds per configuration, with primary selection
at exactly 32,768 transitions. Paired dominance requires no higher cost and no
lower terminal energy across every expected seed, with a strict improvement in
at least one coordinate for at least one seed. Missing/failed seeds invalidate
selection eligibility. Aggregation uses training seeds as the variability unit.
A single pilot seed has undefined across-seed standard deviation. Hourly rows
and overlapping training episodes supply no independent-sample count.

The implementation and artifacts are described in
[the validation guide](ppo_validation_guide.md). Actual results and revised
compute estimates are recorded in the handoff after pilot execution. Research
training and held-out evaluation retain their separate approval gates.

## Part 6.3 infrastructure verification scope

PPO training infrastructure uses the verified chronological train partition
through explicit train-only source loading. Read guards in synthetic tests reject
validation/test file access throughout loading, training, save/reload, and
continuation. Manifest paths that redirect train verification outside train are
rejected before file reads. Shared manifest/preprocessing records may contain
partition boundary descriptions and recorded hashes.

The approved 512-transition demonstration samples overlapping 168-hour episodes
uniformly with replacement. Automated tests use eight-hour synthetic episodes
and 32-transition jobs. Their repeated episodes are correlated training samples.
Fixed unit conversions govern observations throughout. Per-episode costs and
returns remain traceable through Monitor and the window ledger. PPO's timeout
bootstrapping uses each actual final observation before the sampled environment
resets. Reload verification uses a fixed training window and deterministic
inference. No checkpoint selection uses validation/test metrics in this phase.

The infrastructure records seeds, software versions, source/config snapshots,
training-file hashes, sampled windows, optimizer progress, and completion or
failure status. Checkpoint continuation starts fresh episodes after restoring
saved model/optimizer, global RNG, and sampler state. Trajectories can differ
from an uninterrupted run. The source/config contract and parent manifest hash
identify each continuation. See [the training guide](ppo_training_guide.md) and
[the handoff](development_handoff.md) for commands and final verification evidence.

Terminal SOC alternatives and reliability gates below still apply before
comparative research or shortage-aware training. The smoke settings support
infrastructure verification and carry no baseline-superiority conclusion.

## Chronological separation and selection

Use the existing entity-aligned chronological 70% train, 15% validation, 15%
test allocation in [splitting configuration](../configs/splitting.yaml).
Train/validation timestamp counts are floored; test receives the remainder.
Load checksum-verified physical-unit simulation rows from exactly one split.
Every episode, history row, optional stacked history, and forecast context/target
horizon must remain within its own partition. Do not silently borrow training
context to initialize validation or validation context to initialize test.

Fit normalization and any learned forecast model using training only. Freeze
transforms for validation/test and do not clip their values to the training
range. Use validation for hyperparameters, reward revisions, checkpoint
selection, and episode-protocol development. Predeclare the validation metric,
checkpoint frequency, tie-breaking rule, and tuning budget before training.
Do not select a lucky seed as the reported policy.

Before final DRL test evaluation, freeze policy checkpoints, preprocessing,
reward, terminal treatment, seeds, episode windows, scenarios, baselines, and
reporting rules. Test results may not guide further tuning or window selection.
The shared test partition already supported reported Part 5 forecasting
evaluation; disclose that prior use rather than claiming the dataset is newly
untouched. Any additional test inspection for design/debugging must be logged;
subsequent results cannot be described as untouched confirmatory evaluation.

## Matched episodes and controller access

For every comparison, provide all policies with identical:

- Physical input rows, source/entity registry, hashes, partition, and units.
- Scored timestamps and one real history row; H scored hours require H+1 rows.
- Initial SOC, equipment capacity/ratings/efficiencies, grid limits,
  curtailment policy, missing-price policy, and unserved-load handling.
- Cost assumptions, currency, episode duration, and terminal SOC treatment.
- Previous-hour observation timing and any declared historical context.

Use fresh controller/environment state for each run. A policy receives only
AgentObservation-derived features; no future replay rows, current-hour tariffs,
pre-action settlement results, or privileged diagnostics. Baselines retain their
existing `reset()` / `select_action(observation) -> BatteryAction` API.

Include `zero_action`, `self_consumption`, `price_aware`, and
`battery_preserving` with frozen documented parameters. Reuse existing metric
reconciliation and provenance conventions. The existing baseline runner is
registry-specific; a later separately approved common evaluator/controller
adapter is needed for learned policies. Do not assume it accepts an RL model.

Predeclare fixed validation/test windows independently of outcomes. Prefer
non-overlapping scored windows where coverage permits, explicitly recording
any reused history or overlapping windows. Do not treat correlated windows or
repeated deterministic runs as independent observations. Resolve final-window
handling and scored horizon before experiments; 168 hours is only a development
candidate. Log rejected/incomplete episodes, reasons, and coverage without
silently omitting failures or ranking partial runs against complete ones.

## Scenarios

Reuse [existing deterministic transformations](../src/evaluation/baseline_scenarios.py)
and their metadata, applying them identically to each controller's inputs:

| Scenario | Existing definition |
|---|---|
| `normal_operation` | Unmodified rows and physical limits. |
| `low_renewable` | Solar and wind powers scaled by 0.25. |
| `high_campus_demand` | Building loads scaled by 1.25. |
| `electricity_price_spikes` | Deterministic spike window, multiplier 4.0 and minimum spike price 1.0 in configured currency/kWh. |
| `limited_grid_import` | Import limit min(10% of original, 100 kW); explicitly sets unserved-load handling to `record`. |

Retain exact transformation code/configuration hashes and actual affected
timestamps; do not reproduce approximate transformations in an RL-only path.
Spike schedules and future scenario realizations are evaluator metadata, not
policy inputs. Distinguish training scenario exposure from unseen stress tests.
Declare the training scenario distribution before training and report normal
and stress results separately so aggregation cannot hide reliability failures.

## Reward, reliability, and reporting metrics

The initial reward is negative simulator-recorded net operating cost, including
the existing throughput degradation proxy. Report raw economic and physical
metrics independently of reward or any future normalization/shaping:

| Category | Required outputs |
|---|---|
| Economic | Import cost, export revenue, net grid transaction cost, degradation cost, unserved-demand cost, net operating cost. |
| Demand/reliability | Demand, served and unserved energy (kWh), failed/rejected episodes and reasons. |
| Renewables | Available, utilized, curtailed energy (kWh), utilization ratio from summed energies; undefined with reason when availability is zero. |
| Storage | Charge/discharge energy, AC throughput, efficiency losses, initial/final SOC and energy, energy change, SOC extrema. |
| Grid | Import/export energy, recorded connection-limit checks. |
| Physical accounting | Maximum absolute power and energy residuals, physical violation counts/codes. |
| Safety interventions | Intervals with adjusted actions, adjustment counts/codes, requested/executed power differences. |

Storage efficiency losses are available from dispatch battery results; extract
them without inventing a new cost. Distinguish adjustment-event count from
number of affected intervals. Zero-action may still be adjusted by dispatch.
Committed core transitions reject detected physical violations; also report
rejected attempts rather than inferring zero attempted failures from clean
committed traces.

The current unserved-demand cost coefficient is zero. Shortage-aware training
requires an approved reliability objective before using recorded-shortage
scenarios for learning. Alternatives include a justified monetary cost, explicit
reliability constraints, or lexicographic selection; no coefficient is selected
here. Stress evaluation may record shortages with existing accounting, but must
report them prominently. Preserve the baseline safeguard: a policy with worse
unserved energy or more physical violations than zero-action is ineligible for
a cost-superiority claim. Apply declared reliability requirements across other
pairwise claims as well. Reliability safeguards do not resolve terminal SOC bias.

Report absolute cost differences against zero-action and each relevant baseline.
Percentage changes need the existing signed convention and an explicit undefined
case for a zero denominator. Do not equate renewable utilization with campus
self-consumption or assign renewable origin to grid-charged battery discharge.

## Terminal SOC: required decision before comparative claims

Free final SOC can favor a controller that consumes initial stored energy.
Identical initial SOC and reporting final SOC alone do not remove that bias.
The existing runner reports terminal SOC without enforcing or valuing it.
Use free-terminal-SOC only for functional development and descriptive traces;
do not claim comparative DRL performance until an explicit treatment is frozen.

| Feasible alternative | Benefit | Tradeoff / approval needed |
|---|---|---|
| Terminal stored-energy valuation | Account for ending inventory in an adjusted economic score, separately from raw operating cost. A possible form is operating cost minus v times (final minus initial stored kWh). | Requires a defensible, predeclared currency/stored-kWh value and efficiency/market assumptions. A common coefficient may not reflect realized recoverable value. No coefficient is chosen here; never derive it from test outcomes. |
| Common terminal target or band | Require comparable ending inventory, with reliability and feasibility checked for every controller. | Existing baselines cannot be assumed to hit a target. Classify misses explicitly; a hard requirement can make episodes infeasible. Any common recovery controller must be separately approved, preserve original baseline behavior, include all recovery energy/costs, and avoid oracle inputs. |
| Common causal settlement/recovery phase | Extend each scored run using the same declared recovery rule until a shared target/band or fixed recovery budget. | Adds duration, data requirements, operational assumptions, and possible failures. Keep recovery rows inside the split, freeze tariff information rules, report phase costs separately, and do not hide unequal coverage. |
| Long continuous runs with boundary accounting | Reduce repeated reset end effects and report retained inventory explicitly. | Horizon length alone cannot eliminate inventory bias. Still needs a valuation or matched-inventory rule for formal ranking; crossing split boundaries is prohibited. |

No terminal valuation coefficient, forced final action, or physical dispatch
change is approved. Decide whether terminal treatment affects training reward,
selection score, evaluation score, or all three, and document any mismatch.
If using a target, define tolerance, feasibility checks, misses, and costs
before test. Do not discard target misses after observing policy outcomes.

## Seeds, deterministic evaluation, and uncertainty

Propose five predefined training seeds `[0, 1, 2, 3, 4]`, subject to the training
budget gate. Record separate algorithm, Python, NumPy, PyTorch, environment,
action-space, and episode-sampler seeds or their documented derivation.
The core replay is deterministic and has no RNG. Fixed actions/data/configuration
must reproduce transitions; different seeds need not change that replay.

Evaluate each frozen trained policy deterministically, without exploration,
training updates, or normalization-statistic updates, on the same windows and
scenarios. Record deterministic inference mode and network evaluation mode.
Evaluate all trained seeds; disclose failures. Report per-seed and per-scenario
metrics, mean and standard deviation across training seeds, and paired
differences against baselines. Deterministic baselines need only one run per
matched condition; repetition does not create training-seed variability.

If confidence intervals are used, predeclare their unit and method and respect
temporal/window dependence. Five seeds give limited uncertainty resolution;
do not make unsupported significance or generalization claims. Record wall time,
environment steps, optimizer updates, device, and compute budget. For PPO/SAC,
predeclare whether the main comparison matches environment interactions or
wall time, and report both; neither guarantees equal optimization effort.

## Optional forecast ablations: later approval only

Keep no-forecast DRL as the primary initial condition. Later compare matched
conditions using seasonal persistence and the frozen GNN-Transformer, with
optional ordinary persistence/history-average controls. Preserve the existing
adapter's decision-plus-one-hour first target, issue/cutoff rules, node order,
mask, provenance, and negative-prediction policy. Do not relabel targets to fill
the immediately controlled interval or use realized future values as forecasts.

Freeze forecast methods/checkpoints using train/validation only. All methods
use matched decision origins, horizons, split-local context, availability masks,
negative-output handling, training budget, and controller architecture where
feasible. Account for any extra warmup by matching scored windows. Baselines
remain unchanged and do not consume forecasts: label these comparisons as
different information conditions, and use matched DRL ablations to isolate
forecast benefit. Do not claim an equal-information baseline comparison merely
because the physical replay is shared.

## Reproducibility and artifact contract

For each run, retain source/license/origin metadata, synthetic flags, units,
UTC coverage and splits, raw/input/split/config hashes, asset registry, scenario
transformations, history/scored boundaries, initial SOC, observation schema and
scales, action mapping, reward definition, reliability/terminal rules, all seeds,
algorithm/network/hyperparameters, tuning budget, checkpoint-selection rule,
software versions, device/determinism settings, and checkpoint/artifact hashes.

Record source revision plus dirty-worktree status and a source snapshot/diff
identity: a commit hash alone cannot describe this uncommitted checkout.
Record completion/failure state and exact commands. Keep raw simulator metrics,
policy outputs, diagnostics, and reports traceable to each other. Use new output
directories; never overwrite datasets, checkpoints, or existing experiments.
Follow the Windows storage convention without moving existing files. A future
RL-specific schema should preserve the existing forecasting schema unchanged.
No bitwise reproducibility across hardware or nondeterministic kernels is assumed.

## Acceptance and verification gates

Part 6.1: requirements/protocol agree with actual simulator timing, action and
metrics; unresolved terminal/reliability decisions are explicit; only the two
new documents and approved handoff update change. Check Markdown links and
diff scope; no training or runtime tests are necessary for documentation alone.
Do not present earlier 828-test evidence as a fresh run.

Part 6.2: implement only an approved no-forecast wrapper; verify causal feature
encoding, normalized action mapping, physical transition equivalence, accounting,
atomic failures, reset/seeding, and final-observation/truncation behavior using
small synthetic fixtures. Run the framework environment checker after approved
installation, focused simulator/controller regressions, and the full offline
suite when integration warrants it. Do not load the held-out dataset for tests.

Before experiments: validate partition isolation, train-only preprocessing,
seed/checkpoint restoration, no-overwrite artifacts, and absence of test access
in training/selection paths. Before comparative claims: freeze terminal SOC,
reliability criteria, matched episodes, failure handling, selection rules, and
uncertainty reporting. A successful smoke run or high reward is not evidence
of baseline superiority or real-campus validity.

## Verified Part 6.2 result

The final focused regression command passed 164 tests, including all 51 new
adapter cases. The complete offline suite passed 879 tests in 192.36 seconds.
Gymnasium check_env passed with render checks disabled and one advisory warning
about intentionally infinite observation bounds. pip check found no broken
requirements; existing package versions and unrelated files were preserved.
Syntax, relative links, whitespace, and git diff --check passed. Exact commands
and temporary-directory restrictions are recorded in the
[development handoff](development_handoff.md). Functional adapter acceptance is
satisfied; no training, policy comparison, forecast assistance, commit, or push
occurred. Terminal SOC and shortage-aware reliability decisions remain open.

## Verified Part 6.3 result

Final verification on 2026-11-13 passed 178 focused regression tests and 893
complete offline tests, including the 14 new sampling/training cases. The one
Gymnasium infinite-bound warning is expected. The approved 512-transition CPU
run completed four rollout updates and 32 optimizer steps in 38.5280 seconds of
pipeline time, with 53.4839 seconds of measured CLI wall time. Model reload,
deterministic inference, artifact hashes, and synthetic checkpoint continuation
passed. Safe publication uses an atomic manifest commit marker. Exact commands,
saved test reports, artifact paths, timings, and remaining research decisions
are recorded in the [development handoff](development_handoff.md).

## Part 6.4 verified pilot evidence

The approved two seed-0, 4,096-transition pilots and all four baseline validations
completed on 1,314 validation rows (1,313 scored hours), with frozen PPO/optimizer
fingerprints and shared action mapping. Focused verification passed 184 tests;
final offline verification passed 899 tests. Both PPO policies depleted initial
stored energy to 50 kWh and did not recharge. These short runs establish
infrastructure evidence only; no research winner or seed uncertainty is inferred.
See [the handoff](development_handoff.md#current-task-part-64-completed-pilot) for
actual metrics, artifacts, runtimes, diagnostics, and remaining decisions. The
32,768-transition five-seed comparison remains unexecuted pending approval.

## Part 6.4 secondary economic metric: inventory-adjusted net operating cost

To account for boundary inventory distortions across finite evaluation horizons
without altering simulator physics, step rewards, or Pareto dominance criteria,
evaluation reporting includes a secondary economic convention:

`inventory_adjusted_net_operating_cost_usd = raw_net_operating_cost - (terminal_energy_kwh - initial_energy_kwh) * reference_value_usd_per_kwh_dc`

For the synthetic campus scenario, `reference_value_usd_per_kwh_dc` represents
the off-peak replacement cost of stored DC energy, incorporating charging
efficiency $\eta_c$ and AC battery degradation cost:

`reference_value_usd_per_kwh_dc = (p_offpeak + c_degradation) / eta_charge = (0.12 + 0.02) / 0.95 = 0.14736842105263157 USD/kWh DC`

### Evaluation scope and guardrails:
1. **Secondary metric only**: Raw net operating cost, terminal stored energy,
   terminal SOC, and the multi-objective Pareto dominance frontier (`dominates()`)
   remain the primary evaluation criteria.
2. **No training reward leakage**: PPO training continues to optimize committed
   step operational cost without terminal inventory salvage or shadow pricing.
3. **No hardcoding**: Valuation parameters are loaded dynamically from
   `SimulationConfig` (battery efficiency), `CostConfig` (degradation rate), and
   baseline configuration (off-peak tariff threshold).
4. **Convention boundaries**: This metric represents an off-peak replacement cost
   accounting convention, not an electricity-market valuation, dynamic shadow price,
   or physical guarantee of monetization.
