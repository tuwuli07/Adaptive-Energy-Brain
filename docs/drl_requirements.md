# Part 6.1: DRL requirements and environment contracts

## Status and scope

Part 6.1 planning was approved on 2026-11-12. Part 6.2 now implements the
approved fixed-window, no-forecast adapter in `src/rl/environment.py` and its
encoder in `src/rl/observations.py`. The implementation decisions below
supersede the provisional planning language retained in later sections.
Part 6.3 now adds the approved PPO infrastructure smoke workflow described below.
All data/equipment assumptions remain synthetic and hypothetical.

## Part 6.4 approved pilot and validation infrastructure

The approved pilot compares learning rates 0.0003 and 0.0001 at seed 0,
4,096 transitions per run, and 168-hour overlapping training episodes. All
other PPO settings match the existing CPU configuration. Version-2 training
settings add a 2,048-transition publication interval and validation_pilot purpose.
Legacy smoke configurations and metadata remain readable.

A shared partition-selective loader supports validated train/validation access,
while the training entry point remains fixed to train. Frozen validation uses
all 1,314 declared validation rows and 1,313 scored hours, initial SOC 0.5, and
the existing controller/physical replay contract. FrozenPPOController shares
ObservationEncoder and map_normalized_action with the Gymnasium adapter.
Policy/optimizer fingerprints must remain equal before and after evaluation.
Baseline configurations and simulator accounting retain their existing behavior.

The primary coordinates are committed net operating cost and terminal stored
energy, accompanied by initial/final SOC and energy change. Pareto screening
uses declared numerical tolerances without assigning a price to terminal
energy. Failed or missing seeds remain in the experiment record. Pilot records
never select a research winner. Planned multi-seed selection requires the fixed
32,768-transition checkpoint and paired complete seed sets. The executable
matrix is limited to the approved two-run pilot.

Details, commands, artifacts, and diagnostics are specified in
[the validation guide](ppo_validation_guide.md). Verified test results and the
actual pilot measurements are recorded in the development handoff. Terminal
valuation, shortage-aware reliability, forecasts, SAC, and further training
remain subject to their approved phase boundaries.

## Part 6.3 training infrastructure

The approved CPU PPO pipeline is implemented in `src/rl/episodes.py`,
`src/rl/training.py`, and `scripts/train_ppo.py`. Its entry point, configuration,
sampling, artifact contract, and continuation semantics are documented in
[the PPO training guide](ppo_training_guide.md).

Training uses the existing train-selective source loader with explicit
verify_partitions=(train,), followed by physical simulation preparation.
Train-file checksums, labels, declared locations, and full chronological coverage
are checked. Shared manifest/preprocessing metadata supports integrity checks.
Each episode samples uniformly with replacement from L-H valid starts and owns
a fixed-window adapter with H+1 rows. The default H is 168 scored hours.
Overlapping windows give correlated samples and are recorded in a window ledger.
The initial dataset contains 6,132 verified training hours.

The approved smoke configuration uses one CPU environment, one PyTorch thread,
separate [64,64] Tanh actor/value networks, learning rate 0.0003, rollout 128,
batch 64, four epochs, gamma 0.99, GAE lambda 0.95, clip range 0.2, entropy
coefficient 0, value coefficient 0.5, max gradient norm 0.5, and 512 transitions.
The existing observation schema, scaling, physical action mapping, and reward
accounting govern every adapter transition. PPO performs its declared Box action
clipping and truncated-rollout bootstrapping. Monitor logs raw episode returns.

`configs/rl_experiment_metadata.schema.json` describes the PPO run record.
The forecasting metadata schema continues to describe its existing experiments.
Source/config snapshots include uncommitted bytes and hashes. SB3 2.7.1 is
installed through the optional `requirements-rl-training.txt` declaration.
Checkpoint bundles contain model/optimizer state and RNG/sampler sidecars.
An atomically published manifest commits a verified bundle, followed by atomic
latest-pointer publication. Incomplete bundles fail verification. Recovery uses
the last completed update, restores optimizer/RNG/sampler state, and starts a
fresh episode in a new run directory. This can diverge from uninterrupted
training. Tests exercise sampler restoration, local seed reproducibility, saved
optimizer moments, checkpoint corruption, and partial-update interruption.

The 32-transition synthetic test and the separately approved 512-transition
training-partition demonstration verify infrastructure behavior. Exact final
commands, timings, counts, and artifact locations appear in the handoff.
Terminal SOC treatment, shortage-aware reliability objectives, research-scale
budgets, and validation selection remain research decisions for later approval.

## Implemented Part 6.2 contract

Construct `MicrogridGymEnvironment(data, renewables, demand, config=None,
costs=None, price_provenance=..., adapter_config=...)` with one already selected,
validated SimulationData window. The wrapper privately owns an unchanged
MicrogridEnvironment. Production callers remain responsible for checksum-
verified loaders and split-local selection; the adapter does not infer or
certify dataset provenance from arbitrary in-memory metadata.

The approved ordered 15-feature vector below is implemented as float32.
`configs/rl_environment.yaml` fixes schema `drl_observation_v1`, power reference
1000 kW and price reference 1 unit of the configured currency/kWh. Thus powers
are represented in MW and prices retain their numeric currency/kWh values.
These are fixed unit conversions, with no fitted statistics. Reward stays in
unscaled currency. Currency is inherited from the validated physical config and
reported in info; no currency conversion occurs. Config files reject duplicate
keys, unsupported fields, or changed schema/scales.

Observation Box bounds: SOC and availability mask [0,1]; calendar [-1,1]; all
power fields [0,+infinity]; prices [0,+infinity] when negative prices are
disabled, otherwise [-infinity,+infinity]. Actual values must be finite.
Overflow beyond float32 range and nonzero values that underflow to zero are
rejected, without clipping. Ordinary float32 rounding is expected. Construction
preflights each replay row's representation, without exposing future rows to
the policy. State-dependent unexpected later encoding errors use the failure
policy below. Equipment stays fixed; forecasts are rejected by the encoder.

Action Box is float32 shape (1,) in [-1,1]. Real numeric single-entry arrays or
sequences are accepted, including valid integer/float dtypes; scalar, boolean,
complex, object/string, malformed, nonfinite, and out-of-range inputs are
rejected before the core call. Negative commands multiply the static charge
rating, nonnegative commands the static discharge rating, then construct
BatteryAction. There is no clipping or wrapper feasibility projection.

Reward is exactly negative committed `step.metrics.values.net_operating_cost`.
N input rows yield N-1 scored steps, with the first row reserved for history.
There is no default 168-hour horizon, extra TimeLimit, random window selection,
terminal SOC adjustment, or forecast feature. reset accepts None/empty dict
options only; unsupported options fail before resetting. Explicit seeds initialize
Gymnasium's RNG and action-space sampling; unseeded resets follow Gymnasium RNG
semantics without re-seeding actions. Physical replay remains deterministic.
At exhaustion the adapter returns terminated=False, truncated=True and the
actual final boundary observation; info preserves core_terminated and the
replay_window_exhausted reason. Step requires reset initially and after completion.
close invalidates stepping until reset and owns no external rendering resource.

Diagnostics contain independent copies of schema/order/scales/currency,
observation and decision timestamps, episode provenance, committed step/cumulative
metrics, normalized/requested/executed actions, adjustments/violations, balance
residuals, scored count, and completion reason. They are not policy features.
Full dispatch objects are not serialized into info; immutable dispatch results
remain authoritative core records.

Invalid actions and core dispatch/accounting failures do not advance wrapper
counters or physical state and remain retryable. An unexpected encoding/info/
reward error after the core commits increments the committed count, raises
RLAdapterError with its cause and an explicit no-rollback message, and prevents
further stepping until a successful reset. No unsupported rollback or private
core mutation is attempted. Failed reset response construction also leaves the
wrapper unavailable for stepping.

Gymnasium 1.2.3 and its missing mandatory dependency Farama-Notifications 0.0.6
were installed after inspecting the dry-run resolver plan. Existing package
versions were preserved. `requirements-rl.txt` records both pins; base package
imports remain independent of Gymnasium because `src/rl/__init__.py` does not
import the adapter. Part 6.3 adds Stable-Baselines3 through separate optional training requirements.
See the handoff for exact verification commands and results. Infinite observation
bounds intentionally produce an advisory Gymnasium checker warning; they avoid
inventing a cap or clipping unseen prices/powers.

## Historical Part 6.1 specification

The remaining provisional labels record the approved design baseline. Actual
Part 6.2 behavior is specified above; later training/evaluation decisions remain
subject to separate approval. PPO is planned first and SAC later.

The evaluation requirements are in [the DRL evaluation protocol](drl_evaluation_protocol.md).

## Existing architecture and authoritative interfaces

| Package | Implemented responsibility |
|---|---|
| `src/data/` | Acquisition, validation, chronological splits, training-fitted scaling, graph construction, and forecasting sequences. |
| `src/simulation/` | Physical components, constrained dispatch, causal observations, hourly replay, and economic/energy accounting. |
| `src/baselines/` | Zero-action, self-consumption, price-aware, and battery-preserving controllers. |
| `src/evaluation/` | Shared baseline runs, metric reconciliation, provenance/checksums, deterministic stress scenarios, and reports. |
| `src/forecasting/` | Forecast models, training/evaluation, and the standalone forecast-to-observation adapter. |
| `src/rl/` | Gymnasium adapter, observation encoder, training-only episode sampler, and CPU PPO infrastructure. |

Reuse [MicrogridEnvironment](../src/simulation/environment.py),
[AgentObservation and ForecastFeature](../src/simulation/state.py),
[BatteryAction](../src/simulation/actions.py),
[dispatch](../src/simulation/dispatch.py), and
[metrics](../src/simulation/metrics.py). Preserve the existing controller
`reset()` / `select_action(observation) -> BatteryAction` API and all four
baseline behaviors. Do not duplicate the simulator or change physical dispatch.
Older architecture documents contain proposals; executable interfaces determine
current behavior.

## Decision timing and information boundary

At boundary `t`, the observation contains the completed replay interval
`[t-1, t)` and the battery state before the action for `[t, t+1)`.
`state.timestamp_utc` identifies that completed interval's start;
`decision_timestamp_utc` identifies the current boundary. Demand, solar,
wind, import price, and export price are all previous-hour values.
Current battery SOC and feasible battery limits are current boundary information.
Calendar features follow the decision timestamp, not the lagged row timestamp.

Current-hour realized demand, generation, weather, and tariffs are unavailable
to the policy. The core reads them internally only after the action to resolve
dispatch and settlement. No current-hour tariff announcement assumption is
introduced. Post-step results may enter diagnostics, never pre-action features.
The feature encoder accepts an AgentObservation, not a replay dataset or engine.

## Provisional observation schema: no forecasts

Use one ordered float32 vector, schema identifier `drl_observation_v1`.
The following order is part of the proposed contract; normalization metadata
must retain the physical meaning of every entry.

| Index | Feature | Availability / units before scaling |
|---|---|---|
| 0 | `battery_soc` | Current boundary, fraction. |
| 1 | `charge_limit_kw` | Current SOC-feasible charge magnitude, kW. |
| 2 | `discharge_limit_kw` | Current SOC-feasible discharge magnitude, kW. |
| 3 | `campus_demand_kw` | Previous completed hour, kW. |
| 4 | `solar_generation_kw` | Previous completed hour, kW. |
| 5 | `wind_generation_kw` | Previous completed hour, kW. |
| 6 | `grid_import_price` | Previous completed hour, declared currency/kWh. |
| 7 | `grid_export_price_value` | Previous completed hour, declared currency/kWh; explicit zero placeholder if absent. |
| 8 | `grid_export_price_available` | 1 if the observed export price exists, otherwise 0. |
| 9 | `grid_import_limit_kw` | Observable configured connection limit, kW. |
| 10 | `grid_export_limit_kw` | AgentObservation.features() value, including its explicit missing-price policy, kW. |
| 11-12 | `hour_sin`, `hour_cos` | sin/cos of `2*pi*UTC_hour/24` at the decision boundary. |
| 13-14 | `weekday_sin`, `weekday_cos` | sin/cos of `2*pi*UTC_weekday/7`, Monday=0. |

Encode missing export price as `(value=0, available=0)`; an observed zero price
is `(0, 1)`. Preserve the configured physical export-price handling unchanged.
Do not invent startup measurements: the core requires one real history row.

Use declared positive physical reference scales for power, fixed from equipment
configuration or documented training-only choices. SOC, masks, and calendar
terms need no learned scaling. Price scaling, exact power scales, and numeric
space bounds must be frozen in Part 6.2 before training. Any fitted statistics
use training data only. Do not refit or clip validation/test observations.
Permit legitimate out-of-training-range values with appropriate space bounds;
reject nonfinite encodings. Record order, units, scales, bounds, masks, and
schema version. Keep the physical simulator in its existing physical units.

Initially hold equipment configuration fixed. If later experiments vary
capacity, efficiencies, or hardware limits, extend/version the schema to expose
the needed parameters. SOC plus one historical hour does not establish a
Markov state; treat the policy input as partial observation. Longer history,
frame stacking, or recurrence would be separate experiments. Do not add time
remaining for the continuing-task truncation interpretation without revisiting
the task definition.

## Provisional continuous action contract

The proposed RL space is `Box(-1, 1, shape=(1,), dtype=float32)`.
For its single scalar `a`, use the configured static battery ratings:

```text
if a < 0:
    requested_power_kw = a * configured_charge_rating_kw
else:
    requested_power_kw = a * configured_discharge_rating_kw
physical_action = BatteryAction(requested_power_kw)
```

Positive means AC-side discharge; negative means charge; zero requests idle.
With the existing 150 kW ratings, -1 requests -150 kW and +1 requests +150 kW.
Use the appropriate direction's rating even when ratings differ; a zero rating
disables that direction. These static ratings differ from the SOC-feasible
limits exposed in observations. The mapping remains stable as SOC changes.

Validate shape, real numeric type, finiteness, and range before stepping.
Reject booleans, malformed arrays, NaN/infinity, and values outside [-1, 1].
Do not silently clip in the adapter; any algorithm-side clipping must be
explicitly recorded. Mapping creates a request, not a guarantee of execution.
Dispatch retains SOC, efficiency, power, grid, curtailment, and shortage rules.
Record normalized action, requested kW, executed kW, and adjustment reasons.
No independent grid or renewable-curtailment action is introduced.

## Provisional reward contract

```text
reward = -step.metrics.values.net_operating_cost
net_operating_cost = import_cost - export_revenue
                   + battery_degradation_cost + unserved_demand_cost
```

Reward is the negative committed hourly cost in the configured currency, without
additional normalization initially. Use the existing metric result directly;
do not recompute settlement in the wrapper. Episode reward reconciles with
negative cumulative net operating cost, subject to floating-point tolerance.

[Existing economic configuration](../configs/simulation_costs.yaml) specifies
hypothetical USD accounting: degradation is 0.02 USD per kWh of AC charge-plus-
discharge throughput, and unserved demand is 0.0 USD/kWh. These are assumptions,
not calibrated aging or a validated value of lost load. Preserve both unchanged.
Efficiency losses already affect flows/billing and receive no duplicate charge.
Renewable utilization, curtailment, and battery throughput remain explicit
reported metrics; no arbitrary reward bonuses or extra weights are introduced.
Utilization includes energy exported or sent to storage and is not automatically
campus self-consumption. Safety adjustments are not physical violations and
receive no default penalty. Failed, uncommitted transitions produce no reward.

**Shortage-aware training is not ready:** the current zero unserved-demand
coefficient gives no explicit reliability preference in cost reward. Before
training on scenarios that record shortages, approve a reliability objective
(for example, justified shortage cost, a constrained reliability requirement,
or lexicographic reliability-before-cost selection). Define its units, selection
rules, and verification first. Do not infer reliability from cost improvement.
Keep initial functional development under the existing error-on-shortage policy.

## Provisional episode and wrapper lifecycle

Implement a thin composition wrapper in Part 6.2, leaving core methods intact.
Initially accept a validated fixed replay window with explicit provenance.

| Operation | Proposed adapter behavior |
|---|---|
| `reset(seed=None, options=None)` | Return fresh `(observation, info)`; reset core and wrapper counters; establish local RNG through the framework reset contract. Reject unsupported options explicitly. |
| `step(action)` | Validate/map one action, call core once, encode returned observation, calculate reward, return `(observation, reward, terminated, truncated, info)`. |
| Core replay exhaustion | Preserve the core's `terminated=True` as a diagnostic, but return RL `terminated=False, truncated=True` for the continuing-control interpretation. |
| Additional time limit | Also truncation; declare the exact scored horizon and reason. Initially prefer a window whose natural end matches that horizon. |
| Invalid/infeasible transition | Raise, with no fabricated failure reward or absorbing state. Preserve core atomicity and wrapper counters on rejected transitions. |
| Step before reset / after completion | Raise and require explicit reset; do not auto-reset. |
| Final observation | Encode the returned boundary observation without reading another replay row. |

For H scored hourly transitions provide H+1 consecutive rows in one partition.
The first row supplies history and contributes no dispatch, reward, or costs.
Reset restores configured initial SOC and clears the core ledger and metrics.
The existing core has no RNG; a seed does not change a fixed action replay.
Seed adapter/action-space randomness independently and record it. Random episode
sampling belongs in later training infrastructure and must remain split-local.
The proposed 168 scored hours is a development-window candidate, not a frozen
experimental horizon.

Return independent numeric arrays and diagnostics. Info may contain committed
metrics, requested/executed actions, adjustments, violations, timestamps,
episode provenance, and completion reason. Never promote privileged info into
policy features. Validate encodings/configuration before advancing where
possible; Part 6.2 must explicitly test wrapper failure bookkeeping rather
than assuming a post-commit encoding failure can roll back the core.

The interface follows the [Gymnasium Env API](https://gymnasium.farama.org/api/env/).
Truncation supports bootstrapping from the returned final observation; subsequent
training integration must preserve it through vectorized wrappers and avoid
using an auto-reset observation as the final state. A finite-horizon task would
need a separately approved termination, time-feature, and terminal-objective
contract. No physical absorbing failure model is introduced here.

## Forecast integration: deferred separate condition

Preserve [the existing adapter](../src/forecasting/integration.py): exact eight-
node order, BUS01 exclusion, B01-B05 demand aggregation, PV01/W01 mapping,
provenance, and explicit negative-prediction handling. Issue time and cutoff
must not follow the decision; cutoff must not follow issue time. Targets are
consecutive hourly timestamps beginning at decision time plus one hour.

The action controls the interval starting at decision time. Existing forecast
features therefore do not directly predict that immediately controlled
interval. Document this offset rather than retimestamping forecasts or relaxing
causality. Later features need fixed horizon/order, masks, split-local history,
and retained forecast provenance. No checkpoint loading, inference scheduling,
or forecast features are part of Part 6.2's initial no-forecast wrapper.

## Algorithm and dependency plan

Start with Stable-Baselines3 PPO, a small MLP, and CPU execution. Both PPO and
SAC support continuous Box actions. PPO provides an initial on-policy rollout
lifecycle; SAC adds off-policy replay and entropy-related state and is planned
as a subsequent comparison, potentially benefiting from transition reuse.
These choices do not establish expected superiority. Official references:
[PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and
[SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html).

Planning inspection found local NumPy 2.5.4 and CPU PyTorch 2.7.0+cpu;
Gymnasium and Stable-Baselines3 were absent. This is a point-in-time inventory,
not a compatibility certification. Do not install or upgrade dependencies now.
Part 6.2 requires approval of a checked Gymnasium pin as an optional RL
dependency; Stable-Baselines3 is deferred until training infrastructure. Keep
base simulator/baseline imports usable without RL packages and preserve the
existing environment and separate Colab configuration.

## Roadmap and expected files

| Subpart | Outcome and likely files, subject to separate approval |
|---|---|
| 6.1 | These requirements, evaluation protocol, and development handoff only. |
| 6.2 | Thin wrapper: `src/rl/environment.py`, `observations.py`, `actions.py`, `rewards.py`, `configs/rl_environment.yaml`, optional dependency declaration, and focused `tests/test_rl_*.py`. Final file grouping may be smaller if responsibilities remain clear. |
| 6.3 | PPO training configuration/CLI, training-only episode sampling, checkpoints, run metadata and an RL-specific schema. |
| 6.4 | Multiple-seed training and validation-only selection. |
| 6.5 | Shared DRL/baseline evaluation with terminal SOC treatment resolved. |
| 6.6 | SAC under an explicitly defined comparison budget. |
| 6.7 | Optional no-forecast / seasonal-persistence / GNN-Transformer ablations. |
| 6.8 | Research report, reproducibility evidence, limitations, and closure. |

The existing experiment metadata schema is forecasting-specific. Preserve it;
a later RL schema should reuse its provenance principles without forcing RL
records into forecasting fields. Do not implement meta-learning, DRO/CVaR,
RAG, dashboards, or services in Part 6.

## Open decisions and acceptance gates

Before Part 6.2: approve its bounded scope and optional dependency installation;
freeze numeric scales/bounds, options policy, and fixed-window construction.
Before training: freeze horizon, sampling distribution, algorithm settings,
training budget, seeds, and selection rules. Before shortage-aware training:
define reliability. Before comparative performance claims: resolve terminal
SOC treatment using the alternatives in the evaluation protocol.

Part 6.1 acceptance requires traceability to executable interfaces, explicit
causal timing, provisional contracts distinguished from implementation, visible
unresolved decisions, and only the three approved documentation files changed.
No new test result or external forecasting result may be asserted without
verification.

Part 6.2 verification must cover asymmetric/zero ratings and action endpoints;
shape/type/range rejection; observation order/dtype/bounds and missing-price
masks; train-only scaling; future-row perturbation without pre-action feature
changes; one-history-row/H-transition accounting; reset determinism; final
observation and truncation semantics; cost/reward reconciliation; requested
versus executed power; invalid-action/dispatch atomicity; and direct-core versus
wrapped-core transition equivalence. Run Gymnasium's checker once installed,
then focused regressions and the full offline suite when shared integration is
affected. Checker success does not by itself verify causality or physical safety.

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
