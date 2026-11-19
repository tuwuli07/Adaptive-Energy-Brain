# Paired gamma diagnostic

This bounded Part 6.4 experiment compares gamma 0.99 with 0.999 at seed 0,
learning rate 3e-4, and 8,192 transitions each. Other PPO settings match the
approved pilot. It is not a replacement for the unexecuted five-seed study.
The protocol is `configs/ppo_gamma_diagnostic.yaml`; execute with the existing
environment using `scripts/run_ppo_diagnostic.py --split-root <verified split root>
--output <new experiment directory>`. Existing output directories are rejected.

Before training, the runner verifies train-only provenance, identical initial
policy tensors, settings differing only in gamma, and the full paired window/child
seed sequence. Actual initial checkpoints and sampled windows are checked against
this preflight after each run. Training cannot load validation/test partitions.
Validation separately loads validation only and reuses frozen native replay.

Each run publishes completed-update checkpoints at 0/2048/4096/6144/8192. Frozen
4096/8192 validation includes the four existing baselines for reference. Existing
publication, failure preservation, accounting, causal observations, and action
mapping are unchanged. Every failed job remains in the experiment record and is
not silently replaced. There is no research-winner selection.

## Telemetry contract

`src/rl/diagnostics.py` uses SB3 callbacks and removable PyTorch hooks. It does not
replace SB3's training loop, change the installed library, alter gradients, draw
random numbers, or modify rewards. The supported path is SB3 2.7.1, one CPU
environment, separate actor/critic MLPs, unsquashed Gaussian actions, and no gSDE.

- `telemetry/transitions.jsonl` persists each raw transition immediately. It
  records the policy distribution mean/std/entropy, actual sampled and bounded
  actions, physical requested/executed power, input observation, SOC, lagged
  tariffs, accounting cost, raw vector reward, and balance residuals. The decision
  timestamp labels the executed interval; tariff context belongs to its previous
  completed interval. Observation/SOC context retains policy float32 precision.
- `telemetry/rollouts.jsonl` adds actual rollout-buffer value predictions,
  value targets and raw GAE advantages, adjusted buffer rewards, and the actual
  terminal value captured from SB3's timeout-bootstrap forward call. Completed
  episodes include cumulative accounting; terminal stored energy can be recovered
  from initial energy and physical charging/discharging totals. These raw
  advantages precede SB3's per-minibatch normalization; normalized minibatch
  advantages are not instrumented. A failed partial rollout retains raw transitions
  but may lack computed targets.
- `telemetry/gradients.jsonl` records every optimizer step. Parameter hooks observe
  the actual backward gradients before global clipping; an optimizer pre-hook
  observes them after clipping. Actor and critic groups are disjoint and exhaustive.
  Critic norms include the configured value-loss coefficient; these are gradients
  of the actual combined objective, not separately recomputed unweighted losses.
  Norms use float64 accumulation of the observed tensors. The observed post/pre
  norm ratio measures actual global scaling, with normal floating-point tolerance.
- Existing `logs/progress.csv` records explained variance, value/policy losses,
  approximate KL, Gaussian entropy loss/std, and PPO probability-ratio clipping.
  Ratio clipping is distinct from action-bound clipping and physical constraints.
  Reported Gaussian entropy is measured before action clipping, not the entropy
  of physically executed commands.

## Paired reward-scale diagnostic

The approved follow-up compares reward scales 1.0 and 0.0025 with gamma 0.99,
seed 0, learning rate 3e-4, and 8192 transitions per arm. The architecture,
rollout, batch, epochs, value coefficient, clipping, entropy coefficient, episode
length, and environment are held fixed. The protocol is
`configs/ppo_reward_scale_diagnostic.yaml`; the one-shot runner is
`scripts/run_ppo_reward_scale_diagnostic.py`. It requires a new output directory
and verifies paired initial policy tensors and all window/child-seed draws before
training.

`RewardScaleVecEnv` scales the float32 simulator reward at the PPO vector
interface only. The core simulator and `info` accounting remain in original USD
units. Diagnostics retain both raw and scaled reward; value predictions, targets,
advantages, buffer rewards, and timeout bootstraps are also represented in
dollar-equivalent units. SB3 adds the timeout bootstrap after the wrapper returns
the scaled immediate reward. The critic predicts in scaled units, so that
bootstrap is added directly and is not multiplied by the reward scale again.
Validation bypasses the training wrapper and replays frozen policies against
the unchanged USD simulator.

The completed two-run result met both configured critic diagnostic references:
the scaled-reward arm's median explained variance over the final eight updates
was 0.832, and mean per-update dollar-equivalent RMSE fell from $486.57 to
$106.89 (78.03% reduction using one common normalization denominator). This did not
produce deterministic charging: both policies ended at 50 kWh and requested
discharge throughout validation. The result is a single-seed diagnostic, not a
policy winner or approval for the ten-run study. Full measurements are in
[the diagnostic report](ppo_diagnostic_report.md).

Raw rewards are checked against the float32 conversion of negative committed
cost. Bootstrap-adjusted rollout rewards are checked independently. JSON rejects
nonfinite values; the existing trainer checks losses and model parameters.
Synthetic tests compare instrumented and uninstrumented policy/optimizer states,
RNG sidecars, and episode ledgers, as well as bootstrap and gradient identities.

## Interpretation

Training behavior is stochastic; deterministic validation uses the policy mean
subject to action bounds. Report both independently. Gamma changes value targets
and credit assignment as well as economic patience, so differences are not a pure
measurement of preference with critic quality held constant. One paired seed
cannot establish superiority or estimate training variability. Cost and terminal
stored energy remain separate coordinates, without a terminal valuation or reward
change. The held-out test partition remains untouched.

The run manifest contains configuration/source/data hashes, status, actual pairing,
runtime, checkpoint identities, and an artifact inventory. Exact verified results
are recorded in the development handoff after execution. Further experiments and
the ten-run study require additional approval.

Verified results are in [the diagnostic report](ppo_diagnostic_report.md).
