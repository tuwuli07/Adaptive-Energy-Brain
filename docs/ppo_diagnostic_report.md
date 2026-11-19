# Part 6.4 paired gamma diagnostic results

Completed 2026-11-14 (client date, Asia/Dhaka). This is one paired diagnostic seed,
not a research comparison or authorization for further training.

## Outcome

Gamma 0.999 did not produce charging in frozen deterministic validation at either
4,096 or 8,192 transitions. Both configurations depleted initial inventory and
continued requesting discharge at minimum SOC. Neither critic learned meaningful
return variation. Direct telemetry establishes a large actor/critic gradient-scale
imbalance; it does not establish that gradient clipping alone caused the behavior.

## Execution and reproducibility

Two fresh CPU PPO runs used seed 0, LR 3e-4, 168-hour episodes, 128-step rollouts,
batch 64, four epochs, separate [64,64] Tanh actor/critic networks, and 8,192
transitions each. Gamma was the only differing setting: control 0.99, treatment
0.999. Initial policy tensors and all 49 sampled windows/child seeds matched both
preflight and actual execution. Each run completed 64 rollout updates, 256 epochs,
and 512 optimizer steps, with 48 complete episodes and 128 steps in the last one.

Checkpoints at 0/2048/4096/6144/8192 all passed hash verification and reload checks.
The control's 4096-step policy AND optimizer state match the original uninstrumented
Part 6.4 pilot A exactly. This complements the synthetic instrumentation-equivalence
test. All 81 parent experiment artifact hashes passed review. Four frozen PPO
validations preserved policy/optimizer fingerprints. No failed training jobs or
validation cases occurred and no runs were repeated or replaced.

| Timing | Control | Treatment |
| --- | ---: | ---: |
| Training call including final verification (seconds) | 291.297 | 237.979 |
| Training pipeline internal timer (seconds) | 290.844 | 237.589 |
| Learning including telemetry (seconds) | 272.320 | 226.079 |
| 4096 validation replay (seconds) | 16.933 | 15.434 |
| 8192 validation replay (seconds) | 15.715 | 15.897 |

The complete orchestration took 721.945 seconds (12.03 minutes), excluding Python
startup. Six-controller validation stages took 89.212 and 88.988 seconds. Timing
includes telemetry/I/O and host/cache effects; the difference is not evidence that
a gamma value is inherently faster.

## Actor and critic measurements

| Final update / diagnostic | Control | Treatment |
| --- | ---: | ---: |
| Explained variance | -0.00000918 | -0.00002515 |
| Value loss | 171857.99 | 227081.93 |
| Policy gradient loss | -0.00023034 | -0.00018195 |
| Approximate KL | 0.00006002 | 0.00004708 |
| Gaussian entropy | 1.40609 | 1.40730 |
| Gaussian standard deviation | 0.98676 | 0.98798 |
| PPO ratio clipping fraction | 0 | 0 |
| Last rollout mean value prediction | -15.9979 | -15.9980 |
| Last rollout mean value target | -400.7410 | -455.9876 |
| Last rollout mean raw GAE advantage | -384.7431 | -439.9896 |
| Median actor gradient norm before clipping | 0.42282 | 0.41728 |
| Median critic gradient norm before clipping | 3237.70 | 3673.29 |
| Median actor gradient norm after clipping | 0.00006498 | 0.00005718 |
| Median observed global clipping scale | 0.00015443 | 0.00013612 |
| Optimizer steps triggering global clipping | 100% | 100% |

The critic norms include the configured value-loss coefficient 0.5. They are
observed gradients of the actual combined loss; no counterfactual unweighted
backward passes were run. Last-rollout predictions/targets precede the final
update. Their prediction standard deviations were only 0.03146/0.03105, versus
target standard deviations 154.56/183.18. The critic mostly learned a small nearly
constant offset instead of the return scale or variation. Increasing gamma enlarged
value targets without resolving that mismatch. Raw value losses across gamma are
not directly comparable because targets differ, but near-zero explained variance
and the prediction-target mismatch independently support poor fit in both.

All updates remained finite with small KL and zero PPO ratio clipping. There is
no evidence of explosive optimization. Global gradient clipping is pervasive and
critic-dominated. Adam can partly compensate for gradient rescaling, so small
post-clipping actor norms alone do not prove proportionately small parameter
updates or uniquely explain the policy failure.

Timeout bootstrapping was observed directly from SB3's actual forward calls for
all 48 completed episodes per run. Mean bootstrap predictions were -9.1423/-9.1403,
ranging approximately from -15.82 to -0.20. Adjusted rollout rewards matched raw
negative cost plus gamma times this value at truncations. No terminal-energy reward
or accounting adjustment was introduced. Raw rollout advantages are available;
normalized minibatch advantages were deliberately not instrumented.

## Stochastic training behavior

| Across 8192 transitions | Control | Treatment |
| --- | ---: | ---: |
| Sampled charging requests | 41.43% | 42.69% |
| Mean charging-request magnitude (kW, conditional on charging) | 89.69 | 90.82 |
| Actions clipped to [-1,1] | 33.54% | 32.85% |
| Executed charging hours | 39.33% | 40.22% |
| Executed discharging hours | 39.82% | 40.70% |
| Executed charge energy (kWh) | 276268.93 | 284568.12 |
| Executed discharge energy (kWh) | 253806.32 | 260966.64 |
| Physical requested/executed adjustment fraction | 37.62% | 35.74% |
| Mean complete-episode cost (USD) | 5923.48 | 5932.81 |
| Mean complete-episode terminal SOC | 0.3121 | 0.3266 |

Both terminal-SOC ranges were 0.1 to 0.9. Total raw training costs including the
partial last episode were 287718.33/288177.25 USD. These are stochastic training
trajectories with repeated initial SOC resets, not the long deterministic
validation episode. Episode/window distributions are descriptive and are not
independent estimates of seed variability.

Policy means stayed inside action bounds throughout training; large stochastic
standard deviations caused action-bound saturation. Mean normalized policy actions
were 0.2166/0.1804. Conditioning on observed lagged price, mean policy actions at
0.12 USD/kWh were 0.2157/0.1797, and at 0.18 were 0.2194/0.1830. Thus both ordinary
price groups retained a positive mean, with little separation. These pooled groups
also mix training progress, SOC, and calendar context; they are not causal price
response experiments. Spike-price samples were sparse (36 at 0.48 and six at 0.72).

## Frozen deterministic validation

All cases used 1314 validation rows / 1313 scored hours, initial SOC 0.5 and stored
energy 250 kWh, identical equipment, causal information, and native accounting.

| Checkpoint | Policy | Cost (USD) | Final SOC | Final energy (kWh) | Change (kWh) | First SOC-floor hour |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | Control | 43839.8993 | 0.1 | 50 | -200 | 5 |
| 4096 | Treatment | 43841.1123 | 0.1 | 50 | -200 | 6 |
| 8192 | Control | 43835.7713 | 0.1 | 50 | -200 | 3 |
| 8192 | Treatment | 43837.4853 | 0.1 | 50 | -200 | 4 |

All four policies requested discharge at every hour, charged zero kWh, and executed
190 kWh discharge. Mean requested powers at 4096 were 37.08/30.78 kW; at 8192 they
increased to 58.11/48.61 kW. Physical adjustments occurred in 1309/1308 and 1311/1310
hours respectively. These are feasible dispatcher responses, not safety violations.
None of the deterministic actions saturated at +/-1. The higher gamma reduced
positive request magnitude but never changed its sign. More training strengthened
early depletion in both configurations rather than producing sustainable cycling.

All controller cases had zero physical violations or unserved energy. Training
and validation accounting, finite-value, and power/energy balance checks passed.
The treatment costs 1.2130 USD more at 4096 and 1.7140 USD more at 8192 with identical
terminal energy. These small single-seed differences do not establish superiority.
At 8192, the control discharged all 190 kWh during the first three 0.18 USD/kWh
settlement hours. The treatment discharged 28.5661 kWh in the following 0.12
USD/kWh hour instead: 28.5661 times the 0.06 tariff difference explains the
1.7140 USD gap. The lower cost reflects initial-inventory discharge timing, not
learned recharge/arbitrage. These realized settlement tariffs are used only for
post-hoc accounting analysis; they were not policy inputs at those timestamps.

The price-aware reference was identical at both validation stages: cost 43674.7778
USD, terminal SOC 0.9 / energy 450 kWh (+200), 171 charging hours, 166 discharging
hours, 20761.77 kWh charged and 18547.50 kWh discharged. It uses lagged prices with
explicit economic thresholds, efficiency/degradation tests, and a reserve rule.
It has no realized current-hour tariff or forecast advantage. It is the descriptive
cost/terminal-energy frontier on this replay, not a generalized research winner.

## Conclusion and next bounded decision

This experiment does not support gamma 0.999 as a solution to the no-charging
behavior at this budget. It also does not show that gamma is irrelevant after
competent value learning. Both critics remain severely underfit, while immediate
inventory depletion and discounted arbitrage incentives still apply.

Do not launch the ten-run study unchanged yet. Recommend planning a bounded
critic/optimization-scale ablation first, keeping raw accounting, action mapping,
causal features, and simulator physics fixed. Candidate mechanisms to review are
value-target units/normalization and separate actor/critic clipping or optimization.
Select one controlled intervention with explicit reversible semantics and train-only
calibration before approving execution; none is implemented or approved here.
Track explained variance, prediction-target calibration, parameter-update norms,
and deterministic price-conditioned charging to test the mechanism. Resolve the
intended discounted-versus-undiscounted economic objective before comparative
research claims; do not invent a terminal-energy coefficient.

## Paired reward-scale critic diagnostic

Completed 2026-11-14 under the approved scope: exactly two fresh 8192-transition
runs, gamma 0.99, seed 0, learning rate 3e-4, and reward scales 1.0 (control) and
0.0025 (treatment). All other PPO and simulator settings matched. Preflight and
both actual runs verified the same initial policy tensors and all 49 training
window/child-seed draws. Each run completed 64 rollout updates and 512 optimizer
steps. Checkpoints at 0/2048/4096/6144/8192 all reloaded. No additional training
or multi-seed study was started.

Reward scaling occurred only at the PPO vector-environment interface. Simulator
cost accounting remained in USD, and raw and scaled reward were retained
separately. Telemetry verified that the timeout reward in the rollout buffer was
the scaled immediate reward plus gamma times the critic's already-scaled terminal
value; the bootstrap was not scaled a second time. Treatment's value predictions
and targets are shown in dollar-equivalent units by dividing the scaled critic
quantities by 0.0025.

### Critic fit and optimization diagnostics

Measurements below use the final eight updates (1024 rollout transitions) unless
otherwise noted. RMSE is in dollar-equivalent units. For normalized RMSE, both
arms use the common $1514.57 population standard deviation of all dollar-equivalent
targets from those final eight updates pooled across both arms. This fixed
denominator includes the between-arm target-mean difference; it does not rescale
each arm separately. The percentage RMSE reduction is therefore identical to
the absolute RMSE reduction percentage.

| Metric | Control (scale 1.0) | Treatment (scale 0.0025) |
| --- | ---: | ---: |
| Median explained variance | 0.0000303 | 0.83183 |
| Mean value loss (scaled units) | 246955.86 | 0.07659 |
| Mean value prediction (USD-equivalent) | -15.39 | -3509.60 |
| Mean value target (USD-equivalent) | -472.98 | -3473.20 |
| Mean per-update value RMSE (USD-equivalent) | 486.57 | 106.89 |
| Per-update RMSE range (USD-equivalent) | 393.22–712.20 | 50.34–151.04 |
| Normalized value RMSE (mean across updates) | 0.32126 | 0.07058 |
| Median actor gradient norm before global clipping | 0.42282 | 0.47766 |
| Median critic gradient norm before global clipping | 3237.70 | 1.21574 |
| Median actor gradient norm after global clipping | 0.00006498 | 0.18295 |
| Median critic gradient norm after global clipping | 0.50000 | 0.45454 |
| Global clipping frequency | 100% | 90.43% |
| Median observed global clipping scale | 0.00015443 | 0.34948 |
| Median actor parameter-update norm | 0.000384 | 0.008981 |
| Median critic parameter-update norm | 0.011993 | 0.009790 |
| Mean approximate KL | 0.00002751 | 0.00045298 |
| PPO ratio-clipping fraction | 0% | 0% |
| Mean policy entropy | 1.40814 | 1.39478 |

Treatment normalized RMSE improved 78.03%, exceeding the 25% diagnostic reference;
its median explained variance also exceeded 0.2. These are critic-fit signals, not
policy-quality criteria. Targets differed considerably between the learned
trajectories, so their dollar-equivalent means and raw value losses are not
like-for-like measures of performance. The common RMSE denominator is fixed
across arms; the RMSE reduction percentage is the same as the absolute RMSE
reduction percentage.

Control's critic gradients dominated the joint gradient norm by a median factor
of about 7694, and global clipping reduced its actor gradient to about
0.000065. Scaling narrowed the median critic/actor pre-clipping norm ratio to
2.27 and substantially increased actual actor parameter updates. Global clipping
still occurred on 90.43% of treatment steps. Adam's adaptive updates and the
observed actor changes do not isolate reward scale as the sole cause of the
policy's direction.

### Training behavior and frozen validation

Over all 8192 stochastic transitions, control/treatment sampled charging
fractions were 41.43%/25.85%; sampled actions were clipped to [-1,1] on
33.54%/44.45% of steps. Executed charge energies were 276268.93/165339.37 kWh,
and discharge energies were 253806.32/156494.45 kWh. Mean costs across the 48
complete training episodes were $5923.48/$5801.49. Their mean terminal SOCs
were 0.312/0.189, with treatment reaching the 0.1 floor in many episodes.
These stochastic episodes restart from initial SOC and are descriptive, not
independent seed replicates.

Both frozen validations replayed the same 1313 scored timestamps, starting at SOC
0.5 / 250 kWh, with identical equipment and USD accounting. Each policy requested
discharge in every scored hour, requested no charging, and finished at the 50 kWh
minimum. At 4096, control/treatment cost $43839.8993/$43835.7713; at 8192 both
cost $43835.7713. Thus the small 4096 cost difference did not represent learned
cycling, and by 8192 there was no cost difference. Treatment's 8192 requests
saturated at 150 kW. Requested powers stayed positive across observed import
price groups, with no meaningful price response; both policies spent roughly
1310 scored hours at minimum SOC. The four references were also replayed: the
price-aware controller cost $43674.7778 and ended at 450 kWh; battery-preserving
ended at 150 kWh, zero-action at 250 kWh, and self-consumption at 50 kWh.

| Replay | Policy | Net operating cost (USD) | Terminal energy (kWh) |
| --- | --- | ---: | ---: |
| 4096 | Control | 43839.8993 | 50 |
| 4096 | Treatment | 43835.7713 | 50 |
| 8192 | Control | 43835.7713 | 50 |
| 8192 | Treatment | 43835.7713 | 50 |
| Both | Price-aware reference | 43674.7778 | 450 |
| Both | Battery-preserving reference | 43850.9713 | 150 |
| Both | Zero-action reference | 43866.1713 | 250 |
| Both | Self-consumption reference | 43835.7713 | 50 |

At 8192, mean requested discharge power in the $0.12/$0.18 per-kWh observed-price
groups was 57.99/58.53 kW for control and 150/150 kW for treatment; both requested
no charging in any price group. The high-price groups each contained only three
hours. Executed discharge occurred in only 3/2 hours for control/treatment at
8192, while physical adjustment counts were 1311/1312 due to persistent discharge
requests at the SOC floor.

All six controllers had zero physical violations and zero unserved energy.
Maximum absolute power- and energy-balance residuals were about
4.3e-14. PPO validation preserved both policy and optimizer state. Control's
4096-step checkpoint matched the original pilot and previous gamma-control policy
and optimizer exactly. All 83 run-artifact hashes and all ten checkpoint reloads
passed. The 169 pre-existing experiment files were unchanged; no test partition
was accessed.

**Conclusion:** Reward scale 0.0025 strongly improved critic fit and reduced the
observed actor/critic gradient-scale imbalance, supporting the critic optimization
scale hypothesis at this single seed. It did not improve deterministic battery
management: charging remained absent, discharge requests persisted at the SOC
floor, and economically meaningful cycling was not learned. Do not select a
research winner or launch the ten-run study from these results. A next experiment
should be proposed and explicitly approved to investigate why improved critic fit
did not translate into charging or price-responsive behavior.

Training wall times were 283.64 seconds (control) and 256.29 seconds (treatment).
The 4096/8192 validation stages took 130.56/474.41 seconds; total orchestration
time was 1166.97 seconds. Both stages completed successfully.

## Gamma diagnostic artifacts and verification

Run root: `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_gamma_diagnostic_20261114`.
Review: `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_gamma_diagnostic_review_20261114`.
The run includes source/config snapshots, data/software provenance, the paired
preflight, 10 checkpoint bundles, per-step telemetry, losses, raw accounting,
frozen validation JSON/CSV, and plots. The review contains reproducible artifact-only
`analysis.py`, `review.json`, `critic_and_gradients.png`, and `validation_first48.png`.
Both review plots were visually inspected. Existing pilot artifacts were preserved.

Pre-training focused checks: 188 passed in 132.27 seconds, with the existing
Gymnasium warning about an intentionally unbounded observation-space maximum.
Post-run results and exact commands are recorded in the development handoff.
No dependencies, installed SB3 source, simulator, reward, baseline, observation,
action mapping, or forecasting behavior were changed. No held-out test data,
SAC, research-scale study, commit, or push was used.

## Reward-scale diagnostic artifacts

Run root: `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_reward_scale_diagnostic_20261114`.
Review: `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_reward_scale_diagnostic_review_final_20261114`.
The run manifest verified 83 artifact hashes, 10 checkpoint reloads, paired
initialization and sampler draws, and two frozen validation stages. The
artifact-only review includes `analysis.py`, per-update RMSE values in `review.json`, and
`review_manifest.json`. All 169 pre-existing files in the experiments directory
matched the pre-run SHA-256 inventory. No package versions changed.
