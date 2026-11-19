# Part 6.3 PPO training infrastructure

## Entry point and data contract

Run commands from the repository root with the existing Python environment.
The CLI accepts the verified split root and fixes dataset access to its train
partition. Shared split-manifest and preprocessing metadata are read for
integrity checks. Training source CSV and metadata files require recorded
checksums, train labels, matching boundaries, and complete hourly coverage.
The existing simulation data preparation extracts physical units.

```powershell
.\.venv\Scripts\python.exe scripts/train_ppo.py --split-root data/processed/splits_hypothetical_campus_v1 --output results/ppo/example_run
```

The default config is `configs/ppo_training.yaml`. `--assets`,
`--rating-origin`, `--simulation-config`, and `--costs` select explicitly
documented equipment and economic assumptions. The default ratings origin is
the existing hypothetical example. Choose a fresh output directory.
The CLI also accepts `AEB_SPLIT_ROOT` and uses `AEB_EXPERIMENTS_ROOT` for
automatically named experiment directories when configured by the Windows
storage setup. A complete run keeps its checkpoints beside its metadata/logs.

## Sampling and policy inputs

Each reset samples a history-row start uniformly with replacement from all valid
training starts. For L rows and H scored hours there are L-H starts, indexed
0 through L-H-1. A window contains H+1 rows, with the first providing completed
history. The approved smoke configuration uses 168 scored hours and 169 rows.
Windows can overlap and repeat, giving correlated samples. Every episode
restores the configured initial battery SOC.

The new training environment owns each existing fixed-window adapter. It records
the start index, episode number, child seed, history timestamp, scored interval,
and initial SOC in `episodes.jsonl`. Only the causal 15-feature encoder output
reaches PPO. Demand, renewable generation, and tariffs describe the previous
completed hour. Calendar and battery features follow the current decision
boundary. Power/price references are 1000/1. The existing signed static-rating
BatteryAction mapping and committed operating-cost reward govern transitions.

The root seed initializes Python, NumPy, PyTorch, PPO, and episode sampling.
Child seeds are reproducible sampler draws and are recorded. Framework reset
with an explicit seed restarts ordinary sampling. Restored sampler state takes
priority over the first seeded reset during checkpoint continuation. Subsequent
automatic resets advance that restored generator. CPU deterministic algorithms
are requested, and the configured PyTorch thread count is restored on exit.

Monitor records episode returns and lengths. DummyVecEnv retains the final
observation and truncation flag before resetting to the next episode. PPO uses
these fields for truncated-rollout value bootstrapping. Raw rewards and physical
costs remain recorded separately from that algorithmic return calculation.
SB3 clips sampled Gaussian actions to the declared Box before adapter dispatch.

## Approved smoke settings

| Setting | Value |
|---|---|
| Policy | MlpPolicy, separate actor/value [64,64] Tanh networks |
| Execution | One CPU environment through DummyVecEnv, one PyTorch thread |
| Learning rate | Constant 0.0003 |
| Rollout / minibatch | 128 / 64 transitions |
| Epochs | 4 per rollout |
| Gamma / GAE lambda | 0.99 / 0.95 |
| PPO clip range | 0.2 |
| Entropy / value coefficients | 0.0 / 0.5 |
| Maximum gradient norm | 0.5 |
| Advantage normalization | Enabled |
| Seed / transition budget | 0 / 512 |

The runner calls the public PPO learn API once per complete rollout with
`reset_num_timesteps=False`. This permits checkpoint publication after each
returned optimizer update. Constant schedules apply across these calls.
512 transitions yield four rollout updates, sixteen training epochs, and
32 minibatch optimizer steps. PPO's internal n_updates counter counts epochs.
The metadata distinguishes rollout updates from optimizer steps.

Automated tests use eight-hour episodes, 16-step rollouts, batch size eight,
one epoch, [16,16] networks, and 32 transitions. Research-scale budgets and
validation-based selection require their next approved phase.

## Checkpoints and continuation

Each run creates an initial step-zero checkpoint and one bundle after each
completed update. A bundle contains `model.zip`, trusted serialized RNG/sampler
state in `state.pt`, and `manifest.json` with hashes, configuration/data/software
contract, cumulative timesteps, completed rollout updates, and optimizer steps.
PPO serialization preserves model and optimizer state. Reload verification runs
before publication and restores global RNGs afterward because SB3 model loading
reseeds them.

Each checkpoint directory is exclusively created at its step-numbered path.
Readers require a valid manifest. Model/state files are written and verified
before the manifest is atomically published as the bundle commit marker.
`latest_checkpoint.json` is atomically replaced after manifest publication.
An incomplete directory stays available as failure evidence and is rejected by
the checkpoint loader. The preceding valid bundle and latest pointer survive
a failed write. This publication strategy avoids directory-renaming sharing
errors observed on Windows. JSON publication uses flush/fsync and replacement
within the destination filesystem. Checkpoint paths are never reused.

```powershell
.\.venv\Scripts\python.exe scripts/train_ppo.py --split-root data/processed/splits_hypothetical_campus_v1 --resume results/ppo/example_run/checkpoints/step_000000512 --output results/ppo/continued_run
```

The configuration's timestep budget means additional transitions on continuation.
Configuration, source identity, data, equipment, costs, and library versions
must match the saved contract, with the additional transition budget allowed to
change. Resume accepts a trusted local checkpoint directory and verifies its
model/state hashes before loading serialized Python/PyTorch state. Record the
parent manifest hash and original checkpoint path.

Continuation restores the model, optimizer, cumulative counters, global RNGs,
and sampler RNG/counter, then starts a fresh training episode. A partial previous
episode and its rollout are discarded. Consequently continuation can follow a
different trajectory from uninterrupted learning. Source identity includes
uncommitted code/config bytes. If code or dependencies need to change, treat the
result as a separately designed experiment.

On interrupt or failure during collection/optimization, the last completed-update
bundle remains recoverable. Run status and traceback identify the failure, and
metadata distinguishes collected timesteps from checkpoint timesteps. A wrapper
error after a committed physical step is recorded and aborts the run. Recovery
uses the prior checkpoint in a new output directory. Failures before a complete
metadata record exists are written to `failure.json` once the output is created.

## Artifacts and completion checks

`experiment.json` follows `configs/rl_experiment_metadata.schema.json` and records
status, provenance, resolved settings, software, observation/action/reward
contracts, seeds, progress, lineage, verification, diagnostics, and artifact
hashes. `resolved_config.json` captures simulator, cost, asset, adapter, and PPO
configuration. `source_manifest.json` and `source_snapshot.zip` retain the actual
relevant source/configuration bytes and their hashes alongside Git revision and
dirty-worktree status. Secret environment files are excluded from this snapshot.

Other artifacts include `logs/progress.csv`, `monitor.csv`, `episodes.jsonl`,
and checkpoint bundles. Checkpoints live in ignored research-output directories
or configured external experiment storage. Keep their complete run directory.
The source snapshot and library versions support later reproduction, with
hardware/software limits documented in the experiment record.

Completion requires finite policy parameters and PPO metrics, changed policy
parameters, advancing optimizer state, successful checkpoint reload, equal
deterministic actions before/after reload, and a successful action through a
fixed training-window adapter. Exact verification commands/results and the
approved real-training-partition smoke run are recorded in the development handoff.

Terminal SOC treatment remains a research decision before comparative claims.
The current zero unserved-demand coefficient leaves shortage-aware training
dependent on an approved reliability objective. This pipeline uses the existing
error-on-shortage configuration. The infrastructure smoke demonstrates training
and artifact handling on synthetic campus data.

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
