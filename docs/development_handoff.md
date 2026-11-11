# Development Handoff

## Current phase

Part 5 — GNN-Transformer energy forecasting. Parts 1–4 and Parts 5.1–5.8 are
implemented. Part 5.9 provides the Colab GPU training notebook and guide;
successful full GPU training is still unconfirmed. Part 5.10 now provides a
reproducible final evaluation pipeline, but no final held-out results exist
because no full-training checkpoint is available. Windows storage setup
directs project caches, temporary files, and new training artifacts to an
external E: storage root.

## Completed components

- Parts 1–2: Python project foundation; data acquisition, validation,
  integration, chronological train/validation/test splits, train-fitted
  scaling, graph construction, and forecasting sequences.
- Part 3: deterministic single-bus physical simulator, causal observations,
  dispatch, and metrics.
- Part 4: four rule-based baselines (`zero_action`, `self_consumption`,
  `price_aware`, `battery_preserving`) and shared experiment/comparison
  workflow.
- Stack alignment: root `AGENTS.md`, this handoff, an optional integration
  plan, and a JSON Schema for future Part 5 experiment metadata.
- Part 5.1: [forecasting requirements](forecasting_requirements.md), the
  [forecasting data audit](forecasting_data_audit.md), and
  `configs/forecasting_model.yaml`.
- Part 5.2 environment: local CPU-only PyTorch pin, separate optional Colab
  CUDA 12.8 pin, environment smoke-check script and tests; see
  [ML environment setup](ml_environment_setup.md).
- Part 5.3 forecasting baselines: persistence, 24-hour seasonal persistence,
  history-only average, and shape/mask-aware MAE and RMSE in
  [the forecasting package](../src/forecasting/baselines.py). The baselines
  accept system-level or node-level physical-unit histories and preserve
  non-time dimensions.
- Part 5.4 loaders: [forecasting dataset](../src/forecasting/dataset.py) and
  [DataLoader/graph utilities](../src/forecasting/dataloader.py). These wrap
  `prepare_sequences`, preserve physical kW targets and timestamps, use the
  existing train-fitted input scaling, and keep each split's windows isolated.
  Graph indices, node order, attributes, and optional edge weights are
  validated without adding PyTorch Geometric or inventing weights.
- Part 5.5 spatial encoder:
  [GraphMessagePassingEncoder](../src/forecasting/models/gnn_encoder.py) maps
  `[batch, lookback, nodes, features]` to
  `[batch, lookback, nodes, hidden_dim]`. The configurable initial defaults
  are input dimension 1, hidden dimension 64, two layers, 0.1 dropout, and
  ReLU. Directed incoming neighbors use edge-weighted mean aggregation; a
  separate self projection supports isolated nodes without implicit
  self-loops. These defaults are not tuned.
- Part 5.6 temporal encoder:
  [TransformerTemporalEncoder](../src/forecasting/models/transformer_encoder.py)
  accepts and returns `[batch, lookback, nodes, d_model]`. It folds batch and
  node axes so self-attention runs independently over each node's historical
  time steps, with sinusoidal positions and bidirectional attention restricted
  to the supplied historical window. Padded positions can be masked.
- Part 5.7 integrated model:
  [GNNTransformerForecaster](../src/forecasting/models/gnn_transformer.py)
  reuses both encoders and emits direct horizon predictions from the final
  valid historical embedding. Node-level forecasts preserve graph node order
  as `[batch, horizon, graph_nodes, target_channels]`; non-target nodes are
  zero placeholders and must be excluded with the target mask. Optional
  system-level forecasting pools explicitly selected nodes by configured
  `mean` or `sum`; the audited data has no system-level label. The configurable
  direct head supports multiple target channels, but the current dataset has
  one channel. Checkpoints store a reconstruction config and `state_dict`, and
  refuse to overwrite an existing path by default. Graph edges and Part 2
  preprocessing/scaler artifacts remain external; inference must provide the
  matching topology, node order, and prepared historical inputs.
- Part 5.8 training:
  [training utilities](../src/forecasting/training.py) implement AdamW,
  valid-element-count-weighted masked MSE, epoch training/validation, finite
  loss/gradient checks, optional gradient clipping, validation early stopping,
  best and resumable checkpoints, and evaluation loading. The
  [training CLI](../scripts/train_forecaster.py) reads only train and
  validation records, uses the existing train-fitted preprocessing artifacts,
  and writes experiment history and metadata. BUS01 is excluded by the
  dataset-provided target mask; node and target dimensions are checked against
  the model. The pipeline does not evaluate or tune on the held-out test split.
  Additive checkpoint `training_state` preserves the Part 5.7 checkpoint
  format/version and includes optimizer, early-stopping, graph, preprocessing,
  target, and RNG state. Resuming restores optimizer, Python/NumPy/PyTorch
  RNGs and the training-loader generator; there is no scheduler. The `last`
  checkpoint resumes the latest completed epoch; `best` remains the best
  validation-loss model. Seeded runs are not claimed bitwise-identical across
  devices or nondeterministic kernels.
- Part 5.9 Colab runner:
  [Colab notebook](../notebooks/train_gnn_transformer_colab.ipynb) and
  [training guide](colab_training_guide.md) inspect the active GPU/PyTorch
  runtime before dependency installation, clone a configurable Git revision,
  mount Drive, validate saved train/validation artifacts and graph contracts,
  run an actual-data CUDA smoke update, invoke the existing CLI for full
  training, plot recorded history, verify persistent checkpoints, and reload
  the best checkpoint on a validation batch. The notebook does not install a
  CUDA PyTorch wheel automatically, does not use test records, and stops if
  CUDA is unavailable. The CLI's optional `AEB_GIT_COMMIT_SHA` records a
  validated full commit hash, and experiment metadata records CUDA runtime,
  GPU name, and device count.
- Windows storage optimization:
  [setup script](../scripts/setup_windows_storage.ps1) selects a drive with
  sufficient space (E: first, then F:), activates the existing E:-resident
  Python 3.12 environment, and sets process-scoped cache/temp variables.
  Workspace-relative VS Code settings use the ignored `.env.storage`; the
  forecasting CLI accepts local split, experiment, and checkpoint roots while
  preserving relative defaults. Existing datasets, checkpoints, and the
  repository were not moved or deleted. See
  [Windows storage setup](windows_storage_setup.md).
- Part 5.10 evaluation:
  [evaluation utilities](../src/forecasting/evaluation.py), the
  [evaluation CLI](../scripts/evaluate_forecaster.py), and
  `configs/forecasting_evaluation.yaml` evaluate only the checksum-verified
  test partition. They require a matching completed Part 5.8 experiment
  metadata file and checkpoint, verify scaler/split/checkpoint/graph hashes
  and full train/validation epoch counts, and refuse smoke-test checkpoints.
  Model and persistence, seasonal-persistence, and history-average metrics use
  the same forecast origins, target timestamps, masks, and physical kW units.
  BUS01 is excluded. JSON, compressed aligned predictions, and labeled plots
  are written to a new non-overwriting output directory. The actual held-out
  test data has not been evaluated.

Parts 1–4 completion evidence is in
[Part 2](part2_completion_report.md), [Part 3](part3_completion_report.md),
and [Part 4](part4_completion_report.md). The forecasting package now includes
spatial and temporal encoders, an integrated forecasting model, and a reusable
training/validation pipeline. No full-scale training run or production
checkpoint exists.

## Current task

Part 5.9 local readiness review is complete. The Colab notebook preflight now
requires the complete train/validation/test bundle and verifies that the test
partition is represented in the checksum manifest without loading it. The
local Git repository still has no commits or remote; no files have been
published. Run the Part 5.9 Colab workflow only after reviewing and publishing
an approved source revision and uploading the bundle to Drive. No GPU smoke
test, full training, checkpoint, or final Part 5.10 evaluation has occurred.

## Verified tests

Commands run from the project root for the completed Part 5.6 task:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_transformer_encoder.py tests\test_gnn_encoder.py tests\test_forecasting_dataloader.py tests\test_forecasting_data.py tests\test_data_splitting.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

The focused Transformer, GNN, loader, graph, and splitting suite passed
**131 tests**. The complete suite passed **715 tests** in 117.53 seconds. The
Transformer-specific suite passed **34 tests**, covering shape/length
variation, configuration, sinusoidal positions, padding masks, independent
batch/node attention, finite gradients, and GNN compatibility. Pylance
diagnostics for the Transformer module and its tests reported no issues.

An additional loader smoke test against
`data/processed/splits_hypothetical_campus_v1` verified all checked-in splits:
train has 6,085 windows; validation and test have 1,267 each. A batch has
`X [128, 24, 8, 1]`, `global_X [128, 24, 3]`, and `y [128, 24, 8, 1]`.
The shared graph has `edge_index [2, 14]`, the audited eight-node order, no
disconnected nodes, and no fabricated edge-weight tensor. The targets remain
physical kW and the bus remains masked.

An integration smoke test loaded a four-window batch from the existing Part
5.4 training loader and ran the configured encoder forward and backward.
Input shape was `[4, 24, 8, 1]`, embedding shape `[4, 24, 8, 64]`, and the
encoder has **8,576 trainable parameters**. The scalar embedding loss and all
parameter gradients were finite.

The Part 5.6 GNN-to-Transformer smoke test used an actual two-window training
batch. Shapes were input `[2, 24, 8, 1]`, spatial `[2, 24, 8, 64]`, and
temporal `[2, 24, 8, 64]`. The Transformer has **66,944 trainable parameters**;
the scalar loss and gradients in both encoders were finite.

Part 5.7 tests in
`tests/test_gnn_transformer.py` verify target/node ordering, node and optional
system-level output shapes, configuration and input validation, target masks,
finite gradients through all three model components, independent batches,
checkpoint reconstruction/no-overwrite behavior, and Part 5.4 DataLoader
compatibility. The focused command
`.\.venv\Scripts\python.exe -m pytest tests\test_gnn_transformer.py -q -p no:cacheprovider`
passed **19 tests**.

The Part 5.7 CPU end-to-end smoke test used the real Part 5.4 training loader
and the configured shared-bus graph. Input, target, and prediction shapes were
all `[2, 24, 8, 1]`; the configured model has **77,080 trainable parameters**.
Masked MSE loss was finite, and finite gradients reached the GNN, Transformer,
and forecast head. No full training was run.

After Part 5.7, the complete command
`.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`
passed **734 tests** in 167.65 seconds.

Part 5.8 focused regression command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_forecasting_training.py tests\test_forecasting_dataloader.py tests\test_data_splitting.py tests\test_gnn_transformer.py tests\test_gnn_encoder.py tests\test_transformer_encoder.py tests\test_forecasting_baselines.py -q -p no:cacheprovider
```

It passed **171 tests**. The complete suite command
`.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`
passed **762 tests** in 56.90 seconds. Python syntax checks passed for the new
trainer/CLI and the changed loader/splitting modules.

Final post-resume-duration verification: all **28** focused training tests
passed; the complete suite passed **762 tests** in 54.44 seconds. Pylance
diagnostics for the trainer, CLI, and training tests were empty, and the
trainer had no syntax errors. A fresh two-epoch CPU smoke run reproduced the
finite losses above, saved best and resumable checkpoints plus history and
metadata, and reloaded-checkpoint validation inference returned finite
`[2, 24, 8, 1]` predictions with the BUS01 placeholder zero.

The real CPU smoke command used only two train batches and one validation
batch per epoch:

```powershell
.\.venv\Scripts\python.exe scripts\train_forecaster.py --device cpu --epochs 2 --batch-size 2 --max-train-batches 2 --max-validation-batches 1 --output-dir <temporary-directory> --experiment-id part58-cpu-smoke
```

It completed two epochs. Masked train MSEs were **4535.995722** and
**4631.151228 kW²**; validation MSEs were **7167.064732** and **7126.029018
kW²**. The best epoch was 2. A two-sample validation batch produced
`[2, 24, 8, 1]`; reloaded-checkpoint predictions were finite and the BUS01
placeholder was zero. Experiment metadata included the seven targets, kW
units, graph fingerprint, preprocessing references, and a false
`test_partition_used_for_selection` flag. Metadata-required fields and all
artifact SHA-256 values were checked. Best and latest/resume checkpoints were
saved, and the checkpoint-loaded inference passed. Smoke artifacts were
written under the system temporary directory rather than the project results
tree; no full training was performed.

No dependencies, split/tensor artifacts, simulator code, or Part 4 controllers
were changed; no full-model training was run.

Part 5.2 environment evidence remains: Python 3.12.10 and PyTorch 2.7.0+cpu;
CPU tensor and neural-network forward/backward steps passed, CUDA is
unavailable locally, and a live Colab GPU was not available for verification.

Part 5.9 local verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_colab_notebook.py tests\test_forecasting_training.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

The focused notebook/training tests passed **36 tests**. The complete suite
passed **770 tests** in 51.63 seconds. Notebook JSON parsing and Python AST
checks passed for every executable cell. Pylance diagnostics for the updated
CLI and new notebook tests were empty; the CLI syntax check passed. The
notebook's GPU smoke test and full training were not executed because this
workspace has CPU-only PyTorch and no live Colab runtime was available.
An additional one-epoch, one-batch CPU CLI smoke run verified checkpoint and
history output plus structured CPU/CUDA-availability metadata; it did not
perform full training and its temporary outputs were removed.

Windows storage verification used the selected project interpreter after
dot-sourcing `scripts/setup_windows_storage.ps1`. The full suite
`python -m pytest tests -q -p no:cacheprovider` passed **777 tests** in 65.63
seconds. The verified interpreter was Python 3.12.10 at the existing
repository `.venv`; pip cache, PyTorch hub, Python temporary files and
bytecode cache resolved under `E:\AdaptiveEnergyBrainStorage`, and Matplotlib
resolved under its configured external directory. Hugging Face is not
installed or used. The training CLI help worked in the configured environment.
No training, dataset migration, or cache cleanup was performed.

Latest Part 5.9 readiness verification used the selected repository
interpreter:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_colab_notebook.py -q -p no:cacheprovider
```

The notebook structural and Python-cell syntax checks passed **8 tests**.
The complete suite passed **792 tests** in 94.60 seconds after the preflight
change.
The actual split bundle contains 32 files (7,324,231 bytes), including
train, validation, and test directories; its manifest has 31 checksum
entries, 10 of which identify test-partition files. The test partition was
not loaded. The local Git branch is `master`, has no commits, and has no
configured remote; all project source files remain untracked. Dataset
partitions, `.venv`, `.env.storage`, and generated results are ignored by
Git. C:, E:, and F: were present at the time checked, with about 16.00 GB,
30.87 GB, and 33.92 GB free respectively; the existing `.venv` is on E:.
These are point-in-time local facts, not a Colab/Drive readiness result.

Part 5.10 evaluation verification used the selected repository interpreter:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_forecasting_evaluation.py tests\test_forecasting_baselines.py tests\test_forecasting_dataloader.py tests\test_gnn_transformer.py tests\test_forecasting_training.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

The focused forecasting evaluation/regression command passed **100 tests**;
the complete suite passed **792 tests** in 100.59 seconds. Pylance reported no
diagnostics for the new evaluation module, CLI, or tests, and `compileall`
passed. The CLI help worked. A missing-checkpoint CLI check exited with status
2 and created no output. The synthetic end-to-end fixture verified CPU
inference, matching metadata and artifact hashes, test-only partition loading,
full-partition epoch-count gates, seven-target masking, physical-unit baseline
comparison, and non-overwriting JSON/NPZ/plot output. It does not represent
the held-out project dataset or model performance.

## Known issues and constraints

- Available datasets and representative Part 3–4 experiments are synthetic
  and hypothetical; there are no verified measured campus datasets.
- The held-out test set must remain unused for selection and tuning.
- Existing simulator, four rule-based baselines, and their interfaces are
  compatibility boundaries.
- Only a synthetic hypothetical year is available; `data/raw/` contains no
  source CSVs and real data source configurations remain pending.
- The eight-node, 14-edge bidirectional shared-bus graph is a modeling
  assumption, not measured campus topology or weighted electrical data.
- The integrated model consumes only the node-history tensor `X`; historical
  global weather covariates in `global_X` are not currently encoded. No known
  future covariates are available in the audited dataset. CUDA execution
  remains unverified locally.
- Model checkpoints do not embed the graph edges, dataset splits, or train-fitted
  scalers; pair them with the matching Part 2 preprocessing artifacts.
- No trained checkpoint exists in `models/` or the external checkpoint
  storage, and the external experiment directory is empty. The result tree
  contains earlier Part 3–4 outputs but no Part 5 forecasting evaluation.
  Final held-out results are therefore intentionally absent.
- This checkout has no Git commits or configured GitHub remote, so Colab
  cannot yet check out the required project revision. Publishing is a user
  action and requires review/approval; no commit or push was performed.
- The Drive dataset bundle has not been uploaded. Local presence and checksum
  metadata do not establish Drive persistence.
- Default local execution is CPU-only. Full-data multi-epoch training and
  CUDA/Colab behavior remain unverified. The smoke losses use only the first
  two train batches and first validation batch, so they are not performance
  estimates and must not be used for model selection.
- Resume uses the latest epoch's checkpoint and restores optimizer, early
  stopping, and RNG state, but bitwise reproducibility is not guaranteed on
  nondeterministic hardware/kernels. Accumulated training duration and the
  original run start time are retained when resuming from the matching
  experiment history. No learning-rate scheduler is used.
- The Transformer uses bidirectional attention over historical input only;
  the model uses a direct non-autoregressive head. Initial dimensions are
  examples, not tuned settings. System-level forecasting is implemented as an
  optional path only; the actual dataset supports node-level targets only.
- Existing sequence inputs are standardized, whereas the new baselines and
  metrics expect physical-unit history and targets (normally kW). The loader
  retains standardized inputs and physical kW labels; use the provided
  train-scaler inversion helper only when a consumer explicitly needs physical
  historical values.
- Seasonal persistence requires at least one full seasonal period (24
  observations by default) and raises an explicit error otherwise.
- GitHub Spec-Kit, LLM providers, retrieval, Prefect, n8n, and PydanticAI are
  not installed or configured. See the
  [AI stack integration plan](ai_stack_integration_plan.md).
- Local development uses a CPU-only PyTorch wheel; the separately pinned
  Colab CUDA wheel is not installed automatically. The Colab notebook prefers
  a CUDA-capable PyTorch already present in the selected runtime.
- Exact installed ML runtime: `torch==2.7.0+cpu`; PyTorch reports no bundled
  CUDA runtime. The Colab wheel is separately pinned as `torch==2.7.0+cu128`.
- Do not implement RAG in Part 5, call an LLM from the GNN-Transformer training
  loop, or add API keys.

## Next task

Manually execute and review the Part 5.9 notebook in Colab with a GPU runtime.
Confirm the actual Git commit, GPU/software metadata, masked CUDA smoke test,
full train/validation history, Drive checkpoints, and best-checkpoint
validation reload; then run Part 5.10 with the matching `best.pt`,
`experiment.json`, split bundle, and preprocessing artifact. For local Windows
development, open a fresh VS Code PowerShell terminal so the external storage
profile is applied. Do not use held-out test results for selection or modify
the seven-target/graph contracts. Part 5.11 is blocked until the training and
final evaluation evidence have been reviewed.
