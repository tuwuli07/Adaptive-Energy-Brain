# Development Handoff

## Current phase

Phase 13.5 (Complete UI/UX Redesign: Modern Energy Intelligence Platform) is COMPLETE and VERIFIED. Phase 13 (Gemini AI Assistant, RAG, & MCP), Phase 12 (Adaptive Policy Selection), Phase 11 (DRO), Phase 10 (CVaR), Phase 9 (React Frontend), and Part 8 (Unified Comparative Benchmark & Held-Out Test Evaluation) remain authoritative, frozen, and audited.

### 1. Phase 13.5 Completed Outcomes:
- **Design Philosophy ("Make complex energy intelligence feel simple"):**
  - Redesigned the entire frontend experience into a clean, modern, cohesive industrial renewable energy control center.
  - Implemented deep navy `#0B0F19` canvas, translucent glassmorphism cards, and semantic color palette (Solar Amber `#F59E0B`, Wind Cyan `#06B6D4`, Demand Blue `#3B82F6`, Battery Violet `#8B5CF6`, Grid Slate `#64748B`, Clean Savings Emerald `#10B981`).
- **Two-Layer Explanation Architecture:**
  - Layer 1: Human-friendly plain language on all cards and charts (e.g. "Electricity Demand", "Clean Generation", "Battery Charge", "Grid Electricity Needed", "High-Risk Cost Protection", "Uncertainty Protection", "Smart Energy Control").
  - Layer 2: On-demand interactive `InfoTooltip.jsx` and `MetricCard.jsx` popovers providing "What this means", "Why it matters", "How to interpret", and exact mathematical formulations.
- **Interactive Energy Flow Visualization (`frontend/src/components/EnergyFlowDiagram.jsx`):**
  - Visualizes instantaneous power balance across Solar + Wind $\to$ Central Microgrid Hub $\leftrightarrow$ Battery (500 kWh, 150 kW) $\leftrightarrow$ Utility Grid $\to$ Campus Buildings.
  - Interactive scrubber across all 48 trajectory hours with quick presets (*Solar Peak*, *Wind Peak*, *High Tariff*).
  - Strictly enforces physics: non-simultaneous charge/discharge, symmetric $\pm 150\text{ kW}$ inverter limit.
- **Main Command Dashboard (`frontend/src/pages/OverviewDashboard.jsx`):**
  - Hero banner "Your Energy, Smarter." with "Historical Synthetic Simulation Benchmark • Winter Test Partition (1,313 hrs)" badge.
  - 6 prioritized metric cards answering the 5 key questions (Demand, Renewables, Battery SOC, Grid Import, Net Operating Cost, Net Savings vs Baseline).
  - Composed 48-hour area chart and leading strategy snapshot (GNN-MPC capturing 26.3% of theoretical Oracle gap).
- **Subpage Redesigns:**
  - `EnergyMonitoring.jsx`: Clear kW vs. kWh educational distinction, high-resolution telemetry, physical limit callouts.
  - `ForecastingPage.jsx`: Plain-English MAE/RMSE explainer, 24h lookahead, GNN spatial topology + Transformer attention, and transparent *Forecasting vs. Economic Gap* documentation ($5.18 advantage due to 500 kWh battery saturation).
  - `BatteryManagement.jsx`: State of charge gauge with 10%–90% operating window, 150 kW inverter limits, cycle degradation ($0.02/kWh), and arbitrage threshold principle.
  - `ControllerComparison.jsx`: "Which Energy Strategy Works Best?", friendly names with technical subtitles, noncausal oracle ceiling callout, and CSV export.
  - `RiskAnalytics.jsx`: "Energy Risk & Reliability", plain-English CVaR and DRO, peak hourly surge reduction (-30.5% from $164.95 to $114.71/h).
  - `AdaptiveControl.jsx`: "Smart Energy Control", 4-stage visual pipeline (Observe $\to$ Evaluate $\to$ Select $\to$ Act), contextual policy routing, and 72-hour timeline.
  - `AIAssistant.jsx`: "Ask Your Energy AI", 6 recommended questions, clean chat bubbles, grounding citations with RAG chunk relevance and MCP tool tags.
- **Sidebar Reorganization (`frontend/src/components/Sidebar.jsx`):**
  - Grouped into 4 logical sections: OVERVIEW, SMART MANAGEMENT, ANALYSIS, AI & INFORMATION.
- **Verification & QA:**
  - `npm test`: 6/6 formatter unit tests passed.
  - `npm run build`: Clean Vite production bundle in 10.61s (`dist/`).
  - Assistant & Dashboard tests: 15/15 passed.
  - MCP & RAG tests: 14/14 passed.
  - Data Acquisition tests: 29/29 passed unsandboxed.
  - Frozen Phase 8.5 test hash: `305B769D5B98EDB57515010CD59F63B7D58CCB75D5C6AA1BE7864923859758D7` (pristine).
  - Physical limits: 500 kWh storage, strictly 150 kW symmetric inverter limits verified.
  - Git working tree: Clean; zero unauthorized commits, pushes, or deployments.

Next task: Phase 13.5 UI/UX Redesign is COMPLETE. All development milestones are finished. Await project owner authorization before proceeding to submission packaging or final deployment.

### 2. Phase 13 Completed Outcomes:
- **Google Gen AI SDK Integration (`src/assistant/gemini_service.py`):**
  - Built on `google-genai` (3.6.1). Configurable model (`GEMINI_MODEL`, default `gemini-2.5-flash`).
  - Strict key isolation (never exposed to frontend or logged).
  - Bounded retries (exponential backoff on 429/503), 20s timeout, and graceful offline fallback.
- **ChromaDB Vector RAG Pipeline (`src/assistant/rag.py` & `scripts/ingest_rag_docs.py`):**
  - Indexed 275 deterministic chunks across 13 core project documents and completion reports.
  - Zero runtime network dependencies via locally cached ONNX MiniLM embeddings.
  - Relevance-scored semantic retrieval with document and section attribution.
- **Model Context Protocol (MCP) Server (`src/assistant/mcp_server.py`):**
  - Implemented with `mcp` 2.3 SDK exposing 10 strictly read-only tools pulling directly from authoritative data sources (`DATA_SOURCE`).
  - Covers system overview, energy summary, forecast metrics, battery status (500 kWh, 150 kW), 7-controller test comparisons, CVaR results, DRO results, adaptive selection statistics, simulation metadata, and scientific limitations.
- **Assistant Orchestration Engine (`src/assistant/orchestrator.py`):**
  - Unified pipeline: input validation $\to$ MCP structured data access $\to$ RAG context retrieval $\to$ strict grounding prompt assembly $\to$ Gemini inference $\to$ source attribution and provenance delivery.
  - Grounded offline fallback providing structured answers even when `GEMINI_API_KEY` is not configured.
- **Backend REST API (`src/dashboard/api.py`):**
  - Added `POST /api/assistant/chat` and `GET /api/assistant/status`.
  - CORS updated for `GET, POST, OPTIONS`.
- **React AI Energy Assistant (`frontend/src/pages/AIAssistant.jsx`):**
  - Page 11 (`/assistant`) with live status indicators, 6 recommended prompt pills, scrollable chat stream, collapsible source drawers with relevance scores, and clear scientific integrity disclosures.
  - Standalone offline fallback support in `frontend/src/api/client.js` and `frontend/src/api/fallbackData.js`.
- **Test Suite & Verification:**
  - `tests/test_rag.py`: 4/4 passed.
  - `tests/test_mcp_server.py`: 10/10 passed.
  - `tests/test_gemini_assistant.py`: 6/6 passed.
  - `tests/test_dashboard_api.py`: 9/9 passed.
  - Combined Phase 13 tests: 29/29 passed in 6.31s.
  - Full Python regression: 1,096 passed.
  - Frontend unit tests: 5/5 passed.
  - Frontend production build: Clean Vite bundle in 12.89s.
  - Frozen Phase 8.5 test hashes: Pristine (`305B769D...` and `8E12C074...`).
  - Git working tree: Clean; zero commits, zero pushes.

Next task: Phase 13 is COMPLETE. All development milestones are finished ahead of the November 19, 2026 competition submission deadline. Await project owner authorization before proceeding to final deployment or submission packaging.

### 2. Phase 9 Completed Outcomes:
- **Architecture Assessment:** Inspected complete repository structure, GNN forecasting, MPC, PPO, simulator, and comparative runner modules. Identified all authoritative artifacts. Formulated read-only frontend architecture plan in `docs/frontend_architecture_assessment.md`.
- **Modern React (JSX) Application (`frontend/`):**
  - Scaffolded with Vite 6, Tailwind CSS 3, Recharts 2.13, Lucide React, and React Router v6.
  - Implemented 8 dedicated pages:
    1. Overview Dashboard (`/`): Executive KPIs, 48h microgrid power balance, live battery state gauge, savings vs zero-action.
    2. Energy Monitoring (`/energy`): Labeled **Historical Simulation Data**, high-resolution multi-scale telemetry (24h, 7d, 14d, all 1,313h), building demand, solar, wind, imports, exports, and battery dispatch.
    3. AI Energy Forecasting (`/forecasting`): GNN-Transformer model architecture, 24h horizon predictions vs ground truth, masked target loss documentation, and MAE/RMSE breakdown by horizon.
    4. Battery Management (`/battery`): Physical specs (500 kWh, 250 kW, $\eta=95\%$, 10%–90% bounds), degradation rate ($0.02/kWh), FCE, and SOC dispatch curves across any of the 7 controllers.
    5. Controller Benchmark (`/comparison`): 7-controller comparative table with partition switcher (Held-Out Test vs Validation), causality badges, and one-click CSV export.
    6. Simulation Explorer (`/explorer`): Trajectory browser, step-by-step cost breakdown, and raw hourly dispatch downloads (CSV and JSON).
    7. Research Results (`/research`): Formally audited Phase 8.5 outcomes ($36,532.73 GNN-MPC cost, $5.18 advantage, 0.0142% delta, proof of identical 31,540.80 kWh throughput, winter heuristic inversion, and limitations).
    8. About the Project (`/about`): System mission, 4-stage microgrid workflow diagram, MPC vs PPO vs Heuristics comparison, and technology stack reference.
- **Read-Only Python REST API Service (`src/dashboard/api.py`):**
  - Built with Python standard library `http.server.ThreadingHTTPServer` on port 8000 for zero external dependency overhead.
  - Provides 9 endpoints: `/api/health`, `/api/overview`, `/api/energy/timeseries`, `/api/forecast/metrics`, `/api/battery/dispatch`, `/api/controllers`, `/api/controllers/comparison`, `/api/research/results`, `/api/simulations/export`.
  - Supports downsampling, limit, and start parameters for responsive 60 FPS browser charting.
- **Resilient Standalone Fallback:** React API client (`frontend/src/api/client.js`) automatically falls back to bundled Phase 8.5 research data (`frontend/src/api/fallbackData.js`) if the Python backend is stopped, guaranteeing 100% offline functionality.
- **Comprehensive Verification:**
  - Production build: `npm run build` completed cleanly in 50.25s (`dist/assets/index-eodOmOpR.js` 697 kB, `dist/assets/index-B7_hfgH8.css` 28.8 kB).
  - Frontend unit tests: `node --test src/utils/formatters.test.js` passed (5/5 tests in 0.48s).
  - Backend API unit tests: `pytest tests/test_dashboard_api.py` passed (4/4 tests in 0.17s).
  - Full regression test suite: 101/101 tests passed in 160.89s.
  - Data integrity: 24/24 SHA-256 hashes of test evaluation artifacts verified with zero mismatches.
  - Git tree: Clean; zero unauthorized commits or branch mutations.

Next task: Phase 9 is COMPLETE. Await project owner review of the React frontend application and final platform status.

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
  are written to a new non-overwriting output directory. The user-reported
  final test evaluation is summarized in
  [the Part 5 completion report](part5_completion_report.md).
- Part 5.11 integration readiness: [the standalone adapter](../src/forecasting/integration.py)
  validates node order, target masking, physical kW, provenance references,
  hourly targets and issue/cutoff causality, then attaches campus-demand,
  solar and wind `ForecastFeature`s to an existing causal observation.
  Negative predictions are rejected by default; optional projection to zero
  must be explicitly selected and is recorded in provenance. Feature
  `model_id` includes both method and experiment identifiers. It leaves
  simulator dispatch and controller APIs unchanged.

Parts 1–4 completion evidence is in
[Part 2](part2_completion_report.md), [Part 3](part3_completion_report.md),
and [Part 4](part4_completion_report.md). The Part 5.9 run and Part 5.10
evaluation are reported complete; their artifacts are in Google Drive, not
assumed present in this local checkout. This evidence does not establish
real-campus performance or deployment readiness.

## Current task: completed paired reward-scale diagnostic

See the [reward-scale results](ppo_diagnostic_report.md#paired-reward-scale-critic-diagnostic)
and [diagnostic guide](ppo_diagnostic_guide.md#paired-reward-scale-diagnostic).
Exactly two fresh CPU runs completed: control scale 1.0 and treatment scale
0.0025, with gamma 0.99, seed 0, learning rate 3e-4, and 8192 transitions each.
All other PPO settings matched. Paired preflight and both actual runs verified the
same initial policy tensors and all 49 training-window/child-seed draws. Each run
completed 64 rollout updates and 512 optimizer steps; checkpoints at
0/2048/4096/6144/8192 reloaded successfully. No run was repeated or replaced.

Pre-training and post-run focused regression suites each passed 24 tests
(198.11 and 159.44 seconds). The complete offline suite and its final exit status
passed: **910 passed, 0 failures, 0 errors, 0 skips**, in 393.68 seconds
(JUnit duration 393.455 seconds), with exit status 0. Pytest reported one existing
Gymnasium warning that the observation-space Box maximum is infinity.
The saved evidence is `results/verification/part6_4_reward_scale_full.log` and
`results/verification/part6_4_reward_scale_full.xml`.

Training-call wall times were 283.639 seconds (control) and 256.288 seconds
(treatment); 4096/8192 validation took 130.558/474.415 seconds. Orchestration
took 1166.972 seconds. Artifact-only analysis verified all 83 run hashes, all
10 checkpoint reloads, both frozen validation stages, and unchanged policy and
optimizer state during inference. The 169 pre-existing files in the experiments
directory matched their pre-run SHA-256 inventory. Control at 4096 exactly
matched the original pilot and previous gamma-control policy and optimizer.

The treatment's median explained variance over the final eight updates was
0.83183 versus 0.0000303 for control. Mean per-update dollar-equivalent RMSE was
$106.89 for treatment versus $486.57 for control, a 78.03% reduction using a
common denominator of $1514.57. The reference
thresholds (explained variance >=0.2; normalized RMSE reduction >=25%) were both
met as critic diagnostic signals. Control clipped globally on 100% of optimizer
steps; treatment on 90.43%. Median critic/actor pre-clipping norm ratios were
7694.37 and 2.27. Treatment's median actual actor update norm increased to
0.008981 from 0.000384, but this did not produce a better deterministic policy.

At both validation checkpoints, neither PPO policy requested charging; both
requested discharge in all 1313 scored hours and ended at 50 kWh / SOC 0.1. Costs
were $43839.8993/$43835.7713 at 4096 and $43835.7713/$43835.7713 at 8192
(control/treatment). Treatment saturated its discharge request at 150 kW by
8192. All six policies had zero physical violations and unserved energy. The
price-aware baseline ended at 450 kWh for $43674.7778, as a descriptive reference
only. No held-out test partition was accessed, no multi-seed study was run, and
no simulator physics or operating-cost accounting was changed.

Run root: `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_reward_scale_diagnostic_20261114`.
Review root: `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_reward_scale_diagnostic_review_final_20261114`.
The reward-scale artifact-only analysis and review manifest are stored there.

## Previous task: completed paired gamma diagnostic

See [the diagnostic report](ppo_diagnostic_report.md) for complete measurements,
limitations, and recommendations, and [the diagnostic guide](ppo_diagnostic_guide.md)
for the instrumentation contract. The completed original pilot remains below.

New files: `configs/ppo_gamma_diagnostic.yaml`, `src/rl/diagnostics.py`,
`src/rl/diagnostic_experiment.py`, `scripts/run_ppo_diagnostic.py`,
`tests/test_ppo_diagnostics.py`, `docs/ppo_diagnostic_guide.md`, and
`docs/ppo_diagnostic_report.md`. Existing `src/rl/training.py` gained optional
read-only diagnostics and a shared model constructor; the metadata schema permits
`gamma_diagnostic`. Installed SB3 was not edited. All other pre-existing source,
configuration, dependency, and Part 5 work is preserved.

Both runs used seed 0, LR 3e-4, 168-hour training-only windows, and 8192 transitions,
with gamma 0.99/0.999 as the only differing setting. Preflight and actual execution
verified identical initial weights and all 49 window/child-seed draws. Each run
completed 64 rollout updates and 512 optimizer steps. Ten checkpoints at
0/2048/4096/6144/8192 verified and reloaded. The control 4096 policy and optimizer
exactly match original pilot A. All four PPO validations remained frozen.

Training-call wall times were 291.297/237.979 seconds. Overall execution was
721.945 seconds; validation stages took 89.212/88.988 seconds. At 4096 the
control/treatment costs were 43839.8993/43841.1123 USD; at 8192 they were
43835.7713/43837.4853 USD. Every PPO replay ended at SOC 0.1 / 50 kWh (change
-200 kWh), with no charging requests and no physical violations or shortage.
The price-aware reference cost 43674.7778 USD and ended at SOC 0.9 / 450 kWh.
These are descriptive single-seed hypothetical-campus results, not a winner claim.

Critic explained variance remained near zero. Final-rollout value means were
about -16 against targets averaging -401/-456. Median critic gradient norms were
3238/3673 versus actor 0.423/0.417; all updates invoked global clipping. Stochastic
training charged on 41.43%/42.69% of samples, with action-bound clipping on
33.54%/32.85%. This exploratory charging did not become deterministic charging.
Gradient norms observe the actual weighted combined loss. Adam's adaptive scaling
means the clipping measurements alone do not prove causal actor suppression.

### Diagnostic verification and artifact locations

All commands ran from the project root using the existing `.venv`, with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `PYTHONUNBUFFERED=1`.

Pre-training focused command:

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_ppo_diagnostics.py tests/test_ppo_validation.py tests/test_rl_episodes.py tests/test_rl_training.py tests/test_rl_environment.py tests/test_simulation_environment.py tests/test_simulation_state.py tests/test_simulation_metrics.py tests/test_baseline_interface.py tests/test_baseline_runner.py tests/test_forecasting_integration.py -q -p no:cacheprovider --basetemp=results/verification/part6_4_diagnostic_focused --junitxml=results/verification/part6_4_diagnostic_focused.xml
```

Result: **188 passed in 132.27 seconds**, one expected Gymnasium unbounded-space
warning. This includes identical instrumented/uninstrumented tensors, optimizer
state, RNGs, episode accounting, actual bootstrap values, gradient-clipping
identities, failure evidence retention, paired inputs, and partition read guards.

Approved real-data execution (two jobs only):

```powershell
.venv\Scripts\python.exe -B scripts/run_ppo_diagnostic.py --split-root data/processed/splits_hypothetical_campus_v1 --output E:\AdaptiveEnergyBrainStorage\experiments\part6_4_gamma_diagnostic_20261114
```

Post-run regression command:

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_ppo_diagnostics.py tests/test_rl_training.py tests/test_ppo_validation.py -q -p no:cacheprovider --basetemp=results/verification/part6_4_diagnostic_post --junitxml=results/verification/part6_4_diagnostic_post.xml
```

Result: **17 passed in 163.52 seconds**.

Final full-suite command:

```powershell
.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider --basetemp=results/verification/part6_4_diagnostic_full --junitxml=results/verification/part6_4_diagnostic_full.xml
```

The saved full-suite log and JUnit XML are complete; no pytest process remained.
The suite passed **903 tests**, with **0 failures, 0 errors, and 0 skips**, in
**340.85 seconds** (JUnit duration 340.695 seconds). Pytest reported one existing
Gymnasium warning: the observation-space Box maximum is infinity. The completed
successful pytest summary implies exit status **0**; the original shell exit-code
side channel was not separately saved. The suite was not rerun.

```text
results/verification/part6_4_diagnostic_full.log
results/verification/part6_4_diagnostic_full.xml
```

Run artifacts: `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_gamma_diagnostic_20261114`.
The immutable run manifest inventories 81 verified artifact hashes and retains
configuration/source/data/software provenance, statuses, paired-input evidence,
telemetry, checkpoint bundles, and native validation reports/trajectories.
Review artifacts: `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_gamma_diagnostic_review_20261114`.
These contain reproducible artifact-only analysis code, metrics, and two visually
inspected plots. No held-out test files were opened. The previous pilot and review
artifacts remain intact. Logs and JUnit XML are in `results/verification/`.

<a id="current-task-part-64-completed-pilot"></a>

## Part 6.4 original completed pilot

Implemented `src/evaluation/ppo_validation.py`, `src/rl/experiments.py`,
`scripts/run_ppo_experiments.py`, `scripts/validate_ppo.py`,
`configs/ppo_validation.yaml`, and `tests/test_ppo_validation.py`.
Updated RL environment action reuse, partition loading, checkpoint scheduling,
metadata schema, DRL contracts, and [the validation guide](ppo_validation_guide.md).
Training reads only verified train files; validation reads only verified
validation files. The held-out test partition was not evaluated. Frozen PPO
uses the existing observation encoder and shared Gymnasium action conversion.
All controllers use the existing native replay and accounting implementation.

The paired pilot uses seed 0, rates A=3e-4 and B=1e-4, 168-hour sampled training
windows, and otherwise identical CPU PPO settings. Initial policy tensors and
all 25 sampled windows/child seeds were verified identical. Each run completed
4,096 transitions, 32 rollout updates, 128 optimization epochs, and 256 optimizer
steps. Published checkpoints are at 0, 2,048, and 4,096 transitions. Final bundles
reloaded successfully with identical deterministic actions and finite parameters.
Version-1 Part 6.3 smoke artifacts remain readable; continuation still restores
optimizer/RNG/sampler state into fresh episodes without claiming uninterrupted
trajectory equivalence.

| Pilot | Training wall time (s) | Learning time (s) | Validation time (s) |
| --- | ---: | ---: | ---: |
| A | 85.222 | 64.510 | 15.877 |
| B | 73.667 | 65.765 | 17.863 |

The orchestrator recorded 261.808 seconds overall; validation including all six
controllers and reports took 99.537 seconds. Timers exclude interpreter startup.
Both runs and all six evaluations completed; no failed seeds were replaced.

Validation uses 1,313 scored transitions, initial SOC 0.5 and stored energy
250 kWh. The following costs are USD under the configured hypothetical tariff;
this is synthetic campus evidence, not a measured-campus performance claim.

| Controller | Net cost | Final SOC | Final energy (kWh) | Energy change (kWh) | Replay seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO A | 43839.8993 | 0.1 | 50 | -200 | 15.877 |
| PPO B | 43844.5483 | 0.1 | 50 | -200 | 17.863 |
| Zero action | 43866.1713 | 0.5 | 250 | 0 | 20.696 |
| Self consumption | 43835.7713 | 0.1 | 50 | -200 | 16.509 |
| Price aware | 43674.7778 | 0.9 | 450 | +200 | 11.644 |
| Battery preserving | 43850.9713 | 0.3 | 150 | -100 | 10.692 |

All controllers served 438,362.5548 kWh demand, used 112,806.4409 kWh renewable
energy, and had zero shortage, curtailment, or physical violations. Maximum
absolute balance residual was 4.2633e-14. Price aware imported 327,816.9005 kWh,
exported 46.5137 kWh, and incurred 786.1855 USD degradation on 39,309.2729 kWh
throughput. PPO A/B each imported 325,366.1139 kWh, exported zero, and incurred
3.8 USD degradation on 190 kWh throughput. Full physical metrics and trajectories
are retained in validation JSON/CSV artifacts.

Both PPO policies requested discharge at every hour and never recharged. A reached
the SOC floor at scored hour 5; B at hour 14. Mean requested discharge was 37.0791
and 13.1206 kW, respectively. Feasibility adjustments occurred at 1,309 and 1,300
hours; these are safe dispatcher constraints, not physical violations. Frozen
policy and optimizer fingerprints remained unchanged. Low explained variance
(approximately 0.0004/0.0010) and persistent discharge warrant policy/critic review
before longer training; no automatic reward or configuration changes were made.

A costs 4.6490 USD less than B at equal terminal energy in this pilot. Price aware
is the descriptive six-controller Pareto frontier on this one replay. Neither
observation establishes a research winner. One seed cannot estimate training
variability; hourly points and overlapping windows are not independent samples.
The planned primary comparison remains five paired seeds at exactly 32,768
transitions. Tradeoffs between cost and terminal energy imply no unique winner.
Terminal-energy valuation and shortage-aware reliability remain unresolved.

### Verification and provenance

With `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `PYTHONUNBUFFERED=1`, the focused
command was `.venv\Scripts\python.exe -m pytest tests/test_ppo_validation.py
tests/test_rl_episodes.py tests/test_rl_training.py tests/test_rl_environment.py
tests/test_simulation_environment.py tests/test_simulation_state.py
tests/test_simulation_metrics.py tests/test_baseline_interface.py
tests/test_baseline_runner.py tests/test_forecasting_integration.py -q
-p no:cacheprovider --basetemp=results/verification/part6_4_focused_verified
--junitxml=results/verification/part6_4_focused.xml`: **184 passed in 122.80 s**.

The final full command was `.venv\Scripts\python.exe -m pytest tests -q
-p no:cacheprovider --basetemp=results/verification/part6_4_full_final
--junitxml=results/verification/part6_4_full_final.xml`: **899 passed in 296.62 s**.
Both had one existing Gymnasium warning about the intentionally unbounded
observation maximum. Logs/XML are in `results/verification/`. The full suite
also passed before pilot execution (899 tests in 283.77 s).
Final `git diff --check` passed (existing LF/CRLF notices only). All 53 pilot
and five review artifact hashes were rechecked; the post-pilot executable change
is confined to plot layout. Python syntax checks passed.

Pilot command: `.venv\Scripts\python.exe scripts/run_ppo_experiments.py
--split-root data/processed/splits_hypothetical_campus_v1
--output E:\AdaptiveEnergyBrainStorage\experiments\part6_4_pilot_20261113`.

Artifacts: `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_pilot_20261113`.
The immutable pilot has protocol, run statuses, source/config snapshots, data
hashes, software versions, logs, checkpoint bundles, screening, validation JSON,
comparison/hourly CSVs, and diagnostic plots. Its 53 artifact hashes were verified.
A subsequent plot-only layout correction has separate provenance and clearer
plots at `E:\AdaptiveEnergyBrainStorage\experiments\part6_4_pilot_review_20261113`;
`review.json` links parent hashes, paired-run checks, five review artifact hashes,
and compute assumptions. Original pilot artifacts were not rewritten.

Linear extrapolation for ten 32,768-transition runs plus 30 checkpoint validation
replays and one baseline set is approximately 1.6-2.0 hours on this CPU setup.
Reserve 2-3 hours for host/I/O variation and reporting. This is an estimate from
short pilots, not a completed study or guaranteed runtime.

## Part 6.3 implementation

Part 6.3 implements the approved sampling and PPO infrastructure in
`src/rl/episodes.py`, `src/rl/training.py`, and `scripts/train_ppo.py`.
`configs/ppo_training.yaml` defines the CPU smoke settings and
`configs/rl_experiment_metadata.schema.json` defines versioned run metadata.
`requirements-rl-training.txt` pins the optional training dependency.
The bounded synthetic tests are `tests/test_rl_episodes.py` and
`tests/test_rl_training.py`. Usage and recovery semantics appear in
[the PPO guide](ppo_training_guide.md).

The train-selective loader verifies training files and shared preprocessing
metadata. It rejects train manifest entries or source paths escaping their
partition, missing hashes, wrong split labels, and mismatched coverage.
A reset samples uniformly with replacement from valid overlapping 168-hour
training windows, each including one additional completed-history row.
The causal 15-feature adapter, power/price references 1000/1, static-rating
BatteryAction conversion, committed-cost reward, and truncation semantics apply.

PPO uses one CPU environment and one PyTorch thread, MlpPolicy with separate
[64,64] Tanh networks, learning rate 0.0003, 128-step rollouts, batch 64,
four epochs, gamma 0.99, GAE lambda 0.95, clip 0.2, entropy coefficient 0,
value coefficient 0.5, gradient norm 0.5, normalized advantages, and seed 0.
The infrastructure demonstration budget is 512 transitions.

Checkpoints capture completed rollout updates, optimizer moments/counters,
global RNGs, and sampler state. A verified manifest is the atomic bundle commit
marker, followed by atomic latest-pointer publication. Partial directories are
rejected by the loader and retained as diagnostic evidence. An initial step-zero
bundle supports recovery when the first update fails. Continuation verifies
source/config/data/software contracts and begins fresh episodes with restored
optimizer/RNG/sampler state. Its trajectory can differ from uninterrupted
training. Interruptions/failures preserve the last published checkpoint and
record status, traceback, collected steps, and committed-checkpoint steps.

Terminal SOC treatment remains open before comparative claims. Shortage-aware
training requires an approved reliability objective because the unserved-demand
coefficient is zero. This workflow uses error-on-shortage, normal-operation
training data. The demonstration verifies infrastructure behavior.

## Part 6.3 verification

The dependency resolver proposed only Stable-Baselines3 2.7.1, and installation
completed with that addition. The project interpreter's pip check reported no
broken requirements. The exact dependency commands were

```powershell
.\.venv\Scripts\python.exe -B -m pip install --dry-run --no-cache-dir --index-url https://pypi.org/simple stable-baselines3==2.7.1
.\.venv\Scripts\python.exe -B -m pip install --no-cache-dir --index-url https://pypi.org/simple stable-baselines3==2.7.1
.\.venv\Scripts\python.exe -m pip check
```

The final 14 new tests verify train-only read guards through complete training
and continuation, partition boundary safety, seeded overlapping sampling,
32-transition CPU learning, same-seed local parameter reproducibility,
optimizer moments and RNG/sampler restoration, deterministic checkpoint reload,
truncation/final-observation handling, corruption rejection, safe publication,
and interruption during an optimizer update.

The final focused command passed **178 tests** in **88.28 seconds** with one
expected Gymnasium warning about intentionally infinite observation bounds.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rl_episodes.py tests/test_rl_training.py tests/test_rl_environment.py tests/test_simulation_environment.py tests/test_simulation_state.py tests/test_simulation_metrics.py tests/test_baseline_interface.py tests/test_baseline_runner.py tests/test_forecasting_integration.py -q -p no:cacheprovider --basetemp=results/verification/part6_3_focused_verified
```

Tests use fresh ignored project-local temporary directories and run with the
Windows access needed by pytest's private-directory ACLs. Initial development
tests exposed a directory-renaming sharing error during checkpoint publication.
The final implementation publishes a manifest atomically inside an exclusively
created step directory, avoiding that rename. Synthetic continuation advanced
32 to 64 cumulative transitions and four to eight optimizer steps. Initial
continuation bundles preserved saved optimizer moments and global/sampler RNG
state. Failed/interrupted tests retained the preceding valid bundle.

## Part 6.3 final acceptance

Final verification completed on 2026-11-13. The complete offline suite passed
**893 tests** in **368.07 seconds**, with zero failures, errors, or skips.
The single warning is the expected Gymnasium advisory about infinite observation
bounds. The saved XML report independently confirms those counts.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=results/verification/part6_3_full_20261113_final --junitxml=results/verification/part6_3_full_20261113.xml
```

The final run's console output is retained in
`results/verification/part6_3_full_20261113.log` and its JUnit report in
`results/verification/part6_3_full_20261113.xml`. An earlier full-suite session's
final tool output became unavailable. The final rerun supplies durable results.
The approved 512-transition demonstration was retained and verified again.

Source/configuration syntax, documentation links, whitespace, and
`git diff --check` passed. The implementation adds nine files and updates the
three approved DRL/handoff documents. The earlier before/after inventory
confirmed that all other pre-existing files and installed package versions were
preserved, with Stable-Baselines3 2.7.1 the sole package addition. During final
closure, only approved documentation files were edited. All 91 archived
source/config files match the verified smoke-run snapshot, and all 22 recorded
run artifacts retain their hashes. pip check reports no broken requirements.

Part 6.3 acceptance criteria are satisfied for the approved infrastructure
scope. Checkpoint continuation, optimizer moments, RNG/sampler restoration,
train-only access, safe publication, and interruption recovery have synthetic
test coverage. Fresh-episode continuation can diverge from uninterrupted
learning. Terminal SOC and shortage-aware reliability remain unresolved research
decisions. Existing uncommitted work is preserved, and no commit or push was
performed. Part 6.4 and research-scale training await explicit approval.

## Part 6.3 approved training-partition demonstration

The CLI completed successfully with this command

```powershell
.\.venv\Scripts\python.exe scripts/train_ppo.py --split-root data/processed/splits_hypothetical_campus_v1 --output E:\AdaptiveEnergyBrainStorage\experiments\part6_3_ppo_smoke_20261112
```

The loader verified **6,132 training rows** from ten training CSV/metadata files.
The run collected **512 transitions**, completed **four rollout updates** and
**16 epochs**, and performed **32 minibatch optimizer steps**. Four sampled
windows were started, three 168-hour episodes finished, and the final episode
contributed eight transitions. Pipeline runtime was **38.5280 seconds**, including
**13.8840 seconds** in learn calls. Measured CLI wall time, including imports and
final output, was **53.4839 seconds**. The pipeline timer includes data loading,
construction, logging, checkpoint publication, and reload/inference verification,
and is recorded before the final artifact inventory write.

The run directory is
`E:\AdaptiveEnergyBrainStorage\experiments\part6_3_ppo_smoke_20261112`.
Its final bundle is `checkpoints/step_000000512`, alongside step-zero and
128/256/384-step bundles. `experiment.json` records completed status, source
revision/dirty status, source/config snapshots, training provenance, settings,
versions, seeds, runtime, progress, hashes, and verification flags. Additional
artifacts include `resolved_config.json`, `source_manifest.json`,
`source_snapshot.zip`, `episodes.jsonl`, `monitor.csv`, `logs/progress.csv`,
and `latest_checkpoint.json`.

Verification confirmed changed and finite parameters, finite loss metrics,
checkpoint reload, exactly equal deterministic actions after reload, and a
successful inference step through a fixed training window. Independent SHA-256
checks reconciled all **22 recorded artifacts**, the final bundle manifest,
and model/state hashes. The final manifest SHA-256 is
`8baeb22c01ad014e8b10f746940b128ca99e54e802e8b004a41461a04b2dbb04`.
The checkpoint remains in external experiment storage. These checks support
infrastructure acceptance on synthetic data. Terminal SOC, reliability, and
research-scale policy comparisons remain later decisions.

## Part 6.2 verification

The resolver dry run proposed only Gymnasium 1.2.3 and its missing mandatory
Farama-Notifications 0.0.6 dependency. Installation completed with those additions
only; a before/after package inventory verified no existing version changes.
`python -m pip check` using the project interpreter reported no broken requirements.
Stable-Baselines3 was absent at the end of Part 6.2. Exact dependency commands:

```powershell
.\.venv\Scripts\python.exe -B -m pip install --dry-run --no-cache-dir --index-url https://pypi.org/simple gymnasium==1.2.3
.\.venv\Scripts\python.exe -B -m pip install --no-cache-dir --index-url https://pypi.org/simple gymnasium==1.2.3 Farama-Notifications==0.0.6
.\.venv\Scripts\python.exe -m pip check
```

Focused regressions passed **164 tests** in **28.44 seconds**, with one advisory
warning about the intentionally infinite observation-space maximum. This includes
all 51 adapter tests and Gymnasium check_env (render checking disabled because
rendering is not implemented), plus simulator, state, metrics, baseline, and
forecast-adapter regressions:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rl_environment.py tests/test_simulation_environment.py tests/test_simulation_state.py tests/test_simulation_metrics.py tests/test_baseline_interface.py tests/test_baseline_runner.py tests/test_forecasting_integration.py -q -p no:cacheprovider --basetemp=results/verification/part6_2_focused_verified
```

Windows sandbox ACLs prevented access to pytest private temporary directories.
Earlier attempts reported fixture setup errors; the successful command ran
outside that restriction, with a fresh ignored project-local temporary directory.
An earlier test run also identified full dispatch-record serialization as
incompatible with immutable mapping proxies; diagnostics now explicitly select
committed metrics and residuals instead of dumping the entire record.

The complete offline suite passed **879 tests** in **192.36 seconds**, with
the same single advisory warning and no failures or skipped tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=results/verification/part6_2_full_verified
git diff --check
```

The full suite used the same accessible, fresh project-local temp policy as the
successful focused run. git diff --check passed (only existing Windows line-
ending notices appeared). Read-only checks verified syntax of the new Python
files, whitespace/code fences, and all 39 relative links in the three updated
documents. SHA-256 comparison against the pre-edit inventory confirmed only the
three approved documents changed among existing files; the five new files are
exactly the approved adapter, encoder, config, requirements, and tests. Existing
Part 5 files, simulator/baselines/forecasting, and package versions are unchanged.
No source files were removed. Part 6.2 functional acceptance criteria are met.
No held-out project data, training, commit, or push was used for this milestone.

## Part 6.1 verification

Documentation-only checks on 2026-11-12 used the existing interpreter:

```powershell
git diff --check -- docs/development_handoff.md
```

Result: passed, with no whitespace errors in the handoff diff. A read-only
Python check validated relative Markdown link targets in the two new DRL
documents and compared SHA-256 hashes of all 152 pre-existing tracked/untracked
non-ignored files against a pre-edit snapshot. Only the approved handoff changed;
the only added non-ignored files were the two approved DRL documents. Both new
documents also passed a trailing-whitespace check. No files were removed.

No runtime tests were run for this documentation-only milestone. The Part 5.11
126/828-test results below are retained historical evidence, not fresh Part 6.1
verification. No held-out dataset, model checkpoint, or external artifact was
loaded, and no dependency was installed. No commit or push was performed.

## Historical verification record

The following test counts, local readiness findings, and runtime observations
are retained as dated evidence from their respective earlier tasks. They are
not the current status where this handoff gives a later Part 6.2 update.

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
This statement describes that earlier local CPU smoke-test task only; the
separate later Colab training run is summarized in the Part 5 completion report.

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
At that earlier readiness snapshot, the actual split bundle contained 32
files (7,324,231 bytes), including
train, validation, and test directories; its manifest has 31 checksum
entries, 10 of which identify test-partition files. The test partition was
not loaded. That snapshot's local Git branch had no commits or remote and
project files were untracked. The current checkout is connected to
`origin/master`; do not infer current branch cleanliness from that older
observation. Dataset partitions, `.venv`, `.env.storage`, and generated
results are ignored by Git. C:, E:, and F: were present at the time checked,
with about 16.00 GB,
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

## Part 5.11 verification

The adapter, causal observation, baseline interface/runner, and forecasting
evaluation regression command passed **126 tests**:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_forecasting_integration.py tests\test_simulation_state.py tests\test_baseline_interface.py tests\test_simulation_environment.py tests\test_baseline_runner.py tests\test_forecasting_evaluation.py -q -p no:cacheprovider
```

The final complete offline suite passed **828 tests** in **161.17 seconds**:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

No GPU, Google Drive, external service, model retraining, or test-set tuning
was used for Part 5.11 verification.

## Known issues and constraints

- The dataset and all reported forecasting results are synthetic and
  hypothetical; no measured campus dataset or real topology is available.
- The reported final test set is for final evaluation only and must not be
  used for further model selection or tuning.
- The eight-node, 14-edge bidirectional shared-bus graph is an assumed
  modeling topology, not measured electrical infrastructure.
- The forecaster consumes historical node power only; `global_X` weather
  inputs are not encoded, and the data audit contains no known-future
  covariates.
- Training and evaluation outputs are stored in Google Drive, not this local
  repository. Their locations and reported metrics are user-supplied Colab
  run details; the external files were not independently inspected from this
  workspace.
- The model checkpoint must be paired with its matching graph, data split,
  and train-fitted preprocessing artifacts; it does not embed those artifacts.
- Local development remains CPU-only (`torch==2.7.0+cpu`); the supplied
  Colab run reports successful training on an NVIDIA Tesla T4. Local CUDA
  inference is not established by that remote run.
- Seasonal persistence outperformed the GNN-Transformer on the reported
  aggregate held-out metrics. Do not select the model as a universal
  forecasting winner or presume forecast inputs improve DRL control.
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
- GitHub Spec-Kit, LLM providers, retrieval, Prefect, n8n, and PydanticAI
  remain optional and are not required for the completed forecasting workflow.
  See the [AI stack integration plan](ai_stack_integration_plan.md).
- The current PPO policy has infrastructure-smoke evidence. Meta-learning,
  DRO/CVaR, and forecast-serving integration remain later phases. The forecast
  adapter retains its standalone boundary-helper role.

## Part 6.4: Secondary economic metric implementation

Implemented `src/evaluation/inventory_accounting.py` and integrated it into
`src/evaluation/baseline_runner.py` and `src/evaluation/ppo_validation.py`.
- Formula: `reference_value_usd_per_kwh_dc = (p_offpeak + c_degradation) / eta_charge = (0.12 + 0.02) / 0.95 = 0.14736842105263157 USD/kWh DC`.
- Adjustment: `inventory_adjustment_usd = (terminal_energy_kwh - initial_energy_kwh) * reference_value_usd_per_kwh_dc`.
- Metric: `inventory_adjusted_net_operating_cost_usd = raw_net_operating_cost - inventory_adjustment_usd`.
- Guardrails verified:
  1. Primary dimensions (`net_operating_cost`, `terminal_energy_kwh`, `terminal_soc`, `dominates()`) remain identical and untouched.
  2. Training rewards, physical simulator, and episode accounting remain untouched.
  3. No access to the held-out test partition (`data/processed/splits_hypothetical_campus_v1/test`).
  4. Verified against exact historical replay values (e.g. Price-Aware $43,645.304080, Zero-Action $43,866.171334, PPO/Self-Consumption $43,865.245018).
- Verification results:
  - 24 focused evaluation tests passed in 48.03s (`test_inventory_accounting.py`, `test_baseline_runner.py`, `test_ppo_validation.py`).
  - 924 complete offline suite tests passed in 178.72s.
  - All existing historical experiment artifacts in external storage remained pristine with verified SHA-256 hashes.

## Part 6.4: Discrete action adapter implementation and preflight

Implemented `DiscreteActionAdapter` (`gymnasium.spaces.Discrete(3)`) in `src/rl/environment.py`
and integrated discrete model compatibility into `FrozenPPOController` in `src/evaluation/ppo_validation.py`.
- Action Mapping:
  - Action 0 -> -1.0 (requested -150 kW charge, matching `battery.charge_limit_kw`).
  - Action 1 ->  0.0 (requested    0 kW idle).
  - Action 2 -> +1.0 (requested +150 kW discharge, matching `battery.discharge_limit_kw`).
- Guardrails verified:
  1. Base continuous `MicrogridGymEnvironment` remains completely unmodified.
  2. Observation schema (15-dim float32), reward calculation, episode boundaries, and physical simulation constraints remain identical.
  3. Preflight suite passed all 12 tests in `tests/test_discrete_action_adapter.py`.
  4. Full offline suite passed all 936 tests in 173.65s.
  5. No new PPO training was launched; no held-out test data was accessed.
  6. Historical artifacts in external storage remain byte-for-byte identical.

## Part 6.4: Discrete PPO and PBRS diagnostic executions

Executed two paired diagnostic experiments in external storage:
1. Discrete PPO Treatment (`E:\AdaptiveEnergyBrainStorage\experiments\part6_4_discrete_diagnostic_20261115_treatment`):
   - Discrete(3), gamma 0.999, reward scale 0.0025, seed 0, 16,384 steps.
   - Eliminated boundary action clipping (0.0% vs 45% in continuous control).
   - Validation policy collapsed to 100% discharge requests (net cost $43,835.77, terminal SOC 0.10, adjusted cost $43,865.25).
2. Potential-Based Reward Shaping Treatment (`E:\AdaptiveEnergyBrainStorage\experiments\part6_4_pbrs_diagnostic_20261115_treatment`):
   - Mathematically verified policy-invariant shaping: $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ with $\Phi(s) = v_{\text{ref}} \cdot E(s)$.
   - Telescoping sum verified to $< 2.2 \times 10^{-14}$; timeout bootstrapping cancellation verified to $< 1.8 \times 10^{-15}$.
   - Increased exploratory charge steps during training by +33.1% and reduced GAE charge advantage deficit by 25.7% (-$48 to -$35).
   - Deterministic validation policy remained at 100% discharge (net cost $43,835.77, adjusted cost $43,865.25).

## Part 6.4: Observation sufficiency, policy representability, and heuristic validation

1. Information Equivalence Audit:
   - Audited `PriceAwareController._select_action` against `drl_observation_v1` (15 features).
   - Confirmed that baseline decisions depend strictly on `obs[6]` (grid price), `obs[0]` (battery SOC), and static config constants. Zero hidden states or future forecasts are used.
2. Action Discretization Audit:
   - In validation replay, `PriceAwareController` requested intermediate power in 113 intervals (8.61%), exclusively due to boundary headroom/reserve limits.
   - Evaluated deterministic discrete heuristics on 1,313-hour validation split:
     - Continuous `PriceAwareController`: Net cost $43,674.78, adjusted cost $43,645.30.
     - Discretized Price-Aware (Sign): Net cost $43,662.24, adjusted cost $43,632.76 (0 violations, 112 boundary adjustments).
     - Pure Discrete (Price Only): Net cost $43,662.04, adjusted cost $43,632.57 (0 violations, 1,088 adjustments).
   - Confirms that Discrete(3) actions do not degrade economic performance; faster boundary cycling lowers operating cost.
3. Analytical MLP Representability:
   - Proved that the discrete price-aware decision boundary is the intersection of three linear half-spaces in 2D (price, SOC).
   - A 2-layer [64, 64] MLP (5,379 policy parameters) has ample theoretical capacity to represent this boundary.

## Part 6 Formal Closure

Option D was formally approved. Further model-free PPO diagnostics are stopped.
See [Part 6 completion report](part6_completion_report.md) for full synthesis, limitations, and archived artifact locations.

## Phase 10: Conditional Value at Risk (CVaR) Risk-Aware MPC

Implemented multi-scenario Rockafellar-Uryasev (2000) linear programming equivalence in `src/control/cvar_mpc_solver.py`, `src/control/cvar_mpc.py`, and `src/control/scenario_generator.py`:
- Strict nonanticipativity: $P_{\text{chg}, s}[0] = P_{\text{chg}, 0}[0]$ and $P_{\text{dis}, s}[0] = P_{\text{dis}, 0}[0]$.
- Causal rolling history and nominal GNN lookahead scenario synthesis.
- Evaluated across 5 stress conditions; achieved 30.5% reduction in maximum single-hour cost peak under compound stress ($164.95/h to $114.71/h) with $167.37 in adjusted cost savings.
- Formally verified via `tests/test_cvar_math.py`, `tests/test_scenario_generator.py`, and `tests/test_cvar_mpc_controller.py`.

## Phase 11: Distributionally Robust Optimization (DRO) & Project Framing Correction

Implemented distributionally robust optimization over finite-support $L_1$ total-variation probability ambiguity sets $\mathcal{Q} = \{q \ge 0 : \sum_s q_s = 1, \|q - p\|_1 \le \epsilon\}$ via exact LP duality in `src/control/dro_mpc_solver.py`, `src/control/dro_mpc.py`, and `configs/dro_mpc_controller.yaml`:
- Converted inner worst-case expectation into a joint LP solved via HiGHS in ~82.4 ms on CPU.
- Verified strong duality match and analytical 2-scenario thresholds in `tests/test_dro_math.py`.
- Verified nominal recovery at $\epsilon=0$ ($2,669.39 normal, $5,219.65 compound) and physical 150 kW limits in `tests/test_dro_mpc_controller.py`.
- Executed multi-condition benchmarks and ambiguity sweep ($\epsilon \in [0.0, 2.0]$) in `experiments/phase11_dro_experiments/`.
- Integrated DRO into REST API (`src/dashboard/api.py`) and React Risk Analytics (`frontend/src/pages/RiskAnalytics.jsx`).
- Corrected overall project framing: Platform is an autonomous energy management system for renewable communities, microgrids, smart buildings, and commercial facilities; campus is a hypothetical demonstration case study.
- Corrected battery specification to authoritative 150 kW (500 kWh capacity) across all UI cards, API models, and fallbacks.
- Verified 20/20 unit tests, 5/5 frontend tests, Vite production build, and pristine Phase 8.5 held-out test hashes (`305B769D...`, `8E12C074...`).

## Phase 12: Adaptive Policy Selection & Contextual Regime Routing

Implemented causal, context-aware adaptive policy selection among specialized microgrid controllers in `src/control/adaptive_policy_selector.py` and `configs/adaptive_policy_selector.yaml`:
- Resolved Phase 11 DRO duality verification gap: 35 parameterized primal-dual sweep cases passed in `tests/test_dro_math.py` with zero duality gap ($< 10^{-7}$).
- Nominal recovery at $\epsilon=0$ confirmed ($2,669.39 normal, $5,219.65 compound).
- Causal feature extraction: strictly uses observations at decision step $t$ (net load, solar surplus, SOC, current tariff, rolling stress indicator).
- Deterministic regime routing: Solar Self-Consumption (`mpc_gnn_transformer`), Compound Stress / Robust (`mpc_dro_robust`), Peak Tariff Tail-Hedging (`mpc_cvar_balanced`), Nominal Base Dispatch (`mpc_gnn_transformer`).
- Implemented multi-tier fallback cascade (Target Controller $\to$ GNN-MPC $\to$ Seasonal MPC $\to$ Zero-Action) and physical power clamping to $\pm 150.0$ kW.
- Scientifically categorized as Adaptive Policy Selection (Contextual Regime Routing); distinguished from gradient-based meta-learning (MAML/RL²).
- Evaluated on 72-hour validation slice across 5 conditions (`experiments/phase12_adaptive_experiments/`): demonstrated dynamic switching (33.8% DRO, 59.2% GNN, 7.0% CVaR) with 0 physical violations and $< 1.5$ ms feature extraction overhead.
- Extended REST API (`src/dashboard/api.py`): `/api/adaptive/results` and `/api/adaptive/trajectory`.
- Added React Page 10 (`frontend/src/pages/AdaptiveControl.jsx`) with interactive scrubber, composed power chart, regime architecture cards, and comparative matrix.
- Verification: 9 unit tests passed in `tests/test_adaptive_policy_selector.py`, 7 API tests passed, 5/5 frontend unit tests passed, and Vite production build clean (14.87s). Phase 8.5 held-out test hashes remain pristine.

## Phase 13: Gemini AI Energy Assistant, ChromaDB RAG, & MCP

Implemented grounded natural-language assistant with local vector search and read-only tool calling:
- Integrated Google Gen AI SDK (`google-genai` 3.6.1) in `src/assistant/gemini_service.py` with configurable model (`GEMINI_MODEL`, default `gemini-2.5-flash`), 20s timeout, bounded retries on 429/503, and graceful unconfigured/offline fallback.
- Built ChromaDB vector RAG pipeline (`src/assistant/rag.py` & `scripts/ingest_rag_docs.py`), indexing 275 deterministic chunks across 13 core project documents and completion reports. Zero network calls at runtime with cached local ONNX embeddings.
- Developed Model Context Protocol (MCP) server (`src/assistant/mcp_server.py`) with 10 strictly read-only tools pulling directly from authoritative data sources (`DATA_SOURCE`).
- Built Assistant Orchestration Engine (`src/assistant/orchestrator.py`) connecting user queries to RAG context, MCP structured metrics, strict grounding rules, and Gemini synthesis, with grounded offline fallback support.
- Extended REST API (`src/dashboard/api.py`): Added `POST /api/assistant/chat` and `GET /api/assistant/status` with full CORS support.
- Developed React AI Energy Assistant (`frontend/src/pages/AIAssistant.jsx`) as Page 11 (`/assistant`) with live status indicators, 6 quick prompt pills, scrollable chat stream, expandable source drawers with relevance scores, and scientific integrity notices.
- Verification: 29/29 Phase 13 tests passed, 1,096 full regression tests passed, 5/5 frontend unit tests passed, and Vite production build clean (12.89s). Phase 8.5 held-out test hashes remain pristine (`305B769D...` and `8E12C074...`). Zero commits, zero pushes.

## UI presentation refresh - 2026-11-19

Task: make the existing React dashboard more professional and understandable for everyday users.

Completed: campus-inspired green visual treatment, redesigned overview hero, clearer metric hierarchy, simpler navigation and page introductions, mobile icon navigation with accessible labels, focus and reduced-motion styles, and assistant copy without service implementation details. Technical metric subtitles are removed from cards; detailed help remains available. Simulation/demo labels remain explicit. Corrected overview battery percentage, duplicate power units, and energy-flow field compatibility with bundled sample data. Existing local work was preserved.

Verified: `npm run build --prefix frontend` passed; `npm test --prefix frontend` passed (6 tests). These commands required execution outside the sandbox because child process spawning returned EPERM. Production bundle retains Vite's large-chunk warning. Browser visual inspection could not run because the browser access host was unavailable. No backend or Python behavior was changed by this task.

Next bounded task: visually review all routes at desktop and mobile widths when browser access is available; research views retain detailed scientific terminology where needed to interpret results.

## Dataset intake and floating assistant / midnight-blue UI - 2026-11-19

Current task: real CSV descriptive analysis plus UI overhaul, followed by the user's requested floating assistant, fewer navigation entries, and black/blue gradients instead of green.

Completed: isolated upload storage and multipart routes; column mapping and unit confirmation; timestamp-aware analysis; strict uploaded-source selection/gates; compatibility assessment; revised overview and energy monitoring; lazy-loaded pages/charts; midnight-blue styling; seven-entry main sidebar with secondary pages under Study & methods; floating assistant retains chat across minimize/navigation, uses existing assistant service, supports Escape/focus restoration, and explicitly does not analyze uploaded data. The old /assistant URL opens the floating assistant. Existing research URLs remain accessible. No model retraining or algorithm changes were made.

Verification:
- `.\.venv\Scripts\python.exe -m pytest tests/test_user_datasets.py -q -p no:cacheprovider`: 35 passed in 2.32s.
- `.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`: 1,135 passed, 3 Gymnasium warnings in 257.64s.
- `npm test --prefix frontend`: 12 passed.
- `npm run build --prefix frontend`: passed; main bundle 236.70 kB, shared chart bundle 389.86 kB; no large-chunk warning in final build.
- Live multipart upload through the Vite proxy, mapping, and dataset analysis: passed; uploaded sample returned 35.0 kWh; malformed CSV returned HTTP 400. Frontend and health checks returned HTTP 200 at 127.0.0.1:5173. An older backend process was replaced with the updated backend.

Known limitations: browser interaction and desktop/mobile visual acceptance remain unverified after Computer Use could not safely identify the browser URL. The original full redesign of each advanced research page remains partial; their presentation colors/navigation changed, but scientific charts/details remain. Temporary uploads are not durable; deployment needs configured storage and authentication/tenant isolation before public multi-user use. See `docs/ui_redesign_and_dataset_upload_report.md` for full architecture, evidence, and deployment constraints.

Next bounded task: restore browser verification, exercise upload/activation and floating assistant at desktop/mobile widths, and finish advanced research-page visual restructuring against the original brief. Do not claim the original browser acceptance criteria have passed.

## Generalized workspace and local Ollama assistant - 2026-11-19

User direction: make the product site-independent, remove Campus Demo, and run the assistant with local Ollama too.

Completed: removed the campus source option/default demo overview, migrated old campus selections to no selected data, added a generalized data-first empty workspace, retained clearly labeled research references, and kept uploaded analyses scoped to their own records. The floating assistant defaults to local Ollama, lists installed models, accepts an explicit Gemini cloud selection, and includes the active dataset summary. Raw CSVs are not passed to the model. Research RAG/MCP context is opt-in. No automatic model download, cloud fallback, retraining, or equipment control is performed.

Implementation: `src/assistant/ollama_service.py`, `src/assistant/energy_assistant.py`, assistant status/chat API routing, floating provider/model controls, source policy/context, generalized copy, `.env.example`, README, and `tests/test_ollama_assistant.py`. Loopback-only Ollama HTTP uses the official chat/tags APIs. CPU (`OLLAMA_NUM_GPU=0`) is the default; -1 allows automatic GPU offload. Initial GPU startup timed out on this machine; CPU succeeded.

Verified: live Ollama `llama3.2:3b` returned a real small-business energy answer through the Vite proxy and assistant API. A numeric live check caught incorrect local-model arithmetic; direct total/peak/average measurement questions now use deterministic server answers. Final live upload/mapping/chat check returned exactly 35.000 kWh for the sample and labeled it as a verified calculation. Frontend and assistant status returned HTTP 200, with local Ollama available. Browser visual interaction remains unverified due to the earlier URL-verification block.

Tests: initial combined local-model/Gemini/upload checks 56 passed; final focused `tests/test_ollama_assistant.py` 16 passed in 6.82s. `npm test --prefix frontend`: 12 passed. `npm run build --prefix frontend`: passed, 2,022 modules, main bundle 233.83 kB and chart bundle 389.86 kB. Full regression command `.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`: 1,150 passed, 3 Gymnasium warnings in 428.88s. This run began before the final exact-answer safeguard; the final 16-test focused run covers that subsequent change.

Known limits: model-written general explanations can be wrong; only direct recognized measured totals/peak/average questions use the deterministic calculation path. Ollama must run on the backend machine, not automatically on a hosted site's visitor device. Existing trained forecasting topology and preprocessing are unchanged and are not generalized to arbitrary uploads. Temporary upload storage and browser acceptance constraints remain as documented in `docs/ui_redesign_and_dataset_upload_report.md`.
