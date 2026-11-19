# Adaptive Energy Brain

## Project overview

Adaptive Energy Brain is an AI-powered renewable energy management research
project. It proposes combining energy forecasting, adaptive control, policy
selection, and risk optimization to study energy management under uncertainty.

## Research objectives

- Forecast renewable generation and energy demand across connected energy assets.
- Compare multiple reinforcement learning policies for energy management.
- Investigate meta-learning for selecting policies across operating conditions.
- Evaluate distributionally robust optimization (DRO) and conditional value at
  risk (CVaR) methods for managing uncertain outcomes.
- Assess forecasting accuracy, control performance, and risk using reproducible
  experiments and appropriate baselines.

Parts 1–4 provide synthetic-data preparation, a research simulator, and
rule-based controller baselines. Parts 5.1–5.10 have implemented, trained, and
evaluated a GNN-Transformer forecasting research pipeline on a synthetic
hypothetical campus dataset. The forecast model did not beat seasonal
persistence overall. Neither these experiments nor the simulator establish
real-campus performance or operational readiness.

## Research architecture and status

The planned research workflow is:

1. **Data preparation (Parts 1–2 complete):** validate and prepare datasets,
   chronological splits, graph structure, and forecasting sequences.
2. **Energy simulation (Part 3 complete):** run the configurable single-bus
   research simulator.
3. **Rule-based baselines (Part 4 complete):** compare four causal controller
   strategies using shared episode and evaluation interfaces.
4. **GNN-Transformer Forecasting (Parts 5.1–5.10 complete):** train and
   evaluate node-level demand and renewable forecasts; results and integration
   limits are summarized in the
   [Part 5 completion report](docs/part5_completion_report.md).
5. **Deep Reinforcement Learning (next planned phase):** specify and compare
   control policies using the existing simulator; not implemented.
6. **Meta-Learning Policy Selector:** investigate policy selection as conditions
   change.
7. **DRO/CVaR Risk Optimization:** investigate robustness and tail-risk control.
8. **Evaluation and Dashboard:** compare experiments and present their results.
9. **Optional RAG Assistant:** explore retrieval-based access to project knowledge.

See the [development handoff](docs/development_handoff.md) for current status
and the [AI stack integration plan](docs/ai_stack_integration_plan.md) for
optional tooling. Do not treat planned components as implemented.

## Technology stack

- **Configured:** Python 3.12, a local `.venv`, NumPy, pandas, Matplotlib,
  scikit-learn, PyYAML, pytest, and the optional PyTorch ML environment
  (CPU locally; a separate Colab GPU run was reported).
- **Development tools:** Windows, PowerShell, VS Code, and Git.
- **Deferred candidates:** PyTorch Geometric, Stable-Baselines3, Gymnasium,
  CVXPY, Streamlit, and RAG tooling. These are not required by the current
  forecasting or simulator workflows.

## Project directory structure

```text
Adaptive-Energy-Brain/
|-- configs/
|-- dashboard/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|-- results/
|-- src/
|   |-- __init__.py
|   |-- simulation/
|   |-- forecasting/
|   |-- rl/
|   |-- meta_learning/
|   |-- risk/
|   |-- evaluation/
|   `-- rag/
|-- tests/
|-- .gitignore
|-- main.py
|-- README.md
`-- requirements.txt
```

Python packages under `src/` include `__init__.py`. Empty scaffold directories
contain `.gitkeep` files so Git can preserve their structure. The local `.venv/`
is excluded from version control.

- `data/raw/`, `data/processed/`: input datasets and prepared data.
- `configs/`: experiment and application configuration.
- `src/`: Python packages for simulation, forecasting, reinforcement learning,
  meta-learning, risk optimization, evaluation, and retrieval-augmented generation.
- `dashboard/`: dashboard code and assets.
- `notebooks/`: exploratory research notebooks.
- `tests/`: automated tests.
- `models/`: saved model artifacts.
- `results/`: experiment outputs.
- `main.py`: application overview entry point; prints component status.

## Installation on Windows

Install Python 3.12 and Git first. In PowerShell, navigate to the project root
(the directory containing this README and `main.py`), then run:

```powershell
# Create the environment only if .venv does not already exist.
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
python -m pip check
```

The Python version should be 3.12.x. If script execution blocks activation, allow
it for the current PowerShell session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Activation is optional when using the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Requirements currently list the six foundational packages without version pins.
The Part 5 experiment metadata contract is defined in
`configs/experiment_metadata.schema.json`; pinning/locking the training
environment remains necessary before comparable model experiments.

## Run the application

With `.venv` activated, run:

```powershell
python main.py
```

Or without activation:

```powershell
.\.venv\Scripts\python.exe main.py
```

The entry point prints a status overview of implemented and planned research
components. It does not run simulations, train models, or start a dashboard.

## Current development status

Parts 1–4 and the Part 5.1–5.10 forecasting pipeline are complete. The reported
Part 5 Colab experiment used a Tesla T4 and a held-out synthetic test partition;
its aggregate results and caveats are recorded in the
[Part 5 completion report](docs/part5_completion_report.md). DRL is the next
planned research phase. No meta-learning selector, DRO/CVaR optimizer,
dashboard, or RAG assistant is implemented. No result establishes real-campus
performance. See the [development handoff](docs/development_handoff.md) and
[Part 4 completion report](docs/part4_completion_report.md).

Run the complete test suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

This command disables pytest's optional cache. The existing suite requires no
network access, datasets, GPUs, or model training.

Generated datasets in `data/raw/` and `data/processed/`, model artifacts in
`models/`, experiment outputs in `results/`, common checkpoint formats, caches,
and local secrets are excluded from Git. Keep source code and configuration
outside generated-output directories. This checkout has a configured GitHub
remote. Training checkpoints and evaluation artifacts are stored in Google
Drive and are not assumed to be present in this local checkout.

## Development practices and optional integrations

Follow [AGENTS.md](AGENTS.md) for Copilot and Codex instructions. Selected
external providers, retrieval, orchestration, and agent frameworks are not
installed or required; see the [AI stack integration plan](docs/ai_stack_integration_plan.md).
PyTorch is an optional ML dependency; see
[ML environment setup](docs/ml_environment_setup.md) for local CPU and Colab GPU instructions.

## General energy workspace and local assistant

The main workspace supports uploaded site data for homes, buildings, businesses,
renewable communities, and microgrids. There is no default campus demo or
substituted sample dashboard. Published experiments remain separately labeled
research references; they do not establish performance for an uploaded site.

The floating energy assistant defaults to local Ollama. Start Ollama and select
an installed model in the assistant. This checkout's local verification used
`llama3.2:3b`; other installed chat models can be selected. No model is pulled
automatically. If starting from a new machine, install Ollama and run
`ollama pull llama3.2:3b` yourself, then restart/refresh the assistant.

Server configuration (see `.env.example`):

```dotenv
ENERGY_ASSISTANT_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_NUM_GPU=0
```

CPU inference is the default because GPU startup timed out during local
verification; `OLLAMA_NUM_GPU=-1` opts into Ollama's automatic GPU offload.
The adapter uses the official [Ollama chat API](https://docs.ollama.com/api/chat) and [installed-model listing API](https://docs.ollama.com/api/tags).
The local endpoint is restricted to loopback HTTP and ignores system HTTP
proxies. The model receives the question, limited conversation history, and
only the selected dataset's validated summary when present. Full raw CSVs are
not passed to the assistant. Local-model errors are shown explicitly and never
fall back to Gemini or canned project answers.

Gemini remains available as an explicit cloud option when `GEMINI_API_KEY` is
configured on the backend. Selecting it sends the question, conversation
history, and selected dataset summary to Gemini; the UI discloses this.
Published research context (existing RAG and read-only MCP tools) is included
only when the user enables **Include published research references**.

The backend and Ollama run on the server computer. In a hosted deployment,
"local Ollama" means that server, not each visitor's computer; the host must
provide its own running model service. LLM answers do not run forecasting,
retraining, optimization, or equipment control.
