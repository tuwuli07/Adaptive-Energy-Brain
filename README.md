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
rule-based controller baselines. Their short synthetic experiments are smoke
evidence, not real-campus or deployment performance claims. Forecasting and
the other advanced research objectives remain future work.

## Research architecture and status

The planned research workflow is:

1. **Data preparation (Parts 1–2 complete):** validate and prepare datasets,
   chronological splits, graph structure, and forecasting sequences.
2. **Energy simulation (Part 3 complete):** run the configurable single-bus
   research simulator.
3. **Rule-based baselines (Part 4 complete):** compare four causal controller
   strategies using shared episode and evaluation interfaces.
4. **GNN-Transformer Forecasting (Part 5, not started):** model spatial
   relationships and temporal patterns in renewable generation and demand.
5. **Deep Reinforcement Learning:** train and compare multiple control policies.
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
  scikit-learn, PyYAML, and pytest.
- **Development tools:** Windows, PowerShell, VS Code, and Git.
- **Deferred candidates:** PyTorch, PyTorch Geometric, Stable-Baselines3,
  Gymnasium, CVXPY, Streamlit, and RAG tooling. These are not installed or
  integrated as part of the current setup.

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

The entry point prints a status overview of implemented, current, and planned
research components. It does not run simulations, train models, or start a
dashboard.

## Current development status

Parts 1–4 are complete: data preparation, the research simulator, and four
rule-based controller baselines with a shared evaluation workflow. Part 5
GNN-Transformer forecasting has not started. No reinforcement-learning policy,
meta-learning selector, risk optimizer, dashboard, or RAG assistant is
implemented. Parts 1–4 use synthetic/hypothetical data and do not establish
real-campus performance. See [the development handoff](docs/development_handoff.md)
and [Part 4 completion report](docs/part4_completion_report.md).

Run the complete test suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

This command disables pytest's optional cache. The existing suite requires no
network access, datasets, GPUs, or model training.

Generated datasets in `data/raw/` and `data/processed/`, model artifacts in
`models/`, experiment outputs in `results/`, common checkpoint formats, caches,
and local secrets are excluded from Git. Keep source code and configuration
outside generated-output directories. Git is initialized locally; no commit,
GitHub repository, or push is included in this setup.

## Development practices and optional integrations

Follow [AGENTS.md](AGENTS.md) for Copilot and Codex instructions. Selected
external providers, retrieval, orchestration, and agent frameworks are not
installed or required; see the [AI stack integration plan](docs/ai_stack_integration_plan.md).
PyTorch is an optional ML dependency; see
[ML environment setup](docs/ml_environment_setup.md) for local CPU and Colab GPU instructions.
