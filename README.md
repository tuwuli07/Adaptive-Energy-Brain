# Adaptive Energy Brain

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)](#license)

Adaptive Energy Brain is a research dashboard for renewable energy forecasting,
battery dispatch, controller evaluation, risk analysis, and adaptive control.
It combines a React/Vite interface with a Python HTTP API and includes bounded
CSV dataset uploads and an optional energy-assistant interface.

> [!IMPORTANT]
> This project is a research prototype. It uses synthetic and simulated data
> unless a user explicitly uploads and selects a dataset. It is not validated
> for real-world equipment control, safety-critical operations, or grid dispatch.

## Features

- Interactive energy monitoring and battery-discharge dashboards
- Forecasting metrics and comparative controller analysis
- CVaR, DRO, and adaptive-policy research results
- Simulation result exports
- Bounded CSV upload, validation, analysis, and column mapping
- Read-only research API with an illustrative-data fallback
- Gemini-powered energy assistant
- Public-demo safeguards for rate limiting and request validation

## Research status

The data pipeline, simulator, rule-based baselines, GNN-Transformer forecasting
pipeline, and selected risk/adaptive experiments are implemented. The GNN-
Transformer forecast did not outperform seasonal persistence overall in the
reported held-out experiment.

Deep reinforcement learning, meta-learning, real-time control, and
safety-critical grid integration should be treated as research areas rather than
production capabilities. See [Development Handoff](docs/development_handoff.md)
for the verified implementation status and current limitations.

## Architecture

```text
React/Vite frontend
        |
        | /api
        v
Public preview gateway
        |
        | localhost proxy
        v
Python research API
        |
        +-- Simulation and controller artifacts
        +-- CSV dataset store
        +-- Gemini-powered assistant
```

### Main directories

| Path | Purpose |
|---|---|
| `frontend/` | React, Vite, and Tailwind user interface |
| `src/dashboard/` | Python API and public-preview gateway |
| `src/data/` | Dataset ingestion, validation, splitting, and analysis |
| `src/simulation/` | Microgrid simulation environment |
| `src/baselines/` | Causal controller baselines |
| `src/forecasting/` | Forecasting models and evaluation pipeline |
| `src/assistant/` | Gemini, RAG, and MCP integrations |
| `src/risk/` | Risk-management components |
| `tests/` | Python and frontend test coverage |
| `configs/` | Data, experiment, and controller configuration |
| `results/` | Generated and repository-tracked research results |

## Prerequisites

- Python 3.12
- Node.js 20 or newer
- npm
- Git
- Windows, Linux, or macOS

Python dependencies are listed in `requirements.txt`. The frontend dependencies
and exact lockfile are in `frontend/package.json` and
`frontend/package-lock.json`.

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/tuwuli07/Adaptive-Energy-Brain.git
cd Adaptive-Energy-Brain
```

### 2. Create a Python environment

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell blocks virtual-environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 3. Install and build the frontend

```bash
cd frontend
npm ci
npm run build
cd ..
```

## Run locally

Open two terminals from the project root.

**Terminal 1 - API**

```bash
python -m src.dashboard.api 8000
```

**Terminal 2 - Public preview**

```bash
python -m src.dashboard.public_preview
```

Open `http://127.0.0.1:8787`.

For frontend development with hot reload, open a third workflow:

```bash
cd frontend
npm run dev
```

Then open `http://127.0.0.1:5173`. Vite proxies `/api` requests to the Python
API on port 8000.

The `python main.py` command prints a project status summary only. It does not
start the dashboard.

## Assistant configuration

The assistant generates every answer with Gemini; there is no Ollama or saved-
answer fallback. See `.env.example` for variable names. Never commit an actual
`.env` file or API key.

Set environment variables on the backend only:

```dotenv
ENERGY_ASSISTANT_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-2.5-flash
```

Gemini calls send the question, bounded conversation history, and selected
dataset summary to Google. The public preview does not require an assistant
access key; anyone who can reach the public site can send Gemini requests and
consume the configured API quota. Apply usage limits in Google AI Studio and
monitor API usage. Never put `GEMINI_API_KEY` in frontend code.

The uploaded CSV is summarized and validated by the backend; raw upload content
is not included in assistant prompts.

## Tests

Run the Python suite:

```bash
python -m pytest tests -q -p no:cacheprovider
```

Run the frontend unit tests:

```bash
cd frontend
npm test
npm run build
```

Run all relevant tests before releasing changes. This repository has no live equipment
dependency, GPU requirement, or external dataset requirement for the existing
test suite.

## Security boundaries

- The public preview exposes only an allowlisted set of API routes.
- Cross-origin POST requests and direct source, asset traversal, and internal
  internal model endpoints are blocked.
- Uploaded CSV files are size and row bounded.
- Raw uploaded CSV content is not passed to the assistant.
- Local secrets and generated data are excluded by `.gitignore`.
- LLM output never executes forecasting, retraining, optimization, or equipment
  control.
- Review legal, privacy, cost, and data-retention requirements before accepting
  real-world data.

## Optional ML setup

The default API and dashboard do not require training dependencies. PyTorch,
GPU, Colab, or data-training configuration is only needed for reproducible
research experiments.

- [ML environment setup](docs/ml_environment_setup.md)
- [Windows storage setup](docs/windows_storage_setup.md)
- [AI stack integration plan](docs/ai_stack_integration_plan.md)

## License

No license is currently specified. Confirm the intended license before
redistributing or commercializing this project.
