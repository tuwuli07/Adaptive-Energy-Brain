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
- Optional local Ollama or Gemini assistant
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
        +-- Optional Ollama or Gemini assistant
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
| `src/assistant/` | Ollama, Gemini, RAG, and MCP integrations |
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

The default assistant provider is local Ollama. See `.env.example` for variable
names. Never commit an actual `.env` file or API key.

**Local Ollama**

1. Install and start [Ollama](https://ollama.com/).
2. Pull a model:

   ```bash
   ollama pull llama3.2:3b
   ```

3. Configure:

   ```dotenv
   ENERGY_ASSISTANT_PROVIDER=ollama
   OLLAMA_BASE_URL=http://127.0.0.1:11434
   OLLAMA_MODEL=llama3.2:3b
   OLLAMA_NUM_GPU=0
   ```

**Gemini**

Set environment variables on the backend only:

```dotenv
ENERGY_ASSISTANT_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-2.5-flash
```

Gemini calls send the question, bounded conversation history, and selected
dataset summary to Google. The public preview disables Gemini and only permits
its explicitly bounded local-model flow.

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

Run all relevant tests before deployment. This repository has no live equipment
dependency, GPU requirement, or external dataset requirement for the existing
test suite.

## Deploy to Render

The repository includes a Render Blueprint in `render.yaml`. It publishes the
built React interface and Python API as one web service so the browser uses the
same origin.

### One-time deployment

1. Push the repository to GitHub.
2. Open the [Render Dashboard](https://dashboard.render.com/).
3. Select **New > Blueprint**.
4. Connect the GitHub repository `tuwuli07/Adaptive-Energy-Brain`.
5. Render will detect `render.yaml`.
6. Review and create the service.
7. Wait for the build to finish, then open the generated `*.onrender.com` URL.

The Blueprint performs this build:

```bash
cd frontend
npm ci
npm run build
cd ..
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The start command launches the Python API and the public preview gateway:

```bash
python scripts/start_render.py
```

### Manual Render configuration

If the Blueprint is not used, create a Python web service with:

| Setting | Value |
|---|---|
| Root directory | repository root |
| Runtime | Python 3.12 |
| Build command | `cd frontend && npm ci && npm run build && cd .. && pip install -r requirements.txt` |
| Start command | `python scripts/start_render.py` |
| Health-check path | `/api/health` |

Do not set `OLLAMA_BASE_URL` to `127.0.0.1` unless Ollama is running in the same
Render environment. Ollama is not included in the Render Blueprint. For hosted
AI use, configure a reachable remote provider or deploy a separate hosted
Ollama-compatible service and set `OLLAMA_BASE_URL`.

### Environment variables

For a public demo, no secrets are required. The API serves built-in or uploaded
research data and keeps the public preview assistant restricted to Ollama.

Optional variables:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini access, used only when explicitly enabled |
| `GEMINI_MODEL` | Gemini model name |
| `OLLAMA_BASE_URL` | Reachable Ollama-compatible API |
| `OLLAMA_MODEL` | Installed model name |
| `ENERGY_DATASET_STORAGE` | Persistent mounted directory for uploaded datasets |

Store secrets only in Render environment settings. Uploaded datasets should use
a Render disk for persistence; the free ephemeral filesystem will not retain
them across restarts or redeploys.

## Security and deployment boundaries

- The public preview exposes only an allowlisted set of API routes.
- Cross-origin POST requests and direct source, asset traversal, and internal
  Ollama endpoints are blocked.
- Uploaded CSV files are size and row bounded.
- Raw uploaded CSV content is not passed to the assistant.
- Local secrets and generated data are excluded by `.gitignore`.
- LLM output never executes forecasting, retraining, optimization, or equipment
  control.
- A hosted public deployment must be reviewed separately for legal, privacy,
  cost, and data-retention requirements before accepting real-world data.

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
