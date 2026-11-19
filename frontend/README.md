# Adaptive Energy Brain — React Frontend & UX Platform

A modern, responsive, professional web frontend for the **Adaptive Energy Brain** AI microgrid research project, built with **React (JSX)**, **Vite**, **Tailwind CSS**, and **Recharts**.

The interface transforms research-oriented evaluation artifacts, GNN-Transformer neural forecasts, battery storage telemetry, and controller comparative benchmarks into an interactive, intuitive energy analytics dashboard.

---

## 1. Quickstart Guide (Windows PowerShell)

### Prerequisites
- **Node.js**: v18+ or v22+ (`node -v`)
- **Python**: 3.10+ / 3.12 with project `.venv`

### Step 1: Start the Read-Only Python Backend API
In the project root directory, run:
```powershell
.\.venv\Scripts\python.exe src/dashboard/api.py
```
The REST API server will launch at: `http://localhost:8000/`  
Endpoints available:
- `GET /api/health` — System status and test evaluation presence.
- `GET /api/overview` — Microgrid KPIs, total energy flows, and best causal controller savings.
- `GET /api/energy/timeseries` — High-resolution power flows with downsampling.
- `GET /api/forecast/metrics` — GNN-Transformer MAE/RMSE by horizon (1h to 24h).
- `GET /api/battery/dispatch` — Battery SOC, power, throughput, and degradation.
- `GET /api/controllers/comparison` — Cross-controller comparative benchmark.
- `GET /api/research/results` — Audited Phase 8.5 scientific findings.
- `GET /api/simulations/export` — Downloadable hourly dispatch CSV/JSON.

*(Note: If the backend is not running, the frontend automatically activates its pre-bundled standalone research cache, ensuring full functionality offline!)*

### Step 2: Start the React Development Server
In a separate PowerShell terminal, navigate to the `frontend` folder:
```powershell
cd frontend
npm install
npm run dev
```
Open your browser to: **`http://localhost:5173/`**

### Step 3: Production Build & Preview
To verify the production build:
```powershell
cd frontend
npm run build
npm run preview
```

---

## 2. Application Architecture & Pages

The application is structured into 8 dedicated pages:

| Page | Route | Description & Key Metrics |
| :--- | :--- | :--- |
| **Overview Dashboard** | `/` | Executive microgrid KPIs, 48h power balance chart, live battery gauge, savings vs zero-action, best causal controller badge. |
| **Energy Monitoring** | `/energy` | Labeled **Historical Simulation Data**. High-resolution telemetry for campus demand, PV, wind, grid imports/exports, and battery dispatch across presets (24h, 7d, 14d, all 1,313h). |
| **AI Energy Forecasting** | `/forecasting` | Spatial-Temporal GNN-Transformer architecture overview, 24-hour lead-time predictions vs ground truth, masked target loss documentation, and MAE/RMSE breakdown by horizon. |
| **Battery Management** | `/battery` | Physical battery specs (500 kWh, 250 kW, $\eta=95\%$, 10%–90% SOC limits), degradation analytics ($0.02/kWh), FCE, and SOC dispatch curves across any of the 7 controllers. |
| **Controller Benchmark** | `/comparison` | Complete comparative evaluation across all 7 controllers with partition toggle (Held-Out Test vs Validation), sortable metrics table, causality badges, and CSV export. |
| **Simulation Explorer** | `/explorer` | In-depth trajectory inspection, step-by-step cost breakdown, and direct one-click download of hourly dispatch records in CSV and JSON formats. |
| **Research Results** | `/research` | Scientific synthesis of the Phase 8.5 held-out audit: GNN-MPC $5.18 advantage (0.0142% of cost), proof of identical throughput (31,540.80 kWh), winter heuristic inversion analysis, and PPO conservatism. |
| **About the Project** | `/about` | Non-technical project guide, microgrid workflow diagram, MPC vs PPO vs Rule-Based comparison, technology stack details, and reproducibility guidelines. |

---

## 3. Metric-to-Data-Source Mapping

All values displayed in the frontend originate directly from verified Phase 8.5 and Part 7 evaluation files:

| Dashboard Metric | Authoritative Data Source | Phase / Partition |
| :--- | :--- | :--- |
| GNN-MPC Adjusted Cost: **$36,532.73** | `results/part8_comparative_benchmark/test_evaluation/test_benchmark_results.json` | Phase 8.5 (Test) |
| Seasonal MPC Adjusted Cost: **$36,537.91** | `results/part8_comparative_benchmark/test_evaluation/test_benchmark_results.json` | Phase 8.5 (Test) |
| Zero-Action Baseline Cost: **$36,610.65** | `results/part8_comparative_benchmark/test_evaluation/test_benchmark_results.json` | Phase 8.5 (Test) |
| GNN-MPC Savings: **+$77.92** | Recomputed: $\$36,610.65 - \$36,532.73$ | Phase 8.5 (Test) |
| GNN-MPC vs Seasonal MPC: **+$5.18** | Recomputed: $\$36,537.91 - \$36,532.73$ (0.0142% delta) | Phase 8.5 (Test) |
| Battery Throughput: **31,540.80 kWh** | `results/part8_comparative_benchmark/test_evaluation/<id>/summary.json` | Phase 8.5 (Test) |
| Battery Degradation: **$630.82** | `results/part8_comparative_benchmark/test_evaluation/<id>/summary.json` | Phase 8.5 (Test) |
| Validation Benchmark Costs | `results/part8_comparative_benchmark/benchmark_table.json` | Phase 8.1 (Val) |
| GNN Forecasting MAE (22.4 kW) | `results/part7_4_gnn_mpc/gnn_mpc_validation_summary.json` | Part 7.4 (Val) |
| Controller Metadata & Causality | `configs/controller_registry.json` | Phase 8.1 Registry |

---

## 4. Key Design & Scientific Features
- **Strict Read-Only Guarantee:** The application only performs `GET` requests against existing results. It never reruns evaluations, touches held-out partitions, or modifies model checkpoints.
- **Causality Classification:** Strictly separates deployable controllers (`Causal / Deployable`) from the noncausal rolling oracle (`Noncausal Diagnostic`).
- **Resilient Offline Architecture:** If the Python REST API server is stopped, the frontend seamlessly switches to its built-in fallback research cache without UI disruption.
- **Responsive & Accessible:** Dark and light mode toggle, collapsible navigation sidebar, accessible colors, and responsive layouts across desktop, tablet, and mobile.
