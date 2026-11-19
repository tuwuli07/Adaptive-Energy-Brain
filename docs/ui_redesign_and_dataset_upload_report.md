# UI redesign and dataset upload report

Date: 2026-11-19

## Generalized workspace and Ollama follow-up (2026-11-19)

This follow-up supersedes the campus-demo behavior described in the earlier sections. The workspace now starts with no selected site data. The Campus Demo selector and demonstration overview are removed. Existing stored campus selections migrate to an empty workspace. Homes, buildings, businesses, renewable communities, and microgrids use their own uploaded measurements. Published experiments remain behind a Research reference notice and are never represented as the selected site's results.

Added `src/assistant/ollama_service.py` (loopback-only local adapter), `src/assistant/energy_assistant.py` (explicit provider selection and dataset-summary grounding), and `tests/test_ollama_assistant.py`. Updated assistant status/chat endpoints, the floating model/provider controls, source policy, empty states, `.env.example`, and README setup guidance. Ollama defaults to installed `llama3.2:3b` with CPU inference (`OLLAMA_NUM_GPU=0`); GPU auto-offload can be configured with -1. No models are downloaded automatically and no cloud fallback occurs. Gemini remains an explicit cloud option with a disclosure. The active dataset summary is included on chat requests; raw CSV data is not sent. Existing RAG/MCP references are included only on request through the research-reference checkbox.

Live verification returned a real Ollama response through the frontend proxy and assistant API for a general small-business energy question. Initial GPU startup timed out; CPU inference succeeded. A separate live numeric-grounding check exposed an arithmetic error in the 3B model. Direct total-consumption, peak-demand, and average-demand questions are now answered by validated server calculations, labeled Verified calculation, rather than by LLM arithmetic. General model-generated explanations are not guaranteed correct and do not control equipment.

Targeted checks: 56 passed for the initial combined Ollama/Gemini/upload tests; final Ollama-specific tests 16 passed in 6.82s after adding the exact-answer safeguard. Frontend tests: 12 passed. Final production build passed (main bundle 233.83 kB, chart bundle 389.86 kB). Browser visual verification remains unperformed because of the previously documented Computer Use URL-verification block. Full regression: 1,150 passed, 3 Gymnasium warnings in 428.88s; the run began before the last exact-answer safeguard, which passed its subsequent 16-test focused run. The final live upload/mapping/chat check returned the correct 35.000 kWh deterministic total. See the development handoff for commands and details.

## Earlier implementation status and design history

The CSV descriptive-analysis workflow and the revised overview are implemented. The follow-up request replaces the green theme with midnight black / blue gradients, moves the energy assistant to a persistent floating window, and reduces the main sidebar to seven entries. Detailed comparisons, risk, scenario exploration, and project information are grouped under Study & methods; existing URLs remain accessible. This report does not claim that the original full multi-page overhaul or browser acceptance criteria are complete.

## Files changed by this work

- `src/data/user_datasets.py`: isolated CSV validation, storage adapter, mapping, compatibility assessment, and timestamp-aware descriptive analysis.
- `src/dashboard/api.py`: bounded multipart upload, dataset listing/detail/analysis/mapping routes; local-only default bind.
- `tests/test_user_datasets.py`: 35 validation, numerical, storage, HTTP, and artifact-preservation tests.
- `frontend/src/api/datasets.js`: strict dataset requests without research fallback; real upload progress.
- `frontend/src/api/client.js`: research/sample provenance labels.
- `frontend/src/context/DataSourceContext.jsx`, `utils/dataPolicy.js`, `utils/energyAnalysis.js`: source selection, compatibility gates, and interval integration.
- `frontend/src/pages/MyEnergyData.jsx`: drag/drop, browse, preview, mapping, units, quality, and activation.
- `frontend/src/components/DatasetAnalytics.jsx`, `EnergyVisuals.jsx`, `SourceBoundary.jsx`: uploaded analysis, conditional visuals, and unsupported-operation states.
- `frontend/src/components/FloatingAssistant.jsx`: floating chat, retained conversation, real assistant requests, source references, close/minimize controls, focus restoration, Escape support, and uploaded-source limitation.
- `frontend/src/App.jsx`, `components/Layout.jsx`, `components/Sidebar.jsx`: lazy pages, source selector, compact navigation, and legacy assistant route opening the floating window.
- `frontend/src/pages/OverviewDashboard.jsx`, `EnergyMonitoring.jsx`: editorial overview, flow snapshots, renewable generation mix, battery portrait, cost summary, demand heatmap, focused time trend.
- `frontend/src/pages/ResearchResults.jsx`: secondary study navigation.
- `frontend/src/index.css`, `frontend/tailwind.config.js`, and existing JSX presentation colors: black / blue atmospheric surfaces and compatible light appearance.
- Frontend test files and `frontend/package.json`: expanded test command.

The checkout contained substantial pre-existing local changes. They were not reverted, committed, or pushed.

## Upload architecture and schema

`POST /api/datasets/upload` accepts one multipart `file` field containing UTF-8 CSV. Limits: 10 MiB per file, 100,000 rows, 64 unique columns, 2,000 characters per value, 20 datasets per storage area. Malformed CSV, control characters, invalid units, invalid/nonfinite numeric values, duplicate timestamps, and invalid timestamps are rejected with structured errors. Parsed uploads are drafts until mapping succeeds.

`GET /api/datasets` lists metadata and storage policy. `GET /api/datasets/{id}` returns preview and mapping. `POST /api/datasets/{id}/mapping` validates and stores mapping/conversion consent. `GET /api/datasets/{id}/analysis` returns actual normalized uploaded records, summary, and compatibility assessment. No deletion endpoint is implemented.

Map timestamp plus at least one power field. Supported standardized fields: timestamp, demand_kw, solar_kw, wind_kw, grid_import_kw, grid_export_kw, price, battery_soc. Power units: kW, W, MW. Charge: fraction or percent. Price: USD/kWh. ISO timestamps require an offset unless the user explicitly confirms naive timestamps are UTC. kWh is never silently treated as kW. W/MW/percentage conversions require confirmation. Unknown/unmapped columns stay in the preserved source CSV, not fabricated model inputs.

## Storage behavior and deployment

A generated UUID identifies each private dataset directory; uploaded filenames never determine paths. Original CSV bytes, checksum, metadata, mapping, and normalized analysis are retained separately from campus datasets and experiments. No arbitrary serialized Python objects are accepted or executed. The API provides no public raw-file route. External browser origins are rejected for dataset operations unless explicitly configured; the API defaults to 127.0.0.1.

Without `ENERGY_DATASET_STORAGE`, a process-specific temporary directory is used and the UI discloses that uploads become unavailable after server restart. Restart does not securely erase old temporary files; host lifecycle/cleanup governs them. With `ENERGY_DATASET_STORAGE`, the adapter uses the specified volume; durability depends on the deployment's mounted storage and backups. Render's ephemeral disk is not durable storage. Production multi-user use requires authentication, per-user isolation, quotas, TLS, CSRF protections appropriate to the deployment, and a durable object/volume storage adapter. This local service must not be exposed as a public multi-tenant upload service without those additions. `ENERGY_DATASET_ORIGINS` can specify allowed frontend origins.

## Scientific accounting and model limits

Energy integrates interval-start average kW to the next timestamp, using actual elapsed hours. The last row gets no invented duration. Missing field values exclude their starting intervals; field coverage is reported. Irregular sampling is explicitly warned about, including long-gap hold assumptions. Average demand is time-weighted. No consumption-based cost or savings is invented from optional prices. Renewable output-to-demand ratio requires complete aligned demand, solar, and wind measurements; it is explicitly not self-consumption.

The renewable donut divides solar and wind generation only, not imports plus gross generation masquerading as consumption. Exports remain separate in the flow snapshot. Flow arrows describe recorded connections, not invented source-to-building allocations. Uploaded battery capacity, dispatch power, and operating limits are unavailable because the current schema does not collect them; they are not inherited from the campus model.

Compatibility explicitly checks required features, 8 ordered nodes / 14 directed edges, hourly UTC frequency, 24-hour history, 24-hour horizon, units, training-fitted preprocessing, and checkpoint verification. No aggregate upload is declared inference-ready. Forecasting requires adaptation; optimization is unsupported. Unsupported pages show an explanation instead of campus values under an uploaded label. The floating assistant is restricted to Campus Demo and never claims access to uploaded measurements. No training, optimization algorithm, or frozen benchmark artifact was changed.

## Design and chart choices

The updated palette uses nearly black backgrounds, midnight navy translucent panels, electric blue accents, and restrained solar amber / wind cyan. Dark appearance is the default; the user may select light appearance. Glass is concentrated in the navigation, overview and assistant. Tables/forms retain readable surfaces. CSS includes narrow-screen layouts, focus styles, and reduced-motion support.

The overview replaces six repetitive cards with an asymmetric primary energy panel, connected flow snapshot, large battery portrait, renewable generation donut, compact zero-based cost bars, and a study insight. Energy monitoring uses a flow view, renewable mix, daily/hourly demand heatmap, and one focused time-series. Heavy chart code and page routes are lazy-loaded. Other research pages retain their functionality and detailed charts; they have been recolored and moved behind the study entry, rather than fully redesigned in this pass.

## Verification

- `.\.venv\Scripts\python.exe -m pytest tests/test_user_datasets.py -q -p no:cacheprovider`: **35 passed in 2.32s**. Initial sandbox attempt had temporary-directory permission failures; the same tests passed outside the sandbox.
- `npm test --prefix frontend`: **12 passed**, including file limits, strict API failure behavior, real progress event propagation, source-policy gates, and irregular-interval integration.
- `npm run build --prefix frontend`: **passed**, 2,022 modules. Final main bundle 236.70 kB and shared chart bundle 389.86 kB; the earlier >500 kB warning is resolved by lazy-loading uploaded-data charts.
- Live HTTP check through `http://127.0.0.1:5173/api/datasets`: valid multipart upload accepted, mapping saved, returned energy changed to the uploaded sample's **35.0 kWh**, malformed CSV rejected with **HTTP 400**. This is API integration evidence, not browser interaction evidence.
- `.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`: **1,135 passed, 3 Gymnasium warnings in 257.64s**.
- Browser verification: **not completed**. Earlier Computer Use stopped because it could not verify the current browser URL with enough confidence. No desktop/mobile screenshot or click-through results are claimed. Browser-host verification must be restored before satisfying the original complete browser acceptance criteria.

## Remaining limitations

Full browser upload/activation/assistant interaction checks and rendered desktop/mobile QA remain. Several detailed research pages still need their original requested visualization restructuring; the current follow-up simplifies their navigation and recolors them. The existing forecasting demo graph is illustrative and is not an inference result for an upload. No arbitrary uploaded inference, automatic retraining, persistent production hosting, deletion UI, or multi-user storage is advertised as implemented.
