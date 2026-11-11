# AI Development Stack Alignment and Optional Integration Plan

## Scope and repository findings

The selected stack is a direction for development, not a requirement to install
every tool. The current project is a Python research codebase: Parts 1–4
implement data preparation, a physical simulator, rule-based controllers, and
experiment evaluation. Part 5 forecasting is not implemented. Existing scripts
run workflows directly; experiment artifacts already record provenance and
checksums in JSON. No new runtime package or service is required for this
alignment.

| Selected tool or practice | Repository status after this change | Evidence / boundary |
|---|---|---|
| Semantic chunk deduplication | Planned; not implemented or configured | Consider only for a later knowledge-retrieval corpus. |
| Sliding-window/context trimming | Planned; not implemented or configured | No LLM context pipeline exists. Forecast sequence windows are ordinary time-series inputs, not LLM context trimming. |
| Rolling summary memory | Planned; not implemented or configured | No agent memory component exists. |
| Structured JSON outputs | Partially present for experiment/audit artifacts; not an LLM output mechanism | Existing data and baseline workflows write JSON metadata and checksums. A proposed Part 5 run-metadata contract is in [the schema](../configs/experiment_metadata.schema.json); runtime validation is not implemented. |
| Cheap-model routing | Planned; not implemented or configured | No model router or LLM calls exist. |
| Gemini, Grok, Qwen | Planned provider options; no clients, credentials, or configuration | Keep provider calls outside forecasting/training code. |
| Naive RAG | Not implemented | `src/rag/` is an empty package placeholder. The assistant is explicitly out of Part 5 scope. |
| Vector database | Planned only | No vector-store dependency, index, or service is configured. |
| Semantic chunking for retrieval | Planned only | No retrieval corpus or chunker exists. |
| n8n | Planned optional automation | No workflow definitions or integration are configured. |
| Prefect | Planned optional experiment orchestration | Current experiment/data workflows are Python scripts, not Prefect flows. |
| PydanticAI | Planned optional agent framework | No package or agent runtime is configured. |
| GitHub Spec-Kit | Not configured; minimal adoption proposed below | No `.specify/` directory, Spec-Kit configuration, or spec-driven workflow was found. |
| `AGENTS.md` | Configured by this alignment | Root [AGENTS.md](../AGENTS.md) gives shared instructions for Copilot and Codex. |

These findings are based on the project files and dependency manifest. They do
not imply that any absent tool is installed outside this repository.

## GitHub Spec-Kit: minimal adoption proposal

Do not install, initialize, or add Spec-Kit templates as part of this task. If
the team chooses to adopt its specification-first workflow:

1. Before Part 5.1 coding, write one forecasting feature specification under
   `.specify/specs/part5-gnn-transformer/spec.md`. State scope, requirements,
   acceptance criteria, data/forecast interfaces, leakage rules, and explicit
   exclusions (no simulator rewrite, baseline changes, RAG, or training-time
   LLM calls).
2. Derive a bounded implementation plan and tasks from that specification;
   link them from the specification and this handoff. Keep the existing
   `docs/` completion reports as historical evidence rather than migrating
   them.
3. Use Spec-Kit's commands/templates only after the project owner approves
   adding and configuring the tool. Until then, ordinary Markdown
   requirements and existing pytest tests remain sufficient.

This adopts traceable specifications without creating an unreviewed tooling
dependency or rewriting the existing documentation.

## Part 5 experiment metadata

The project already uses YAML for checked-in configuration and JSON for
generated run/audit metadata. Continue that division: keep experiment
parameters in YAML; serialize each run's metadata in JSON. The new
[JSON Schema](../configs/experiment_metadata.schema.json) defines a
versioned, dependency-free contract for Part 5 runs, including:

- experiment ID, phase, status, creation time, and random seed;
- dataset manifest and split hashes, source IDs/origins, features, target,
  lookback/horizon, and the training-only preprocessing boundary;
- model architecture/configuration and training/selection settings;
- Python/framework/package versions, device, determinism, source revision,
  measured metrics, and artifact hashes.

The contract requires validation-based model selection and explicitly forbids
test-based selection. It is a schema only: no experiment runner, validator
dependency, or training implementation is added. Part 5.1 should emit metadata
and add validation at the experiment-artifact boundary, selecting an existing
or optional validator only if justified. If this contract proves too
restrictive for the chosen model, revise it with a schema version bump rather
than silently changing old run metadata.

## Optional integration sequence

These integrations are proposals for later phases, not current requirements.
Adopt each only when a concrete use case justifies its maintenance, secrets,
and deployment costs.

### Forecasting experiments with Prefect

If experiment volume or scheduling makes manual scripts inadequate, wrap the
existing validated steps in an optional Prefect flow: verify immutable input
manifest and split hashes; load a versioned YAML config; train with recorded
seed; select/checkpoint using validation only; run a separately approved final
test evaluation; validate metadata and artifacts; then publish to a new
versioned results directory. Make retries idempotent, preserve failure
visibility, and never let orchestration alter simulator behavior, baselines,
or data. Keep direct script execution available for offline/local use. Add
Prefect only with an agreed execution environment and a tested flow contract.

### n8n

Use n8n, if needed, for external triggers, human approval, notifications, or
handoff between systems—not as a replacement for Python model training,
simulation, or reproducible experiment code. Pass versioned run IDs and
artifact references rather than datasets or credentials; require explicit
approval before accessing held-out test evaluation or external services.

### PydanticAI and LLM providers

If a later assistant or bounded research helper needs an LLM, put PydanticAI
and provider adapters (Gemini, Grok, or Qwen) behind a separate optional
application boundary. Configure provider selection and secrets outside source
control, return validated structured data, and log provider/model identifiers
without logging credentials or private prompts. Cheap-model routing, context
trimming, rolling summaries, and semantic deduplication apply to that assistant
only. Never add provider calls or model routing to the Part 5 training loop.

### Retrieval

Defer retrieval until after the forecasting workflow is established. If a
later knowledge assistant has a demonstrated need, begin with a small
source-attributed naive RAG prototype. Define chunk boundaries, deduplication,
source hashes, access rules, and retrieval evaluation before selecting a vector
database. Keep semantic chunking, embeddings, vector storage, and the assistant
as optional replaceable components; do not introduce them during Part 5.

## Guardrails for all phases

- Do not change the existing physical simulator or completed rule-based
  baselines to support these optional tools.
- Keep training and evaluation reproducible and prevent future-data leakage;
  do not fit preprocessing on validation/test or select models on test.
- Do not implement RAG in Part 5, add LLM calls to training, commit secrets,
  or make optional services a prerequisite for offline tests.
- Record any later adoption, dependency, configuration, and verification in
  [the development handoff](development_handoff.md).
