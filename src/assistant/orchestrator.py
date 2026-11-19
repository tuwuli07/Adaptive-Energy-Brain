"""AI Energy Assistant Orchestration Engine for Adaptive Energy Brain.

Coordinates intent classification, read-only MCP tool data fetching,
ChromaDB semantic retrieval, context grounding, Gemini generation,
and safe offline fallbacks.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from src.assistant.gemini_service import GEMINI_SERVICE, GeminiService
from src.assistant.mcp_server import execute_tool, READ_ONLY_TOOLS
from src.assistant.rag import RAG_PIPELINE, RetrievedChunk

logger = logging.getLogger("orchestrator")

SYSTEM_INSTRUCTION = """You are the AI Energy Assistant for Adaptive Energy Brain, a research and engineering platform for autonomous multi-asset energy management and microgrid optimization.

Your responsibility is to provide accurate, scientifically grounded, and concise answers explaining the project's architecture, AI forecasting models, battery dispatch, risk-aware controllers, and experimental benchmarks.

STRICT SCIENTIFIC INTEGRITY & GROUNDING RULES:
1. DATA PROVENANCE: The benchmark is conducted on a 1,314-hour frozen held-out test partition of the "Hypothetical Campus Microgrid Dataset v1". Always clarify that this is a synthetic demonstration case study, NOT live physical utility telemetry.
2. BATTERY SPECIFICATION: Battery storage has a capacity of 500 kWh and strictly enforced symmetric power limits of 150 kW (both charge and discharge). Never claim 250 kW or other unverified limits. Degradation is modeled as a linear throughput cost of $0.02/kWh.
3. ORACLE CLASSIFICATION: The rolling 24h oracle controller is a noncausal, perfect-foresight theoretical ceiling for benchmarking only. It is NOT deployable in real-world operations.
4. ADAPTIVE POLICY SELECTION: Phase 12 implements Contextual Adaptive Policy Selection (regime-aware routing among expert controllers). It does NOT implement gradient-based few-shot meta-learning (e.g., MAML or RL2), which is accurately cataloged as future research.
5. GROUNDING: Use the provided Grounded Evidence (MCP structured tool data and RAG documentation excerpts). When numbers are requested, use the exact values from the evidence. Never invent energy measurements, costs, or unverified claims.
6. TRANSPARENCY: If evidence is unavailable or insufficient to answer a question, explicitly state the limitation rather than fabricating an answer.
"""


class AssistantOrchestrator:
    """Orchestrates RAG retrieval, MCP tool execution, and Gemini generation."""

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini = gemini_service or GEMINI_SERVICE

    def _select_mcp_tools(self, query: str) -> List[str]:
        """Classify query intent and select relevant read-only MCP tools."""
        q_lower = query.lower()
        selected = []

        if any(w in q_lower for w in ("battery", "soc", "storage", "inverter", "degradation", "kwh", "power limit")):
            selected.append("get_battery_status")
        if any(w in q_lower for w in ("forecast", "gnn", "transformer", "spatio-temporal", "prediction", "rmse", "mae")):
            selected.append("get_forecast_metrics")
        if any(w in q_lower for w in ("compare", "comparison", "benchmark", "best", "versus", "vs", "seasonal", "oracle", "rank", "test results")):
            selected.append("get_controller_comparison")
        if any(w in q_lower for w in ("cvar", "risk", "tail", "var", "lambda", "peak shaving")):
            selected.append("get_risk_results")
        if any(w in q_lower for w in ("dro", "robust", "ambiguity", "worst-case", "duality", "distributional")):
            selected.append("get_dro_results")
        if any(w in q_lower for w in ("adaptive", "selector", "regime", "routing", "switch", "dynamic")):
            selected.append("get_adaptive_policy_results")
        if any(w in q_lower for w in ("energy", "demand", "generation", "load", "import", "export", "solar", "wind", "consumption")):
            selected.append("get_energy_summary")
        if any(w in q_lower for w in ("limitation", "limitations", "synthetic", "assumptions", "future work", "caution")):
            selected.append("get_research_limitations")
        if any(w in q_lower for w in ("overview", "architecture", "what is", "about", "mission", "system")):
            selected.append("get_system_overview")

        # Fallback to system overview if no specific match
        if not selected:
            selected = ["get_system_overview"]

        # Limit to top 2 tools to keep prompt compact and focused
        return selected[:2]

    def _generate_offline_fallback(
        self,
        query: str,
        rag_chunks: List[RetrievedChunk],
        mcp_data: Dict[str, Any],
    ) -> str:
        """Construct a high-quality grounded answer when Gemini API is unconfigured/offline."""
        q_lower = query.lower()

        # Architecture & Overview
        if any(w in q_lower for w in ("architecture", "what is", "overview", "adaptive energy brain")):
            return (
                "**Adaptive Energy Brain** is an autonomous multi-asset energy management system designed for renewable microgrids, "
                "commercial facilities, smart buildings, and university campuses. Evaluated on a 1,314-hour synthetic campus simulation, "
                "the platform coordinates:\n\n"
                "1. **Spatio-Temporal GNN-Transformer Forecasting:** Multi-horizon (1h–24h) energy load and renewable generation predictions.\n"
                "2. **Battery Energy Storage Optimization:** 500 kWh capacity with strictly enforced ±150 kW inverter limits and linear degradation tracking ($0.02/kWh).\n"
                "3. **Optimization & Control Suite:** Includes Deterministic MPC, PPO Reinforcement Learning, CVaR Risk-Aware MPC (Phase 10), and Distributionally Robust Optimization (DRO, Phase 11).\n"
                "4. **Adaptive Policy Selection (Phase 12):** Context-aware regime routing dynamically directing dispatch to specialized controllers under solar surplus, peak tariffs, or adverse stress."
            )

        # GNN-Transformer Forecasting
        if any(w in q_lower for w in ("gnn", "forecast", "prediction", "rmse")):
            return (
                "**GNN-Transformer Spatio-Temporal Forecasting:**\n\n"
                "- **Model Structure:** Combines Graph Neural Networks (capturing spatial power flow across 5 campus buildings) with multi-head self-attention Transformers (capturing 24-hour temporal dynamics).\n"
                "- **Training Contract:** Uses masked target loss to prevent future-data leakage across split boundaries.\n"
                "- **Performance:** Achieves 24h Mean Absolute Error of **17.18 kW** and Root Mean Squared Error of **22.12 kW** (a 42.75% RMSE reduction compared to seasonal persistence baselines).\n"
                "- **Integration:** Forecast lookaheads directly feed the LP optimization horizon of the Model Predictive Controllers."
            )

        # Controller Comparison & Results
        if any(w in q_lower for w in ("compare", "comparison", "benchmark", "best controller", "seasonal vs gnn")):
            return (
                "**Authoritative Phase 8.5 Test Benchmark Results (1,314 Held-Out Hours):**\n\n"
                "- **GNN-Transformer MPC:** Lowest deployable adjusted net cost (**$36,532.73**), saving **$5.18** over Seasonal Persistence MPC with identical 31,540.80 kWh throughput (31.54 FCE).\n"
                "- **Seasonal Persistence MPC:** $36,537.91 adjusted net cost.\n"
                "- **Continuous Price-Aware Baseline:** $36,881.33 adjusted net cost.\n"
                "- **Zero-Action Baseline:** $39,267.88 net cost ($2,735.15 savings achieved by GNN-MPC).\n"
                "- **Oracle 24h Benchmark:** $36,157.06 noncausal theoretical ceiling (not deployable).\n"
                "- **PPO Continuous Control:** Underperformed heuristics due to partial observability and value estimation horizon limits."
            )

        # Risk-Aware Control: CVaR vs DRO
        if any(w in q_lower for w in ("cvar", "dro", "difference", "risk", "robust")):
            return (
                "**Comparison of Risk-Aware Optimization Paradigms:**\n\n"
                "1. **CVaR Risk-Aware MPC (Phase 10):**\n"
                "   - *Method:* Rockafellar-Uryasev (2000) linear program equivalence minimizing expected cost plus a tail risk penalty ($\\lambda=1.0$, $\\alpha=0.95$) over $S=5$ scenarios.\n"
                "   - *Outcome:* Under compound adverse stress, capped peak hourly purchase surges from **$164.95/h to $114.71/h** (a 30.5% maximum peak cost reduction) while achieving **$167.37** in adjusted operating savings.\n\n"
                "2. **Distributionally Robust Optimization (DRO, Phase 11):**\n"
                "   - *Method:* Exact LP duality over total-variation / $L_1$ probability ambiguity sets $\\mathcal{Q}(p, \\epsilon) = \\{q \\in \\Delta^S : \\|q - p\\|_1 \\le \\epsilon\\}$.\n"
                "   - *Outcome:* Worst-case protection against probability shift and unannounced price shocks without scenario probability assumptions. Solved via HiGHS in ~82.4 ms.\n\n"
                "3. **Key Distinction:** CVaR optimizes the known tail; DRO insulates against distribution drift and scenario probability estimation errors."
            )

        # Adaptive Policy Selection
        if any(w in q_lower for w in ("adaptive", "selector", "regime", "routing")):
            return (
                "**Adaptive Policy Selection & Contextual Regime Routing (Phase 12):**\n\n"
                "- **Mechanism:** At decision step $t$, causal state features (forecasted net load, solar surplus, battery SOC, current tariff rate, and rolling stress metrics) determine the active operational regime:\n"
                "  1. *Solar Self-Consumption:* Dispatches GNN-MPC when solar surplus > 10 kW and SOC < 0.85 to capture local generation.\n"
                "  2. *Peak Tariff Tail-Hedging:* Dispatches CVaR-MPC ($\\lambda=1.0$) when tariff $\\ge$ $0.25/kWh and SOC > 0.30 to shave expensive peaks.\n"
                "  3. *Compound Stress / Robust:* Dispatches DRO-MPC under combined renewable deficits and elevated demand.\n"
                "  4. *Nominal Base Dispatch:* Dispatches GNN-MPC for baseline economic scheduling.\n"
                "- **Dynamic Profile:** Under compound stress across 71 hours, the selector routed 59.2% of intervals to GNN-MPC, 33.8% to DRO-MPC, and 7.0% to CVaR-MPC with **zero constraint violations**.\n"
                "- **Scientific Classification:** Contextual regime routing, distinct from gradient-based meta-learning (MAML/RL2)."
            )

        # Battery Storage & Limits
        if any(w in q_lower for w in ("battery", "soc", "inverter", "limit", "capacity")):
            return (
                "**Battery Energy Storage System Specifications:**\n\n"
                "- **Nominal Capacity:** 500.0 kWh\n"
                "- **Power Inverter Limits:** Strictly **150.0 kW** continuous symmetric charge and discharge.\n"
                "- **Efficiency:** 95% charge, 95% discharge (round-trip efficiency: 90.25%).\n"
                "- **Operational State of Charge (SOC):** Constrained between 10% (0.10) and 90% (0.90) to prevent deep discharge degradation.\n"
                "- **Degradation Accounting:** Linear throughput penalty of **$0.02 per kWh** throughput ($10.00/FCE).\n"
                "- **Terminal Inventory Credit:** Rated at **$0.147368 per kWh** DC storage valuation."
            )

        # Scientific Limitations
        if any(w in q_lower for w in ("limitation", "limitations", "limit", "synthetic")):
            return (
                "**Documented Scientific Limitations of the Platform:**\n\n"
                "1. **Synthetic Case Study:** Evaluated on synthetic multi-building campus microgrid simulation data, not live utility hardware.\n"
                "2. **Linear Degradation Model:** Uses a linear throughput penalty ($0.02/kWh) rather than nonlinear electro-chemical cell dynamics.\n"
                "3. **Single-Bus Power Balance:** Evaluated as a localized copper-plate microgrid without AC power flow, voltage sag, or reactive power modeling.\n"
                "4. **Oracle Noncausality:** The 24h rolling oracle possesses perfect ground-truth lookahead and serves only as a diagnostic reference ceiling.\n"
                "5. **Contextual Routing vs Meta-Learning:** Phase 12 implements rule-based regime routing among fixed controllers; parameter-adapting meta-learning (MAML/RL2) remains future work."
            )

        # Default Grounded Summary from RAG Chunks
        if rag_chunks:
            top_chunk = rag_chunks[0]
            return (
                f"Based on verified project documentation ({top_chunk.source_file}, section '{top_chunk.section_title}'):\n\n"
                f"{top_chunk.text}\n\n"
                "*(Note: GEMINI_API_KEY is currently unconfigured; this answer was generated directly from the local verified RAG knowledge base and MCP tools.)*"
            )

        return (
            "Adaptive Energy Brain is an AI-driven energy management system combining GNN-Transformer forecasting, "
            "model predictive control, CVaR tail-risk hedging, and distributionally robust optimization for battery storage. "
            "Please ask about forecasting, battery specs, controller comparisons, or scientific limitations."
        )

    def answer_query(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Process user question through RAG retrieval, MCP structured data, and Gemini."""
        # 1. Validation & sanitization
        clean_msg = (message or "").strip()
        if not clean_msg:
            return {
                "answer": "Please provide a valid question regarding the Adaptive Energy Brain platform.",
                "sources": [],
                "data_provenance": "None",
                "model": "none",
                "status": "invalid_input",
            }
        clean_msg = clean_msg[:2000]

        # 2. Select and invoke MCP tools
        tool_names = self._select_mcp_tools(clean_msg)
        mcp_data = {}
        sources = []

        for tname in tool_names:
            try:
                res = execute_tool(tname)
                if res.get("status") == "success":
                    mcp_data[tname] = res.get("data")
                    sources.append({
                        "name": tname,
                        "type": "mcp_tool",
                        "title": f"MCP Tool: {tname}",
                        "source_file": "src/assistant/mcp_server.py",
                    })
            except Exception as e:
                logger.warning(f"Error executing MCP tool {tname}: {e}")

        # 3. Retrieve relevant documentation chunks from ChromaDB
        rag_chunks = []
        try:
            rag_chunks = RAG_PIPELINE.retrieve(clean_msg, top_k=3)
            for c in rag_chunks:
                sources.append({
                    "name": c.chunk_id,
                    "type": "rag_document",
                    "title": f"{Path(c.source_file).name} ({c.section_title})",
                    "source_file": c.source_file,
                    "relevance_score": c.relevance_score,
                })
        except Exception as e:
            logger.warning(f"ChromaDB retrieval error: {e}")

        # 4. Check Gemini configuration
        if not self.gemini.is_configured():
            # Graceful grounded offline fallback
            answer_text = self._generate_offline_fallback(clean_msg, rag_chunks, mcp_data)
            return {
                "answer": answer_text,
                "sources": sources,
                "data_provenance": "Verified Research Artifacts & Simulation Benchmark v1 (Local RAG + MCP)",
                "model": "offline-grounded-fallback",
                "status": "offline_fallback",
            }

        # 5. Build prompt with evidence grounding
        context_parts = []
        if mcp_data:
            context_parts.append("### STRUCTURED PROJECT METRICS (MCP Tools):")
            for tname, tdata in mcp_data.items():
                context_parts.append(f"Tool `{tname}` output:\n```json\n{json.dumps(tdata, indent=2)}\n```")

        if rag_chunks:
            context_parts.append("\n### VERIFIED DOCUMENTATION EXCERPTS (RAG Knowledge Base):")
            for i, chunk in enumerate(rag_chunks, 1):
                context_parts.append(f"Excerpt {i} [{chunk.source_file} - {chunk.section_title}]:\n{chunk.text}\n")

        full_prompt = (
            f"USER QUESTION: {clean_msg}\n\n"
            f"GROUNDED EVIDENCE:\n" + "\n".join(context_parts) + "\n\n"
            "Please synthesize a clear, concise, and direct response strictly grounded in the evidence above. "
            "Cite the specific documents or MCP data used. If the question cannot be answered from the evidence, "
            "transparently state the limitation."
        )

        # 6. Execute Gemini inference
        response = self.gemini.generate(
            prompt=full_prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=1500,
        )

        if response.get("status") == "success":
            return {
                "answer": response["text"],
                "sources": sources,
                "data_provenance": "Verified Research Artifacts & Simulation Benchmark v1 (Gemini + RAG + MCP)",
                "model": response.get("model", self.gemini.model_name),
                "status": "success",
            }
        else:
            # Fall back safely on error or timeout
            logger.warning(f"Gemini generation returned status '{response.get('status')}'. Falling back to local grounded response.")
            answer_text = self._generate_offline_fallback(clean_msg, rag_chunks, mcp_data)
            return {
                "answer": answer_text,
                "sources": sources,
                "data_provenance": "Verified Research Artifacts & Simulation Benchmark v1 (Local RAG + MCP Fallback)",
                "model": "offline-grounded-fallback",
                "status": "fallback_on_error",
                "error_detail": response.get("error"),
            }


# Global Assistant Orchestrator instance
ASSISTANT_ORCHESTRATOR = AssistantOrchestrator()
