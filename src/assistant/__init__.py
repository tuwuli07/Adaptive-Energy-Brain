"""Assistant package for Adaptive Energy Brain.

Integrates Gemini AI Assistant, ChromaDB RAG, and Model Context Protocol (MCP) tools.
"""

from src.assistant.gemini_service import GEMINI_SERVICE, GeminiService
from src.assistant.mcp_server import list_registered_tools, execute_tool, READ_ONLY_TOOLS
from src.assistant.orchestrator import ASSISTANT_ORCHESTRATOR, AssistantOrchestrator
from src.assistant.rag import RAG_PIPELINE, ProjectRAGPipeline

__all__ = [
    "GEMINI_SERVICE",
    "GeminiService",
    "RAG_PIPELINE",
    "ProjectRAGPipeline",
    "ASSISTANT_ORCHESTRATOR",
    "AssistantOrchestrator",
    "list_registered_tools",
    "execute_tool",
    "READ_ONLY_TOOLS",
]
