"""Unit tests for Gemini service integration and assistant orchestration."""

from unittest.mock import MagicMock, patch
import pytest

from src.assistant.gemini_service import GeminiService
from src.assistant.orchestrator import AssistantOrchestrator


def test_gemini_service_unconfigured():
    service = GeminiService(api_key="")
    assert not service.is_configured()
    res = service.generate("Hello")
    assert res["status"] == "unconfigured"
    assert "GEMINI_API_KEY" in res["error"]


def test_gemini_service_mocked_success():
    service = GeminiService(api_key="mock_key_for_testing", model="gemini-2.5-flash")
    assert service.is_configured()

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Adaptive Energy Brain combines GNN forecasting with MPC."
    mock_response.usage_metadata = {"prompt_token_count": 50, "candidates_token_count": 20}
    mock_client.models.generate_content.return_value = mock_response

    service._client = mock_client
    res = service.generate("What is Adaptive Energy Brain?")
    assert res["status"] == "success"
    assert "Adaptive Energy Brain" in res["text"]
    assert res["model"] == "gemini-2.5-flash"


def test_gemini_service_mocked_transient_retry():
    service = GeminiService(api_key="mock_key", model="gemini-2.5-flash")
    mock_client = MagicMock()

    # Fail once with rate limit (429), succeed on retry
    mock_response = MagicMock()
    mock_response.text = "Succeeded after rate-limit backoff."
    mock_client.models.generate_content.side_effect = [
        Exception("429 ResourceExhausted: rate limit exceeded"),
        mock_response,
    ]

    service._client = mock_client
    with patch("time.sleep", return_value=None):
        res = service.generate("Test prompt")
        assert res["status"] == "success"
        assert res["text"] == "Succeeded after rate-limit backoff."
        assert mock_client.models.generate_content.call_count == 2


def test_orchestrator_query_sanitization():
    orchestrator = AssistantOrchestrator()
    res = orchestrator.answer_query("")
    assert res["status"] == "invalid_input"
    assert len(res["sources"]) == 0


def test_orchestrator_grounded_offline_fallback():
    # When Gemini is unconfigured, orchestrator should return structured factual fallback
    service = GeminiService(api_key="")
    orchestrator = AssistantOrchestrator(gemini_service=service)

    res = orchestrator.answer_query("Explain the battery storage specifications and inverter limits.")
    assert res["status"] == "offline_fallback"
    assert "150" in res["answer"]
    assert "500" in res["answer"]
    assert len(res["sources"]) > 0
    assert "MCP Tool: get_battery_status" in [s["title"] for s in res["sources"]]


def test_orchestrator_sources_attribution_with_mocked_gemini():
    service = GeminiService(api_key="mock_key")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "According to Phase 12 documentation, the selector routes to DRO under adverse stress."
    mock_client.models.generate_content.return_value = mock_response
    service._client = mock_client

    orchestrator = AssistantOrchestrator(gemini_service=service)
    res = orchestrator.answer_query("Why did the controller select DRO?")
    assert res["status"] == "success"
    assert "DRO" in res["answer"]
    assert len(res["sources"]) >= 2
    # Check that at least one MCP source and one RAG source were included
    source_types = {s["type"] for s in res["sources"]}
    assert "mcp_tool" in source_types
    assert "rag_document" in source_types
