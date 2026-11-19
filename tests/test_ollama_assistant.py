"""Local model routing, failure isolation and site-specific evidence."""
from unittest.mock import Mock, patch

import pytest

from src.assistant.ollama_service import OllamaService
from src.assistant import energy_assistant as service


@pytest.mark.parametrize("url", ["https://remote.example", "http://192.168.1.2:11434", "http://localhost:11434/path", "http://user:secret@localhost:11434", "file:///models"])
def test_ollama_only_uses_local_http(url):
    with pytest.raises(ValueError):
        OllamaService(base_url=url)


def test_ollama_chat_contract_and_history():
    local = OllamaService(model="local:3b")
    local._request = Mock(side_effect=[{"models": [{"name": "local:3b"}]}, {"message": {"content": "Use demand data to identify peaks."}}])
    result = local.generate("How?", "Energy rules", history=[{"role": "model", "content": "Previous answer"}], max_output_tokens=50)
    assert result["status"] == "success"
    path, payload = local._request.call_args.args
    assert path == "/api/chat"
    assert payload["stream"] is False
    assert payload["messages"][0] == {"role": "system", "content": "Energy rules"}
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["options"]["num_predict"] == 50


def test_unavailable_ollama_and_uninstalled_model_are_explicit():
    local = OllamaService()
    local._request = Mock(side_effect=OSError("offline"))
    assert local.status()["available"] is False
    assert "not running" in local.generate("Question")["error"]
    local._request = Mock(return_value={"models": []})
    assert "not installed" in local.generate("Question")["error"]


def test_ollama_timeout_and_empty_response():
    local = OllamaService(model="x")
    for response in (OSError("timeout"), {"message": {"content": ""}}):
        local._request = Mock(side_effect=[{"models": [{"name": "x"}]}, response])
        assert local.generate("Question")["status"] == "error"


def test_default_assistant_uses_gemini_only():
    with patch.object(service.GEMINI_SERVICE, "is_configured", return_value=True), patch.object(service.GEMINI_SERVICE, "generate", return_value={"status": "success", "text": "Gemini answer.", "model": "gemini-test"}) as cloud:
        result = service.answer_energy_query("Explain demand")
        assert result["status"] == "success"
        assert result["provider"] == "gemini"
        cloud.assert_called_once()


def test_assistant_status_reports_gemini_only():
    with patch.object(service.GEMINI_SERVICE, "is_configured", return_value=True):
        status = service.assistant_status()
    assert status["default_provider"] == "gemini"
    assert status["gemini_configured"]
    assert "ollama" not in status


def test_unconfigured_gemini_does_not_fall_back_to_ollama():
    with patch.object(service.GEMINI_SERVICE, "is_configured", return_value=False), patch.object(service.GEMINI_SERVICE, "generate") as cloud:
        result = service.answer_energy_query("Explain demand")
        assert result["status"] == "error"
        assert "Gemini" in result["error"]
        cloud.assert_not_called()


def test_ollama_is_rejected_as_an_assistant_provider():
    with patch.object(service.GEMINI_SERVICE, "generate") as cloud:
        result = service.answer_energy_query("Explain demand", provider="ollama")
        assert result["status"] == "invalid_input"
        assert "Gemini API only" in result["answer"]
        cloud.assert_not_called()


def test_uploaded_summary_is_context_for_gemini_only():
    summary = {"record_count": 3, "energy_kwh": {"demand_kw": 35}, "warnings": ["Irregular intervals"]}
    with patch.object(service.GEMINI_SERVICE, "is_configured", return_value=True), patch.object(service.GEMINI_SERVICE, "generate", return_value={"status": "success", "text": "Your consumption is 35 kWh.", "model": "gemini-test"}) as cloud:
        result = service.answer_energy_query("Explain my consumption", dataset_summary=summary)
        assert result["sources"][0]["type"] == "dataset"
        assert result["provider"] == "gemini"
        assert '"demand_kw": 35' in cloud.call_args.args[0]
        assert "campus" not in cloud.call_args.args[0].lower()


def test_explicit_gemini_selection_uses_gemini():
    with patch.object(service.GEMINI_SERVICE, "is_configured", return_value=True), patch.object(service.GEMINI_SERVICE, "generate", return_value={"status": "success", "text": "A kWh measures energy.", "model": "gemini-test"}) as cloud, patch("src.assistant.ollama_service.OLLAMA_SERVICE.generate") as local:
        result = service.answer_energy_query("What is a kWh?", provider="gemini")
        assert result["provider"] == "gemini"
        cloud.assert_called_once()
        local.assert_not_called()


def test_exact_dataset_calculation_is_given_to_gemini_to_answer():
    summary = {"energy_kwh": {"demand_kw": 35}}
    with patch.object(service.GEMINI_SERVICE, "is_configured", return_value=True), patch.object(service.GEMINI_SERVICE, "generate", return_value={"status": "success", "text": "Total electricity consumption is 35.000 kWh.", "model": "gemini-test"}) as cloud:
        result = service.answer_energy_query("What is my total electricity consumption?", dataset_summary=summary)
        assert result["answer"] == "Total electricity consumption is 35.000 kWh."
        assert result["provider"] == "gemini"
        assert "35.000 kWh" in cloud.call_args.args[0]


def test_dataset_calculation_is_not_returned_without_gemini():
    with patch.object(service.GEMINI_SERVICE, "is_configured", return_value=False), patch.object(service.GEMINI_SERVICE, "generate") as cloud:
        result = service.answer_energy_query(
            "What is my total electricity consumption?",
            dataset_summary={"energy_kwh": {"demand_kw": 35}},
        )
    assert result["status"] == "error"
    assert "Gemini" in result["error"]
    cloud.assert_not_called()


@pytest.mark.parametrize("message,provider", [(None, "gemini"), ("", "gemini"), ("x" * 4001, "gemini"), ("Hi", "unrecognized")])
def test_invalid_requests(message, provider):
    assert service.answer_energy_query(message, provider=provider)["status"] == "invalid_input"
