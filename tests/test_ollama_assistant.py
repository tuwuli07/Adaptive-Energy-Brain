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


def test_default_local_routing_and_no_cloud_fallback(monkeypatch):
    monkeypatch.delenv("ENERGY_ASSISTANT_PROVIDER", raising=False)
    with patch.object(service.OLLAMA_SERVICE, "generate", return_value={"status": "error", "error": "offline"}) as local, patch.object(service.GEMINI_SERVICE, "generate") as cloud:
        result = service.answer_energy_query("Explain demand")
        assert result["status"] == "error"
        assert result["provider"] == "ollama"
        local.assert_called_once()
        cloud.assert_not_called()


def test_uploaded_summary_goes_to_selected_local_model_only():
    summary = {"record_count": 3, "energy_kwh": {"demand_kw": 35}, "warnings": ["Irregular intervals"]}
    with patch.object(service.OLLAMA_SERVICE, "generate", return_value={"status": "success", "text": "Your consumption is 35 kWh.", "model": "tiny"}) as local:
        result = service.answer_energy_query("Explain my consumption", provider="ollama", model="tiny", dataset_summary=summary)
        assert result["sources"][0]["type"] == "dataset"
        assert result["provider"] == "ollama"
        assert '"demand_kw": 35' in local.call_args.args[0]
        assert local.call_args.kwargs["model"] == "tiny"
        assert "campus" not in local.call_args.args[0].lower()


def test_explicit_gemini_selection_preserves_cloud_option():
    with patch.object(service.GEMINI_SERVICE, "is_configured", return_value=True), patch.object(service.GEMINI_SERVICE, "generate", return_value={"status": "success", "text": "A kWh measures energy.", "model": "gemini-test"}) as cloud, patch.object(service.OLLAMA_SERVICE, "generate") as local:
        result = service.answer_energy_query("What is a kWh?", provider="gemini")
        assert result["provider"] == "gemini"
        cloud.assert_called_once()
        local.assert_not_called()


def test_exact_dataset_totals_do_not_use_llm_arithmetic():
    with patch.object(service.OLLAMA_SERVICE, "generate") as local:
        result = service.answer_energy_query("What is my total electricity consumption?", provider="ollama", dataset_summary={"energy_kwh": {"demand_kw": 35}})
        assert "35.000 kWh" in result["answer"]
        assert result["provider"] == "calculated"
        local.assert_not_called()


@pytest.mark.parametrize("message,provider", [(None, "ollama"), ("", "ollama"), ("x" * 4001, "ollama"), ("Hi", "unrecognized")])
def test_invalid_requests(message, provider):
    assert service.answer_energy_query(message, provider=provider)["status"] == "invalid_input"
