from src.dashboard.public_preview import (
    allowed_api,
    gemini_preview_enabled,
    safe_static,
)


def test_preview_never_serves_repository_or_encoded_traversal(tmp_path):
    (tmp_path / "index.html").write_text("built app")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("app")
    assert safe_static("/data", tmp_path) == tmp_path / "index.html"
    assert safe_static("/assets/app.js", tmp_path) == tmp_path / "assets" / "app.js"
    for path in ("/.env", "/src/main.jsx", "/configs/data_sources.yaml", "/assets/%2e%2e/.env", "/assets/..%5c.env", "/@vite/client"):
        assert safe_static(path, tmp_path) is None


def test_public_api_has_explicit_route_and_method_allowlist():
    assert allowed_api("POST", "/api/assistant/chat")
    assert allowed_api("GET", "/api/datasets")
    assert allowed_api("POST", "/api/datasets/" + "a" * 32 + "/mapping")
    assert not allowed_api("GET", "/api/datasets/../../config")
    assert not allowed_api("POST", "/api/simulations/run")
    assert not allowed_api("DELETE", "/api/datasets/" + "a" * 32)


def test_gemini_preview_requires_provider_and_api_key_but_not_access_key():
    configured = {
        "ENERGY_ASSISTANT_PROVIDER": "gemini",
        "GEMINI_API_KEY": "provider-secret",
    }

    assert gemini_preview_enabled(configured)
    assert gemini_preview_enabled({**configured, "ENERGY_ASSISTANT_ACCESS_KEY": ""})
    assert not gemini_preview_enabled({**configured, "GEMINI_API_KEY": ""})
    assert not gemini_preview_enabled({**configured, "ENERGY_ASSISTANT_PROVIDER": "ollama"})
