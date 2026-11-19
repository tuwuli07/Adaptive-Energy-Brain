from src.dashboard.public_preview import (
    allowed_api,
    gemini_preview_enabled,
    safe_static,
    valid_gemini_access_key,
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


def test_gemini_preview_requires_configured_provider_key_and_access_key():
    access_key = "a" * 43
    configured = {
        "ENERGY_ASSISTANT_PROVIDER": "gemini",
        "GEMINI_API_KEY": "provider-secret",
        "ENERGY_ASSISTANT_ACCESS_KEY": access_key,
    }

    assert gemini_preview_enabled(configured)
    assert valid_gemini_access_key("Bearer " + access_key, configured)
    assert not valid_gemini_access_key("Bearer wrong", configured)
    assert not valid_gemini_access_key(access_key, configured)
    assert not gemini_preview_enabled({**configured, "ENERGY_ASSISTANT_ACCESS_KEY": "short"})
    assert not gemini_preview_enabled({**configured, "ENERGY_ASSISTANT_PROVIDER": "ollama"})
