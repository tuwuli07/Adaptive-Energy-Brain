"""Local Ollama chat adapter. No model downloads and no cloud fallback."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaService:
    def __init__(self, base_url=None, model=None, timeout=90):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1") or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("Ollama must use a loopback HTTP URL without credentials or a path.")
        self.model_name = model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def _request(self, path, payload=None, timeout=None):
        raw = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.base_url + path, data=raw, headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=timeout or self.timeout) as response:
            data = response.read(2 * 1024 * 1024 + 1)
            if len(data) > 2 * 1024 * 1024:
                raise ValueError("Ollama response exceeded the supported size.")
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("Invalid Ollama response.")
            return result

    def status(self):
        try:
            result = self._request("/api/tags", timeout=3)
            models = [m["name"] for m in result.get("models", []) if isinstance(m, dict) and isinstance(m.get("name"), str)]
            return {"available": True, "models": models, "model": self.model_name, "configured_model_available": self.model_name in models}
        except (OSError, ValueError, KeyError, TypeError):
            return {"available": False, "models": [], "model": self.model_name, "configured_model_available": False}

    def is_configured(self):
        return self.status()["configured_model_available"]

    def generate(self, prompt, system_instruction=None, temperature=.2, max_output_tokens=600, model=None, history=None):
        selected = model or self.model_name
        status = self.status()
        if not status["available"]:
            return {"status": "error", "text": "", "model": selected, "error": "Ollama is not running. Start Ollama on this computer and try again."}
        if selected not in status["models"]:
            return {"status": "error", "text": "", "model": selected, "error": "The selected model is not installed in Ollama. Choose an installed model or pull it manually."}
        messages = [{"role": "system", "content": system_instruction or "You are a helpful energy assistant."}]
        for item in (history or [])[-6:]:
            if isinstance(item, dict) and item.get("role") in ("user", "model", "assistant") and isinstance(item.get("content"), str):
                messages.append({"role": "assistant" if item["role"] == "model" else item["role"], "content": item["content"][:2000]})
        messages.append({"role": "user", "content": prompt[:24000]})
        try:
            # CPU is predictable on machines where GPU cold-start may stall.
            # Administrators can opt into automatic offload with -1.
            num_gpu = int(os.environ.get("OLLAMA_NUM_GPU", "0"))
            result = self._request("/api/chat", {"model": selected, "messages": messages, "stream": False, "keep_alive": "5m", "options": {"temperature": temperature, "num_predict": min(1500, max(1, max_output_tokens)), "num_ctx": 8192, "num_gpu": num_gpu}})
            text = result.get("message", {}).get("content", "").strip()
            if not text:
                raise ValueError("Ollama returned an empty response.")
            return {"status": "success", "text": text, "model": selected, "provider": "ollama"}
        except (OSError, ValueError, AttributeError):
            return {"status": "error", "text": "", "model": selected, "error": "Ollama could not finish the answer. It may be busy or the request timed out. Please try again."}


OLLAMA_SERVICE = OllamaService()
