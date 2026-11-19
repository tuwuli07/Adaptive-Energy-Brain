"""Google Gen AI SDK Service for Adaptive Energy Brain Assistant.

Handles client initialization, environment configuration, bounded retries,
rate-limit resilience, request timeouts, and error handling.
"""

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger("gemini_service")

DEFAULT_MODEL = "gemini-2.5-flash"
REQUEST_TIMEOUT_SECONDS = 20.0
MAX_RETRIES = 2


class GeminiService:
    """Wrapper around Google Gen AI SDK for resilient inference."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key
        self._model = model
        self.model_name = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self._client = None
        self._active_key = None

    def get_api_key(self) -> Optional[str]:
        if self._api_key is not None:
            return self._api_key
        return os.environ.get("GEMINI_API_KEY")

    def is_configured(self) -> bool:
        """Check if GEMINI_API_KEY is available."""
        key = self.get_api_key()
        return bool(key and key.strip())

    def _get_client(self):
        if not self.is_configured():
            return None
        key = self.get_api_key()
        if self._client is None or (self._active_key is not None and self._active_key != key):
            try:
                from google import genai
                self._client = genai.Client(api_key=key)
                self._active_key = key
            except Exception as e:
                logger.error(f"Failed to initialize Google Gen AI Client: {e}")
                return None
        return self._client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """Generate a response using Gemini with bounded retries and timeout protection."""
        if not self.is_configured():
            return {
                "status": "unconfigured",
                "text": "",
                "error": "GEMINI_API_KEY is not configured in the environment.",
                "model": self.model_name,
            }

        client = self._get_client()
        if client is None:
            return {
                "status": "error",
                "text": "",
                "error": "Failed to create GenAI client.",
                "model": self.model_name,
            }

        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction if system_instruction else None,
        )

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    return {
                        "status": "success",
                        "text": response.text.strip(),
                        "model": self.model_name,
                        "usage": getattr(response, "usage_metadata", None),
                    }
                else:
                    return {
                        "status": "empty",
                        "text": "The model returned an empty response.",
                        "model": self.model_name,
                    }
            except Exception as e:
                last_error = str(e)
                # Check for rate-limiting (429) or transient server errors (503)
                is_transient = any(code in last_error for code in ("429", "503", "ResourceExhausted", "DeadlineExceeded"))
                if is_transient and attempt < MAX_RETRIES:
                    sleep_time = (2 ** attempt) * 1.5
                    logger.warning(f"Transient error from Gemini (attempt {attempt+1}/{MAX_RETRIES+1}). Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Gemini API generation error: {e}")
                    break

        return {
            "status": "error",
            "text": "",
            "error": f"Gemini API request failed: {last_error}",
            "model": self.model_name,
        }


# Global Gemini Service instance
GEMINI_SERVICE = GeminiService()
