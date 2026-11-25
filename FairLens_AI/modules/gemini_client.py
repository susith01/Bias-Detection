"""Official Google GenAI Python SDK wrapper with availability check and offline fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from config import CONFIG

logger = logging.getLogger(__name__)

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore


class GeminiClient:
    """Wrapper around the official `google-genai` SDK for FairLens AI."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else CONFIG.gemini_api_key).strip()
        self.model_name = (model_name or CONFIG.gemini_model or self.DEFAULT_MODEL).strip()
        self._client: Optional[Any] = None
        self.last_error: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and genai is not None)

    def _get_client(self) -> Optional[Any]:
        if not self.is_available:
            return None
        if self._client is None:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                self.last_error = f"Failed to initialize Google GenAI client: {exc}"
                logger.warning(self.last_error)
                return None
        return self._client

    def verify_model_availability(self) -> Dict[str, Any]:
        """Verify configured Gemini model availability before generating."""
        client = self._get_client()
        if client is None:
            return {
                "available": False,
                "model": self.model_name,
                "mode": "offline_fallback",
                "reason": self.last_error or "GEMINI_API_KEY not configured or google-genai missing.",
            }
        try:
            models = list(client.models.list())
            names = [getattr(m, "name", "") for m in models]
            matched = any(self.model_name in name for name in names)
            return {
                "available": matched or len(names) > 0,
                "model": self.model_name,
                "mode": "gemini_live",
                "reason": "Model verified via Google GenAI SDK." if matched else "Using configured model.",
            }
        except Exception as exc:
            self.last_error = str(exc)
            return {
                "available": False,
                "model": self.model_name,
                "mode": "offline_fallback",
                "reason": f"Model check failed ({exc}); offline template fallback active.",
            }

    def generate_text(
        self,
        prompt: str,
        system_instruction: str,
        fallback_text: str,
    ) -> Dict[str, str]:
        """
        Invoke Gemini with strictly aggregated metrics (never PII or keys).
        Falls back cleanly to `fallback_text` if the API is unavailable or errors.
        """
        client = self._get_client()
        if client is None:
            return {
                "text": fallback_text,
                "source": "offline_fallback",
                "model": "template-engine",
            }
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"system_instruction": system_instruction},
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                return {
                    "text": fallback_text,
                    "source": "offline_fallback",
                    "model": "template-engine",
                }
            return {
                "text": text,
                "source": "gemini_api",
                "model": self.model_name,
            }
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Gemini API call failed, using offline fallback: %s", exc)
            return {
                "text": fallback_text,
                "source": "offline_fallback",
                "model": "template-engine",
            }
