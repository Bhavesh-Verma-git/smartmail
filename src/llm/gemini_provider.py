"""
src/llm/gemini_provider.py
--------------------------
Direct Google Gemini REST client implementation with bounded retries and rate limit handling.

Free Tier Considerations:
- Connects to Google AI Studio Gemini API via HTTPS REST.
- Model ID is dynamically configurable (defaults to gemini-2.0-flash).
- Handles HTTP 429 RESOURCE_EXHAUSTED with exponential backoff.
- Treats model responses as untrusted: parses and validates JSON safely without eval().
- Never prints, logs, or returns raw GEMINI_API_KEY.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

from src.config import get_gemini_api_key, get_gemini_model, get_gemini_timeout
from src.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _clean_json_markdown(text: str) -> str:
    """Strip markdown code fence ```json ... ``` wrapper if present."""
    trimmed = text.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```[a-zA-Z]*\n?", "", trimmed)
        trimmed = re.sub(r"\n?```$", "", trimmed)
    return trimmed.strip()


class GeminiProvider(LLMProvider):
    """Client for Google Gemini API via REST."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: int | None = None,
        max_retries: int = 2,
        initial_backoff: float = 1.0,
    ) -> None:
        self._api_key = api_key.strip() if api_key is not None else get_gemini_api_key()
        self._model = (model_name or get_gemini_model()).strip()
        self._timeout = timeout or get_gemini_timeout()
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def _call_api(self, prompt: str, system_instruction: str = "", response_mime_type: str = "") -> LLMResponse:
        """Call generateContent REST endpoint with retry and backoff."""
        if not self.is_configured:
            return LLMResponse(
                success=False,
                provider_name=self.provider_name,
                model_name=self._model,
                error_message="Gemini API key is not configured.",
            )

        url = f"{GEMINI_BASE_URL}/{self._model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        body: dict[str, Any] = {"contents": contents}

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        generation_config: dict[str, Any] = {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        }
        if response_mime_type == "application/json":
            generation_config["responseMimeType"] = "application/json"

        body["generationConfig"] = generation_config

        retries = 0
        backoff = self._initial_backoff

        while retries <= self._max_retries:
            try:
                res = requests.post(url, json=body, headers=headers, timeout=self._timeout)

                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return LLMResponse(
                            success=False,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Gemini returned no candidates (possible safety filter block).",
                        )
                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in content_parts)
                    return LLMResponse(
                        success=True,
                        text=text,
                        provider_name=self.provider_name,
                        model_name=self._model,
                    )

                elif res.status_code == 429:
                    retries += 1
                    if retries > self._max_retries:
                        logger.warning("Gemini rate limit exceeded after max retries.")
                        return LLMResponse(
                            success=False,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Gemini rate limit exceeded (HTTP 429).",
                            is_rate_limited=True,
                        )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                elif res.status_code in (400, 401, 403):
                    # Bad request or auth error
                    err_msg = f"HTTP {res.status_code}: Check GEMINI_API_KEY or model availability ({self._model})."
                    logger.warning(f"Gemini API error: {err_msg}")
                    return LLMResponse(
                        success=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=err_msg,
                        is_auth_error=res.status_code in (401, 403),
                    )

                else:
                    return LLMResponse(
                        success=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=f"Gemini server error HTTP {res.status_code}",
                    )

            except requests.exceptions.Timeout:
                retries += 1
                if retries > self._max_retries:
                    return LLMResponse(
                        success=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=f"Gemini request timed out after {self._timeout}s.",
                    )
                time.sleep(backoff)
                backoff *= 2

            except Exception as exc:
                return LLMResponse(
                    success=False,
                    provider_name=self.provider_name,
                    model_name=self._model,
                    error_message=f"Gemini request failed: {type(exc).__name__}",
                )

        return LLMResponse(
            success=False,
            provider_name=self.provider_name,
            model_name=self._model,
            error_message="Gemini request exhausted retries.",
        )

    def generate_text(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        return self._call_api(prompt, system_instruction=system_instruction)

    def generate_json(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        resp = self._call_api(prompt, system_instruction=system_instruction, response_mime_type="application/json")
        if not resp.success:
            return resp

        # Parse JSON defensively
        raw_text = _clean_json_markdown(resp.text)
        try:
            parsed = json.loads(raw_text)
            resp.parsed_json = parsed if isinstance(parsed, dict) else {"data": parsed}
            return resp
        except json.JSONDecodeError as exc:
            resp.success = False
            resp.error_message = f"Failed to parse LLM JSON output: {exc}"
            return resp
