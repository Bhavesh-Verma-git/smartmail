"""
src/llm/groq_provider.py
------------------------
Groq API client implementation using OpenAI-compatible chat completions REST endpoint.

Designed as an optional fallback LLM:
- Configurable model (defaults to llama-3.3-70b-versatile).
- Implements timeouts and bounded backoff.
- Completely optional: Does not throw or break if GROQ_API_KEY is not set.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

from src.config import get_groq_api_key, get_groq_model, get_groq_timeout
from src.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def _clean_json_markdown(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```[a-zA-Z]*\n?", "", trimmed)
        trimmed = re.sub(r"\n?```$", "", trimmed)
    return trimmed.strip()


class GroqProvider(LLMProvider):
    """Client for Groq free-tier LLM models via REST."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: int | None = None,
        max_retries: int = 2,
        initial_backoff: float = 1.0,
    ) -> None:
        self._api_key = api_key.strip() if api_key is not None else get_groq_api_key()
        self._model = (model_name or get_groq_model()).strip()
        self._timeout = timeout or get_groq_timeout()
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def _call_api(self, prompt: str, system_instruction: str = "", response_format_json: bool = False) -> LLMResponse:
        if not self.is_configured:
            return LLMResponse(
                success=False,
                provider_name=self.provider_name,
                model_name=self._model,
                error_message="Groq API key is not configured.",
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        retries = 0
        backoff = self._initial_backoff

        while retries <= self._max_retries:
            try:
                res = requests.post(GROQ_API_ENDPOINT, json=payload, headers=headers, timeout=self._timeout)

                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return LLMResponse(
                            success=False,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Groq returned no choices.",
                        )
                    text = choices[0].get("message", {}).get("content", "")
                    return LLMResponse(
                        success=True,
                        text=text,
                        provider_name=self.provider_name,
                        model_name=self._model,
                    )

                elif res.status_code == 429:
                    retries += 1
                    if retries > self._max_retries:
                        logger.warning("Groq rate limit exceeded after max retries.")
                        return LLMResponse(
                            success=False,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Groq rate limit exceeded (HTTP 429).",
                            is_rate_limited=True,
                        )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                elif res.status_code in (401, 403):
                    return LLMResponse(
                        success=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=f"HTTP {res.status_code}: Check GROQ_API_KEY.",
                        is_auth_error=True,
                    )

                else:
                    return LLMResponse(
                        success=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=f"Groq server error HTTP {res.status_code}",
                    )

            except requests.exceptions.Timeout:
                retries += 1
                if retries > self._max_retries:
                    return LLMResponse(
                        success=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=f"Groq request timed out after {self._timeout}s.",
                    )
                time.sleep(backoff)
                backoff *= 2

            except Exception as exc:
                return LLMResponse(
                    success=False,
                    provider_name=self.provider_name,
                    model_name=self._model,
                    error_message=f"Groq request failed: {type(exc).__name__}",
                )

        return LLMResponse(
            success=False,
            provider_name=self.provider_name,
            model_name=self._model,
            error_message="Groq request exhausted retries.",
        )

    def generate_text(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        return self._call_api(prompt, system_instruction=system_instruction)

    def generate_json(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        resp = self._call_api(prompt, system_instruction=system_instruction, response_format_json=True)
        if not resp.success:
            return resp

        raw_text = _clean_json_markdown(resp.text)
        try:
            parsed = json.loads(raw_text)
            resp.parsed_json = parsed if isinstance(parsed, dict) else {"data": parsed}
            return resp
        except json.JSONDecodeError as exc:
            resp.success = False
            resp.error_message = f"Failed to parse Groq JSON output: {exc}"
            return resp
