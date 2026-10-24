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

from src.config import get_groq_api_key, get_groq_api_keys, get_groq_model, get_groq_timeout
from src.llm.base import CredentialPool, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def _clean_json_markdown(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```[a-zA-Z]*\n?", "", trimmed)
        trimmed = re.sub(r"\n?```$", "", trimmed)
    return trimmed.strip()


class GroqProvider(LLMProvider):
    """Client for Groq free-tier LLM models via REST with multi-key pool and cooldown management."""

    def __init__(
        self,
        api_key: str | list[str] | None = None,
        model_name: str | None = None,
        timeout: int | None = None,
        max_retries: int = 1,
        initial_backoff: float = 0.5,
    ) -> None:
        if api_key is None:
            keys = get_groq_api_keys()
        elif isinstance(api_key, str):
            keys = [api_key.strip()] if api_key.strip() else []
        else:
            keys = [k.strip() for k in api_key if k and k.strip()]

        self._pool = CredentialPool(keys)
        self._model = (model_name or get_groq_model()).strip()
        self._timeout = timeout or get_groq_timeout()
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def is_configured(self) -> bool:
        return self._pool.has_keys

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def _api_key(self) -> str:
        """Compatibility accessor for primary key."""
        return self._pool.credentials[0].api_key if self._pool.has_keys else ""

    def get_pool_status(self) -> list[dict[str, Any]]:
        return self._pool.get_status()

    def check_health(self) -> dict[str, Any]:
        pool_status = self.get_pool_status()
        available_keys = sum(1 for p in pool_status if not p["is_disabled"] and not p["in_cooldown"])
        return {
            "provider": self.provider_name,
            "configured": self.is_configured,
            "model": self._model,
            "key_count": len(pool_status),
            "available_keys": available_keys,
            "keys": pool_status,
            "status": "HEALTHY" if available_keys > 0 else ("COOLDOWN_OR_DISABLED" if self.is_configured else "UNCONFIGURED"),
        }

    def _call_api(self, prompt: str, system_instruction: str = "", response_format_json: bool = False) -> LLMResponse:
        if not self.is_configured:
            return LLMResponse(
                success=False,
                provider_name=self.provider_name,
                model_name=self._model,
                error_message="Groq API key is not configured.",
                error_category="AUTH_FAILURE",
                is_auth_error=True,
            )

        messages = []
        user_content = prompt
        if response_format_json and "json" not in (system_instruction + " " + user_content).lower():
            user_content = f"{user_content}\n\nRespond with a valid JSON object."

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": user_content})

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
        last_cred = None

        while retries <= self._max_retries:
            cred = self._pool.get_next_available()
            if cred is None:
                return LLMResponse(
                    success=False,
                    provider_name=self.provider_name,
                    model_name=self._model,
                    error_message="All Groq API keys are currently cooling down or disabled.",
                    error_category="RATE_LIMIT",
                    is_rate_limited=True,
                    retries_attempted=retries,
                )

            last_cred = cred
            headers = {
                "Authorization": f"Bearer {cred.api_key}",
                "Content-Type": "application/json",
            }

            try:
                res = requests.post(GROQ_API_ENDPOINT, json=payload, headers=headers, timeout=self._timeout)

                if res.status_code == 200:
                    cred.record_success()
                    data = res.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return LLMResponse(
                            success=False,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Groq returned no choices.",
                            error_category="INVALID_RESPONSE",
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    text = choices[0].get("message", {}).get("content", "")
                    return LLMResponse(
                        success=True,
                        text=text,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        used_key_masked=cred.masked_key,
                        retries_attempted=retries,
                    )

                elif res.status_code == 429:
                    cred.set_cooldown(duration_seconds=60.0, reason="HTTP 429 Rate Limit")
                    retries += 1
                    if retries > self._max_retries:
                        logger.warning("Groq rate limit exceeded after max retries.")
                        return LLMResponse(
                            success=False,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Groq rate limit exceeded (HTTP 429).",
                            error_category="RATE_LIMIT",
                            is_rate_limited=True,
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    if self._pool.get_next_available() is None:
                        time.sleep(backoff)
                        backoff *= 2
                    continue

                elif res.status_code in (401, 403):
                    cred.mark_disabled(reason=f"HTTP {res.status_code} Auth Failure")
                    err_msg = f"HTTP {res.status_code}: Check GROQ_API_KEY."
                    if self._pool.get_next_available() is not None and retries < self._max_retries:
                        retries += 1
                        continue
                    return LLMResponse(
                        success=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=err_msg,
                        error_category="AUTH_FAILURE",
                        is_auth_error=True,
                        used_key_masked=cred.masked_key,
                        retries_attempted=retries,
                    )

                elif res.status_code == 400:
                    err_msg = f"HTTP 400: Bad request or unsupported model ({self._model})."
                    return LLMResponse(
                        success=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=err_msg,
                        error_category="INVALID_REQUEST",
                        used_key_masked=cred.masked_key,
                        retries_attempted=retries,
                    )

                else:
                    retries += 1
                    if retries > self._max_retries:
                        return LLMResponse(
                            success=False,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message=f"Groq server error HTTP {res.status_code}: {res.text[:200]}",
                            error_category="SERVER_ERROR",
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

            except requests.exceptions.Timeout:
                retries += 1
                if retries > self._max_retries:
                    return LLMResponse(
                        success=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=f"Groq request timed out after {self._timeout}s.",
                        error_category="TIMEOUT",
                        used_key_masked=last_cred.masked_key if last_cred else "",
                        retries_attempted=retries,
                    )
                time.sleep(backoff)
                backoff *= 2

            except Exception as exc:
                return LLMResponse(
                    success=False,
                    provider_name=self.provider_name,
                    model_name=self._model,
                    error_message=f"Groq request failed: {type(exc).__name__}",
                    error_category="SERVER_ERROR",
                    used_key_masked=last_cred.masked_key if last_cred else "",
                    retries_attempted=retries,
                )

        return LLMResponse(
            success=False,
            provider_name=self.provider_name,
            model_name=self._model,
            error_message="Groq request exhausted retries.",
            error_category="RATE_LIMIT",
            used_key_masked=last_cred.masked_key if last_cred else "",
            retries_attempted=retries,
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
            resp.error_category = "PARSE_ERROR"
            resp.error_message = f"Failed to parse Groq JSON output: {exc}"
            return resp
