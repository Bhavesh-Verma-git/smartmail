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
import random
import re
import time
from typing import Any

import requests

from src.config import get_groq_api_key, get_groq_api_keys, get_groq_model, get_groq_timeout
from src.llm.base import (
    CredentialPool,
    LLMProvider,
    LLMResponse,
    get_circuit_breaker,
    get_rate_limiter,
    parse_retry_after,
    safe_parse_llm_json,
)

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

        raw_model = (model_name or get_groq_model()).strip() or "openai/gpt-oss-120b"
        self._pool = CredentialPool(keys)
        self._model = raw_model
        self._timeout = timeout or get_groq_timeout()
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self.circuit_breaker = get_circuit_breaker("groq")
        self.rate_limiter = get_rate_limiter("groq")

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

        allowed, block_reason = self.circuit_breaker.allow_request()
        if not allowed:
            logger.warning(f"Groq request rejected by circuit breaker: {block_reason}")
            return LLMResponse(
                success=False,
                provider_name=self.provider_name,
                model_name=self._model,
                error_message=block_reason,
                error_category="CIRCUIT_OPEN",
                circuit_open=True,
                is_transient=False,
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
            "max_tokens": 2048,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        retries = 0
        backoff = self._initial_backoff
        last_cred = None

        while retries <= self._max_retries:
            cred = self._pool.get_next_available()
            if cred is None:
                self.circuit_breaker.record_failure("All Groq keys in cooldown", is_permanent=False, cooldown_override=30.0)
                return LLMResponse(
                    success=False,
                    provider_name=self.provider_name,
                    model_name=self._model,
                    error_message="All Groq API keys are currently cooling down or disabled.",
                    error_category="RATE_LIMIT",
                    is_rate_limited=True,
                    is_transient=True,
                    retries_attempted=retries,
                )

            last_cred = cred
            headers = {
                "Authorization": f"Bearer {cred.api_key}",
                "Content-Type": "application/json",
            }

            try:
                self.rate_limiter.acquire()
                res = requests.post(GROQ_API_ENDPOINT, json=payload, headers=headers, timeout=self._timeout)

                if res.status_code == 200:
                    cred.record_success()
                    self.circuit_breaker.record_success()
                    data = res.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return LLMResponse(
                            success=False,
                            http_status=200,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Groq returned no choices.",
                            error_category="INVALID_RESPONSE",
                            finish_reason="NO_CHOICES",
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason", "")
                    text = choice.get("message", {}).get("content", "")
                    if not text.strip():
                        return LLMResponse(
                            success=False,
                            http_status=200,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message=f"Groq returned empty text (finish_reason: {finish_reason or 'UNKNOWN'}).",
                            error_category="EMPTY_RESPONSE",
                            finish_reason=finish_reason,
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    return LLMResponse(
                        success=True,
                        text=text,
                        http_status=200,
                        finish_reason=finish_reason,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        used_key_masked=cred.masked_key,
                        retries_attempted=retries,
                    )

                elif res.status_code == 404:
                    err_detail = ""
                    try:
                        err_detail = res.json().get("error", {}).get("message", "")
                    except Exception:
                        pass
                    err_msg = f"HTTP 404 Model Not Found ({self._model}): {err_detail or 'The requested model does not exist on Groq'}"
                    logger.error(err_msg)
                    self.circuit_breaker.record_failure(reason=err_msg, is_permanent=True)
                    return LLMResponse(
                        success=False,
                        http_status=404,
                        is_transient=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=err_msg,
                        error_category="MODEL_NOT_FOUND",
                        used_key_masked=cred.masked_key,
                        retries_attempted=retries,
                    )

                elif res.status_code == 429:
                    retries += 1
                    retry_after = parse_retry_after(res.headers.get("Retry-After"))
                    self.circuit_breaker.record_failure(reason="HTTP 429 Rate Limit", is_permanent=False, cooldown_override=retry_after)
                    has_other_keys = len(self._pool.credentials) > 1 and self._pool.get_next_available() is not None

                    if has_other_keys:
                        cred.set_cooldown(duration_seconds=60.0, reason="HTTP 429 Rate Limit")
                    else:
                        if retries > self._max_retries:
                            cred.set_cooldown(duration_seconds=60.0, reason="HTTP 429 Rate Limit (exhausted retries)")
                            logger.warning("Groq rate limit exceeded after max retries.")
                            return LLMResponse(
                                success=False,
                                http_status=429,
                                is_transient=True,
                                provider_name=self.provider_name,
                                model_name=self._model,
                                error_message="Groq rate limit exceeded (HTTP 429).",
                                error_category="RATE_LIMIT",
                                is_rate_limited=True,
                                used_key_masked=cred.masked_key,
                                retries_attempted=retries,
                            )
                        delay = min(retry_after if retry_after > 0 else (backoff + random.uniform(0.1, 0.4)), 3.0)
                        time.sleep(delay)
                        backoff *= 2
                        continue

                    if retries > self._max_retries:
                        logger.warning("Groq rate limit exceeded after max retries.")
                        return LLMResponse(
                            success=False,
                            http_status=429,
                            is_transient=True,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Groq rate limit exceeded (HTTP 429).",
                            error_category="RATE_LIMIT",
                            is_rate_limited=True,
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    continue

                elif res.status_code in (401, 403):
                    cred.mark_disabled(reason=f"HTTP {res.status_code} Auth Failure")
                    err_msg = f"HTTP {res.status_code}: Check GROQ_API_KEY."
                    self.circuit_breaker.record_failure(reason=err_msg, is_permanent=True)
                    if self._pool.get_next_available() is not None and retries < self._max_retries:
                        retries += 1
                        continue
                    return LLMResponse(
                        success=False,
                        http_status=res.status_code,
                        is_transient=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=err_msg,
                        error_category="AUTH_FAILURE",
                        is_auth_error=True,
                        used_key_masked=cred.masked_key,
                        retries_attempted=retries,
                    )

                elif res.status_code == 400:
                    err_detail = ""
                    try:
                        err_detail = res.json().get("error", {}).get("message", "")
                    except Exception:
                        pass
                    err_msg = f"HTTP 400: {err_detail or 'Bad request or unsupported model'} ({self._model})."
                    logger.warning(f"Groq bad request: {err_msg}")
                    self.circuit_breaker.record_failure(reason=err_msg, is_permanent=True)
                    return LLMResponse(
                        success=False,
                        http_status=400,
                        is_transient=False,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=err_msg,
                        error_category="INVALID_REQUEST",
                        used_key_masked=cred.masked_key,
                        retries_attempted=retries,
                    )

                elif res.status_code in (500, 502, 503, 504):
                    retries += 1
                    err_detail = ""
                    try:
                        err_detail = res.json().get("error", {}).get("message", "")
                    except Exception:
                        pass
                    if retries > self._max_retries:
                        return LLMResponse(
                            success=False,
                            http_status=res.status_code,
                            is_transient=True,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message=f"Groq server error HTTP {res.status_code}: {err_detail or 'Internal Server Error'}",
                            error_category="SERVER_ERROR",
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    delay = min(backoff + random.uniform(0.1, 0.5), 3.0)
                    time.sleep(delay)
                    backoff *= 2
                    continue

                else:
                    retries += 1
                    if retries > self._max_retries:
                        return LLMResponse(
                            success=False,
                            http_status=res.status_code,
                            is_transient=False,
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
                        http_status=408,
                        is_transient=True,
                        provider_name=self.provider_name,
                        model_name=self._model,
                        error_message=f"Groq request timed out after {self._timeout}s.",
                        error_category="TIMEOUT",
                        used_key_masked=last_cred.masked_key if last_cred else "",
                        retries_attempted=retries,
                    )
                time.sleep(min(backoff + random.uniform(0.1, 0.3), 2.0))
                backoff *= 2

            except Exception as exc:
                return LLMResponse(
                    success=False,
                    provider_name=self.provider_name,
                    model_name=self._model,
                    error_message=f"Groq request failed: {type(exc).__name__}",
                    error_category="SERVER_ERROR",
                    is_transient=True,
                    used_key_masked=last_cred.masked_key if last_cred else "",
                    retries_attempted=retries,
                )

        return LLMResponse(
            success=False,
            http_status=429,
            is_transient=True,
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

        parsed, err_cat, err_msg = safe_parse_llm_json(resp.text, finish_reason=resp.finish_reason)
        if parsed is None:
            resp.success = False
            resp.error_category = err_cat
            resp.error_message = err_msg
            return resp

        resp.parsed_json = parsed
        return resp
