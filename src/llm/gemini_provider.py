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
import random
import re
import time
from typing import Any

import requests

from src.config import get_gemini_api_key, get_gemini_api_keys, get_gemini_model, get_gemini_timeout
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

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _clean_json_markdown(text: str) -> str:
    """Strip markdown code fence ```json ... ``` wrapper if present."""
    trimmed = text.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```[a-zA-Z]*\n?", "", trimmed)
        trimmed = re.sub(r"\n?```$", "", trimmed)
    return trimmed.strip()


class GeminiProvider(LLMProvider):
    """Client for Google Gemini API via REST with multi-key pool and cooldown management."""

    def __init__(
        self,
        api_key: str | list[str] | None = None,
        model_name: str | None = None,
        timeout: int | None = None,
        max_retries: int = 1,
        initial_backoff: float = 0.2,
    ) -> None:
        if api_key is None:
            keys = get_gemini_api_keys()
        elif isinstance(api_key, str):
            keys = [api_key.strip()] if api_key.strip() else []
        else:
            keys = [k.strip() for k in api_key if k and k.strip()]

        self._pool = CredentialPool(keys)
        self._model = (model_name or get_gemini_model()).strip() or "gemini-3.8-flash"
        self._timeout = timeout or get_gemini_timeout()
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self.circuit_breaker = get_circuit_breaker("gemini")
        self.rate_limiter = get_rate_limiter("gemini")

    @property
    def provider_name(self) -> str:
        return "gemini"

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

    def _call_api(self, prompt: str, system_instruction: str = "", response_mime_type: str = "") -> LLMResponse:
        """Call generateContent REST endpoint with multi-key rotation and cooldown handling."""
        if not self.is_configured:
            return LLMResponse(
                success=False,
                provider_name=self.provider_name,
                model_name=self._model,
                error_message="Gemini API key is not configured.",
                error_category="AUTH_FAILURE",
                is_auth_error=True,
            )

        allowed, block_reason = self.circuit_breaker.allow_request()
        if not allowed:
            logger.warning(f"Gemini request rejected by circuit breaker: {block_reason}")
            return LLMResponse(
                success=False,
                provider_name=self.provider_name,
                model_name=self._model,
                error_message=block_reason,
                error_category="CIRCUIT_OPEN",
                circuit_open=True,
                is_transient=False,
            )

        url = f"{GEMINI_BASE_URL}/{self._model}:generateContent"
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        body: dict[str, Any] = {"contents": contents}

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        generation_config: dict[str, Any] = {
            "temperature": 0.2,
            "maxOutputTokens": 2048,
        }
        if response_mime_type == "application/json":
            generation_config["responseMimeType"] = "application/json"

        body["generationConfig"] = generation_config

        retries = 0
        backoff = self._initial_backoff
        last_cred = None

        while retries <= self._max_retries:
            cred = self._pool.get_next_available()
            if cred is None:
                # All configured keys are either cooling down or disabled
                self.circuit_breaker.record_failure("All Gemini keys in cooldown", is_permanent=False, cooldown_override=30.0)
                return LLMResponse(
                    success=False,
                    provider_name=self.provider_name,
                    model_name=self._model,
                    error_message="All Gemini API keys are currently cooling down or disabled.",
                    error_category="RATE_LIMIT",
                    is_rate_limited=True,
                    is_transient=True,
                    retries_attempted=retries,
                )

            last_cred = cred
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": cred.api_key,
            }

            try:
                self.rate_limiter.acquire()
                res = requests.post(url, json=body, headers=headers, timeout=self._timeout)

                if res.status_code == 200:
                    cred.record_success()
                    self.circuit_breaker.record_success()
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return LLMResponse(
                            success=False,
                            http_status=200,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Gemini returned no candidates (possible safety filter block).",
                            error_category="INVALID_RESPONSE",
                            finish_reason="NO_CANDIDATES",
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    candidate = candidates[0]
                    finish_reason = candidate.get("finishReason", "")
                    if finish_reason == "SAFETY":
                        return LLMResponse(
                            success=False,
                            http_status=200,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Gemini candidate blocked by safety filter.",
                            error_category="SAFETY_BLOCKED",
                            finish_reason="SAFETY",
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    content_parts = candidate.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in content_parts)
                    if not text.strip():
                        return LLMResponse(
                            success=False,
                            http_status=200,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message=f"Gemini returned empty text (finishReason: {finish_reason or 'UNKNOWN'}).",
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
                    err_msg = f"HTTP 404 Model Not Found ({self._model}): {err_detail or 'The requested model does not exist on Gemini'}"
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
                        # Multi-key pool: put current key on cooldown and switch immediately
                        cred.set_cooldown(duration_seconds=60.0, reason="HTTP 429 RESOURCE_EXHAUSTED")
                    else:
                        # Single-key pool or no alternative keys: do NOT lockout key before exhausting retries!
                        if retries > self._max_retries:
                            cred.set_cooldown(duration_seconds=60.0, reason="HTTP 429 RESOURCE_EXHAUSTED (exhausted retries)")
                            logger.warning("Gemini rate limit exceeded after max retries.")
                            return LLMResponse(
                                success=False,
                                http_status=429,
                                is_transient=True,
                                provider_name=self.provider_name,
                                model_name=self._model,
                                error_message="Gemini rate limit exceeded (HTTP 429).",
                                error_category="RATE_LIMIT",
                                is_rate_limited=True,
                                used_key_masked=cred.masked_key,
                                retries_attempted=retries,
                            )
                        # Back off with jitter and respect Retry-After
                        delay = min(retry_after if retry_after > 0 else (backoff + random.uniform(0.1, 0.4)), 3.0)
                        time.sleep(delay)
                        backoff *= 2
                        continue

                    if retries > self._max_retries:
                        logger.warning("Gemini rate limit exceeded after max retries.")
                        return LLMResponse(
                            success=False,
                            http_status=429,
                            is_transient=True,
                            provider_name=self.provider_name,
                            model_name=self._model,
                            error_message="Gemini rate limit exceeded (HTTP 429).",
                            error_category="RATE_LIMIT",
                            is_rate_limited=True,
                            used_key_masked=cred.masked_key,
                            retries_attempted=retries,
                        )
                    continue

                elif res.status_code in (401, 403):
                    cred.mark_disabled(reason=f"HTTP {res.status_code} Auth Failure")
                    err_msg = f"HTTP {res.status_code}: Check GEMINI_API_KEY or permissions ({self._model})."
                    logger.warning(f"Gemini API auth error: {err_msg}")
                    self.circuit_breaker.record_failure(reason=err_msg, is_permanent=True)
                    # If other keys available in pool, attempt next key
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
                    logger.warning(f"Gemini bad request: {err_msg}")
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
                            error_message=f"Gemini server error HTTP {res.status_code}: {err_detail or 'The model is overloaded.'}",
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
                            error_message=f"Gemini server error HTTP {res.status_code}: {res.text[:200]}",
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
                        error_message=f"Gemini request timed out after {self._timeout}s.",
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
                    error_message=f"Gemini request failed: {type(exc).__name__}",
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
            error_message="Gemini request exhausted retries.",
            error_category="RATE_LIMIT",
            used_key_masked=last_cred.masked_key if last_cred else "",
            retries_attempted=retries,
        )

    def generate_text(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        return self._call_api(prompt, system_instruction=system_instruction)

    def generate_json(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        resp = self._call_api(prompt, system_instruction=system_instruction, response_mime_type="application/json")
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
