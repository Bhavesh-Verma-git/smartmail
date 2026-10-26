"""
src/llm/base.py
---------------
Minimal abstract base class and common result models for LLM providers.
Designed to be completely decoupled from any specific vendor (Gemini, Groq, local fallback).
"""
from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class KeyCredential:
    """Represents a single API key credential with rate-limit and health state."""
    api_key: str
    masked_key: str
    cooldown_until: float = 0.0
    is_disabled: bool = False
    last_error: str = ""
    failure_count: int = 0
    success_count: int = 0

    def is_available(self, now: float | None = None) -> bool:
        if self.is_disabled:
            return False
        if now is None:
            now = time.time()
        return now >= self.cooldown_until

    def set_cooldown(self, duration_seconds: float = 60.0, reason: str = "", jitter: float = 2.0) -> None:
        actual_cooldown = duration_seconds + random.uniform(0.0, jitter)
        self.cooldown_until = time.time() + actual_cooldown
        self.last_error = reason
        self.failure_count += 1

    def mark_disabled(self, reason: str = "") -> None:
        self.is_disabled = True
        self.last_error = reason
        self.failure_count += 1

    def record_success(self) -> None:
        self.success_count += 1
        self.failure_count = 0
        self.last_error = ""


class CredentialPool:
    """Manages rotation, cooldowns, and error tracking across multiple API keys."""

    def __init__(self, keys: list[str], mask_fn: Callable[[str], str] | None = None) -> None:
        if mask_fn is None:
            from src.config import mask_secret
            mask_fn = mask_secret
        self._credentials = [
            KeyCredential(api_key=k.strip(), masked_key=mask_fn(k.strip()))
            for k in keys
            if k and k.strip()
        ]
        self._current_idx = 0

    @property
    def total_count(self) -> int:
        return len(self._credentials)

    @property
    def has_keys(self) -> bool:
        return len(self._credentials) > 0

    @property
    def credentials(self) -> list[KeyCredential]:
        return list(self._credentials)

    def get_next_available(self) -> KeyCredential | None:
        if not self._credentials:
            return None
        now = time.time()
        n = len(self._credentials)
        for offset in range(n):
            idx = (self._current_idx + offset) % n
            cred = self._credentials[idx]
            if cred.is_available(now):
                self._current_idx = (idx + 1) % n
                return cred
        return None

    def get_status(self) -> list[dict[str, Any]]:
        now = time.time()
        return [
            {
                "masked_key": cred.masked_key,
                "is_disabled": cred.is_disabled,
                "in_cooldown": now < cred.cooldown_until,
                "cooldown_remaining": max(0.0, round(cred.cooldown_until - now, 1)),
                "failure_count": cred.failure_count,
                "success_count": cred.success_count,
                "last_error": cred.last_error,
            }
            for cred in self._credentials
        ]


@dataclass
class LLMResponse:
    """Safe structured container for LLM generation results."""
    success: bool
    text: str = ""
    parsed_json: dict[str, Any] | None = None
    provider_name: str = ""
    model_name: str = ""
    error_message: str = ""
    error_category: str = ""  # "RATE_LIMIT", "AUTH_FAILURE", "TIMEOUT", "SERVER_ERROR", "INVALID_REQUEST", "PARSE_ERROR", "EMPTY_RESPONSE", "TRUNCATED_RESPONSE", "SAFETY_BLOCKED"
    is_rate_limited: bool = False
    is_auth_error: bool = False
    used_key_masked: str = ""
    retries_attempted: int = 0
    fallback_reason: str = ""
    http_status: int = 0
    is_transient: bool = False
    primary_error: str = ""
    fallback_error: str = ""
    finish_reason: str = ""
    circuit_open: bool = False


class CircuitBreaker:
    """
    Thread-safe circuit breaker preventing cascading failures, repeated permanent errors,
    and quota thrashing across concurrent batch workers.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        provider_name: str = "",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.provider_name = provider_name
        self._lock = threading.Lock()
        self._state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._consecutive_failures: int = 0
        self._last_failure_time: float = 0.0
        self._trip_reason: str = ""
        self._is_permanent: bool = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "OPEN":
                if self._is_permanent:
                    return "OPEN"
                if time.time() - self._last_failure_time >= self.cooldown_seconds:
                    self._state = "HALF_OPEN"
                    return "HALF_OPEN"
            return self._state

    def allow_request(self) -> tuple[bool, str]:
        """Check if request is allowed. Returns (allowed, reason_if_blocked)."""
        st = self.state
        with self._lock:
            if st == "OPEN":
                perm_suffix = " (Permanent configuration error; will not retry)" if self._is_permanent else f" (Cooldown remaining: {max(0.0, round(self.cooldown_seconds - (time.time() - self._last_failure_time), 1))}s)"
                return False, f"Circuit for {self.provider_name} is OPEN: {self._trip_reason}{perm_suffix}"
            return True, ""

    def record_success(self) -> None:
        with self._lock:
            self._state = "CLOSED"
            self._consecutive_failures = 0
            self._trip_reason = ""
            self._is_permanent = False

    def record_failure(
        self,
        reason: str,
        is_permanent: bool = False,
        cooldown_override: float | None = None,
    ) -> None:
        with self._lock:
            self._last_failure_time = time.time()
            if is_permanent:
                self._state = "OPEN"
                self._is_permanent = True
                self._trip_reason = reason
                return

            self._consecutive_failures += 1
            if cooldown_override and cooldown_override > 0:
                self.cooldown_seconds = min(cooldown_override, 300.0)

            if self._consecutive_failures >= self.failure_threshold or (cooldown_override and cooldown_override > 10):
                self._state = "OPEN"
                self._trip_reason = reason

    def reset(self) -> None:
        with self._lock:
            self._state = "CLOSED"
            self._consecutive_failures = 0
            self._trip_reason = ""
            self._is_permanent = False


class SharedRateLimiter:
    """Thread-safe token bucket rate limiter to space requests across batch workers."""

    def __init__(self, min_interval_seconds: float = 0.5) -> None:
        self.min_interval = min_interval_seconds
        self._last_call_time = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            if self.min_interval <= 0:
                self._last_call_time = time.time()
                return
            now = time.time()
            elapsed = now - self._last_call_time
            if 0 < elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call_time = time.time()

    def reset(self) -> None:
        with self._lock:
            self._last_call_time = 0.0


_circuit_breakers: dict[str, CircuitBreaker] = {}
_circuit_lock = threading.Lock()


def get_circuit_breaker(provider_name: str, failure_threshold: int = 3, cooldown_seconds: float = 30.0) -> CircuitBreaker:
    """Retrieve or create the shared CircuitBreaker singleton for a provider."""
    with _circuit_lock:
        if provider_name not in _circuit_breakers:
            _circuit_breakers[provider_name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                cooldown_seconds=cooldown_seconds,
                provider_name=provider_name,
            )
        return _circuit_breakers[provider_name]


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breaker singletons to CLOSED state (useful for tests and manual admin reset)."""
    with _circuit_lock:
        for cb in _circuit_breakers.values():
            cb.reset()


_rate_limiters: dict[str, SharedRateLimiter] = {}
_rate_lock = threading.Lock()
_default_rate_limit_interval: float = 0.5


def get_rate_limiter(provider_name: str, min_interval_seconds: float | None = None) -> SharedRateLimiter:
    """Retrieve or create the shared SharedRateLimiter singleton for a provider."""
    with _rate_lock:
        if provider_name not in _rate_limiters:
            interval = min_interval_seconds if min_interval_seconds is not None else _default_rate_limit_interval
            _rate_limiters[provider_name] = SharedRateLimiter(min_interval_seconds=interval)
        return _rate_limiters[provider_name]


def reset_all_rate_limiters(min_interval_seconds: float | None = None) -> None:
    """Reset all rate limiter singletons (useful for tests)."""
    global _default_rate_limit_interval
    with _rate_lock:
        if min_interval_seconds is not None:
            _default_rate_limit_interval = min_interval_seconds
        for rl in _rate_limiters.values():
            rl.reset()
            if min_interval_seconds is not None:
                rl.min_interval = min_interval_seconds



def parse_retry_after(header_val: str | None, default_delay: float = 1.0, max_delay: float = 5.0) -> float:
    """Safely parse Retry-After HTTP header value into seconds bounded to max_delay."""
    if not header_val:
        return default_delay
    try:
        val = float(str(header_val).strip())
        return max(0.1, min(val, max_delay))
    except (ValueError, TypeError):
        return default_delay


def safe_parse_llm_json(
    raw_text: str,
    finish_reason: str = "",
) -> tuple[dict[str, Any] | None, str, str]:
    """
    Safely extract and parse JSON from LLM output.
    Returns:
        (parsed_dict, error_category, error_message)
    """
    if not raw_text or not raw_text.strip():
        if finish_reason in ("SAFETY", "RECITATION", "BLOCKLIST"):
            return None, "SAFETY_BLOCKED", f"LLM content generation blocked by safety policy (finishReason: {finish_reason})."
        return None, "EMPTY_RESPONSE", "LLM returned an empty response."

    # If truncated by token limits
    if finish_reason in ("MAX_TOKENS", "length"):
        return None, "TRUNCATED_RESPONSE", "LLM output was truncated due to output token limit."

    text = raw_text.strip()

    # 1. First attempt: Direct JSON parse
    import json
    import re

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, "", ""
        return {"data": parsed}, "", ""
    except json.JSONDecodeError:
        pass

    # 2. Second attempt: Markdown code fence extraction (```json ... ```) with or without preamble
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if match:
        fenced_content = match.group(1).strip()
        try:
            parsed = json.loads(fenced_content)
            if isinstance(parsed, dict):
                return parsed, "", ""
            return {"data": parsed}, "", ""
        except json.JSONDecodeError:
            pass

    # 3. Third attempt: Substring between first '{' and last '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate_json = text[first_brace : last_brace + 1].strip()
        try:
            parsed = json.loads(candidate_json)
            if isinstance(parsed, dict):
                return parsed, "", ""
            return {"data": parsed}, "", ""
        except json.JSONDecodeError as exc:
            return None, "PARSE_ERROR", f"Malformed JSON: {exc}"

    # If it starts with '{' but has no closing '}', it is clearly truncated
    if first_brace != -1 and last_brace <= first_brace:
        return None, "TRUNCATED_RESPONSE", "LLM output was truncated before closing JSON structure."

    return None, "PARSE_ERROR", f"Could not parse valid JSON structure: {raw_text[:100]}..."


class LLMProvider(ABC):
    """Abstract interface for LLM calls."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'groq')."""
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True if the provider has necessary credentials."""
        pass

    def get_pool_status(self) -> list[dict[str, Any]]:
        """Safe status list for configured credentials without revealing raw keys."""
        return []

    def check_health(self) -> dict[str, Any]:
        """Safe explicit health check for the provider without exposing secrets."""
        return {
            "provider": self.provider_name,
            "configured": self.is_configured,
            "status": "HEALTHY" if self.is_configured else "UNCONFIGURED",
        }

    @abstractmethod
    def generate_text(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        """Generate unstructured plain text response."""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        """Generate and parse structured JSON response."""
        pass

