"""
src/llm/base.py
---------------
Minimal abstract base class and common result models for LLM providers.
Designed to be completely decoupled from any specific vendor (Gemini, Groq, local fallback).
"""
from __future__ import annotations

import random
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
    error_category: str = ""  # "RATE_LIMIT", "AUTH_FAILURE", "TIMEOUT", "SERVER_ERROR", "INVALID_REQUEST", "PARSE_ERROR", ""
    is_rate_limited: bool = False
    is_auth_error: bool = False
    used_key_masked: str = ""
    retries_attempted: int = 0
    fallback_reason: str = ""


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

