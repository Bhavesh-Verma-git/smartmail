"""
src/llm/base.py
---------------
Minimal abstract base class and common result models for LLM providers.
Designed to be completely decoupled from any specific vendor (Gemini, Groq, local fallback).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Safe structured container for LLM generation results."""
    success: bool
    text: str = ""
    parsed_json: dict[str, Any] | None = None
    provider_name: str = ""
    model_name: str = ""
    error_message: str = ""
    is_rate_limited: bool = False
    is_auth_error: bool = False


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

    @abstractmethod
    def generate_text(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        """Generate unstructured plain text response."""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, system_instruction: str = "") -> LLMResponse:
        """Generate and parse structured JSON response."""
        pass
