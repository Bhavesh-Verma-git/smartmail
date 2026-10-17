"""
src/config.py
-------------
Centralized configuration and environment loader for SmartMail.

Strict Rules:
- Never log, print, return, or expose raw API keys or secrets in logs, exceptions, or UI.
- Never overwrite an existing .env file.
- Treat all external providers as optional: missing keys gracefully default to False/offline.
- Maintain configurable model names and timeouts for free-tier resilience.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Compute repository root directory relative to this file (src/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_environment(override: bool = False) -> None:
    """
    Load environment variables from .env located at project root,
    with fallback to current working directory.
    Never raises an error if .env is missing.
    """
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=override)
    else:
        # Fallback to local .env in current working directory if different
        cwd_env = Path(".env")
        if cwd_env.exists():
            load_dotenv(dotenv_path=cwd_env, override=override)
        else:
            load_dotenv(override=override)

# Load environment on import
load_environment(override=False)


@dataclass(frozen=True)
class ProviderStatus:
    """Safe status object that indicates provider availability without exposing keys."""
    tavily_configured: bool
    tavily_model_or_desc: str
    gemini_configured: bool
    gemini_model: str
    groq_configured: bool
    groq_model: str
    share_cv_with_llm: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tavily_configured": self.tavily_configured,
            "tavily_model_or_desc": self.tavily_model_or_desc,
            "gemini_configured": self.gemini_configured,
            "gemini_model": self.gemini_model,
            "groq_configured": self.groq_configured,
            "groq_model": self.groq_model,
            "share_cv_with_llm": self.share_cv_with_llm,
        }


def get_tavily_api_key() -> str:
    """Retrieve Tavily API key from environment."""
    return os.getenv("TAVILY_API_KEY", "").strip()


def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment."""
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_groq_api_key() -> str:
    """Retrieve Groq API key from environment."""
    return os.getenv("GROQ_API_KEY", "").strip()


def get_gemini_model() -> str:
    """Configurable Gemini model identifier (defaults to gemini-2.0-flash)."""
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"


def get_groq_model() -> str:
    """Configurable Groq model identifier (defaults to llama-3.3-70b-versatile)."""
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"


def get_tavily_timeout() -> int:
    try:
        return int(os.getenv("TAVILY_TIMEOUT_SECONDS", "10"))
    except ValueError:
        return 10


def get_gemini_timeout() -> int:
    try:
        return int(os.getenv("GEMINI_TIMEOUT_SECONDS", "15"))
    except ValueError:
        return 15


def get_groq_timeout() -> int:
    try:
        return int(os.getenv("GROQ_TIMEOUT_SECONDS", "15"))
    except ValueError:
        return 15


def is_cv_sharing_allowed() -> bool:
    """
    CRITICAL PRIVACY SAFEGUARD:
    Returns True ONLY if user explicitly enabled SMARTMAIL_SHARE_CV_WITH_EXTERNAL_LLM=true.
    Defaults to False.
    """
    val = os.getenv("SMARTMAIL_SHARE_CV_WITH_EXTERNAL_LLM", "false").strip().lower()
    return val in ("true", "1", "yes")


def get_provider_status() -> ProviderStatus:
    """
    Check configuration state of all providers.
    NEVER returns or logs raw keys.
    """
    tavily_key = get_tavily_api_key()
    gemini_key = get_gemini_api_key()
    groq_key = get_groq_api_key()

    return ProviderStatus(
        tavily_configured=bool(tavily_key),
        tavily_model_or_desc="Tavily Search API (REST)",
        gemini_configured=bool(gemini_key),
        gemini_model=get_gemini_model(),
        groq_configured=bool(groq_key),
        groq_model=get_groq_model(),
        share_cv_with_llm=is_cv_sharing_allowed(),
    )


def format_config_status_report() -> str:
    """Produce clean, human-readable status for CLI without leaking secrets."""
    status = get_provider_status()
    tavily_text = "CONFIGURED" if status.tavily_configured else "NOT CONFIGURED"
    tavily_note = "" if status.tavily_configured else " (fallback: direct HTTP)"

    gemini_text = "CONFIGURED" if status.gemini_configured else "NOT CONFIGURED"
    gemini_note = f" [Model: {status.gemini_model}]" if status.gemini_configured else " (fallback: deterministic)"

    groq_text = "CONFIGURED" if status.groq_configured else "NOT CONFIGURED"
    groq_note = f" [Model: {status.groq_model}]" if status.groq_configured else ""

    cv_text = "ALLOWED (User opt-in)" if status.share_cv_with_llm else "DISABLED (Candidate data stays local)"

    lines = [
        "SmartMail External Services Configuration Status:",
        "=" * 56,
        f"  Tavily Web Search:       [{tavily_text}]{tavily_note}",
        f"  Gemini Primary LLM:      [{gemini_text}]{gemini_note}",
        f"  Groq Fallback LLM:       [{groq_text}]{groq_note}",
        f"  Local HTTP Scraper:      [READY] (Built-in BeautifulSoup fallback)",
        "-" * 56,
        f"  External CV Sharing:     [{cv_text}]",
        "=" * 56,
    ]
    return "\n".join(lines)
