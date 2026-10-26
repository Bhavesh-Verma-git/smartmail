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

import json
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
    gemini_key_count: int = 0
    groq_key_count: int = 0
    gemini_masked_keys: tuple[str, ...] = ()
    groq_masked_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tavily_configured": self.tavily_configured,
            "tavily_model_or_desc": self.tavily_model_or_desc,
            "gemini_configured": self.gemini_configured,
            "gemini_model": self.gemini_model,
            "groq_configured": self.groq_configured,
            "groq_model": self.groq_model,
            "share_cv_with_llm": self.share_cv_with_llm,
            "gemini_key_count": self.gemini_key_count,
            "groq_key_count": self.groq_key_count,
            "gemini_masked_keys": list(self.gemini_masked_keys),
            "groq_masked_keys": list(self.groq_masked_keys),
        }


def mask_secret(secret: str) -> str:
    """Safely mask an API key or token for audit logs and UI displays."""
    if not secret:
        return ""
    s = secret.strip()
    if len(s) <= 8:
        return "..." + s[-2:]
    return f"{s[:4]}...{s[-4:]}"


def get_tavily_api_key() -> str:
    """Retrieve Tavily API key from environment."""
    return os.getenv("TAVILY_API_KEY", "").strip()


def _parse_key_entries(raw: str) -> list[str]:
    """Parse comma, newline, or JSON list formatted API key strings safely."""
    if not raw:
        return []
    cleaned = raw.strip().strip("\"'")
    if not cleaned:
        return []
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(k).strip().strip("\"'") for k in parsed if str(k).strip().strip("\"'")]
        except Exception:
            pass
    # Split by comma or newline, strip each entry
    parts: list[str] = []
    for item in cleaned.replace("\n", ",").split(","):
        k = item.strip().strip("\"'")
        if k:
            parts.append(k)
    return parts


def get_gemini_api_key() -> str:
    """Retrieve primary Gemini API key from environment."""
    keys = get_gemini_api_keys()
    return keys[0] if keys else ""


def get_gemini_api_keys() -> list[str]:
    """
    Retrieve configured Gemini API keys supporting single-key, comma-separated,
    JSON list, and numbered environment variables (GEMINI_API_KEY_1..5).
    Handles whitespace, quotes, and empty entries defensively.
    Returns deduplicated list preserving order.
    """
    keys: list[str] = []
    keys.extend(_parse_key_entries(os.getenv("GEMINI_API_KEY", "")))
    keys.extend(_parse_key_entries(os.getenv("GEMINI_API_KEYS", "")))

    for i in range(1, 10):
        keys.extend(_parse_key_entries(os.getenv(f"GEMINI_API_KEY_{i}", "")))

    seen: set[str] = set()
    deduped: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


def get_groq_api_key() -> str:
    """Retrieve primary Groq API key from environment."""
    keys = get_groq_api_keys()
    return keys[0] if keys else ""


def get_groq_api_keys() -> list[str]:
    """
    Retrieve configured Groq API keys supporting single-key, comma-separated,
    JSON list, and numbered environment variables (GROQ_API_KEY_1..5).
    Handles whitespace, quotes, and empty entries defensively.
    Returns deduplicated list preserving order.
    """
    keys: list[str] = []
    keys.extend(_parse_key_entries(os.getenv("GROQ_API_KEY", "")))
    keys.extend(_parse_key_entries(os.getenv("GROQ_API_KEYS", "")))

    for i in range(1, 10):
        keys.extend(_parse_key_entries(os.getenv(f"GROQ_API_KEY_{i}", "")))

    seen: set[str] = set()
    deduped: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


def get_gemini_model() -> str:
    """Configurable Gemini model identifier (defaults to gemini-3.8-flash)."""
    return os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"


def get_groq_model() -> str:
    """
    Configurable Groq model identifier (defaults to openai/gpt-oss-120b).
    Preserves exact configured model name from environment without silent mutation.
    """
    raw = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    if not raw or raw.lower() in ("none", "null", ""):
        return "openai/gpt-oss-120b"
    return raw


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


def get_authoritative_candidate_email() -> str:
    """Explicitly configured authoritative candidate email, if any."""
    return (
        os.getenv("SMARTMAIL_AUTHORITATIVE_EMAIL", "").strip()
        or os.getenv("SMARTMAIL_CANDIDATE_EMAIL", "").strip()
    )


def get_authoritative_candidate_phone() -> str:
    """Explicitly configured authoritative candidate phone, if any."""
    return (
        os.getenv("SMARTMAIL_AUTHORITATIVE_PHONE", "").strip()
        or os.getenv("SMARTMAIL_CANDIDATE_PHONE", "").strip()
    )


def get_provider_status() -> ProviderStatus:
    """
    Check configuration state of all providers.
    NEVER returns or logs raw keys.
    """
    tavily_key = get_tavily_api_key()
    gemini_keys = get_gemini_api_keys()
    groq_keys = get_groq_api_keys()

    return ProviderStatus(
        tavily_configured=bool(tavily_key),
        tavily_model_or_desc="Tavily Search API (REST)",
        gemini_configured=bool(gemini_keys),
        gemini_model=get_gemini_model(),
        groq_configured=bool(groq_keys),
        groq_model=get_groq_model(),
        share_cv_with_llm=is_cv_sharing_allowed(),
        gemini_key_count=len(gemini_keys),
        groq_key_count=len(groq_keys),
        gemini_masked_keys=tuple(mask_secret(k) for k in gemini_keys),
        groq_masked_keys=tuple(mask_secret(k) for k in groq_keys),
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
