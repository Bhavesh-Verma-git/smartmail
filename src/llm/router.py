"""
src/llm/router.py
-----------------
LLM fallback router with privacy safeguards and grounded evidence validation.

Routing Priority:
1. Gemini (Primary)
2. Groq (Secondary Fallback)
3. Offline / Deterministic Fallback (if all LLMs unconfigured or failing)

Privacy Enforcements:
- Default: CV and candidate details are NEVER included in prompts unless
  SMARTMAIL_SHARE_CV_WITH_EXTERNAL_LLM=true.
- Recipient email address is NEVER included in prompts.
- All extracted company facts are treated as untrusted and must be grounded
  in retrieved EvidenceItems before being accepted.
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import is_cv_sharing_allowed
from src.llm.base import LLMProvider, LLMResponse
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Coordinates primary (Gemini) and fallback (Groq) providers with privacy controls.
    """

    def __init__(
        self,
        primary_provider: LLMProvider | None = None,
        fallback_provider: LLMProvider | None = None,
        allow_cv_sharing: bool | None = None,
    ) -> None:
        self.primary = primary_provider or GeminiProvider()
        self.fallback = fallback_provider or GroqProvider()
        self.allow_cv_sharing = is_cv_sharing_allowed() if allow_cv_sharing is None else allow_cv_sharing

    @property
    def has_any_llm(self) -> bool:
        return self.primary.is_configured or self.fallback.is_configured

    def get_router_status(self) -> dict[str, Any]:
        """Return safe overview of router configuration and key pool states without leaking secrets."""
        return {
            "has_any_llm": self.has_any_llm,
            "primary": {
                "name": self.primary.provider_name,
                "configured": self.primary.is_configured,
                "pool": self.primary.get_pool_status(),
            },
            "fallback": {
                "name": self.fallback.provider_name,
                "configured": self.fallback.is_configured,
                "pool": self.fallback.get_pool_status(),
            },
            "allow_cv_sharing": self.allow_cv_sharing,
        }

    def check_all_providers_health(self) -> dict[str, Any]:
        """Safe explicit health check across all configured providers."""
        return {
            "primary": self.primary.check_health(),
            "fallback": self.fallback.check_health(),
        }

    def generate_json_with_fallback(
        self,
        prompt: str,
        system_instruction: str = "",
    ) -> LLMResponse:
        """
        Execute structured JSON generation with Gemini -> Groq fallback.
        Preserves error context from both primary and fallback attempts.
        """
        primary_resp: LLMResponse | None = None
        fallback_resp: LLMResponse | None = None
        primary_failure_reason = ""

        # Try Primary (Gemini)
        if self.primary.is_configured:
            resp = self.primary.generate_json(prompt, system_instruction=system_instruction)
            if resp.success:
                return resp
            primary_resp = resp
            primary_failure_reason = f"Primary {self.primary.provider_name} failed: {resp.error_message}"
            logger.warning(f"{primary_failure_reason}. Trying fallback...")

        # Try Fallback (Groq)
        if self.fallback.is_configured:
            resp = self.fallback.generate_json(prompt, system_instruction=system_instruction)
            if resp.success:
                if primary_failure_reason:
                    resp.fallback_reason = primary_failure_reason
                    resp.primary_error = primary_resp.error_message if primary_resp else ""
                return resp
            fallback_resp = resp
            logger.warning(f"Fallback LLM ({self.fallback.provider_name}) failed: {resp.error_message}.")

        err_parts = []
        if primary_resp:
            err_parts.append(f"Primary ({self.primary.provider_name}): {primary_resp.error_message}")
        if fallback_resp:
            err_parts.append(f"Fallback ({self.fallback.provider_name}): {fallback_resp.error_message}")
        combined_error = "; ".join(err_parts) if err_parts else "No configured LLM succeeded."

        return LLMResponse(
            success=False,
            provider_name="none",
            error_message=f"No configured LLM succeeded. {combined_error}",
            fallback_reason=primary_failure_reason,
            primary_error=primary_resp.error_message if primary_resp else "",
            fallback_error=fallback_resp.error_message if fallback_resp else "",
            http_status=primary_resp.http_status if primary_resp and primary_resp.http_status else (fallback_resp.http_status if fallback_resp else 0),
            is_transient=bool((primary_resp and primary_resp.is_transient) or (fallback_resp and fallback_resp.is_transient)),
        )

    def generate_text_with_fallback(
        self,
        prompt: str,
        system_instruction: str = "",
    ) -> LLMResponse:
        """
        Execute plain text generation with Gemini -> Groq fallback.
        Preserves error context from both primary and fallback attempts.
        """
        primary_resp: LLMResponse | None = None
        fallback_resp: LLMResponse | None = None
        primary_failure_reason = ""

        # Try Primary (Gemini)
        if self.primary.is_configured:
            resp = self.primary.generate_text(prompt, system_instruction=system_instruction)
            if resp.success:
                return resp
            primary_resp = resp
            primary_failure_reason = f"Primary {self.primary.provider_name} failed: {resp.error_message}"
            logger.warning(f"{primary_failure_reason}. Trying fallback...")

        # Try Fallback (Groq)
        if self.fallback.is_configured:
            resp = self.fallback.generate_text(prompt, system_instruction=system_instruction)
            if resp.success:
                if primary_failure_reason:
                    resp.fallback_reason = primary_failure_reason
                    resp.primary_error = primary_resp.error_message if primary_resp else ""
                return resp
            fallback_resp = resp
            logger.warning(f"Fallback LLM ({self.fallback.provider_name}) failed: {resp.error_message}.")

        err_parts = []
        if primary_resp:
            err_parts.append(f"Primary ({self.primary.provider_name}): {primary_resp.error_message}")
        if fallback_resp:
            err_parts.append(f"Fallback ({self.fallback.provider_name}): {fallback_resp.error_message}")
        combined_error = "; ".join(err_parts) if err_parts else "No configured LLM succeeded."

        return LLMResponse(
            success=False,
            provider_name="none",
            error_message=f"No configured LLM succeeded. {combined_error}",
            fallback_reason=primary_failure_reason,
            primary_error=primary_resp.error_message if primary_resp else "",
            fallback_error=fallback_resp.error_message if fallback_resp else "",
            http_status=primary_resp.http_status if primary_resp and primary_resp.http_status else (fallback_resp.http_status if fallback_resp else 0),
            is_transient=bool((primary_resp and primary_resp.is_transient) or (fallback_resp and fallback_resp.is_transient)),
        )

    def extract_company_ai_evidence(
        self,
        company_name: str,
        evidence_texts: list[str],
    ) -> dict[str, Any] | None:
        """
        Extract structured technical focus areas and verified products from public web text.
        PRIVACY GUARANTEE: Does NOT receive any candidate information or recipient emails.
        """
        if not self.has_any_llm or not evidence_texts:
            return None

        joined_evidence = "\n\n---\n\n".join(t[:1500] for t in evidence_texts[:4])
        system_prompt = (
            "You are a strict technical analyst. Analyze the provided company evidence text. "
            "Output ONLY valid JSON with keys: 'is_ai_relevant' (bool), 'technical_areas' (list of strings), "
            "'evidence_summary' (short string citing facts from text), and 'grounding_confidence' ('HIGH', 'MEDIUM', 'LOW'). "
            "Never invent products or facts not directly supported in the text."
        )

        user_prompt = (
            f"Company Name: {company_name}\n\n"
            f"Evidence Excerpts:\n{joined_evidence}\n\n"
            "Analyze the text and extract verifiable AI/ML technology focus."
        )

        resp = self.generate_json_with_fallback(prompt=user_prompt, system_instruction=system_prompt)
        if resp.success and resp.parsed_json:
            return resp.parsed_json
        return None
