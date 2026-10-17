"""
tests/test_external_providers.py
---------------------------------
Automated test suite for external service integrations (Tavily, Gemini, Groq) and config management.

All external HTTP calls are strictly mocked with unittest.mock or requests_mock patterns:
ZERO real network requests are executed during tests.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.config import (
    ProviderStatus,
    format_config_status_report,
    get_gemini_api_key,
    get_gemini_model,
    get_provider_status,
    get_tavily_api_key,
    is_cv_sharing_allowed,
)
from src.llm.base import LLMResponse
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.router import LLMRouter
from src.research.models import EvidenceItem, RETRIEVAL_BLOCKED, RETRIEVAL_ERROR, RETRIEVAL_OK
from src.research.providers import HttpPageProvider, get_default_research_provider
from src.research.tavily_provider import TavilySearchProvider


# ===========================================================================
# 1. Configuration & Key Secrecy Tests
# ===========================================================================

class TestConfigurationAndKeySecrecy:
    """Ensure keys are never exposed in reports, strings, or logs."""

    def test_default_state_unconfigured_without_env(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("SMARTMAIL_SHARE_CV_WITH_EXTERNAL_LLM", raising=False)

        status = get_provider_status()
        assert status.tavily_configured is False
        assert status.gemini_configured is False
        assert status.groq_configured is False
        assert status.share_cv_with_llm is False

        report = format_config_status_report()
        assert "[NOT CONFIGURED]" in report
        assert "DISABLED" in report
        assert "Candidate data stays local" in report

    def test_report_never_leaks_actual_api_keys(self, monkeypatch):
        secret_tavily = "tvly-secret-123456789"
        secret_gemini = "AIzaSy-super-secret-gemini-key"
        secret_groq = "gsk_super_secret_groq_key"

        monkeypatch.setenv("TAVILY_API_KEY", secret_tavily)
        monkeypatch.setenv("GEMINI_API_KEY", secret_gemini)
        monkeypatch.setenv("GROQ_API_KEY", secret_groq)

        status = get_provider_status()
        assert status.tavily_configured is True
        assert status.gemini_configured is True
        assert status.groq_configured is True

        report = format_config_status_report()
        assert "[CONFIGURED]" in report
        # ABSOLUTE RULE: Raw key secrets MUST NOT appear in report string
        assert secret_tavily not in report
        assert secret_gemini not in report
        assert secret_groq not in report

    def test_load_environment_finds_project_root_env_from_any_cwd(self, tmp_path, monkeypatch):
        """Verify .env loading succeeds even if current working directory is completely different."""
        from src.config import PROJECT_ROOT, load_environment, get_tavily_api_key

        # Create a mock .env file inside a temporary root
        fake_env = tmp_path / ".env"
        fake_env.write_text("TAVILY_API_KEY=tvly-test-project-root-key\n", encoding="utf-8")

        # Patch PROJECT_ROOT to point to tmp_path
        with patch("src.config.PROJECT_ROOT", tmp_path):
            monkeypatch.delenv("TAVILY_API_KEY", raising=False)
            load_environment(override=True)
            assert get_tavily_api_key() == "tvly-test-project-root-key"

            # Check that config status report does not leak the key
            report = format_config_status_report()
            assert "[CONFIGURED]" in report
            assert "tvly-test-project-root-key" not in report



# ===========================================================================
# 2. Tavily Search Provider Mock Tests
# ===========================================================================

class TestTavilySearchProvider:
    """Test Tavily search logic, domain caching, retries, and errors."""

    def test_unconfigured_tavily_returns_empty_and_gracefully_skips(self):
        provider = TavilySearchProvider(api_key="")
        assert not provider.is_configured
        results = provider.search_company(company_name="Acme AI", domain="acme.ai")
        assert results == []

    @patch("src.research.tavily_provider.requests.post")
    def test_successful_tavily_search_returns_grounded_evidence(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "url": "https://acme.ai/about",
                    "title": "About Acme AI",
                    "content": "Acme AI develops autonomous multi-agent systems and retrieval augmented generation pipelines.",
                }
            ]
        }
        mock_post.return_value = mock_resp

        provider = TavilySearchProvider(api_key="tvly-mock-key")
        items = provider.search_company(company_name="Acme AI", domain="acme.ai")

        assert len(items) == 1
        item = items[0]
        assert item.source_type == "tavily_search"
        assert item.source_url == "https://acme.ai/about"
        assert "multi-agent systems" in item.extracted_text
        assert item.retrieval_status == RETRIEVAL_OK

        # Verify caching: second call uses cache without second HTTP request
        items_second = provider.search_company(company_name="Acme AI", domain="acme.ai")
        assert len(items_second) == 1
        assert mock_post.call_count == 1  # only called once

    @patch("src.research.tavily_provider.requests.post")
    def test_tavily_rate_limit_retry_and_backoff(self, mock_post):
        resp_429 = MagicMock()
        resp_429.status_code = 429

        mock_post.return_value = resp_429

        provider = TavilySearchProvider(api_key="tvly-mock-key", max_retries=1, initial_backoff=0.01)
        items = provider.search_company(company_name="HighLoad Corp", domain="highload.example")

        assert len(items) == 1
        assert items[0].retrieval_status == RETRIEVAL_ERROR
        assert "429" in items[0].retrieval_note

    @patch("src.research.tavily_provider.requests.post")
    def test_tavily_auth_error_captured_as_blocked(self, mock_post):
        resp_401 = MagicMock()
        resp_401.status_code = 401
        mock_post.return_value = resp_401

        provider = TavilySearchProvider(api_key="tvly-bad-key", max_retries=0)
        items = provider.search_company(company_name="Secure Co")

        assert len(items) == 1
        assert items[0].retrieval_status == RETRIEVAL_BLOCKED


# ===========================================================================
# 3. Gemini LLM Provider Mock Tests
# ===========================================================================

class TestGeminiProvider:
    """Test Gemini REST client with mock JSON & text generation."""

    def test_unconfigured_gemini_returns_safe_failure(self):
        gemini = GeminiProvider(api_key="")
        assert not gemini.is_configured
        res = gemini.generate_text("Hello")
        assert not res.success
        assert "not configured" in res.error_message

    @patch("src.llm.gemini_provider.requests.post")
    def test_gemini_successful_json_generation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '```json\n{"is_ai_relevant": true, "technical_areas": ["Generative AI", "RAG"]}\n```'
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        gemini = GeminiProvider(api_key="mock-gemini-key")
        resp = gemini.generate_json("Analyze this company")

        assert resp.success is True
        assert resp.parsed_json == {"is_ai_relevant": True, "technical_areas": ["Generative AI", "RAG"]}
        assert resp.provider_name == "gemini"

    @patch("src.llm.gemini_provider.requests.post")
    def test_gemini_rate_limit_handled_with_backoff(self, mock_post):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        mock_post.return_value = resp_429

        gemini = GeminiProvider(api_key="mock-gemini-key", max_retries=1, initial_backoff=0.01)
        resp = gemini.generate_text("Prompt")

        assert resp.success is False
        assert resp.is_rate_limited is True


# ===========================================================================
# 4. LLM Router Fallback Chain Tests
# ===========================================================================

class TestLLMRouterFallback:
    """Test primary (Gemini) -> fallback (Groq) -> offline fallback chain."""

    def test_router_falls_back_to_groq_when_gemini_fails(self):
        mock_gemini = MagicMock(spec=GeminiProvider)
        mock_gemini.is_configured = True
        mock_gemini.provider_name = "gemini"
        mock_gemini.generate_json.return_value = LLMResponse(
            success=False,
            error_message="Gemini Quota Exceeded (429)",
            is_rate_limited=True,
        )

        mock_groq = MagicMock(spec=GroqProvider)
        mock_groq.is_configured = True
        mock_groq.provider_name = "groq"
        mock_groq.generate_json.return_value = LLMResponse(
            success=True,
            parsed_json={"is_ai_relevant": True, "technical_areas": ["Computer Vision"]},
            provider_name="groq",
        )

        router = LLMRouter(primary_provider=mock_gemini, fallback_provider=mock_groq)
        resp = router.generate_json_with_fallback("Analyze text")

        assert resp.success is True
        assert resp.provider_name == "groq"
        assert resp.parsed_json["technical_areas"] == ["Computer Vision"]

    def test_router_evidence_extraction_preserves_privacy_default(self):
        mock_gemini = MagicMock(spec=GeminiProvider)
        mock_gemini.is_configured = True
        mock_gemini.generate_json.return_value = LLMResponse(
            success=True,
            parsed_json={
                "is_ai_relevant": True,
                "technical_areas": ["RAG, Retrieval & Knowledge Systems"],
                "evidence_summary": "Company builds RAG search index.",
            },
        )

        router = LLMRouter(primary_provider=mock_gemini)
        # Call extract_company_ai_evidence
        res = router.extract_company_ai_evidence(
            company_name="VectorFlow",
            evidence_texts=["VectorFlow provides real-time vector search and RAG indexing."],
        )

        assert res is not None
        assert res["is_ai_relevant"] is True
        # Verify prompt passed to LLM only has company name and public excerpts, no candidate personal data
        call_args = mock_gemini.generate_json.call_args[0][0]
        assert "VectorFlow" in call_args
        assert "Bhavesh" not in call_args
        assert "7067419231" not in call_args
        assert "@gmail.com" not in call_args


# ===========================================================================
# 5. Default Factory Behavior
# ===========================================================================

class TestDefaultResearchProviderFactory:
    """Verify fallback to HttpPageProvider when TAVILY_API_KEY is not set."""

    def test_factory_returns_http_provider_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        provider = get_default_research_provider()
        assert isinstance(provider, HttpPageProvider)

    def test_factory_returns_tavily_provider_when_key_set(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        provider = get_default_research_provider()
        assert isinstance(provider, TavilySearchProvider)
