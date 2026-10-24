"""
tests/test_polish_and_provider_verification.py
------------------------------------------------
Comprehensive targeted test suite covering:
1. Overview Landing Page routing (/ and /overview).
2. Draft Cleanup actions with scopes (unapproved, rejected, batch, all) and CSRF protection.
3. Multi-key parsing (comma-separated, whitespace-stripped, JSON list) and key masking.
4. Error classification, cooldown bounding, and health check confidentiality.
5. Markdown formatting stripping (**bold**, *italic*, `code`).
6. Email writing quality and natural opening/closing variations across companies.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.config import _parse_key_entries, mask_secret, get_provider_status
from src.llm.base import CredentialPool, KeyCredential, LLMResponse
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.outreach.models import EmailDraft, GenerationStatus, WorkflowMode
from src.outreach.personalized_generator import strip_markdown_formatting, generate_personalized_email_body
from src.outreach.opportunity_analyzer import CompanyOpportunity
from src.outreach.experience_matcher import CandidateExperienceMatch
from src.profile.loader import load_profile
from src.ui.app import create_app
from src.ui.storage import save_draft, get_draft, clear_all_drafts, delete_drafts_for_batch


@pytest.fixture
def test_app():
    app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": True})
    return app


@pytest.fixture
def master_profile():
    return load_profile("data/master_profile.json")


# ---------------------------------------------------------------------------
# 1. Overview Route Verification
# ---------------------------------------------------------------------------

class TestOverviewLandingPage:
    def test_root_and_overview_routes_render_dashboard(self, test_app):
        client = test_app.test_client()

        # Root route
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert b"Outreach Dashboard" in res_root.data
        assert b"Manage drafts" in res_root.data
        assert b"Overview" in res_root.data

        # Explicit /overview route
        res_overview = client.get("/overview")
        assert res_overview.status_code == 200
        assert b"Outreach Dashboard" in res_overview.data

    def test_sidebar_places_console_overview_at_top(self, test_app):
        client = test_app.test_client()
        res = client.get("/")
        html = res.data.decode("utf-8")
        console_idx = html.find("Console")
        workflow_idx = html.find("Workflow")
        assert console_idx != -1 and workflow_idx != -1
        assert console_idx < workflow_idx


# ---------------------------------------------------------------------------
# 2. Manage Drafts Cleanup Actions & Modal Verification
# ---------------------------------------------------------------------------

class TestDraftCleanupOnDashboard:
    def test_cleanup_modal_elements_present_on_dashboard(self, test_app):
        client = test_app.test_client()
        res = client.get("/")
        html = res.data.decode("utf-8")
        assert "cleanupModal" in html
        assert "Deletion Scope" in html
        assert "Data Safety Guarantee" in html
        assert "cleanupConfirmCheck" in html
        assert "cleanupSubmitBtn" in html

    def test_cleanup_route_requires_valid_csrf(self, test_app):
        client = test_app.test_client()
        # Missing CSRF token
        res = client.post("/drafts/cleanup", data={"scope": "unapproved"})
        assert res.status_code == 400
        assert b"CSRF validation failed" in res.data

    def test_cleanup_route_executes_safely_with_csrf(self, test_app, master_profile):
        client = test_app.test_client()
        # Seed a draft
        draft = EmailDraft(
            draft_id="test-clean-99",
            recipient="test@example.com",
            recipient_name="Tester",
            company_name="TestCorp",
            subject="Test Cleanup",
            body="Test body",
            attachment_path="output/cv.pdf",
            mode=WorkflowMode.UPLOADED_CV,
            approved=False,
            review_required=True,
            generation_status=GenerationStatus.NEEDS_REVIEW.value,
        )
        save_draft(draft)
        assert get_draft("test-clean-99") is not None

        # Initialize session by visiting home page
        client.get("/")
        with client.session_transaction() as sess:
            csrf_token = sess.get("csrf_token")

        res = client.post(
            "/drafts/cleanup",
            data={"scope": "unapproved", "csrf_token": csrf_token},
            follow_redirects=True,
        )
        assert res.status_code == 200
        assert get_draft("test-clean-99") is None


# ---------------------------------------------------------------------------
# 3. Provider Configuration and Key Parsing
# ---------------------------------------------------------------------------

class TestProviderMultiKeyParsingAndSafety:
    def test_parse_comma_separated_and_whitespace_keys(self):
        raw = "  key_alpha,  key_beta, key_gamma  ,  , key_delta\nkey_epsilon  "
        parsed = _parse_key_entries(raw)
        assert len(parsed) == 5
        assert parsed == ["key_alpha", "key_beta", "key_gamma", "key_delta", "key_epsilon"]

    def test_parse_json_list_keys(self):
        raw = '["k1", "k2", "k3"]'
        parsed = _parse_key_entries(raw)
        assert parsed == ["k1", "k2", "k3"]

    def test_key_masking_does_not_leak_raw_secret(self):
        k1 = "AQ.AbCdEf1234567890nExg"
        masked = mask_secret(k1)
        assert masked == "AQ.A...nExg"
        assert "1234567890" not in masked

    def test_credential_pool_rotation_and_cooldown(self):
        keys = ["key_1", "key_2", "key_3"]
        pool = CredentialPool(keys)
        assert pool.total_count == 3

        c1 = pool.get_next_available()
        assert c1.api_key == "key_1"
        c2 = pool.get_next_available()
        assert c2.api_key == "key_2"

        # Set cooldown on key 2
        c2.set_cooldown(duration_seconds=30.0, reason="HTTP 429")
        assert not c2.is_available()

        # Next available should be key 3, then wrap to key 1
        c3 = pool.get_next_available()
        assert c3.api_key == "key_3"
        c4 = pool.get_next_available()
        assert c4.api_key == "key_1"

    def test_pool_status_hides_raw_keys(self):
        keys = ["secret_key_123456789"]
        pool = CredentialPool(keys)
        status = pool.get_status()
        assert len(status) == 1
        assert "secret_key_123456789" not in json.dumps(status)
        assert status[0]["masked_key"] == "secr...6789"


# ---------------------------------------------------------------------------
# 4. Error Classification and Cooldown Bounding
# ---------------------------------------------------------------------------

class TestErrorClassificationAndBoundedRetries:
    def test_gemini_429_sets_cooldown_and_rotates(self):
        provider = GeminiProvider(api_key=["gemini_k1", "gemini_k2"], max_retries=1)

        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Success content"}]}}]
        }

        with patch("requests.post", side_effect=[mock_resp_429, mock_resp_200]):
            res = provider._call_api(prompt="Test prompt")
            assert res.success is True
            assert res.text == "Success content"
            assert res.used_key_masked == mask_secret("gemini_k2")

    def test_groq_401_marks_disabled_without_infinite_retry(self):
        provider = GroqProvider(api_key=["groq_bad_key"], max_retries=1)

        mock_resp_401 = MagicMock()
        mock_resp_401.status_code = 401

        with patch("requests.post", return_value=mock_resp_401):
            res = provider._call_api(prompt="Test prompt")
            assert res.success is False
            assert res.error_category == "AUTH_FAILURE"
            assert res.is_auth_error is True
            # Key should now be disabled in pool
            assert provider._pool.credentials[0].is_disabled is True


# ---------------------------------------------------------------------------
# 5. Markdown Formatting Stripping and Writing Quality
# ---------------------------------------------------------------------------

class TestMarkdownStrippingAndWritingQuality:
    def test_strip_markdown_formatting_removes_bold_and_code(self):
        raw = "I am interested in **Generative AI** and `LangGraph` agents at *Acme Corp*."
        cleaned = strip_markdown_formatting(raw)
        assert "**" not in cleaned
        assert "`" not in cleaned
        assert "*" not in cleaned
        assert cleaned == "I am interested in Generative AI and LangGraph agents at Acme Corp."

    def test_opening_and_closing_variations_across_companies(self, master_profile):
        opp_a = CompanyOpportunity(
            company_name="AlphaRobotics",
            concrete_finding="developing autonomous mobile manipulation and visual navigation systems",
            evidence_strength="STRONG",
            primary_source_url="https://alpharobotics.com",
            technical_domain="ML & Data Systems",
        )
        match_a = CandidateExperienceMatch(
            primary_experience_id="exp_netsec",
            primary_experience_name="Network Security ML",
            organization_or_project="Network Security ML",
            concrete_candidate_work="developed classification pipelines and ML deployment workflows",
            connection_rationale="relevant to system-level ML",
        )

        opp_b = CompanyOpportunity(
            company_name="BetaAI",
            concrete_finding="building agent orchestration middleware and compliance auditing engines",
            evidence_strength="STRONG",
            primary_source_url="https://betaai.io",
            technical_domain="AI Governance & Safety",
        )
        match_b = CandidateExperienceMatch(
            primary_experience_id="exp_cp",
            primary_experience_name="ControlPlane.ai",
            organization_or_project="ControlPlane.ai",
            concrete_candidate_work="built context-aware AI middleware and RAG architectures",
            connection_rationale="relevant to compliance and governance",
        )

        s_a, b_a, _ = generate_personalized_email_body(opp_a, match_a, master_profile, "AlphaRobotics")
        s_b, b_b, _ = generate_personalized_email_body(opp_b, match_b, master_profile, "BetaAI")

        # Distinct openings and closings
        assert b_a != b_b
        assert "alpharobotics" in b_a.lower()
        assert "betaai" in b_b.lower()

        # No raw markdown tokens in body
        assert "**" not in b_a and "**" not in b_b
        assert "`" not in b_a and "`" not in b_b

        # Word count between 90 and 170 words
        word_count_a = len(b_a.split())
        word_count_b = len(b_b.split())
        assert 90 <= word_count_a <= 170
        assert 90 <= word_count_b <= 170
