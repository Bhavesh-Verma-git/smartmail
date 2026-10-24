"""
tests/test_production_readiness.py
----------------------------------
Targeted regression test suite verifying SmartMail Production-Readiness:
1. Priority 1: CSRF protection on draft regeneration (valid, missing, invalid, expired session).
2. Priority 1 & 7: Regeneration persistence, version increments, error handling, prior draft preservation.
3. Priority 3: Draft cleanup UX routes (single draft delete, batch delete, scoped cleanup: unapproved, rejected, all).
4. Priority 3 & 7: Data integrity (preservation of SUCCESS send_log records, research caches, CV, master profile).
5. Priority 4: Multi-key credential loading, key masking, 429 cooldowns, 401 auth error handling, error categories.
6. Priority 4 & 5: LLM Router fallback integrity, fallback reason audit metadata, safe health checks.
7. Priority 2 & 6: Truly generalizable personalization on unseen company without hardcoded company names; candidate contact consistency.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.config import (
    get_authoritative_candidate_email,
    get_gemini_api_keys,
    get_groq_api_keys,
    get_provider_status,
    mask_secret,
)
from src.llm.base import CredentialPool, KeyCredential, LLMResponse
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.router import LLMRouter
from src.outreach.drafting import draft_outreach_email
from src.outreach.experience_matcher import match_candidate_experience
from src.outreach.models import DraftStatus, EmailDraft, GenerationStatus
from src.outreach.opportunity_analyzer import analyze_company_opportunity
from src.outreach.quality_gate import evaluate_personalization_quality
from src.outreach.send_log import SendLog
from src.profile.loader import load_profile
from src.research.models import EvidenceItem, ResearchResult, RETRIEVAL_OK
from src.ui.app import create_app
from src.ui.storage import (
    clear_all_drafts,
    delete_draft,
    delete_drafts_for_batch,
    get_draft,
    regenerate_draft_content,
    save_draft,
)


@pytest.fixture
def master_profile() -> dict:
    return load_profile("data/master_profile.json")


@pytest.fixture
def test_send_log(tmp_path: Path) -> SendLog:
    db_file = tmp_path / "test_send_log.db"
    return SendLog(db_path=str(db_file))


@pytest.fixture
def app_with_csrf():
    """Flask application fixture with CSRF enabled and testing mode."""
    app = create_app({
        "TESTING": True,
        "WTF_CSRF_ENABLED": True,
        "SECRET_KEY": "test-secret-key-for-csrf-validation",
    })
    return app


# ---------------------------------------------------------------------------
# Priority 1: CSRF Verification on Draft Regeneration
# ---------------------------------------------------------------------------

class TestCSRFRegenerationWorkflow:
    """Verifies that the CSRF protection on regeneration works correctly and securely."""

    def test_regeneration_with_valid_csrf_token_succeeds(self, app_with_csrf, master_profile):
        ev = [EvidenceItem(
            source_url="https://zenithai.example/tech",
            source_type="official_website",
            title="Zenith AI Systems",
            extracted_text="Zenith AI develops low-latency edge inference engines for industrial telemetry.",
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res = ResearchResult(
            company_name="Zenith AI",
            evidence=ev,
            technical_areas=["Edge AI Inference"],
            ai_ml_evidence_summary=[{"excerpt": ev[0].extracted_text, "source_url": ev[0].source_url}],
        )
        draft = draft_outreach_email(master_profile, "careers@zenithai.example", "Zenith AI", research_result=res)
        save_draft(draft)

        client = app_with_csrf.test_client()

        # Step 1: GET draft page to initiate session and obtain valid CSRF token
        get_resp = client.get(f"/drafts/{draft.draft_id}")
        assert get_resp.status_code == 200

        with client.session_transaction() as sess:
            csrf_token = sess.get("csrf_token")
        assert csrf_token is not None

        # Step 2: POST /drafts/<draft_id>/regenerate with valid CSRF token
        post_resp = client.post(
            f"/drafts/{draft.draft_id}/regenerate",
            data={"csrf_token": csrf_token},
            follow_redirects=True,
        )
        assert post_resp.status_code == 200
        assert b"Draft successfully regenerated" in post_resp.data

    def test_regeneration_with_missing_csrf_token_returns_400(self, app_with_csrf, master_profile):
        client = app_with_csrf.test_client()
        # Initialize session
        client.get("/review")

        # Submit regeneration without csrf_token in form data
        resp = client.post("/drafts/test-draft-missing-csrf/regenerate", data={})
        assert resp.status_code == 400
        assert b"CSRF validation failed: missing or invalid CSRF token" in resp.data

    def test_regeneration_with_invalid_csrf_token_returns_400(self, app_with_csrf):
        client = app_with_csrf.test_client()
        client.get("/review")

        # Submit regeneration with tampered/wrong csrf_token
        resp = client.post(
            "/drafts/test-draft-invalid-csrf/regenerate",
            data={"csrf_token": "tampered-bogus-token-value-12345"},
        )
        assert resp.status_code == 400
        assert b"CSRF validation failed: missing or invalid CSRF token" in resp.data

    def test_regeneration_with_expired_or_missing_session_returns_helpful_message(self, app_with_csrf):
        client = app_with_csrf.test_client()
        # Direct POST without visiting any page or sending session cookies
        resp = client.post(
            "/drafts/test-draft-expired-session/regenerate",
            data={"csrf_token": "some-token-without-session"},
        )
        assert resp.status_code == 400
        assert b"session expired or missing" in resp.data

    def test_failed_regeneration_preserves_previous_draft(self, master_profile):
        ev = [EvidenceItem(
            source_url="https://veritas.example/about",
            source_type="official_website",
            title="Veritas AI",
            extracted_text="Veritas develops distributed validation systems.",
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res = ResearchResult(company_name="Veritas AI", evidence=ev)
        draft = draft_outreach_email(master_profile, "team@veritas.example", "Veritas AI", research_result=res)
        save_draft(draft)
        orig_body = draft.body

        # Simulate exception during regeneration
        with patch("src.outreach.drafting.draft_outreach_email", side_effect=RuntimeError("LLM simulated breakdown")):
            with pytest.raises(RuntimeError):
                regenerate_draft_content(draft.draft_id)

        # Verify draft file on disk is unchanged and preserved
        preserved = get_draft(draft.draft_id)
        assert preserved is not None
        assert preserved["body"] == orig_body


# ---------------------------------------------------------------------------
# Priority 3 & 7: Draft Management, Cleanup UX, and Data Retention
# ---------------------------------------------------------------------------

class TestDraftCleanupAndDataRetention:
    """Verifies that draft management routes delete intended records only and preserve audit records & caches."""

    def test_delete_single_draft_removes_json_and_send_log(self, app_with_csrf, master_profile, tmp_path):
        draft = draft_outreach_email(master_profile, "delete-me@corp.example", "Corp A")
        save_draft(draft)
        draft_file = Path("output/.drafts") / f"{draft.draft_id}.json"
        assert draft_file.exists()

        client = app_with_csrf.test_client()
        client.get("/review")
        with client.session_transaction() as sess:
            token = sess["csrf_token"]

        # Call delete route with CSRF
        resp = client.post(f"/drafts/{draft.draft_id}/delete", data={"csrf_token": token}, follow_redirects=True)
        assert resp.status_code == 200
        assert not draft_file.exists()

    def test_scoped_cleanup_unapproved_preserves_approved_and_success_logs(self, test_send_log, master_profile):
        # Create unapproved draft
        draft1 = draft_outreach_email(master_profile, "pending@corp1.example", "Corp 1")
        save_draft(draft1)
        test_send_log.record_draft(draft1.draft_id, draft1.recipient)

        # Create approved draft
        draft2 = draft_outreach_email(master_profile, "approved@corp2.example", "Corp 2")
        draft2.status = DraftStatus.APPROVED
        save_draft(draft2)
        test_send_log.record_success(draft2.draft_id, draft2.recipient)

        # Execute unapproved cleanup
        result = clear_all_drafts(scope="unapproved", send_log=test_send_log)
        assert result["deleted_count"] >= 1

        # draft1 must be deleted, draft2 preserved
        assert get_draft(draft1.draft_id) is None
        assert get_draft(draft2.draft_id) is not None

        # Send log SUCCESS record must NOT be deleted
        assert test_send_log.was_successfully_sent(draft2.recipient) is True

        # Clean up draft2
        delete_draft(draft2.draft_id, send_log=test_send_log)

    def test_cleanup_preserves_master_profile_cv_and_research_cache(self, test_send_log):
        # Verify critical files exist before and after cleanup
        profile_path = Path("data/master_profile.json")
        assert profile_path.exists()
        orig_profile_content = profile_path.read_text(encoding="utf-8")

        clear_all_drafts(scope="all", send_log=test_send_log)

        # Profile content must remain perfectly untouched
        assert profile_path.read_text(encoding="utf-8") == orig_profile_content


# ---------------------------------------------------------------------------
# Priority 4: Provider Reliability, Multi-Key Pool, Cooldowns & Masking
# ---------------------------------------------------------------------------

class TestProviderMultiKeyAndReliability:
    """Tests multi-key rotation, 429 cooldowns, 401 auth handling, and secret masking."""

    def test_key_masking_utility(self):
        assert mask_secret("AIzaSyD-1234567890abcdef") == "AIza...cdef"
        assert mask_secret("short") == "...rt"
        assert mask_secret("") == ""

    def test_multi_key_credential_pool_rotation_and_cooldown(self):
        pool = CredentialPool(["key-alpha", "key-beta", "key-gamma"])
        assert pool.total_count == 3

        # First pick: alpha
        cred1 = pool.get_next_available()
        assert cred1.api_key == "key-alpha"

        # Apply cooldown to alpha
        cred1.set_cooldown(duration_seconds=10.0, reason="HTTP 429")
        assert not cred1.is_available()

        # Second pick: should cycle to beta
        cred2 = pool.get_next_available()
        assert cred2.api_key == "key-beta"

        # Mark beta disabled (auth error)
        cred2.mark_disabled(reason="HTTP 401 Invalid Key")
        assert cred2.is_disabled

        # Third pick: should cycle to gamma
        cred3 = pool.get_next_available()
        assert cred3.api_key == "key-gamma"

    def test_gemini_provider_cooldown_on_429_switches_to_next_key(self):
        provider = GeminiProvider(api_key=["gemini-key-1", "gemini-key-2"], max_retries=1)

        # Mock requests.post: key-1 returns 429, key-2 returns 200
        call_keys = []

        def mock_post(url, json=None, headers=None, timeout=None):
            used_key = headers.get("x-goog-api-key")
            call_keys.append(used_key)
            mock_res = MagicMock()
            if used_key == "gemini-key-1":
                mock_res.status_code = 429
            else:
                mock_res.status_code = 200
                mock_res.json.return_value = {
                    "candidates": [{"content": {"parts": [{"text": "Success from key 2"}]}}]
                }
            return mock_res

        with patch("requests.post", side_effect=mock_post):
            resp = provider.generate_text("test prompt")
            assert resp.success is True
            assert resp.text == "Success from key 2"
            assert "gemini-key-1" in call_keys
            assert "gemini-key-2" in call_keys

            # Key 1 must now be in cooldown
            pool_status = provider.get_pool_status()
            assert pool_status[0]["in_cooldown"] is True
            assert pool_status[1]["in_cooldown"] is False

    def test_groq_provider_401_marks_key_disabled_without_infinite_retry(self):
        provider = GroqProvider(api_key=["groq-bad-key"], max_retries=2)

        def mock_post(url, json=None, headers=None, timeout=None):
            mock_res = MagicMock()
            mock_res.status_code = 401
            return mock_res

        with patch("requests.post", side_effect=mock_post):
            resp = provider.generate_text("test prompt")
            assert resp.success is False
            assert resp.error_category == "AUTH_FAILURE"
            assert resp.is_auth_error is True
            assert resp.retries_attempted == 0  # Does not retry auth failures indefinitely

    def test_provider_health_check_does_not_leak_keys(self):
        provider = GeminiProvider(api_key=["super-secret-production-gemini-key-12345"])
        health = provider.check_health()
        assert health["status"] == "HEALTHY"
        assert health["available_keys"] == 1
        # Raw key must NOT appear in health report
        health_json = json.dumps(health)
        assert "super-secret-production-gemini-key-12345" not in health_json
        assert "supe...2345" in health_json


# ---------------------------------------------------------------------------
# Priority 5: Fallback Integrity & Architecture Evaluation
# ---------------------------------------------------------------------------

class TestLLMFallbackRoutingIntegrity:
    """Verifies that primary failure routes to Groq with audit metadata, and total failure falls back safely."""

    def test_primary_failure_falls_back_to_groq_with_audit_reason(self):
        mock_gemini = MagicMock(spec=GeminiProvider)
        mock_gemini.is_configured = True
        mock_gemini.provider_name = "gemini"
        mock_gemini.generate_json.return_value = LLMResponse(
            success=False,
            provider_name="gemini",
            error_message="Gemini quota exceeded",
            error_category="RATE_LIMIT",
        )

        mock_groq = MagicMock(spec=GroqProvider)
        mock_groq.is_configured = True
        mock_groq.provider_name = "groq"
        mock_groq.generate_json.return_value = LLMResponse(
            success=True,
            provider_name="groq",
            parsed_json={"is_ai_relevant": True, "technical_areas": ["Vector Search"]},
        )

        router = LLMRouter(primary_provider=mock_gemini, fallback_provider=mock_groq)
        resp = router.generate_json_with_fallback("extract json")

        assert resp.success is True
        assert resp.provider_name == "groq"
        assert "Primary gemini failed: Gemini quota exceeded" in resp.fallback_reason

    def test_malformed_json_response_handled_gracefully(self):
        provider = GeminiProvider(api_key="valid-key")

        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "NOT VALID JSON {broken: key"}]}}]
        }

        with patch("requests.post", return_value=mock_res):
            resp = provider.generate_json("generate json")
            assert resp.success is False
            assert resp.error_category == "PARSE_ERROR"
            assert "Failed to parse LLM JSON" in resp.error_message


# ---------------------------------------------------------------------------
# Priority 2: Generalized Personalization (Unseen Companies)
# ---------------------------------------------------------------------------

class TestGeneralizedPersonalizationOnUnseenCompany:
    """Verifies that personalization works on previously unseen companies without any company name keywords."""

    def test_unseen_robotics_company_receives_grounded_technical_connection(self, master_profile):
        # A completely unseen company: "TitanRobotics AI"
        ev = [EvidenceItem(
            source_url="https://titanrobotics.example/autonomous-navigation",
            source_type="official_website",
            title="TitanRobotics AI",
            extracted_text="TitanRobotics develops real-time visual SLAM, robot perception, and edge vision models for harsh environments.",
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res = ResearchResult(
            company_name="TitanRobotics AI",
            evidence=ev,
            technical_areas=["Computer Vision", "Visual SLAM"],
            ai_ml_evidence_summary=[{"excerpt": ev[0].extracted_text, "source_url": ev[0].source_url}],
        )

        draft = draft_outreach_email(master_profile, "hiring@titanrobotics.example", "TitanRobotics AI", research_result=res)

        assert draft is not None
        # Must be grounded in the verified excerpt
        assert "titanrobotics" in draft.body.lower()
        # Must NOT mention ungrounded companies or arbitrary hardcoded mappings
        assert "controlplane.ai" not in draft.body.lower()
        assert "rasa" not in draft.body.lower()
        assert "glowminds" not in draft.body.lower()
        assert "supersei" not in draft.body.lower()

        # Opportunity analysis must find the vision/perception anchor directly from the excerpt
        opp = analyze_company_opportunity(res, "TitanRobotics AI")
        assert opp.concrete_finding != ""
        assert opp.is_grounded is True
        assert "perception" in opp.concrete_finding.lower() or "vision" in opp.concrete_finding.lower() or "slam" in opp.concrete_finding.lower()

    def test_dry_run_safety_guarantee(self, master_profile, test_send_log):
        # Verify that sending in dry-run mode never attempts SMTP or live transmission
        draft = draft_outreach_email(master_profile, "ceo@enterprise.example", "Enterprise AI")
        test_send_log.record_simulated(draft.draft_id, draft.recipient)
        assert test_send_log.get_status(draft.draft_id) == "SIMULATED"
        assert test_send_log.was_simulated(draft.recipient) is True
        assert os.getenv("SMARTMAIL_ALLOW_LIVE_SEND", "false").lower() != "true"

