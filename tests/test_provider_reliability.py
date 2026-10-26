"""
tests/test_provider_reliability.py
------------------------------------
Focused offline regression tests covering provider failure modes, retries,
JSON parsing robustness, fallback chaining, and batch processing reliability.

Strict constraints:
- Zero real network requests (all HTTP calls mocked).
- Zero email sending (dry_run enforced).
- Fast execution with zero artificial sleep delays (mocked time.sleep).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from src.config import get_groq_model
from src.llm.base import CredentialPool, LLMResponse, safe_parse_llm_json
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.router import LLMRouter
from src.outreach.context import OutreachContext
from src.outreach.experience_matcher import CandidateExperienceMatch
from src.outreach.opportunity_analyzer import CompanyOpportunity
from src.outreach.models import (
    DraftStatus,
    WorkflowMode,
)
from src.outreach.personalized_generator import generate_personalized_email_body
from src.workflow.batch_runner import BackgroundBatchRunner
from src.workflow.engine import OutreachWorkflow, WorkflowResult


@pytest.fixture(autouse=True)
def reset_singletons():
    """Ensure circuit breakers and rate limiters start fresh in CLOSED state with zero delays."""
    from src.llm.base import reset_all_circuit_breakers, reset_all_rate_limiters
    reset_all_circuit_breakers()
    reset_all_rate_limiters(min_interval_seconds=0.0)
    yield
    reset_all_circuit_breakers()
    reset_all_rate_limiters(min_interval_seconds=0.5)


# ===========================================================================
# 1. Gemini HTTP 503 Overload Recovery
# ===========================================================================

def test_gemini_http_503_eventually_recovers():
    """Verify Gemini retries HTTP 503 and succeeds on subsequent attempt."""
    resp_503 = MagicMock(spec=requests.Response)
    resp_503.status_code = 503
    resp_503.json.return_value = {"error": {"message": "The model is overloaded."}}

    resp_200 = MagicMock(spec=requests.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": '{"opening_paragraph": "Hello", "alignment_paragraph": "Fit", "closing_paragraph": "Bye"}'}]},
                "finishReason": "STOP",
            }
        ]
    }

    with patch("src.llm.gemini_provider.requests.post", side_effect=[resp_503, resp_200]) as mock_post, \
         patch("src.llm.gemini_provider.time.sleep") as mock_sleep:
        provider = GeminiProvider(api_key="mock-key", max_retries=2, initial_backoff=0.01)
        resp = provider.generate_json("Test prompt")

        assert resp.success is True
        assert resp.http_status == 200
        assert resp.parsed_json == {"opening_paragraph": "Hello", "alignment_paragraph": "Fit", "closing_paragraph": "Bye"}
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1


# ===========================================================================
# 2. Gemini HTTP 429 Bounded Retries & Single-Key Behavior
# ===========================================================================

def test_gemini_http_429_bounded_retry_and_no_unbounded_loop():
    """Verify Gemini HTTP 429 retries up to max_retries without unbounded looping."""
    resp_429 = MagicMock(spec=requests.Response)
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "1"}

    with patch("src.llm.gemini_provider.requests.post", return_value=resp_429) as mock_post, \
         patch("src.llm.gemini_provider.time.sleep") as mock_sleep:
        provider = GeminiProvider(api_key="mock-key", max_retries=2, initial_backoff=0.01)
        resp = provider.generate_text("Test prompt")

        assert resp.success is False
        assert resp.http_status == 429
        assert resp.is_rate_limited is True
        assert resp.is_transient is True
        # Initial attempt + 2 retries = 3 calls total
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2


# ===========================================================================
# 3. Permanent Groq HTTP 400 Is Not Retried
# ===========================================================================

def test_groq_permanent_http_400_not_retried():
    """Verify Groq HTTP 400 bad request / invalid model fails immediately without retry."""
    resp_400 = MagicMock(spec=requests.Response)
    resp_400.status_code = 400
    resp_400.json.return_value = {
        "error": {"message": "Model 'unsupported-model' does not exist"}
    }

    with patch("src.llm.groq_provider.requests.post", return_value=resp_400) as mock_post, \
         patch("src.llm.groq_provider.time.sleep") as mock_sleep:
        provider = GroqProvider(api_key="mock-groq-key", model_name="unsupported-model", max_retries=3)
        resp = provider.generate_text("Test prompt")

        assert resp.success is False
        assert resp.http_status == 400
        assert resp.is_transient is False
        assert "unsupported-model" in resp.error_message
        # Should not retry at all on 400
        assert mock_post.call_count == 1
        assert mock_sleep.call_count == 0


# ===========================================================================
# 4. Explicit Model Configuration Preserved Without Silent Mutation
# ===========================================================================

def test_groq_explicit_model_preserved_without_mutation():
    """Verify configured model ID (e.g. 'openai/gpt-oss-20b') is preserved without silent mutation."""
    provider = GroqProvider(api_key="mock-groq-key", model_name="openai/gpt-oss-20b")
    assert provider.model_name == "openai/gpt-oss-20b"


# ===========================================================================
# 5. Empty Output Handled Safely Without JSONDecodeError
# ===========================================================================

def test_gemini_empty_output_handled_safely():
    """Verify Gemini returning empty text parts produces typed EMPTY_RESPONSE without unhandled exception."""
    resp_200 = MagicMock(spec=requests.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "candidates": [
            {
                "content": {"parts": []},
                "finishReason": "STOP",
            }
        ]
    }

    with patch("src.llm.gemini_provider.requests.post", return_value=resp_200):
        provider = GeminiProvider(api_key="mock-key", max_retries=0)
        resp = provider.generate_json("Test prompt")

        assert resp.success is False
        assert resp.error_category == "EMPTY_RESPONSE"
        assert "empty text" in resp.error_message


# ===========================================================================
# 6. Malformed and Truncated JSON Rejected Cleanly
# ===========================================================================

def test_safe_parse_llm_json_handles_truncation_and_malformed_syntax():
    """Verify safe_parse_llm_json classifies truncated output and malformed syntax correctly."""
    # Truncated JSON with MAX_TOKENS
    truncated_text = '{"opening_paragraph": "Hello", "alignment_paragraph": "We do ML and '
    parsed, err_cat, err_msg = safe_parse_llm_json(truncated_text, finish_reason="MAX_TOKENS")
    assert parsed is None
    assert err_cat == "TRUNCATED_RESPONSE"

    # Completely malformed text
    malformed_text = "I am an AI assistant and cannot output JSON right now."
    parsed, err_cat, err_msg = safe_parse_llm_json(malformed_text)
    assert parsed is None
    assert err_cat == "PARSE_ERROR"

    # Valid JSON inside markdown fences with conversational preamble
    fenced_text = (
        "Here is the generated output:\n"
        "```json\n"
        '{"opening_paragraph": "Hi", "alignment_paragraph": "Work", "closing_paragraph": "Thanks"}\n'
        "```\nHope this helps!"
    )
    parsed, err_cat, err_msg = safe_parse_llm_json(fenced_text)
    assert parsed is not None
    assert parsed["opening_paragraph"] == "Hi"


# ===========================================================================
# 7. Schema-Invalid Response Falls Back to Grounded Deterministic Composer
# ===========================================================================

def test_schema_invalid_response_triggers_deterministic_fallback():
    """Verify that if LLM returns JSON missing required paragraph schema, deterministic composer handles it."""
    mock_router = MagicMock(spec=LLMRouter)
    mock_router.has_any_llm = True
    # Missing 'closing_paragraph'
    mock_router.generate_json_with_fallback.return_value = LLMResponse(
        success=True,
        parsed_json={"opening_paragraph": "Hi", "other_key": "No closing"},
        provider_name="gemini",
        model_name="gemini-2.0-flash",
    )

    opp = CompanyOpportunity(
        company_name="DeepTech AI",
        concrete_finding="Building large scale diffusion models.",
        source_excerpt="Our research specializes in diffusion models.",
        evidence_strength="STRONG",
        has_advertised_internship=False,
    )
    match = CandidateExperienceMatch(
        primary_experience_id="proj-01",
        primary_experience_name="ControlPlane.ai",
        organization_or_project="ControlPlane.ai",
        concrete_candidate_work="Developed distributed inference runtime",
        connection_rationale="Both involve scaling deep neural architectures",
    )
    profile = {
        "personal_info": {"full_name": "Bhavesh Verma", "preferred_name": "Bhavesh"},
        "education": [{"institution": "IIT Kharagpur"}],
    }

    subject, body, meta = generate_personalized_email_body(
        opportunity=opp,
        candidate_match=match,
        profile=profile,
        company_name="DeepTech AI",
        contact_name="Alex",
        target_role="AI Engineer Intern",
        context=OutreachContext.DIRECT_APPLICATION,
        router=mock_router,
    )

    assert meta["method"] == "grounded_deterministic"
    assert meta.get("llm_fallback") is True
    assert "DeepTech AI" in body
    assert "Developed distributed inference runtime" in body


# ===========================================================================
# 8. Gemini Failure Followed by Successful Groq Fallback
# ===========================================================================

def test_router_gemini_failure_groq_fallback_preserves_audit_trail():
    """Verify primary failure triggers Groq fallback and records primary error and fallback reason."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.is_configured = True
    mock_gemini.provider_name = "gemini"
    mock_gemini.generate_json.return_value = LLMResponse(
        success=False,
        http_status=503,
        error_message="Gemini server error HTTP 503: The model is overloaded.",
        is_transient=True,
    )

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.is_configured = True
    mock_groq.provider_name = "groq"
    mock_groq.model_name = "llama-3.3-70b-versatile"
    mock_groq.generate_json.return_value = LLMResponse(
        success=True,
        http_status=200,
        parsed_json={"opening_paragraph": "P1", "alignment_paragraph": "P2", "closing_paragraph": "P3"},
        provider_name="groq",
        model_name="llama-3.3-70b-versatile",
    )

    router = LLMRouter(primary_provider=mock_gemini, fallback_provider=mock_groq)
    resp = router.generate_json_with_fallback("Draft inquiry")

    assert resp.success is True
    assert resp.provider_name == "groq"
    assert "Primary gemini failed" in resp.fallback_reason
    assert "HTTP 503" in resp.primary_error


# ===========================================================================
# 9. Failure of Both Providers Produces Terminal Outcome with Combined Diagnostics
# ===========================================================================

def test_router_both_providers_failing_preserves_all_diagnostics():
    """Verify failure of both primary and secondary retains both error messages and HTTP status."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.is_configured = True
    mock_gemini.provider_name = "gemini"
    mock_gemini.generate_text.return_value = LLMResponse(
        success=False,
        http_status=429,
        error_message="Gemini rate limit exceeded (HTTP 429).",
        is_transient=True,
    )

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.is_configured = True
    mock_groq.provider_name = "groq"
    mock_groq.generate_text.return_value = LLMResponse(
        success=False,
        http_status=400,
        error_message="HTTP 400: Bad request or unsupported model (llama-old).",
        is_transient=False,
    )

    router = LLMRouter(primary_provider=mock_gemini, fallback_provider=mock_groq)
    resp = router.generate_text_with_fallback("Draft text")

    assert resp.success is False
    assert resp.provider_name == "none"
    assert "HTTP 429" in resp.primary_error
    assert "HTTP 400" in resp.fallback_error
    assert resp.http_status == 429


# ===========================================================================
# 10. No Nested Retry Explosion
# ===========================================================================

def test_no_nested_retry_explosion():
    """Verify that a single call to router with retries does not multiply across layers."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.is_configured = True
    mock_gemini.provider_name = "gemini"
    # gemini does internal retries, so generate_json is called exactly once by the router
    mock_gemini.generate_json.return_value = LLMResponse(
        success=False,
        error_message="Rate limit",
        retries_attempted=1,
    )

    mock_groq = MagicMock(spec=GroqProvider)
    mock_groq.is_configured = True
    mock_groq.provider_name = "groq"
    mock_groq.generate_json.return_value = LLMResponse(
        success=False,
        error_message="Server error",
        retries_attempted=1,
    )

    router = LLMRouter(primary_provider=mock_gemini, fallback_provider=mock_groq)
    router.generate_json_with_fallback("Prompt")

    # Each provider is called exactly once by the router
    assert mock_gemini.generate_json.call_count == 1
    assert mock_groq.generate_json.call_count == 1


# ===========================================================================
# 11 & 12. Batch Processing Reliability, Terminal States & Draft Preservation
# ===========================================================================

def test_batch_runner_isolates_contact_failure_and_preserves_other_drafts(tmp_path, monkeypatch):
    """Verify that a failed contact does not stop batch execution or corrupt successful drafts."""
    from src.outreach.models import EmailDraft
    from src.ui.storage import get_batch_record, get_draft

    # Point UI storage to temporary directory
    monkeypatch.setattr("src.ui.storage._BATCHES_DIR", tmp_path / "batches")
    monkeypatch.setattr("src.ui.storage._DRAFTS_DIR", tmp_path / "drafts")

    runner = BackgroundBatchRunner()

    mock_workflow = MagicMock()
    mock_workflow.send_log.was_successfully_sent.return_value = False
    mock_workflow.sender.dry_run = True
    mock_workflow.send_log.was_simulated.return_value = False
    mock_workflow.research_cache = MagicMock()

    contact_good = MagicMock()
    contact_good.email_result.normalized_email = "good@example.com"
    contact_good.company_name = "Good Co"
    contact_good.contact_name = "Good Lead"
    contact_good.row_number = 1

    contact_bad = MagicMock()
    contact_bad.email_result.normalized_email = "bad@example.com"
    contact_bad.company_name = "Bad Co"
    contact_bad.contact_name = "Bad Lead"
    contact_bad.row_number = 2

    # First contact succeeds, second contact throws unexpected exception
    draft_good = EmailDraft(
        draft_id="draft-good-1",
        recipient="good@example.com",
        recipient_name="Good Lead",
        company_name="Good Co",
        subject="Internship Inquiry",
        body="Hello Good Lead, I am interested in Good Co.",
        attachment_path="cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
    )
    result_good = WorkflowResult(
        mode=WorkflowMode.UPLOADED_CV,
        contact_email="good@example.com",
        company_name="Good Co",
        status="PROCEED",
        draft=draft_good,
        row_number=1,
    )

    def side_effect_process(contact, **kwargs):
        if contact.email_result.normalized_email == "good@example.com":
            return result_good
        raise RuntimeError("External network connection reset")

    mock_workflow.process_contact.side_effect = side_effect_process

    import_result = MagicMock()
    import_result.valid_contacts = [contact_good, contact_bad]
    import_result.errors = []

    batch_id = "test-batch-reliability-001"
    runner.start_batch(
        batch_id=batch_id,
        import_result=import_result,
        workflow=mock_workflow,
        wait_for_completion=True,
    )

    record = get_batch_record(batch_id)
    assert record is not None
    assert record["status"] == "COMPLETED"
    assert record["total_rows"] == 2
    assert record["processed_count"] == 2
    assert record["drafted_count"] == 1
    assert record["failed_count"] == 1

    # Check that contact 1 was drafted and contact 2 is an ERROR
    results = record["results"]
    assert len(results) == 2
    assert results[0]["contact_email"] == "good@example.com"
    assert results[0]["status"] == "PROCEED"
    assert results[0]["draft_id"] == "draft-good-1"

    assert results[1]["contact_email"] == "bad@example.com"
    assert results[1]["status"] == "ERROR"
    assert "External network connection reset" in results[1]["error"]

    # Draft for contact 1 is preserved and loadable from disk
    saved_draft = get_draft("draft-good-1")
    assert saved_draft is not None
    assert saved_draft["recipient"] == "good@example.com"


# ===========================================================================
# 13. Batch Status Polling Is Read-Only and Idempotent
# ===========================================================================

def test_batch_status_endpoint_is_strictly_read_only(tmp_path, monkeypatch):
    """Verify that querying batch status endpoint does not trigger LLM calls or email sending."""
    from src.ui.app import create_app
    from src.ui.storage import save_batch_record

    monkeypatch.setattr("src.ui.storage._BATCHES_DIR", tmp_path / "batches")
    monkeypatch.setattr("src.ui.storage._DRAFTS_DIR", tmp_path / "drafts")

    batch_id = "poll-test-batch"
    save_batch_record(batch_id, {
        "batch_id": batch_id,
        "status": "PROCESSING",
        "total_rows": 5,
        "processed_count": 2,
        "drafted_count": 2,
        "review_required_count": 0,
        "blocked_count": 0,
        "failed_count": 0,
    })

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    with patch("src.workflow.engine.OutreachWorkflow.process_contact") as mock_process, \
         patch("src.llm.gemini_provider.requests.post") as mock_gemini_post, \
         patch("src.llm.groq_provider.requests.post") as mock_groq_post:
        resp = client.get(f"/batch/{batch_id}/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["batch_id"] == batch_id
        assert data["status"] == "PROCESSING"
        assert data["processed_count"] == 2

        # Absolutely NO workflow or LLM calls triggered by polling
        assert mock_process.call_count == 0
        assert mock_gemini_post.call_count == 0
        assert mock_groq_post.call_count == 0


# ===========================================================================
# 14. Groq HTTP 404 Model Not Found Fails Fast and Trips Circuit Permanently
# ===========================================================================

def test_groq_http_404_model_not_found_fails_fast_and_trips_circuit():
    """Verify Groq HTTP 404 fails fast without retrying and permanently trips circuit breaker."""
    resp_404 = MagicMock(spec=requests.Response)
    resp_404.status_code = 404
    resp_404.json.return_value = {
        "error": {
            "message": "The model `llama-3.3-70b-versatile` does not exist or you do not have access to it.",
            "type": "invalid_request_error",
            "code": "model_not_found",
        }
    }

    with patch("src.llm.groq_provider.requests.post", return_value=resp_404) as mock_post, \
         patch("src.llm.groq_provider.time.sleep") as mock_sleep:
        provider = GroqProvider(api_key="mock-groq-key", model_name="llama-3.3-70b-versatile", max_retries=3)
        resp1 = provider.generate_text("Test prompt 1")

        assert resp1.success is False
        assert resp1.http_status == 404
        assert resp1.is_transient is False
        assert resp1.error_category == "MODEL_NOT_FOUND"
        assert "llama-3.3-70b-versatile" in resp1.error_message
        assert mock_post.call_count == 1
        assert mock_sleep.call_count == 0

        # Second call must be blocked immediately by the open circuit breaker without touching network
        resp2 = provider.generate_text("Test prompt 2")
        assert resp2.success is False
        assert resp2.error_category == "CIRCUIT_OPEN"
        assert resp2.circuit_open is True
        # Total network calls must still be 1!
        assert mock_post.call_count == 1


# ===========================================================================
# 15. Quality Gate Categorically Rejects Bloated or Unsupported Claims
# ===========================================================================

def test_quality_gate_rejects_bloated_candidate_claims():
    """Verify quality gate rejects drafts claiming bloated 9-node state machines or RRF judges."""
    from src.outreach.models import EmailDraft
    from src.outreach.quality_gate import evaluate_personalization_quality

    opp = CompanyOpportunity(
        company_name="LivingAI Solutions",
        concrete_finding="building retrieval-augmented generation (RAG) systems and knowledge bases",
        source_excerpt="We build enterprise RAG systems.",
        evidence_strength="STRONG",
        has_advertised_internship=False,
    )
    candidate_match = CandidateExperienceMatch(
        primary_experience_id="proj-001",
        primary_experience_name="ControlPlane.ai",
        organization_or_project="ControlPlane.ai",
        concrete_candidate_work="In my ControlPlane.ai project, I architected a nine-node state machine with parallel auditing",
        connection_rationale="Grounded engineering match",
        is_specific_match=True,
    )
    profile = {
        "personal_info": {"full_name": "Bhavesh Verma", "confirmed_authoritative_email": "bhaveshverma7890@gmail.com"},
        "education": [{"institution": "IIT Kharagpur"}],
    }

    # Draft with bloated claim
    bloated_draft = EmailDraft(
        draft_id="bloated-001",
        recipient="careers@livingai.in",
        recipient_name="Hiring Team",
        company_name="LivingAI Solutions",
        subject="AI/ML Intern Application — LivingAI Solutions",
        body=(
            "Dear Hiring Team at LivingAI Solutions,\n\n"
            "I wanted to reach out regarding potential AI/ML Intern roles at LivingAI Solutions. "
            "I am a fourth-year Dual Degree student at IIT Kharagpur.\n\n"
            "Given LivingAI Solutions's work building retrieval-augmented generation (RAG) systems and knowledge bases, "
            "I believe my experience is directly relevant. In my ControlPlane.ai project, I architected a nine-node state machine "
            "with parallel auditing, compliance retrieval via Neo4j, and integrated an XGBoost hallucination classifier.\n\n"
            "My CV is attached for your review.\n\n"
            "Thank you for your time and consideration.\n\n"
            "Best regards,\n\nBhavesh Verma (IIT Kharagpur)\n+91 7067419231\nbhaveshverma7890@gmail.com"
        ),
        attachment_path="templates/reference_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
    )

    qg_res = evaluate_personalization_quality(bloated_draft, opp, candidate_match, profile)
    assert qg_res.passed is False
    assert any("Bloated/unsupported claim" in f for f in qg_res.failure_reasons)


# ===========================================================================
# 16. Four Problematic Companies Produce Pairwise-Distinct Grounded Drafts
# ===========================================================================

def test_four_problematic_companies_produce_pairwise_distinct_grounded_drafts():
    """Verify LivingAI, Tekravio, S3K, and Cehpoint produce distinct, grounded, non-repetitive drafts."""
    from src.outreach.experience_matcher import match_candidate_experience
    from src.outreach.models import EmailDraft
    from src.outreach.personalized_generator import generate_personalized_email_body
    from src.outreach.quality_gate import evaluate_personalization_quality
    from src.profile.loader import load_profile

    profile = load_profile("data/master_profile.json")

    # 1. LivingAI Solutions (Verified RAG focus)
    opp_livingai = CompanyOpportunity(
        company_name="LivingAI Solutions",
        concrete_finding="building retrieval-augmented generation (RAG) systems and knowledge bases",
        source_excerpt="We ship real systems — 22-language NLP, enterprise RAG and knowledge bases.",
        evidence_strength="STRONG",
        technical_domain="RAG & Retrieval Systems",
        has_advertised_internship=False,
    )
    match_livingai = match_candidate_experience(opp_livingai, profile)
    s1, b1, m1 = generate_personalized_email_body(
        opportunity=opp_livingai,
        candidate_match=match_livingai,
        profile=profile,
        company_name="LivingAI Solutions",
        contact_name="Hiring Team",
        target_role="AI/ML Intern",
        mode=WorkflowMode.UPLOADED_CV,
    )
    draft_livingai = EmailDraft(
        draft_id="d-livingai",
        recipient="careers@livingai.in",
        recipient_name="Hiring Team",
        company_name="LivingAI Solutions",
        subject=s1,
        body=b1,
        attachment_path="templates/reference_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
    )

    # 2. Tekravio (Verified AI Agents focus)
    opp_tekravio = CompanyOpportunity(
        company_name="Tekravio",
        concrete_finding="engineering AI agents, conversational assistants, and custom RAG pipelines",
        source_excerpt="Ship AI agents and RAG pipelines with vector DBs and LangChain.",
        evidence_strength="STRONG",
        technical_domain="AI Agents & Workflows",
        has_advertised_internship=False,
    )
    match_tekravio = match_candidate_experience(opp_tekravio, profile)
    s2, b2, m2 = generate_personalized_email_body(
        opportunity=opp_tekravio,
        candidate_match=match_tekravio,
        profile=profile,
        company_name="Tekravio",
        contact_name="Hiring Team",
        target_role="AI/ML Intern",
        mode=WorkflowMode.UPLOADED_CV,
    )
    draft_tekravio = EmailDraft(
        draft_id="d-tekravio",
        recipient="careers@tekravio.com",
        recipient_name="Hiring Team",
        company_name="Tekravio",
        subject=s2,
        body=b2,
        attachment_path="templates/reference_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
    )

    # 3. S3K Technologies (Unverified / Held for human review)
    opp_s3k = CompanyOpportunity(
        company_name="S3K Technologies",
        concrete_finding="full-stack AI ecosystem and enterprise services",
        source_excerpt="With S3K Tech, you can Learn AI AI-Ready India.",
        evidence_strength="WEAK",
        technical_domain="General AI",
        has_advertised_internship=False,
    )
    match_s3k = match_candidate_experience(opp_s3k, profile)
    s3, b3, m3 = generate_personalized_email_body(
        opportunity=opp_s3k,
        candidate_match=match_s3k,
        profile=profile,
        company_name="S3K Technologies",
        contact_name="Hiring Team",
        target_role="AI/ML Intern",
        mode=WorkflowMode.UPLOADED_CV,
    )
    draft_s3k = EmailDraft(
        draft_id="d-s3k",
        recipient="hr@s3ktech.ai",
        recipient_name="Hiring Team",
        company_name="S3K Technologies",
        subject=s3,
        body=b3,
        attachment_path="templates/reference_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
    )

    # 4. Cehpoint (Autonomous Agent Architect / Prompt Engineer)
    opp_cehpoint = CompanyOpportunity(
        company_name="Cehpoint",
        concrete_finding="developing autonomous AI agents and intelligent workflow automation systems",
        source_excerpt="Under 1-on-1 mentorship, you will build autonomous agents that solve real enterprise tasks.",
        evidence_strength="STRONG",
        technical_domain="AI Agents & Workflows",
        has_advertised_internship=False,
    )
    match_cehpoint = match_candidate_experience(opp_cehpoint, profile)
    s4, b4, m4 = generate_personalized_email_body(
        opportunity=opp_cehpoint,
        candidate_match=match_cehpoint,
        profile=profile,
        company_name="Cehpoint",
        contact_name="Hiring Team",
        target_role="AI/ML Intern",
        mode=WorkflowMode.UPLOADED_CV,
    )
    draft_cehpoint = EmailDraft(
        draft_id="d-cehpoint",
        recipient="ai-engineering-intern-careers@cehpoint.co.in",
        recipient_name="Hiring Team",
        company_name="Cehpoint",
        subject=s4,
        body=b4,
        attachment_path="templates/reference_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
    )

    # Assert pairwise distinct bodies
    bodies = [
        draft_livingai.body.strip(),
        draft_tekravio.body.strip(),
        draft_s3k.body.strip(),
        draft_cehpoint.body.strip(),
    ]
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            assert bodies[i] != bodies[j], f"Drafts {i} and {j} must not be identical!"

    # Assert none of them contain bloated buzzword dumps
    for body in bodies:
        b_lower = body.lower()
        assert "nine-node state machine" not in b_lower
        assert "9-node" not in b_lower
        assert "reciprocal rank fusion" not in b_lower
        assert "neo4j" not in b_lower
        assert "xgboost hallucination" not in b_lower
        assert "combining faiss and bm25 with reciprocal rank fusion" not in b_lower

    # Verify quality gates pass
    qg_livingai = evaluate_personalization_quality(draft_livingai, opp_livingai, match_livingai, profile)
    assert qg_livingai.passed is True

    qg_tekravio = evaluate_personalization_quality(draft_tekravio, opp_tekravio, match_tekravio, profile)
    assert qg_tekravio.passed is True


# ===========================================================================
# 17. Security & Privacy: No API Keys Leaked in Diagnostics or Reports
# ===========================================================================

def test_no_api_keys_leaked_in_provider_diagnostics_or_reports():
    """Verify that provider logs, pool status, and error representations mask secrets completely."""
    secret_key = "AIzaSySecretGeminiKey1234567890"
    provider = GeminiProvider(api_key=secret_key)

    status = provider.get_pool_status()
    assert len(status) == 1
    masked = status[0]["masked_key"]
    assert secret_key not in masked
    assert secret_key not in repr(provider)

    groq_secret = "gsk_SecretGroqKey1234567890"
    groq_provider = GroqProvider(api_key=groq_secret)
    groq_status = groq_provider.get_pool_status()
    assert len(groq_status) == 1
    assert groq_secret not in groq_status[0]["masked_key"]

