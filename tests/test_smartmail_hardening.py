"""
tests/test_smartmail_hardening.py
---------------------------------
Regression tests for the branch-M hardening work (uploaded-CV-only,
evidence grounding, suppression, approval binding, ambiguous sends,
send-log integrity, and localhost-only deployment).

All external providers are mocked. No network, no live LLM/search/email
quotas, no real emails.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.outreach.approval import approval_hash_for
from src.outreach.compliance import check_subject_honest, pre_send_compliance_check
from src.outreach.mailer import EmailReviewManager, EmailSender
from src.outreach.models import DraftStatus, EmailDraft, WorkflowMode
from src.outreach.send_log import SendLog
from src.research.cache import CachedResearchProvider
from src.research.models import (
    EvidenceItem,
    ResearchInput,
    ResearchResult,
    RETRIEVAL_ERROR,
    RETRIEVAL_OK,
    build_structured_evidence,
)
from src.research.providers import HttpPageProvider
from src.workflow.engine import OutreachWorkflow


def _draft(**overrides) -> EmailDraft:
    base = dict(
        draft_id="draft-hard-1",
        recipient="lead@example.ai",
        recipient_name="Lead",
        company_name="ExampleAI",
        subject="AI/ML Internship Inquiry",
        body="Dear Lead, I am writing about AI/ML work.",
        attachment_path="",
        mode=WorkflowMode.UPLOADED_CV,
    )
    base.update(overrides)
    return EmailDraft(**base)


# ---------------------------------------------------------------------------
# 1. Uploaded-CV-only + legacy tailored drafts
# ---------------------------------------------------------------------------

class TestUploadedOnly:
    def test_tailored_mode_rejected(self):
        with pytest.raises(ValueError, match="Tailored-CV mode was removed"):
            OutreachWorkflow(default_mode=WorkflowMode.TAILORED_CV)
        wf = OutreachWorkflow()
        with pytest.raises(ValueError, match="Tailored-CV mode was removed"):
            wf.set_mode(WorkflowMode.TAILORED_CV)

    def test_legacy_tailored_draft_regenerate_blocked(self, tmp_path, monkeypatch):
        import src.ui.storage as storage
        monkeypatch.setattr(storage, "_DRAFTS_DIR", tmp_path)
        draft = {
            "draft_id": "draft-legacy-1", "mode": "tailored_cv",
            "recipient": "a@example.ai", "subject": "s", "body": "b",
            "company_name": "C", "attachment_path": "",
        }
        (tmp_path / "draft-legacy-1.json").write_text(json.dumps(draft))
        with pytest.raises(ValueError, match="removed tailored-CV mode"):
            storage.regenerate_draft_content("draft-legacy-1")
        # And it was NOT silently converted.
        assert json.loads((tmp_path / "draft-legacy-1.json").read_text())["mode"] == "tailored_cv"

    def test_legacy_tailored_draft_cli_loader_blocked(self, tmp_path, monkeypatch):
        import src.cli as cli
        monkeypatch.setattr(cli, "_DRAFTS_DIR", tmp_path)
        draft = {
            "draft_id": "draft-legacy-2", "mode": "tailored_cv",
            "recipient": "a@example.ai", "recipient_name": "", "company_name": "C",
            "subject": "s", "body": "b", "attachment_path": "",
            "status": "DRAFT", "approved": False, "review_required": False,
            "review_flags": [],
        }
        (tmp_path / "draft-legacy-2.json").write_text(json.dumps(draft))
        assert cli._get_draft_and_workflow("draft-legacy-2") is None

    def test_cli_ui_rejects_non_loopback_host(self):
        from src.cli import main
        assert main(["ui", "--host", "0.0.0.0"]) == 1
        assert main(["ui", "--host", "192.168.1.10"]) == 1


# ---------------------------------------------------------------------------
# 2. Structured evidence + cache + HTTP evidence honesty
# ---------------------------------------------------------------------------

class TestEvidenceGrounding:
    def test_structured_evidence_only_from_ok_items_with_excerpts(self):
        items = [
            EvidenceItem(source_url="https://a.example", source_type="official_website",
                         title="A", extracted_text="We build RAG systems.",
                         retrieval_status=RETRIEVAL_OK, retrieved_at="2026-01-01T00:00:00Z"),
            EvidenceItem(source_url="https://b.example", source_type="web_page",
                         extracted_text="", retrieval_status=RETRIEVAL_OK),
            EvidenceItem(source_url="https://c.example", source_type="web_page",
                         extracted_text="Error page", retrieval_status=RETRIEVAL_ERROR),
        ]
        facts = build_structured_evidence(items)
        assert len(facts) == 1
        assert facts[0].fact_id == "fact-001"
        assert facts[0].excerpt == "We build RAG systems."
        assert facts[0].verification_status == RETRIEVAL_OK

    def test_cache_never_relabels_source_type(self):
        inner = MagicMock()
        inner.fetch.return_value = EvidenceItem(
            source_url="https://co.example", source_type="official_website",
            title="Co", extracted_text="Hello", retrieval_status=RETRIEVAL_OK,
            supports=["company_identity"],
        )
        cached = CachedResearchProvider(inner)
        first = cached.fetch("https://co.example", "official_website", ["company_identity"])
        assert first.source_type == "official_website"
        second = cached.fetch("https://co.example", "job_listing", ["role_evidence"])
        assert second.source_type == "official_website"
        assert "role_evidence" not in second.supports

    @patch("requests.Session.get")
    def test_http_redirect_cites_final_url(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.ok = True
        resp.url = "https://final.example/page"
        resp.headers = {"content-type": "text/html"}
        resp.text = "<html><head><title>Final</title></head><body><p>Real company text here.</p></body></html>"
        mock_get.return_value = resp
        ev = HttpPageProvider().fetch("https://start.example/old", "official_website", ["company_identity"])
        assert ev.source_url == "https://final.example/page"
        assert ev.retrieval_status == RETRIEVAL_OK

    @patch("requests.Session.get")
    def test_non_html_is_not_counted_as_evidence(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.ok = True
        resp.url = "https://co.example/doc.pdf"
        resp.headers = {"content-type": "application/pdf"}
        resp.text = "%PDF-binary-bytes"
        mock_get.return_value = resp
        ev = HttpPageProvider().fetch("https://co.example/doc.pdf", "web_page", ["technical_relevance"])
        assert ev.retrieval_status == RETRIEVAL_ERROR
        assert ev.extracted_text == ""

    @patch("requests.Session.get")
    def test_private_ip_fetch_blocked(self, mock_get):
        ev = HttpPageProvider().fetch("http://127.0.0.1/admin", "web_page", [])
        assert ev.retrieval_status == "BLOCKED"
        mock_get.assert_not_called()

    def test_researcher_llm_exception_does_not_raise_nameerror(self, monkeypatch):
        from src.research import researcher as researcher_mod
        from src.research.researcher import research_contact

        class BoomRouter:
            def __init__(self, *a, **k):
                raise RuntimeError("LLM unavailable")
        monkeypatch.setattr(researcher_mod, "LLMRouter", BoomRouter, raising=False)

        class StubProvider:
            def fetch(self, url, source_type, supports):
                return EvidenceItem(
                    source_url=url, source_type=source_type, title="Co",
                    extracted_text="Acme builds machine learning systems for logistics.",
                    retrieval_status=RETRIEVAL_OK, supports=list(supports),
                )
        inp = ResearchInput(
            company_name="Acme", normalized_email="hr@acme.example",
            email_domain="acme.example", registered_domain="acme.example",
            website="https://acme.example",
        )
        # LLMRouter is imported inside research_contact from src.llm.router;
        # patch that module attribute instead.
        import src.llm.router as router_mod
        monkeypatch.setattr(router_mod, "LLMRouter", BoomRouter)
        result = research_contact(inp, provider=StubProvider())
        assert result is not None
        assert any("LLM extraction skipped" in lim for lim in result.limitations)


# ---------------------------------------------------------------------------
# 3. Classifier / matcher / opportunity / gate
# ---------------------------------------------------------------------------

class TestGroundingDecisions:
    def test_classifier_review_never_becomes_confident(self):
        from src.contacts.recipient_resolver import NAME_VERIFIED, RecipientResolution
        from src.verification.classifier import classify_contact
        from src.verification.models import MatchResult, PipelineResult, VerificationResult, RELEVANT
        rr = ResearchResult(
            company_name="Acme", company_identity_status="REVIEW",
            domain_relationship_status="SUPPORTED", technical_areas=["AI/ML"],
            evidence=[],
        )
        pipe = PipelineResult(
            verification=VerificationResult(company_status="REVIEW", overall_status="REVIEW"),
            match=MatchResult(relevance_status=RELEVANT, matched_areas=["AI/ML"]),
        )
        rec = RecipientResolution(
            supplied_name="Alice", inferred_name="Alice", verified_name="Alice",
            name_verification_status=NAME_VERIFIED, is_generic_inbox=False,
            inferred_role_or_title="", explanation="",
        )
        clf = classify_contact(rr, pipe, recipient_resolution=rec, original_email="alice@acme.example")
        assert clf.primary_outcome != "CONFIDENT"
        assert clf.is_outreach_ready is False

    def test_matcher_alias_label_matches(self):
        from src.verification.matcher import match_profile
        rr = ResearchResult(company_name="Acme", technical_areas=["Deep Learning & NLP"])
        profile = {
            "technical_skills": {"ml_and_data": ["NLP", "PyTorch"], "frameworks_and_libraries": [], "languages": ["Python"]},
            "preferences": {"target_roles": ["AI/ML Intern"]},
            "projects": [], "experience": [],
        }
        res = match_profile(rr, profile)
        assert "Deep Learning & NLP" in res.matched_areas
        assert res.relevance_status == "RELEVANT"

    def test_matcher_single_generic_indicator_does_not_match(self):
        from src.verification.matcher import _area_matches_profile
        assert _area_matches_profile("Generative AI", frozenset(["analytics"])) is False
        assert _area_matches_profile("Generative AI", frozenset(["pytorch"])) is True
        assert _area_matches_profile("Generative AI", frozenset(["analytics", "python"])) is False

    def test_opportunity_strong_requires_evidence_excerpt(self):
        from src.outreach.opportunity_analyzer import analyze_company_opportunity
        summary_only = ResearchResult(
            company_name="DeepAI",
            ai_ml_evidence_summary="DeepAI builds RAG pipelines.",
            technical_areas=["RAG"],
        )
        opp = analyze_company_opportunity(summary_only, "DeepAI")
        assert opp.evidence_strength in ("WEAK", "NONE", "MODERATE")
        assert opp.evidence_strength != "STRONG"
        with_evidence = ResearchResult(
            company_name="DeepAI",
            ai_ml_evidence_summary="DeepAI builds RAG pipelines.",
            technical_areas=["RAG"],
            evidence=[EvidenceItem(
                source_url="https://deepai.example", source_type="official_website",
                title="DeepAI", extracted_text="DeepAI builds enterprise retrieval-augmented generation and autonomous agent pipelines for customers.",
                retrieval_status=RETRIEVAL_OK,
            )],
        )
        opp2 = analyze_company_opportunity(with_evidence, "DeepAI")
        assert opp2.evidence_strength == "STRONG"

    def test_gate_weak_evidence_does_not_auto_pass(self):
        from src.outreach.experience_matcher import CandidateExperienceMatch
        from src.outreach.opportunity_analyzer import CompanyOpportunity
        from src.outreach.quality_gate import evaluate_personalization_quality
        opp = CompanyOpportunity(company_name="Acme", concrete_finding="AI work",
                                 evidence_strength="WEAK")
        match = CandidateExperienceMatch(
            primary_experience_id="p1", primary_experience_name="Proj",
            organization_or_project="Proj", concrete_candidate_work="Did ML work.",
            connection_rationale="Relevant.", relevant_technologies=["Python"],
            is_specific_match=False,
        )
        draft = _draft(body="Dear Lead, I am writing about AI/ML work at Acme. Thank you.")
        res = evaluate_personalization_quality(draft=draft, opportunity=opp, candidate_match=match, profile={})
        assert res.checks["has_supported_anchor"] is False


# ---------------------------------------------------------------------------
# 4. Suppression, approval binding, send-log integrity, ambiguous sends
# ---------------------------------------------------------------------------

class TestSendSafety:
    def test_suppression_blocks_claim_and_mailer_send(self):
        log = SendLog(db_path=":memory:")
        log.suppress("optout@example.ai", reason="opted out")
        assert log.is_suppressed("optout@example.ai") is True
        claimed, reason = log.claim_for_sending("draft-x", "optout@example.ai", "camp")
        assert claimed is False and reason == "SUPPRESSED"

        mgr = EmailReviewManager()
        mgr.register_draft(_draft(draft_id="draft-x", recipient="optout@example.ai",
                                  attachment_path="", approved=True, status=DraftStatus.APPROVED))
        sender = EmailSender(review_manager=mgr, dry_run=True, send_log=log)
        res = sender.send("draft-x")
        assert res.error == "SUPPRESSED"

    def test_upsert_never_downgrades_success_or_uncertain(self):
        log = SendLog(db_path=":memory:")
        log.record_success("draft-s", "a@example.ai", "camp")
        log.record_draft("draft-s", "a@example.ai", "camp")
        assert log.get_status("draft-s") == "SUCCESS"
        log.record_uncertain("draft-u", "b@example.ai", "camp")
        log.record_approved("draft-u", "b@example.ai", "camp")
        assert log.get_status("draft-u") == "UNCERTAIN"

    def test_approval_hash_invalidated_by_edit(self):
        import tempfile
        from pypdf import PdfWriter
        mgr = EmailReviewManager()
        draft = _draft()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            w = PdfWriter(); w.add_blank_page(72, 72); w.write(fh)
            draft.attachment_path = fh.name
        mgr.register_draft(draft)
        mgr.approve_draft(draft.draft_id)
        assert draft.approval_hash
        # Direct mutation after approval (bypassing edit_draft) must be caught.
        draft.body = "Dear Lead, totally different body."
        sender = EmailSender(review_manager=mgr, dry_run=True)
        res = sender.send(draft.draft_id)
        assert res.error == "APPROVAL_INVALIDATED"
        assert draft.approved is False

    def test_edit_clears_approval_and_hash(self):
        mgr = EmailReviewManager()
        draft = _draft()
        mgr.register_draft(draft)
        mgr.approve_draft(draft.draft_id)
        mgr.edit_draft(draft.draft_id, subject="New subject")
        assert draft.approved is False
        assert draft.approval_hash == ""

    def test_transient_provider_retries_bounded_then_uncertain(self):
        mgr = EmailReviewManager()
        draft = _draft(draft_id="draft-t", approved=True, status=DraftStatus.APPROVED,
                       attachment_path="")
        # Attachment checks require a real PDF; use a mock-free path by
        # giving the draft a valid temp PDF below.
        import tempfile
        from pypdf import PdfWriter
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            w = PdfWriter(); w.add_blank_page(72, 72); w.write(fh)
            draft.attachment_path = fh.name
        mgr.register_draft(draft)
        calls = {"n": 0}
        class Flaky:
            def send_email(self, **kwargs):
                calls["n"] += 1
                raise RuntimeError("Gateway timeout waiting for server")
        sender = EmailSender(review_manager=mgr, provider=Flaky(), dry_run=False)
        res = sender.send("draft-t")
        assert res.error == "UNCERTAIN_SEND"
        assert calls["n"] == 3  # initial + 2 bounded retries, no more

    def test_permanent_provider_failure_not_retried(self):
        mgr = EmailReviewManager()
        import tempfile
        from pypdf import PdfWriter
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            w = PdfWriter(); w.add_blank_page(72, 72); w.write(fh)
            path = fh.name
        draft = _draft(draft_id="draft-p", approved=True, status=DraftStatus.APPROVED,
                       attachment_path=path)
        mgr.register_draft(draft)
        calls = {"n": 0}
        class Bad:
            def send_email(self, **kwargs):
                calls["n"] += 1
                raise RuntimeError("Invalid recipient rejected by provider")
        sender = EmailSender(review_manager=mgr, provider=Bad(), dry_run=False)
        res = sender.send("draft-p")
        assert res.error == "PROVIDER_ERROR"
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 5. Compliance, batch fail-fast, batch status, XSS escaping (static)
# ---------------------------------------------------------------------------

class TestComplianceAndBatch:
    def test_subject_honesty_flags_fake_reply(self):
        assert check_subject_honest("Re: Your application") != []
        assert check_subject_honest("AI/ML Internship Inquiry") == []

    def test_pre_send_check_blocks_suppressed_not_jurisdiction_notes(self):
        ok, notes = pre_send_compliance_check(
            recipient="a@example.ai", subject="Hello", sender_address="me@example.com",
            recipient_jurisdiction=None, is_suppressed=False,
        )
        assert ok is True
        assert any("jurisdiction unknown" in n for n in notes)
        ok2, _ = pre_send_compliance_check(
            recipient="a@example.ai", subject="Hello", sender_address="me@example.com",
            is_suppressed=True,
        )
        assert ok2 is False

    def test_batch_fail_fast_without_registered_cv(self):
        from src.workflow.batch_runner import BackgroundBatchRunner
        wf = OutreachWorkflow()
        assert wf.uploaded_cv_path is None
        runner = BackgroundBatchRunner()
        with pytest.raises(ValueError, match="requires a registered CV"):
            runner.start_batch(
                batch_id="batch-nocv",
                import_result=SimpleNamespace(contacts=[], errors=[]),
                workflow=wf, campaign_id="camp",
            )

    def test_batch_status_never_defaults_to_completed(self):
        from src.ui.app import create_app
        app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False})
        # get_batch_record is imported into the app module namespace; patch there.
        import src.ui.app as app_mod
        with patch.object(app_mod, "get_batch_record", return_value={"batch_id": "b1", "total_rows": 0}):
            with app.test_client() as client:
                res = client.get("/batch/b1/status")
                assert res.status_code == 200
                assert res.get_json()["status"] == "UNKNOWN"

    def test_batch_evidence_rendering_is_escaped(self):
        html = Path("src/ui/templates/batch.html").read_text()
        assert "esc(it.extracted_text)" in html
        assert "esc(it.source_url)" in html
        assert "esc(c)" in html  # contradictions escaped

    def test_tavily_cache_is_bounded(self):
        from src.research.tavily_provider import TavilySearchProvider
        prov = TavilySearchProvider(api_key="test-key")
        prov._cache = {f"q{i}": [] for i in range(200)}
        prov._execute_search = lambda query, max_results=5: []
        prov.search_company("NewCo", domain="newco.example")
        assert len(prov._cache) <= 200
