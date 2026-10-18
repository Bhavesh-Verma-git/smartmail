"""
tests/test_ui.py
----------------
Automated test suite for the SmartMail Functional Testing UI.
Verifies all web routes, CSV upload, pre-upload validation preview,
batch execution, draft review and editing safeguards, CV mode inspection,
and launch-readiness checklist.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from flask.testing import FlaskClient

from src.outreach.models import DraftStatus, EmailDraft, WorkflowMode
from src.outreach.send_log import SendLog
from src.ui.app import create_app
from src.ui.storage import (
    approve_draft,
    clear_registered_cv,
    get_draft,
    get_launch_readiness_checklist,
    register_uploaded_cv,
    save_draft_record,
    update_draft,
)


@pytest.fixture
def test_send_log(tmp_path: Path) -> SendLog:
    """Fixture providing an isolated SQLite send log."""
    db_file = tmp_path / "test_ui_send_log.db"
    return SendLog(db_path=db_file)


@pytest.fixture
def client(tmp_path: Path, test_send_log: SendLog) -> Generator[FlaskClient, None, None]:
    """Fixture providing a configured Flask test client."""
    # Ensure reference CV exists for uploaded CV mode
    dummy_cv = tmp_path / "candidate_cv.pdf"
    import pypdf
    _w = pypdf.PdfWriter()
    _w.add_blank_page(width=72, height=72)
    with open(dummy_cv, "wb") as _f:
        _w.write(_f)
    register_uploaded_cv(dummy_cv)

    app = create_app({
        "TESTING": True,
        "SEND_LOG_PATH": str(test_send_log._db_path),
        "WTF_CSRF_ENABLED": False,
    })

    with app.test_client() as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Dashboard Tests
# ---------------------------------------------------------------------------

def test_dashboard_route_loads(client: FlaskClient):
    """Dashboard loads with HTTP 200, correct titles and metric cards."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Outreach Dashboard" in html
    assert "Imported" in html
    assert "Drafts" in html
    assert "Needs Review" in html
    assert "DRY RUN: SAFE" in html


def test_launch_checklist_route(client: FlaskClient):
    """Launch checklist loads and displays truthful status for all items."""
    res = client.get("/checklist")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Launch Readiness &amp; Safety Audit" in html
    assert "Custom Domain Connected" in html
    assert "Production Email-Sending Safeguards" in html
    assert "NOT_CONFIGURED" in html or "Not Configured" in html or "NOT CONFIGURED" in html


def test_send_log_route(client: FlaskClient, test_send_log: SendLog):
    """Send log route renders records from the SQLite database."""
    test_send_log.record_draft("draft-test-1", "prospect@example.com", "camp-1")
    test_send_log.record_approved("draft-test-1", "prospect@example.com", "camp-1")

    res = client.get("/send-log")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "prospect@example.com" in html
    assert "draft-test-1" in html


# ---------------------------------------------------------------------------
# CSV Pre-Upload Validation Preview Tests
# ---------------------------------------------------------------------------

def test_csv_preview_endpoint_with_mixed_contacts(client: FlaskClient):
    """Preview endpoint correctly separates valid contacts from malformed rows."""
    csv_content = (
        "email,company_name,contact_name,role_or_job_url,notes\n"
        "founder@acme.ai,Acme AI,Alice Founder,AI/ML Intern,Found on web\n"
        "not-an-email,Bad Corp,Bob,ML Intern,Bad row\n"
        "recruiter@gmail.com,Beta Labs,Charlie,Data Intern,Free provider\n"
    )

    data = {
        "csv_file": (io.BytesIO(csv_content.encode("utf-8")), "mixed_contacts.csv")
    }

    res = client.post("/import/preview", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    assert json_data["valid_count"] == 2
    assert json_data["error_count"] == 1
    # Check that free provider is flagged
    gmail_contact = next(c for c in json_data["contacts"] if c["email"] == "recruiter@gmail.com")
    assert gmail_contact["is_free_provider"] is True
    # Check malformed error reason
    err = json_data["errors"][0]
    assert "syntax" in err["reason"].lower() or "invalid" in err["reason"].lower()


def test_csv_preview_missing_file_returns_400(client: FlaskClient):
    """Submitting preview without a file returns 400 Bad Request."""
    res = client.post("/import/preview", data={})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# Batch Processing Execution Tests
# ---------------------------------------------------------------------------

def test_process_batch_run_end_to_end(client: FlaskClient):
    """Batch processing imports CSV, runs pipeline, persists drafts, and renders results."""
    csv_content = (
        "email,company_name,contact_name,website,role_or_job_url,notes\n"
        "cto@fitsol.example,Fitsol,Dave CTO,https://fitsol.example,AI/ML Intern,Good fit\n"
        "invalid-email-address,Broken Co,Eve,https://broken.co,Intern,Broken row\n"
    )

    data = {
        "csv_file": (io.BytesIO(csv_content.encode("utf-8")), "test_batch.csv"),
        "target_role": "AI/ML Intern",
        "workflow_mode": "uploaded_cv",
        "campaign_id": "test_camp_ui_1",
    }

    res = client.post("/import/process", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    assert "Batch Processing Results" in html
    assert "cto@fitsol.example" in html
    assert "invalid-email-address" in html
    assert "Invalid or Skipped Rows" in html

    # Verify error report downloads
    # Find batch id in HTML
    import re
    match = re.search(r"batch_[a-f0-9]+", html)
    assert match is not None
    batch_id = match.group(0)

    # Test error report CSV download
    csv_res = client.get(f"/batch/{batch_id}/errors/csv")
    assert csv_res.status_code == 200
    assert "invalid-email-address" in csv_res.data.decode("utf-8")

    # Test error report JSON download
    json_res = client.get(f"/batch/{batch_id}/errors/json")
    assert json_res.status_code == 200
    err_json = json_res.get_json()
    assert err_json["failed_count"] >= 1


# ---------------------------------------------------------------------------
# Draft Review, Editing, and Approval Safeguards Tests
# ---------------------------------------------------------------------------

def test_draft_editing_resets_approval(client: FlaskClient, test_send_log: SendLog):
    """Editing draft subject or body must reset approval status to DRAFT."""
    original_subject = "AI/ML Internship Inquiry — TechCorp"
    original_body = "Dear Hiring Team, I am writing to apply..."
    draft = EmailDraft(
        draft_id="draft-ui-test-1",
        recipient="hiring@techcorp.example",
        recipient_name="Hiring Team",
        company_name="TechCorp",
        subject=original_subject,
        body=original_body,
        attachment_path="output/test_uploaded_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
        status=DraftStatus.DRAFT,
        approved=False,
        review_required=True,
        review_flags=["Company identity requires verification"],
    )
    save_draft_record(draft, campaign_id="camp-edit-test")

    # 1. Explicitly approve draft
    appr_res = client.post("/drafts/draft-ui-test-1/approve", follow_redirects=True)
    assert appr_res.status_code == 200
    d_after_appr = get_draft("draft-ui-test-1")
    assert d_after_appr["approved"] is True
    assert d_after_appr["status"] == "APPROVED"

    # 2. Resave / edit draft with unmodified subject and body
    edit_data = {
        "recipient": "hiring@techcorp.example",
        "recipient_name": "Hiring Team",
        "subject": original_subject,
        "body": original_body,
    }
    edit_res = client.post("/drafts/draft-ui-test-1/edit", data=edit_data, follow_redirects=True)
    assert edit_res.status_code == 200

    # 3. Assert approval is revoked without adding artificial test markers
    d_after_edit = get_draft("draft-ui-test-1")
    assert d_after_edit["subject"] == original_subject
    assert "[Human Reviewed]" not in d_after_edit["subject"]
    assert d_after_edit["body"] == original_body
    assert d_after_edit["approved"] is False
    assert d_after_edit["status"] == "DRAFT"

    # 4. Also test update_draft direct helper with reset_approval=True
    client.post("/drafts/draft-ui-test-1/approve", follow_redirects=True)
    assert get_draft("draft-ui-test-1")["approved"] is True
    d_direct_reset = update_draft("draft-ui-test-1", reset_approval=True)
    assert d_direct_reset["approved"] is False
    assert d_direct_reset["status"] == "DRAFT"
    assert d_direct_reset["subject"] == original_subject
    assert d_direct_reset["body"] == original_body


def test_draft_send_in_dry_run_mode(client: FlaskClient, test_send_log: SendLog):
    """Sending an approved draft in dry-run mode succeeds safely without network calls."""
    class MockProvider:
        def send_email(self, **kwargs):
            pass

    client.application.config["EMAIL_SENDER_PROVIDER"] = MockProvider()

    draft = EmailDraft(
        draft_id="draft-send-dry-1",
        recipient="lead@innovate.example",
        recipient_name="Lead",
        company_name="Innovate",
        subject="AI Application",
        body="Application body",
        attachment_path="output/test_uploaded_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
        status=DraftStatus.APPROVED,
        approved=True,
    )
    save_draft_record(draft, campaign_id="camp-send-dry")
    approve_draft("draft-send-dry-1", send_log=test_send_log)

    send_res = client.post("/drafts/draft-send-dry-1/send", follow_redirects=True)
    assert send_res.status_code == 200
    html = send_res.data.decode("utf-8")
    assert "Send outcome" in html or "Email sent successfully" in html or "Sent" in html

    # Resending to the same recipient should now be blocked by duplicate protection
    retry_res = client.post("/drafts/draft-send-dry-1/send", follow_redirects=True)
    assert retry_res.status_code == 200
    retry_html = retry_res.data.decode("utf-8")
    assert "DUPLICATE_RECIPIENT" in retry_html or "Send blocked" in retry_html


def test_unconfigured_gmail_reports_honest_error(client: FlaskClient, test_send_log: SendLog):
    """When Gmail is not configured and no mock provider is injected, UI honestly reports GMAIL_NOT_CONFIGURED."""
    client.application.config["EMAIL_SENDER_PROVIDER"] = None
    client.application.config["DRY_RUN"] = False

    draft = EmailDraft(
        draft_id="draft-gmail-unconf",
        recipient="founder@unconf.example",
        recipient_name="Founder",
        company_name="Unconf Co",
        subject="AI Application",
        body="Application body",
        attachment_path="output/test_uploaded_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
        status=DraftStatus.APPROVED,
        approved=True,
    )
    save_draft_record(draft, campaign_id="camp-unconf")
    approve_draft("draft-gmail-unconf", send_log=test_send_log)

    send_res = client.post("/drafts/draft-gmail-unconf/send", follow_redirects=True)
    assert send_res.status_code == 200
    html = send_res.data.decode("utf-8")
    assert "GMAIL_NOT_CONFIGURED" in html
    assert "Gmail integration is not configured" in html


# ---------------------------------------------------------------------------
# CV Settings & Tailoring Preview Tests
# ---------------------------------------------------------------------------

def test_cv_settings_page_and_tailoring_preview(client: FlaskClient):
    """CV settings page displays uploaded CV details and runs tailored CV generation preview."""
    res = client.get("/cv")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Uploaded CV Mode" in html
    assert "Tailored CV Mode" in html
    assert "SHA-256 Digest" in html

    # Test Tailored CV generator test form
    tailor_data = {
        "target_company": "Apex Deep Learning",
        "target_role": "AI/ML Intern",
        "job_description": "PyTorch, diffusion models, computer vision, distributed training",
    }
    tailor_res = client.post("/cv/test-tailor", data=tailor_data)
    assert tailor_res.status_code == 200
    tailor_html = tailor_res.data.decode("utf-8")
    assert "ControlPlane.ai" in tailor_html
    assert "Generated Markdown Content" in tailor_html


# ---------------------------------------------------------------------------
# Stage 2 & 3 Import Page Regression Tests
# ---------------------------------------------------------------------------

def test_import_page_renders_with_registered_cv(client: FlaskClient, tmp_path: Path):
    """GET /import returns HTTP 200 and renders CV details, wizard stepper, and upload controls."""
    test_pdf = tmp_path / "my_verified_cv.pdf"
    test_pdf.write_bytes(b"%PDF-1.4 test cv")
    register_uploaded_cv(test_pdf, original_filename="my_verified_cv.pdf")
    res = client.get("/import")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Outreach Campaign Wizard" in html
    assert "Master CV" in html
    assert "Contacts Spreadsheet" in html
    assert "Column Mapping" in html
    assert "Confirm &amp; Run" in html
    assert "my_verified_cv.pdf" in html
    assert "standardFields" in html
    assert "fieldDescriptions" in html


def test_import_page_renders_without_registered_cv(client: FlaskClient):
    """GET /import returns HTTP 200 when no CV has been uploaded yet."""
    clear_registered_cv()
    res = client.get("/import")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Outreach Campaign Wizard" in html
    assert "Click to choose or drag &amp; drop your CV PDF" in html


def test_import_page_template_defensive_against_missing_context(client: FlaskClient):
    """
    Ensure import.html does not raise TypeError or UndefinedError even if rendered
    with sparse context (such as missing standard_fields or registered_cv=None).
    """
    from flask import render_template
    from src.ui.app import create_app

    app = create_app({"TESTING": True})
    with app.test_request_context("/import"):
        # Explicitly pass None or omitted variables to simulate uninitialized session/context
        rendered = render_template(
            "import.html",
            active_tab="import",
            registered_cv=None,
            standard_fields=None,
            field_descriptions=None,
        )
        assert "Outreach Campaign Wizard" in rendered
        assert "standardFields: []" in rendered


def test_saved_draft_openai_subject_and_metadata_preserved():
    """Verify saved draft-97f974b1 subject is clean and all metadata is preserved."""
    draft = get_draft("draft-97f974b1")
    assert draft is not None, "Saved draft draft-97f974b1 must exist on disk"
    assert draft["subject"] == "AI/ML Internship Inquiry — OpenAI"
    assert "[Human Reviewed]" not in draft["subject"]
    assert "[Human Reviewed]" not in draft["body"]
    assert draft["recipient"] == "careers@openai.com"
    assert "Dear Hiring Team at OpenAI" in draft["body"]
    assert len(draft.get("research_evidence", [])) > 0
    assert len(draft.get("evidence_summary", [])) > 0
    assert draft["draft_id"] == "draft-97f974b1"


def test_draft_evidence_rendering_structured(client: FlaskClient):
    """Verify structured company evidence entries render readable title, excerpt, and external links."""
    draft = EmailDraft(
        draft_id="draft-ev-structured-test",
        recipient="research@deepai.example",
        recipient_name="Research Team",
        company_name="DeepAI",
        subject="AI/ML Internship Inquiry — DeepAI",
        body="Dear Research Team,\n\nI am writing to inquire...",
        attachment_path="output/test_uploaded_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
        status=DraftStatus.DRAFT,
        approved=False,
        review_required=False,
        evidence_summary=[
            {
                "keyword": "llm_analysis",
                "excerpt": "DeepAI builds low-latency transformer inference pipelines and retrieval engines.",
                "source_url": "https://deepai.example/research/transformers",
            },
            {
                "keyword": "diffusion_models",
                "excerpt": "Scaling generative vision models across multi-GPU distributed clusters.",
                "source_url": "https://deepai.example/careers/vision-engineer",
            },
        ],
    )
    save_draft_record(draft, campaign_id="camp-ev-test")

    res = client.get("/drafts/draft-ev-structured-test")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Clean titles/keywords
    assert "LLM Research Analysis" in html
    assert "Diffusion Models" in html

    # Excerpts
    assert "DeepAI builds low-latency transformer inference pipelines" in html
    assert "Scaling generative vision models" in html

    # Clickable source URLs opening in new tab
    assert 'href="https://deepai.example/research/transformers"' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html
    assert "Source link &nearr;" in html

    # No raw python dict representation exposed
    assert "[{'keyword':" not in html
    assert "[{&#39;keyword&#39;:" not in html
    assert "[{'" not in html


def test_draft_evidence_rendering_missing(client: FlaskClient):
    """Verify missing or empty evidence renders a clear friendly message."""
    for did, ev_val in [("draft-ev-none", None), ("draft-ev-empty-list", []), ("draft-ev-empty-str", "")]:
        draft = EmailDraft(
            draft_id=did,
            recipient="info@example.com",
            recipient_name="Team",
            company_name="Example Corp",
            subject="AI/ML Internship Inquiry — Example",
            body="Hello,\n\nInquiry...",
            attachment_path="output/test_uploaded_cv.pdf",
            mode=WorkflowMode.UPLOADED_CV,
            status=DraftStatus.DRAFT,
            approved=False,
            review_required=True,
            evidence_summary=ev_val,
        )
        save_draft_record(draft, campaign_id="camp-ev-missing")

        res = client.get(f"/drafts/{did}")
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        assert "No source evidence available." in html


def test_draft_evidence_rendering_malformed(client: FlaskClient):
    """Verify malformed evidence data is handled gracefully without crashing or showing traceback."""
    malformed_values = [
        [123, None, {}],
        {"invalid_key": 999},
        "   ",
        [{"keyword": "", "excerpt": "", "source_url": ""}],
    ]
    for idx, bad_val in enumerate(malformed_values):
        did = f"draft-ev-malformed-{idx}"
        draft = EmailDraft(
            draft_id=did,
            recipient="test@example.com",
            recipient_name="Team",
            company_name="Malformed Test Corp",
            subject="AI/ML Internship Inquiry — Malformed",
            body="Hello,\n\nInquiry...",
            attachment_path="output/test_uploaded_cv.pdf",
            mode=WorkflowMode.UPLOADED_CV,
            status=DraftStatus.DRAFT,
            approved=False,
            review_required=True,
            evidence_summary=bad_val,
        )
        save_draft_record(draft, campaign_id="camp-ev-malformed")

        res = client.get(f"/drafts/{did}")
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        assert "No source evidence available." in html


def test_draft_evidence_rendering_escaping_and_wrapping(client: FlaskClient):
    """Verify untrusted HTML and javascript URLs are safely sanitized and long URLs wrap."""
    draft = EmailDraft(
        draft_id="draft-ev-security-test",
        recipient="sec@example.com",
        recipient_name="Security Team",
        company_name="SecCorp",
        subject="AI/ML Internship Inquiry — SecCorp",
        body="Hello,\n\nInquiry...",
        attachment_path="output/test_uploaded_cv.pdf",
        mode=WorkflowMode.UPLOADED_CV,
        status=DraftStatus.DRAFT,
        approved=False,
        review_required=True,
        evidence_summary=[
            {
                "keyword": "<script>alert('kw')</script>",
                "excerpt": "<b onclick='alert(1)'>untrusted excerpt content</b>",
                "source_url": "javascript:alert('xss')",
            },
            {
                "keyword": "secure_system",
                "excerpt": "A" * 150,
                "source_url": "https://example.com/very/long/path/" + "B" * 120,
            },
        ],
    )
    save_draft_record(draft, campaign_id="camp-ev-sec")

    res = client.get("/drafts/draft-ev-security-test")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Safe escaping: Raw script tag must NOT exist in DOM
    assert "<script>alert('kw')</script>" not in html
    assert "&lt;Script&gt;Alert(&#39;Kw&#39;)&lt;/Script&gt;" in html or "&lt;script&gt;" in html or "Alert" in html

    # Raw onclick handler must NOT exist
    assert "<b onclick" not in html
    assert "untrusted excerpt content" in html

    # javascript: scheme must NOT be rendered as a link
    assert 'href="javascript:' not in html

    # Long safe URL must be present and have wrapping styles
    assert 'word-break: break-all' in html
    assert "https://example.com/very/long/path/" in html


def test_secret_key_production_enforcement(monkeypatch):
    """Production mode must raise RuntimeError if insecure or missing secret key."""
    from src.ui.app import create_app
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY must be set"):
        create_app()


def test_file_token_regex_security(client: FlaskClient):
    """Malicious or path-traversing file tokens are rejected with HTTP 400."""
    res1 = client.post("/contacts/preview", json={"file_token": "../../etc/passwd"})
    assert res1.status_code == 400
    assert "Invalid file token format" in res1.get_json()["error"]

    res2 = client.post("/contacts/download-report", json={"file_token": "../token"})
    assert res2.status_code == 400


def test_cv_download_security_restrictions(client: FlaskClient):
    """Only PDF files in approved directories can be downloaded."""
    # Attempting to download non-pdf file
    res = client.get("/download/cv/.session.json", follow_redirects=True)
    html = res.data.decode("utf-8")
    assert "Access denied: only PDF files may be downloaded" in html


def test_atomic_write_storage_helper(tmp_path: Path):
    """_atomic_write_json ensures atomic, uncorrupted writes to disk."""
    from src.ui.storage import _atomic_write_json
    target = tmp_path / "test.json"
    data = {"key": "initial_value", "items": [1, 2, 3]}

    _atomic_write_json(target, data)
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == data

    # Update atomically
    data["key"] = "updated_value"
    _atomic_write_json(target, data)
    assert json.loads(target.read_text(encoding="utf-8")) == data


def test_csrf_protection_enforced_when_enabled(tmp_path: Path, test_send_log: SendLog):
    """When WTF_CSRF_ENABLED is True, POST routes require a valid session CSRF token."""
    app = create_app({
        "TESTING": True,
        "SEND_LOG_PATH": str(test_send_log._db_path),
        "WTF_CSRF_ENABLED": True,
        "SECRET_KEY": "test-csrf-secret-key",
    })
    with app.test_client() as c:
        # Establish session by visiting a GET route
        res_get = c.get("/")
        assert res_get.status_code == 200

        # POST without CSRF token must fail with 400
        res_fail = c.post("/cv/clear")
        assert res_fail.status_code == 400
        assert "CSRF validation failed" in res_fail.data.decode("utf-8")

        # Now get the valid token from the session
        with c.session_transaction() as sess:
            valid_token = sess.get("csrf_token")
        assert valid_token is not None

        # POST with header X-CSRFToken succeeds (redirects back to settings page)
        res_ok = c.post("/cv/clear", headers={"X-CSRFToken": valid_token})
        assert res_ok.status_code in (200, 302)


def test_draft_path_traversal_rejected(client: FlaskClient):
    """Path traversal characters in draft_id cannot escape storage directory."""
    from src.ui.storage import get_draft
    # Storage layer rejects path traversal and returns None
    assert get_draft("../../etc/passwd") is None
    assert get_draft("..\\..\\windows\\win.ini") is None
    assert get_draft("valid_id_with_safe_chars_123") is None

    # Route layer safely redirects without crashing or leaking file paths
    res = client.get("/drafts/.._invalid_id", follow_redirects=True)
    assert res.status_code == 200
    assert "not found" in res.data.decode("utf-8").lower()


def test_cv_streaming_size_limit_rejection():
    """CV validator streaming rejects files larger than limit without unbounded memory buffering."""
    from src.ui.cv_validator import validate_cv_stream

    class OversizedDummyStream:
        def __init__(self, total_bytes: int):
            self.total = total_bytes
            self.read_so_far = 0

        def read(self, chunk_size=65536):
            if self.read_so_far >= self.total:
                return b""
            if self.read_so_far == 0:
                chunk = b"%PDF-1.4 " + b"X" * (chunk_size - 9)
            else:
                chunk = b"X" * min(chunk_size, self.total - self.read_so_far)
            self.read_so_far += len(chunk)
            return chunk

    # 11MB stream
    stream = OversizedDummyStream(11 * 1024 * 1024)
    res = validate_cv_stream(stream, "large.pdf", max_size_bytes=10 * 1024 * 1024)
    assert not res.is_valid
    assert "exceeds" in res.error.lower()


def test_csrf_rejection_across_state_changing_routes(tmp_path: Path, test_send_log: SendLog):
    """
    Verify that missing or invalid CSRF tokens are rejected with HTTP 400 across
    every class of state-changing route.
    """
    app = create_app({
        "TESTING": True,
        "SEND_LOG_PATH": str(test_send_log._db_path),
        "WTF_CSRF_ENABLED": True,
        "SECRET_KEY": "csrf-security-audit-key",
    })

    routes = [
        ("/cv/clear", "POST", {}),
        ("/cv/test-tailor", "POST", {"target_company": "Acme AI"}),
        ("/drafts/sample-draft/approve", "POST", {}),
        ("/drafts/sample-draft/reject", "POST", {}),
        ("/drafts/sample-draft/send", "POST", {}),
        ("/contacts/preview", "POST", {"file_token": "token123"}),
        ("/contacts/download-report", "POST", {"file_token": "token123"}),
    ]

    with app.test_client() as c:
        # Establish session
        c.get("/")
        with c.session_transaction() as sess:
            valid_token = sess["csrf_token"]

        for route, method, payload in routes:
            # 1. Missing CSRF token -> must reject with 400
            res_missing = c.post(route, data=payload)
            assert res_missing.status_code == 400, f"Route {route} did not reject missing CSRF token"
            assert "CSRF validation failed" in res_missing.data.decode("utf-8")

            # 2. Invalid CSRF token header -> must reject with 400
            res_invalid = c.post(route, data=payload, headers={"X-CSRFToken": "invalid-token-xyz"})
            assert res_invalid.status_code == 400, f"Route {route} did not reject invalid CSRF token"
            assert "CSRF validation failed" in res_invalid.data.decode("utf-8")

            # 3. Valid CSRF token in header -> passes CSRF layer (does not return CSRF 400)
            res_valid = c.post(route, data=payload, headers={"X-CSRFToken": valid_token})
            assert res_valid.status_code != 400 or "CSRF validation failed" not in res_valid.data.decode("utf-8"), \
                f"Route {route} rejected valid CSRF token"


def test_failed_spreadsheet_inspection_cleans_up_saved_file(tmp_path: Path, test_send_log: SendLog, monkeypatch):
    """
    Prove that if spreadsheet inspection fails, any uploaded file saved to
    output/batches_input/ is removed and not orphaned on disk.
    """
    app = create_app({
        "TESTING": True,
        "SEND_LOG_PATH": str(test_send_log._db_path),
        "WTF_CSRF_ENABLED": False,
    })

    batches_dir = Path("output/batches_input")
    batches_dir.mkdir(parents=True, exist_ok=True)
    initial_files = set(batches_dir.iterdir())

    # Simulate inspection exception
    def broken_inspect(path, sheet_name=None):
        raise ValueError("Corrupted workbook headers")

    monkeypatch.setattr("src.ui.app.inspect_spreadsheet", broken_inspect)

    with app.test_client() as c:
        fake_file = (io.BytesIO(b"not a valid sheet content"), "corrupted.xlsx")
        res = c.post("/contacts/inspect", data={"contacts_file": fake_file}, content_type="multipart/form-data")
        assert res.status_code == 400
        json_data = res.get_json()
        assert json_data["success"] is False
        assert "Ensure the file is not corrupted" in json_data["error"]

    # Verify no new files were left behind in output/batches_input
    current_files = set(batches_dir.iterdir())
    new_files = current_files - initial_files
    assert len(new_files) == 0, f"Orphaned temporary files left behind: {new_files}"


def test_dispatch_state_consistency_across_outcomes(tmp_path: Path, test_send_log: SendLog):
    """
    Verify complete synchronization between UI draft JSON and SQLite send-log across:
    - Simulated send
    - Mock live provider send in isolated test
    - Hard provider error
    - Uncertain outcome
    - Edit invalidation
    """
    import pypdf
    # 1. Create a genuine 1-page PDF
    valid_pdf = tmp_path / "dispatch_valid.pdf"
    w = pypdf.PdfWriter()
    w.add_blank_page(width=72, height=72)
    with open(valid_pdf, "wb") as f:
        w.write(f)

    # 2. Mock sending provider with configurable behavior
    class ControllableProvider:
        def __init__(self):
            self.mode = "success"
            self.sent = []

        def send_email(self, recipient, subject, body, attachment_path):
            if self.mode == "fail":
                raise RuntimeError("SMTP connection dropped unexpectedly")
            if self.mode == "timeout":
                raise RuntimeError("Gateway timeout waiting for server response")
            self.sent.append(recipient)

    mock_prov = ControllableProvider()

    app = create_app({
        "TESTING": True,
        "SEND_LOG_PATH": str(test_send_log._db_path),
        "WTF_CSRF_ENABLED": False,
        "DRY_RUN": True,
        "EMAIL_SENDER_PROVIDER": mock_prov,
    })

    # Setup draft
    draft_id = "draft-sync-audit"
    draft = EmailDraft(
        draft_id=draft_id,
        recipient="sync-audit@example.com",
        recipient_name="Auditor",
        company_name="AuditCorp",
        subject="Audit Subject",
        body="Audit Body",
        attachment_path=str(valid_pdf),
        mode=WorkflowMode.UPLOADED_CV,
        status=DraftStatus.DRAFT,
        approved=False,
    )
    save_draft_record(draft, campaign_id="sync-camp")
    test_send_log.record_draft(draft_id, "sync-audit@example.com", "sync-camp")

    with app.test_client() as c:
        # A. Unapproved send attempt is blocked
        res_unapproved = c.post(f"/drafts/{draft_id}/send", follow_redirects=True)
        assert res_unapproved.status_code == 200
        assert get_draft(draft_id)["status"] == "DRAFT"
        assert test_send_log.get_status(draft_id) in ("DRAFT", None)

        # B. Approve draft
        c.post(f"/drafts/{draft_id}/approve", follow_redirects=True)
        assert get_draft(draft_id)["status"] == "APPROVED"
        assert test_send_log.get_status(draft_id) == "APPROVED"

        # C. Simulated send (dry_run=True)
        c.post(f"/drafts/{draft_id}/send", follow_redirects=True)
        assert get_draft(draft_id)["status"] == "SIMULATED_SENT"
        assert test_send_log.get_status(draft_id) == "SIMULATED"
        assert test_send_log.was_simulated("sync-audit@example.com", "sync-camp")
        assert not test_send_log.was_successfully_sent("sync-audit@example.com", "sync-camp")

        # D. Edit draft -> must reset approval in BOTH draft JSON and SQLite send-log
        c.post(
            f"/drafts/{draft_id}/edit",
            data={"subject": "Revised Audit Subject", "body": "Revised Body"},
            follow_redirects=True,
        )
        assert get_draft(draft_id)["status"] == "DRAFT"
        assert get_draft(draft_id)["approved"] is False
        assert test_send_log.get_status(draft_id) == "DRAFT"

        # E. Live mock provider send in isolated test (DRY_RUN=False)
        app.config["DRY_RUN"] = False
        # Re-approve
        c.post(f"/drafts/{draft_id}/approve", follow_redirects=True)
        assert get_draft(draft_id)["status"] == "APPROVED"
        assert test_send_log.get_status(draft_id) == "APPROVED"

        # Send live with mock provider
        c.post(f"/drafts/{draft_id}/send", follow_redirects=True)
        assert get_draft(draft_id)["status"] == "SENT"
        assert test_send_log.get_status(draft_id) == "SUCCESS"
        assert test_send_log.was_successfully_sent("sync-audit@example.com", "sync-camp")
        assert len(mock_prov.sent) == 1

        # F. Re-dispatch already-sent recipient -> blocked as DUPLICATE_RECIPIENT
        res_dup = c.post(f"/drafts/{draft_id}/send", follow_redirects=True)
        assert "already been sent" in res_dup.data.decode("utf-8")

        # G. Hard provider error on new draft
        d_fail_id = "draft-fail-test"
        d_fail = EmailDraft(
            draft_id=d_fail_id,
            recipient="fail-target@example.com",
            recipient_name="Fail Target",
            company_name="FailCorp",
            subject="Fail Test",
            body="Fail Body",
            attachment_path=str(valid_pdf),
            mode=WorkflowMode.UPLOADED_CV,
            status=DraftStatus.DRAFT,
            approved=False,
        )
        save_draft_record(d_fail, campaign_id="sync-camp")
        test_send_log.record_draft(d_fail_id, "fail-target@example.com", "sync-camp")

        c.post(f"/drafts/{d_fail_id}/approve", follow_redirects=True)
        mock_prov.mode = "fail"
        c.post(f"/drafts/{d_fail_id}/send", follow_redirects=True)
        assert get_draft(d_fail_id)["status"] == "FAILED"
        assert test_send_log.get_status(d_fail_id) == "FAILED"

        # H. Uncertain outcome on new draft
        d_unc_id = "draft-unc-test"
        d_unc = EmailDraft(
            draft_id=d_unc_id,
            recipient="unc-target@example.com",
            recipient_name="Unc Target",
            company_name="UncCorp",
            subject="Unc Test",
            body="Unc Body",
            attachment_path=str(valid_pdf),
            mode=WorkflowMode.UPLOADED_CV,
            status=DraftStatus.DRAFT,
            approved=False,
        )
        save_draft_record(d_unc, campaign_id="sync-camp")
        test_send_log.record_draft(d_unc_id, "unc-target@example.com", "sync-camp")

        c.post(f"/drafts/{d_unc_id}/approve", follow_redirects=True)
        mock_prov.mode = "timeout"
        c.post(f"/drafts/{d_unc_id}/send", follow_redirects=True)
        assert get_draft(d_unc_id)["status"] == "UNCERTAIN"
        assert test_send_log.get_status(d_unc_id) == "UNCERTAIN"
        assert test_send_log.has_uncertain_send("unc-target@example.com", "sync-camp")



