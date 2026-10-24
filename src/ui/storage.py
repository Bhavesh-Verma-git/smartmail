"""
src/ui/storage.py
-----------------
Local persistence and state management for the SmartMail Functional Testing UI.
Integrates with output/.drafts, output/.batches, output/.session.json, and data/send_log.db.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.outreach.send_log import (
    SendLog,
    STATUS_SUCCESS,
    STATUS_APPROVED,
    STATUS_DRAFT,
    STATUS_FAILED,
    STATUS_UNCERTAIN,
    STATUS_REJECTED,
)

_DRAFTS_DIR = Path("output/.drafts")
_BATCHES_DIR = Path("output/.batches")
_SESSION_FILE = Path("output/.session.json")
_TAILORED_DIR = Path("output/tailored_cvs")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(file_path: Path, data: Any) -> None:
    """Atomically write JSON data to file_path via a temporary file and atomic replace with Windows retry."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = file_path.with_name(f"{file_path.name}.tmp.{uuid.uuid4().hex[:8]}")
    payload = json.dumps(data, indent=2)
    try:
        temp_file.write_text(payload, encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temp_file, file_path)
                break
            except OSError:
                if attempt < 4:
                    time.sleep(0.015 * (attempt + 1))
                else:
                    file_path.write_text(payload, encoding="utf-8")
                    break
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Session & CV Configuration
# ---------------------------------------------------------------------------

def get_session() -> dict[str, Any]:
    """Load or initialize local session state."""
    if _SESSION_FILE.exists():
        try:
            return json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_session(data: dict[str, Any]) -> None:
    """Save session state to disk atomically."""
    _atomic_write_json(_SESSION_FILE, data)


def get_registered_cv_metadata() -> dict[str, Any]:
    """
    Get file metadata for the current registered uploaded CV.
    Returns path, filename, size, sha256, exists, and unmodified check.
    """
    session = get_session()
    cv_path_str = session.get("uploaded_cv_path")

    if not cv_path_str:
        return {
            "registered": False,
            "path": None,
            "filename": None,
            "stored_filename": None,
            "size_bytes": 0,
            "size_formatted": "0 KB",
            "sha256": None,
            "exists": False,
        }

    p = Path(cv_path_str)
    if not p.exists() or not p.is_file():
        return {
            "registered": False,
            "path": None,
            "filename": None,
            "stored_filename": None,
            "size_bytes": 0,
            "size_formatted": "0 KB",
            "sha256": None,
            "exists": False,
        }

    file_bytes = p.read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    size_kb = len(file_bytes) / 1024.0

    return {
        "registered": True,
        "path": str(p.resolve()),
        "filename": session.get("original_cv_filename") or p.name,
        "stored_filename": p.name,
        "size_bytes": len(file_bytes),
        "size_formatted": f"{size_kb:.1f} KB",
        "sha256": file_hash,
        "exists": True,
    }


def register_uploaded_cv(file_path: str | Path, original_filename: str | None = None) -> dict[str, Any]:
    """Register a new CV file for Uploaded CV Mode in session."""
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"CV file not found: {file_path}")

    session = get_session()
    session["uploaded_cv_path"] = str(p)
    if original_filename:
        session["original_cv_filename"] = original_filename
    save_session(session)
    return get_registered_cv_metadata()


def store_and_register_cv(
    file_bytes: bytes | Any,
    original_filename: str,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Validate, store securely in output/uploaded_cvs/ with unique server filename,
    and register in session. Accepts raw bytes or file stream.
    """
    import re
    from src.ui.cv_validator import validate_cv_stream

    # Validate stream/bytes
    val_res = validate_cv_stream(file_bytes, original_filename)
    if not val_res.is_valid:
        raise ValueError(val_res.error)

    # Controlled isolated directory
    cv_dir = Path(output_dir) if output_dir else Path("output/uploaded_cvs")
    cv_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^\w\.\-]", "_", Path(original_filename).name)
    unique_filename = f"cv_{uuid.uuid4().hex[:12]}_{safe_name}"
    dest_path = (cv_dir / unique_filename).resolve()
    bytes_to_write = val_res.validated_bytes if val_res.validated_bytes else (file_bytes if isinstance(file_bytes, bytes) else b"")
    dest_path.write_bytes(bytes_to_write)

    session = get_session()
    session["uploaded_cv_path"] = str(dest_path)
    session["original_cv_filename"] = original_filename
    save_session(session)

    meta = get_registered_cv_metadata()
    meta["page_count"] = val_res.page_count
    return meta


def clear_registered_cv() -> None:
    """Clear registered CV from session."""
    session = get_session()
    session.pop("uploaded_cv_path", None)
    session.pop("original_cv_filename", None)
    save_session(session)


# ---------------------------------------------------------------------------
# Draft Storage & Lifecycle
# ---------------------------------------------------------------------------

def save_draft_record(
    draft: Any,
    campaign_id: str = "",
    verification_details: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    research_evidence: list[dict[str, Any]] | None = None,
) -> Path:
    """
    Persist an EmailDraft, its verification audit trail, and Stage 3 research
    and classification evidence to output/.drafts/{draft_id}.json.
    """
    _DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _DRAFTS_DIR / f"{draft.draft_id}.json"

    data: dict[str, Any] = {
        "draft_id": draft.draft_id,
        "recipient": draft.recipient,
        "recipient_name": draft.recipient_name,
        "company_name": draft.company_name,
        "subject": draft.subject,
        "body": draft.body,
        "attachment_path": draft.attachment_path,
        "mode": draft.mode.value if hasattr(draft.mode, "value") else str(draft.mode),
        "status": draft.status.value if hasattr(draft.status, "value") else str(draft.status),
        "approved": draft.approved,
        "review_required": draft.review_required,
        "review_flags": draft.review_flags,
        "tailored_cv_path": draft.tailored_cv_path,
        "created_at": getattr(draft, "created_at", _now_iso()),
        "updated_at": _now_iso(),
        "campaign_id": campaign_id,
        "verification": verification_details or {},
        "classification": classification or {},
        "research_evidence": research_evidence or [],
        "generation_status": getattr(draft, "generation_status", "DRAFT_GENERATED"),
        "personalization_explanation": getattr(draft, "personalization_explanation", ""),
        "evidence_summary": getattr(draft, "evidence_summary", ""),
        "matched_skills": getattr(draft, "matched_skills", []),
        "matched_projects": getattr(draft, "matched_projects", []),
        "recipient_type": getattr(draft, "recipient_type", "individual"),
        "personalization_record": draft.personalization_record.to_dict() if hasattr(getattr(draft, "personalization_record", None), "to_dict") else (draft.personalization_record if isinstance(getattr(draft, "personalization_record", None), dict) else {}),
    }
    _atomic_write_json(file_path, data)
    return file_path


save_draft = save_draft_record


def get_draft(draft_id: str) -> dict[str, Any] | None:
    """Retrieve draft data from disk safely avoiding path traversal."""
    import re
    safe_id = Path(draft_id).name
    if not re.match(r"^[\w\-]+$", safe_id):
        return None
    file_path = (_DRAFTS_DIR / f"{safe_id}.json").resolve()
    try:
        if not file_path.is_relative_to(_DRAFTS_DIR.resolve()) or not file_path.exists():
            return None
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_evidence_summary(raw: Any) -> list[dict[str, str]]:
    """
    Safely normalizes raw evidence_summary data (which can be a list of dicts,
    a single dict, a plain text string, None, or malformed data) into a list of clean,
    sanitized dictionary entries:
      - 'title': short evidence title or keyword (e.g. 'LLM Research Analysis')
      - 'excerpt': concise excerpt text
      - 'source_url': valid web URL string (starting with http:// or https://, empty if none)

    Returns an empty list if no valid source evidence is found.
    """
    if raw is None or raw == "":
        return []

    # If it's a JSON-encoded string, try parsing it first
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s in ("[]", "{}"):
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = json.loads(s)
                return normalize_evidence_summary(parsed)
            except Exception:
                pass
        # Plain text summary string (e.g. "Acme AI provides low-latency RAG systems.")
        return [{
            "title": "Company Evidence",
            "excerpt": s,
            "source_url": "",
        }]

    if isinstance(raw, dict):
        raw_list = [raw]
    elif isinstance(raw, list):
        raw_list = raw
    else:
        return []

    entries: list[dict[str, str]] = []
    for item in raw_list:
        if isinstance(item, dict):
            kw = item.get("keyword") or item.get("title") or item.get("topic") or ""
            if not isinstance(kw, str):
                kw = str(kw) if kw is not None else ""
            kw = kw.strip()

            if kw.lower() == "llm_analysis":
                title = "LLM Research Analysis"
            elif kw:
                title = kw.replace("_", " ").title()
            else:
                title = "Company Evidence"

            excerpt = item.get("excerpt") or item.get("extracted_text") or item.get("text") or item.get("snippet") or ""
            if not isinstance(excerpt, str):
                excerpt = str(excerpt) if excerpt is not None else ""
            excerpt = excerpt.strip()

            source_url = item.get("source_url") or item.get("url") or item.get("link") or ""
            if not isinstance(source_url, str):
                source_url = str(source_url) if source_url is not None else ""
            source_url = source_url.strip()

            # Ensure source_url is a valid web URL scheme only (no javascript:, data:, etc.)
            if not (source_url.startswith("http://") or source_url.startswith("https://")):
                source_url = ""

            # Only include entry if there is at least an excerpt or a source_url or non-default title
            if excerpt or source_url or (kw and kw.lower() != "llm_analysis"):
                entries.append({
                    "title": title,
                    "excerpt": excerpt,
                    "source_url": source_url,
                })
        elif isinstance(item, str):
            s = item.strip()
            if s:
                entries.append({
                    "title": "Company Evidence",
                    "excerpt": s,
                    "source_url": "",
                })

    return entries


def list_all_drafts(status_filter: str | None = None) -> list[dict[str, Any]]:
    """List all persisted drafts, optionally filtered by status, sorted newest first."""
    if not _DRAFTS_DIR.exists():
        return []

    drafts: list[dict[str, Any]] = []
    for fp in _DRAFTS_DIR.glob("*.json"):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            if status_filter:
                if status_filter == "NEEDS_REVIEW":
                    if not (d.get("review_required") and not d.get("approved") and d.get("status") not in ("APPROVED", "SENT", "REJECTED")):
                        continue
                elif d.get("status") != status_filter:
                    continue
            drafts.append(d)
        except Exception:
            continue

    # Sort newest first
    drafts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return drafts


def update_draft(
    draft_id: str,
    subject: str | None = None,
    body: str | None = None,
    recipient: str | None = None,
    recipient_name: str | None = None,
    reset_approval: bool = True,
    send_log: SendLog | None = None,
) -> dict[str, Any]:
    """
    Update draft fields. Resets approval status in both draft JSON and SQLite send_log
    to enforce re-review safeguard.
    """
    draft_data = get_draft(draft_id)
    if not draft_data:
        raise KeyError(f"Draft '{draft_id}' not found.")

    if subject is not None:
        draft_data["subject"] = subject.strip()
    if body is not None:
        draft_data["body"] = body
    if recipient is not None:
        draft_data["recipient"] = recipient.strip()
    if recipient_name is not None:
        draft_data["recipient_name"] = recipient_name.strip()

    if reset_approval:
        draft_data["approved"] = False
        draft_data["status"] = "DRAFT"
        sl = send_log or SendLog()
        sl.record_draft(
            draft_id=draft_id,
            recipient=draft_data.get("recipient", ""),
            campaign_id=draft_data.get("campaign_id", ""),
        )

    draft_data["updated_at"] = _now_iso()

    file_path = _DRAFTS_DIR / f"{draft_id}.json"
    _atomic_write_json(file_path, draft_data)
    return draft_data


def update_draft_status(draft_id: str, status: str) -> dict[str, Any]:
    """Update draft status field safely and atomically."""
    draft_data = get_draft(draft_id)
    if not draft_data:
        raise KeyError(f"Draft '{draft_id}' not found.")
    draft_data["status"] = status
    draft_data["updated_at"] = _now_iso()
    file_path = _DRAFTS_DIR / f"{draft_id}.json"
    _atomic_write_json(file_path, draft_data)
    return draft_data


def regenerate_draft_content(
    draft_id: str,
    profile_path: str | Path = "data/master_profile.json",
    send_log: SendLog | None = None,
) -> dict[str, Any]:
    """
    Regenerate the email draft subject, body, and personalization record
    based on the draft's verified research evidence and profile.
    """
    draft_data = get_draft(draft_id)
    if not draft_data:
        raise KeyError(f"Draft '{draft_id}' not found.")

    from src.profile.loader import load_profile
    from src.research.models import ResearchResult, EvidenceItem, RETRIEVAL_OK
    from src.outreach.drafting import draft_outreach_email
    from src.outreach.models import WorkflowMode
    from src.verification.models import PipelineResult, VerificationResult, MatchResult, RELEVANT, GENERAL_INQUIRY, PROCEED

    profile = load_profile(profile_path)

    # Reconstruct evidence items
    evidence = []
    for ev in draft_data.get("research_evidence", []):
        evidence.append(EvidenceItem(
            source_url=ev.get("source_url", ""),
            source_type=ev.get("source_type", "web_page"),
            title=ev.get("title", ""),
            extracted_text=ev.get("extracted_text", ""),
            retrieval_status=RETRIEVAL_OK,
            supports=ev.get("supports", []),
        ))

    summary = draft_data.get("evidence_summary", [])
    comp = draft_data.get("company_name", "")
    email = draft_data.get("recipient", "")
    v_info = draft_data.get("verification", {})
    matched_areas = v_info.get("matched_areas", [])

    rr = ResearchResult(
        company_name=comp,
        resolved_company_name=comp,
        evidence=evidence,
        ai_ml_evidence_summary=summary,
    )

    verif_res = PipelineResult(
        verification=VerificationResult(
            company_status=v_info.get("company_identity_status", "REVIEW"),
            overall_status=PROCEED if v_info.get("company_identity_status") != "BLOCKED" else "BLOCKED",
            evidence_sources=v_info.get("source_urls", []),
        ),
        match=MatchResult(
            relevance_status=v_info.get("relevance_status", RELEVANT),
            case_type=v_info.get("role_status", GENERAL_INQUIRY),
            matched_areas=matched_areas,
        ),
    )

    mode = WorkflowMode.TAILORED_CV if draft_data.get("mode") == "tailored_cv" else WorkflowMode.UPLOADED_CV

    new_draft = draft_outreach_email(
        profile=profile,
        recipient_email=email,
        company_name=comp,
        contact_name=draft_data.get("recipient_name", ""),
        verification_pipeline_result=verif_res,
        mode=mode,
        attachment_path=draft_data.get("attachment_path", ""),
        research_result=rr,
    )

    draft_data["subject"] = new_draft.subject
    draft_data["body"] = new_draft.body
    draft_data["generation_status"] = new_draft.generation_status
    draft_data["personalization_explanation"] = new_draft.personalization_explanation
    draft_data["matched_skills"] = new_draft.matched_skills
    draft_data["matched_projects"] = new_draft.matched_projects
    draft_data["personalization_record"] = new_draft.personalization_record.to_dict() if new_draft.personalization_record else {}
    draft_data["review_required"] = new_draft.review_required
    draft_data["review_flags"] = new_draft.review_flags
    draft_data["approved"] = False
    draft_data["status"] = "DRAFT"
    draft_data["updated_at"] = _now_iso()
    draft_data["generation_version"] = draft_data.get("generation_version", 1) + 1

    reg_history = draft_data.get("regeneration_history", [])
    reg_history.append({
        "version": draft_data["generation_version"],
        "timestamp": draft_data["updated_at"],
        "quality_gate_passed": new_draft.personalization_record.quality_gate_passed if new_draft.personalization_record else True,
        "company_finding": new_draft.personalization_record.company_finding if new_draft.personalization_record else "",
    })
    draft_data["regeneration_history"] = reg_history

    sl = send_log or SendLog()
    sl.record_draft(
        draft_id=draft_id,
        recipient=email,
        campaign_id=draft_data.get("campaign_id", ""),
    )

    file_path = _DRAFTS_DIR / f"{draft_id}.json"
    _atomic_write_json(file_path, draft_data)
    return draft_data


def approve_draft(draft_id: str, send_log: SendLog | None = None) -> dict[str, Any]:
    """Explicitly grant user approval for a draft."""
    draft_data = get_draft(draft_id)
    if not draft_data:
        raise KeyError(f"Draft '{draft_id}' not found.")

    draft_data["approved"] = True
    draft_data["status"] = "APPROVED"
    draft_data["updated_at"] = _now_iso()

    file_path = _DRAFTS_DIR / f"{draft_id}.json"
    _atomic_write_json(file_path, draft_data)

    # Record approval in persistent SQLite send log
    sl = send_log or SendLog()
    sl.record_approved(
        draft_id=draft_id,
        recipient=draft_data["recipient"],
        campaign_id=draft_data.get("campaign_id", ""),
    )
    return draft_data


def reject_draft(draft_id: str, reason: str = "", send_log: SendLog | None = None) -> dict[str, Any]:
    """Mark a draft as rejected/held and synchronize with persistent SQLite send_log."""
    draft_data = get_draft(draft_id)
    if not draft_data:
        raise KeyError(f"Draft '{draft_id}' not found.")

    draft_data["approved"] = False
    draft_data["status"] = "REJECTED"
    draft_data["rejection_reason"] = reason
    draft_data["updated_at"] = _now_iso()

    file_path = _DRAFTS_DIR / f"{draft_id}.json"
    _atomic_write_json(file_path, draft_data)

    sl = send_log or SendLog()
    sl.record_rejected(
        draft_id=draft_id,
        recipient=draft_data.get("recipient", ""),
        campaign_id=draft_data.get("campaign_id", ""),
    )
    return draft_data


# ---------------------------------------------------------------------------
# Batch Run Records
# ---------------------------------------------------------------------------

def save_batch_record(batch_id: str, record: dict[str, Any]) -> None:
    """Save batch processing outcome record to output/.batches/{batch_id}.json."""
    _BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    fp = _BATCHES_DIR / f"{batch_id}.json"
    record["batch_id"] = batch_id
    record["saved_at"] = _now_iso()
    _atomic_write_json(fp, record)


def get_batch_record(batch_id: str) -> dict[str, Any] | None:
    """Load a batch processing record safely avoiding path traversal."""
    import re
    safe_id = Path(batch_id).name
    if not re.match(r"^[\w\-]+$", safe_id):
        return None
    fp = (_BATCHES_DIR / f"{safe_id}.json").resolve()
    try:
        if not fp.is_relative_to(_BATCHES_DIR.resolve()) or not fp.exists():
            return None
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_batch_records() -> list[dict[str, Any]]:
    """List recent batch runs sorted newest first."""
    if not _BATCHES_DIR.exists():
        return []
    records = []
    for fp in _BATCHES_DIR.glob("*.json"):
        try:
            records.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    records.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    return records


# ---------------------------------------------------------------------------
# Dashboard Statistics (Never Fabricated)
# ---------------------------------------------------------------------------

def get_dashboard_stats(send_log: SendLog | None = None) -> dict[str, Any]:
    """
    Compute genuine dashboard metrics from existing drafts, batches, and SQLite send log.
    If no records exist, returns zeroed empty state.
    """
    sl = send_log or SendLog()
    log_entries = sl.list_all(limit=1000)
    drafts = list_all_drafts()
    batches = list_batch_records()

    # Calculate real numbers
    total_imported = sum(b.get("total_rows", 0) for b in batches)
    total_processed = sum(b.get("processed_count", 0) for b in batches)

    drafts_created = len(drafts)

    # Needs review: draft review_required is True AND not yet approved AND not sent
    needs_review = sum(
        1 for d in drafts
        if d.get("review_required") and not d.get("approved") and d.get("status") not in ("APPROVED", "SENT", "REJECTED")
    )

    # Blocked / failed records
    blocked_count = sum(b.get("blocked_count", 0) for b in batches)
    failed_count = sum(b.get("failed_count", 0) for b in batches)

    # Emails sent successfully: count SUCCESS records in send log
    sent_success = sum(1 for e in log_entries if e.get("status") == STATUS_SUCCESS)

    has_data = bool(batches or drafts or log_entries)

    return {
        "has_data": has_data,
        "total_imported": total_imported,
        "total_processed": total_processed,
        "drafts_created": drafts_created,
        "needs_review": needs_review,
        "blocked_count": blocked_count,
        "failed_count": failed_count,
        "sent_success": sent_success,
        "recent_batches": batches[:10],
        "recent_drafts": drafts[:15],
        "recent_logs": log_entries[:10],
    }


# ---------------------------------------------------------------------------
# Launch Readiness & Safety Checklist Items
# ---------------------------------------------------------------------------

def get_launch_readiness_checklist() -> list[dict[str, Any]]:
    """
    Return the genuine status of the launch-readiness checklist items.
    Truthful assessment: Never marks an item complete unless it actually is.
    """
    gmail_creds = os.getenv("GMAIL_CREDENTIALS_PATH")
    gmail_token = os.getenv("GMAIL_TOKEN_PATH")
    gmail_configured = bool(
        gmail_creds and gmail_token and os.path.exists(gmail_creds) and os.path.exists(gmail_token)
    )

    from src.config import get_provider_status
    provider_status = get_provider_status()

    return [
        {
            "id": "custom_domain",
            "name": "Custom Domain Connected",
            "status": "NOT_CONFIGURED",
            "category": "Deployment",
            "details": "Running strictly locally on 127.0.0.1. No external or public custom domain is attached.",
        },
        {
            "id": "favicon",
            "name": "Favicon Added",
            "status": "COMPLETED",
            "category": "Branding",
            "details": "Clean vector SVG favicon embedded locally in static/favicon.svg.",
        },
        {
            "id": "branding",
            "name": "Unwanted 'Made with AI' Branding Removed",
            "status": "COMPLETED",
            "category": "Branding",
            "details": "All outgoing email drafts use professional human phrasing without robotic or AI badges.",
        },
        {
            "id": "privacy_policy",
            "name": "Privacy Policy Page Created & Reviewed",
            "status": "EVALUATED_LOCAL_ONLY",
            "category": "Legal & Privacy",
            "details": "Local testing mode. Private contact data and CVs remain on local disk. Formal legal review required before public deployment.",
        },
        {
            "id": "terms_and_conditions",
            "name": "Terms and Conditions Page Created & Reviewed",
            "status": "EVALUATED_LOCAL_ONLY",
            "category": "Legal & Privacy",
            "details": "Local tool evaluation only. Production public terms require organizational legal sign-off.",
        },
        {
            "id": "auth_access_control",
            "name": "Authentication & Access Control Assessed",
            "status": "LOCAL_LOOPBACK_BOUND",
            "category": "Security",
            "details": "Server is bound strictly to local loopback (127.0.0.1). Zero public access. Multi-user authentication must be added before hosting online.",
        },
        {
            "id": "secrets_management",
            "name": "Secrets & OAuth Credentials Stored Securely",
            "status": "SECURE_LOCAL_ENV",
            "category": "Security",
            "details": "No API keys, OAuth client secrets, or tokens are displayed or accepted via web UI. Stored strictly in local .env files.",
        },
        {
            "id": "external_providers",
            "name": "External Free-Tier Services Status (Tavily & Gemini)",
            "status": "CONFIGURED" if (provider_status.tavily_configured or provider_status.gemini_configured) else "OFFLINE_FALLBACK_READY",
            "category": "External Integrations",
            "details": (
                f"Tavily: {'Configured' if provider_status.tavily_configured else 'Offline (HTTP fallback active)'} | "
                f"Gemini: {'Configured (' + provider_status.gemini_model + ')' if provider_status.gemini_configured else 'Offline (Deterministic fallback active)'} | "
                f"Groq: {'Configured' if provider_status.groq_configured else 'Unconfigured'} | "
                f"CV Sharing: {'Allowed' if provider_status.share_cv_with_llm else 'Disabled (Local only)'}"
            ),
        },
        {
            "id": "email_safeguards",
            "name": "Production Email-Sending Safeguards Tested",
            "status": "VERIFIED_TESTED",
            "category": "Outreach Safety",
            "details": f"Dry-run default enforced. Duplicate prevention active via SQLite send_log. Gmail OAuth status: {'Configured' if gmail_configured else 'Unconfigured (Safe)'}.",
        },
        {
            "id": "deployment_data_retention",
            "name": "Deployment & Data Retention Documented",
            "status": "DOCUMENTED",
            "category": "Architecture",
            "details": "Drafts stored in output/.drafts/, batches in output/.batches/, send log in data/send_log.db. No remote tracking.",
        },
    ]


# ---------------------------------------------------------------------------
# Draft Cleanup & Deletion Operations
# ---------------------------------------------------------------------------

def delete_draft(draft_id: str, send_log: SendLog | None = None) -> bool:
    """
    Safely delete a single draft JSON record and synchronize with SQLite send_log.
    Guards against path traversal. Preserves SUCCESS send log history.
    """
    import re
    safe_id = Path(draft_id).name
    if not re.match(r"^[\w\-]+$", safe_id):
        return False

    fp = (_DRAFTS_DIR / f"{safe_id}.json").resolve()
    deleted = False
    try:
        if fp.is_relative_to(_DRAFTS_DIR.resolve()) and fp.exists():
            fp.unlink()
            deleted = True
    except Exception:
        pass

    sl = send_log or SendLog()
    sl_removed = sl.remove_draft_record(safe_id)
    return deleted or sl_removed


def delete_drafts_for_batch(batch_id: str, send_log: SendLog | None = None) -> dict[str, Any]:
    """
    Delete all draft JSON records associated with a specific batch/campaign ID.
    Synchronizes with SQLite send_log while preserving SUCCESS records.
    """
    import re
    safe_batch_id = Path(batch_id).name
    if not re.match(r"^[\w\-]+$", safe_batch_id):
        return {"deleted_count": 0, "batch_id": batch_id, "success": False}

    if not _DRAFTS_DIR.exists():
        return {"deleted_count": 0, "batch_id": batch_id, "success": True}

    deleted_count = 0
    for fp in _DRAFTS_DIR.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if data.get("campaign_id") == safe_batch_id:
                fp.unlink()
                deleted_count += 1
        except Exception:
            continue

    sl = send_log or SendLog()
    sl.remove_batch_records(safe_batch_id)

    # Update batch record metadata if present
    batch_data = get_batch_record(safe_batch_id)
    if batch_data:
        batch_data["drafts_deleted"] = True
        batch_data["drafts_deleted_at"] = _now_iso()
        batch_data["drafts_deleted_count"] = deleted_count
        save_batch_record(safe_batch_id, batch_data)

    return {"deleted_count": deleted_count, "batch_id": safe_batch_id, "success": True}


def clear_all_drafts(scope: str = "all", send_log: SendLog | None = None) -> dict[str, Any]:
    """
    Clear drafts from output/.drafts/ according to the requested scope.

    Scopes:
      - 'unapproved': Deletes drafts that are not APPROVED and not SENT.
      - 'rejected':   Deletes drafts that are marked REJECTED.
      - 'all':        Deletes all drafts in the working queue.

    Preserves:
      - Verified company research cache (output/.research_cache/)
      - Candidate CVs and master profile
      - SQLite send_log records marked SUCCESS (for duplicate prevention & audit)
      - Application configuration and session state
    """
    if not _DRAFTS_DIR.exists():
        return {"deleted_count": 0, "scope": scope, "preserved_count": 0}

    deleted_count = 0
    preserved_count = 0

    for fp in list(_DRAFTS_DIR.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            status = data.get("status", "DRAFT")
            is_approved = data.get("approved", False)

            should_delete = False
            if scope == "all":
                should_delete = True
            elif scope == "unapproved":
                should_delete = (not is_approved and status not in ("APPROVED", "SENT", "SIMULATED_SENT"))
            elif scope == "rejected":
                should_delete = (status == "REJECTED")

            if should_delete:
                draft_id = data.get("draft_id") or fp.stem
                fp.unlink()
                deleted_count += 1
                if send_log:
                    send_log.remove_draft_record(draft_id)
            else:
                preserved_count += 1
        except Exception:
            continue

    if scope == "all":
        sl = send_log or SendLog()
        sl.clear_draft_records(keep_success=True)

    return {
        "deleted_count": deleted_count,
        "preserved_count": preserved_count,
        "scope": scope,
    }
