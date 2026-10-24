"""
src/cli.py
----------
Command-line interface for the AI Internship Outreach Assistant.

Usage
-----
  python -m src.cli --help
  python -m src.cli upload-cv --file path/to/cv.pdf
  python -m src.cli run-uploaded --contact contacts.csv --row 2 --role "AI/ML Intern" --jd "..."
  python -m src.cli run-tailored --contact contacts.csv --row 2 --role "AI/ML Intern" --jd "..."
  python -m src.cli review-draft --id draft-abc12345
  python -m src.cli approve-draft --id draft-abc12345
  python -m src.cli send-draft   --id draft-abc12345
  python -m src.cli log          [--limit 20]
  python -m src.cli gmail-setup

All send operations require explicit --confirm flag to guard against accidents.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Internal imports (defer heavy imports so --help stays instant)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartmail",
        description="AI Internship Outreach Assistant — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- upload-cv ---
    p_upload = sub.add_parser("upload-cv", help="Register an existing CV file for Uploaded CV Mode.")
    p_upload.add_argument("--file", required=True, help="Path to CV file (PDF format)")

    # --- run-uploaded ---
    p_run_up = sub.add_parser(
        "run-uploaded",
        help="Run outreach workflow in Uploaded CV Mode for one contact.",
    )
    _add_run_args(p_run_up)
    p_run_up.add_argument("--cv", required=False, help="Path to CV file (overrides stored value)")

    # --- run-tailored ---
    p_run_tail = sub.add_parser(
        "run-tailored",
        help="Run outreach workflow in Tailored CV Mode (generates PDF from profile).",
    )
    _add_run_args(p_run_tail)

    # --- review-draft ---
    p_review = sub.add_parser("review-draft", help="Print a draft for human inspection.")
    p_review.add_argument("--id", required=True, help="Draft ID (e.g. draft-abc12345)")

    # --- approve-draft ---
    p_approve = sub.add_parser("approve-draft", help="Explicitly approve a draft for sending.")
    p_approve.add_argument("--id", required=True, help="Draft ID")
    p_approve.add_argument(
        "--confirm", action="store_true",
        help="Confirm you have reviewed the draft. Required.",
    )

    # --- send-draft ---
    p_send = sub.add_parser("send-draft", help="Send an approved draft (requires Gmail config).")
    p_send.add_argument("--id", required=True, help="Draft ID")
    p_send.add_argument(
        "--confirm", action="store_true",
        help="Final safety confirmation. Required.",
    )

    # --- log ---
    p_log = sub.add_parser("log", help="Show send log entries.")
    p_log.add_argument("--limit", type=int, default=20, help="Maximum entries to show (default 20).")

    # --- gmail-setup ---
    sub.add_parser(
        "gmail-setup",
        help="Run the one-time Gmail OAuth authorisation flow.",
    )

    # --- validate-profile ---
    sub.add_parser("validate-profile", help="Validate data/master_profile.json.")

    # --- config-status ---
    sub.add_parser("config-status", help="Check status of optional external services (Tavily, Gemini, Groq) safely.")

    # --- ui ---
    p_ui = sub.add_parser("ui", help="Launch the local SmartMail testing web UI.")
    p_ui.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1).")
    p_ui.add_argument("--port", type=int, default=5000, help="Port to bind (default 5000).")

    return parser


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--contact", required=True, help="Path to contacts CSV file.")
    p.add_argument("--row", type=int, required=True, help="Row number in CSV (1-indexed header).")
    p.add_argument("--role", default="", help="Target role title.")
    p.add_argument("--jd", default="", help="Job description text (or path to .txt file).")
    p.add_argument("--campaign", default="", help="Campaign identifier for duplicate tracking.")
    p.add_argument("--dry-run", action="store_true", default=True, help="Do not send live email.")


# ---------------------------------------------------------------------------
# Session state — stored in a simple JSON file next to the script
# ---------------------------------------------------------------------------
_SESSION_PATH = Path("output/.session.json")


def _load_session() -> dict:
    if _SESSION_PATH.exists():
        try:
            return json.loads(_SESSION_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_session(data: dict) -> None:
    _SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SESSION_PATH.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Draft store — in-memory + disk persistence in output/.drafts/
# ---------------------------------------------------------------------------
_draft_store: dict = {}
_DRAFTS_DIR = Path("output/.drafts")


def _save_draft_record(draft: Any, workflow: Any = None, campaign_id: str = "") -> None:
    _DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _DRAFTS_DIR / f"{draft.draft_id}.json"
    data = {
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
        "created_at": draft.created_at,
        "campaign_id": campaign_id,
    }
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_draft_and_workflow(draft_id: str) -> tuple[Any, Any] | None:
    if draft_id in _draft_store:
        return _draft_store[draft_id]

    file_path = _DRAFTS_DIR / f"{draft_id}.json"
    if not file_path.exists():
        return None

    try:
        from src.outreach.models import DraftStatus, EmailDraft, WorkflowMode
        from src.outreach.send_log import SendLog
        from src.workflow.engine import OutreachWorkflow

        data = json.loads(file_path.read_text(encoding="utf-8"))
        draft = EmailDraft(
            draft_id=data["draft_id"],
            recipient=data["recipient"],
            recipient_name=data.get("recipient_name", ""),
            company_name=data.get("company_name", ""),
            subject=data["subject"],
            body=data["body"],
            attachment_path=data["attachment_path"],
            mode=WorkflowMode(data["mode"]),
            status=DraftStatus(data.get("status", "DRAFT")),
            approved=data.get("approved", False),
            review_required=data.get("review_required", False),
            review_flags=data.get("review_flags", []),
            tailored_cv_path=data.get("tailored_cv_path"),
            created_at=data.get("created_at", ""),
        )
        send_log = SendLog(db_path="data/send_log.db")
        workflow = OutreachWorkflow(
            default_mode=draft.mode,
            uploaded_cv_path=draft.attachment_path if draft.mode == WorkflowMode.UPLOADED_CV else None,
            dry_run=True,
            send_log=send_log,
            campaign_id=data.get("campaign_id", ""),
        )
        workflow.review_manager._drafts[draft.draft_id] = draft
        _draft_store[draft.draft_id] = (draft, workflow)
        return draft, workflow
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_upload_cv(args: argparse.Namespace) -> int:
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        return 1
    session = _load_session()
    session["uploaded_cv_path"] = str(file_path.resolve())
    _save_session(session)
    print(f"Registered CV: {file_path.resolve()}")
    print("Mode: Uploaded CV Mode will use this file as the attachment.")
    return 0


def cmd_run(args: argparse.Namespace, mode_name: str) -> int:
    from src.contacts.csv_importer import import_contacts_from_csv
    from src.outreach.models import WorkflowMode
    from src.outreach.send_log import SendLog
    from src.workflow.engine import OutreachWorkflow

    # --- Load contact ---
    import_result = import_contacts_from_csv(args.contact)
    matching = [c for c in import_result.contacts if c.row_number == args.row]
    if not matching:
        print(f"ERROR: No contact found at row {args.row}.", file=sys.stderr)
        return 1
    contact = matching[0]

    # --- Job description ---
    jd = args.jd or ""
    if jd and Path(jd).exists():
        jd = Path(jd).read_text(encoding="utf-8")

    # --- Build workflow ---
    campaign_id = getattr(args, "campaign", "")
    send_log = SendLog(db_path="data/send_log.db")

    if mode_name == "uploaded":
        mode = WorkflowMode.UPLOADED_CV
        session = _load_session()
        cv_path = getattr(args, "cv", None) or session.get("uploaded_cv_path")
        if not cv_path:
            print(
                "ERROR: No CV registered. Run: python -m src.cli upload-cv --file <path>",
                file=sys.stderr,
            )
            return 1
        workflow = OutreachWorkflow(
            default_mode=mode,
            uploaded_cv_path=cv_path,
            dry_run=True,
            send_log=send_log,
            campaign_id=campaign_id,
        )
    else:
        mode = WorkflowMode.TAILORED_CV
        workflow = OutreachWorkflow(
            default_mode=mode,
            dry_run=True,
            send_log=send_log,
            campaign_id=campaign_id,
        )

    print(f"\n[Workflow: {mode_name.upper()} CV MODE]")
    print(f"Contact: {contact.contact_name} <{contact.email_result.normalized_email}>")
    print(f"Company: {contact.company_name}")
    print(f"Role:    {args.role or '(not specified)'}")

    result = workflow.process_contact(
        contact=contact,
        target_role=args.role,
        job_description=jd,
    )

    print(f"\nStatus:  {result.status}")
    print(f"Message: {result.message}")

    if result.status in ("BLOCKED", "ERROR"):
        return 1

    if result.draft:
        d = result.draft
        _draft_store[d.draft_id] = (d, workflow)
        _save_draft_record(d, workflow, campaign_id=campaign_id)
        _save_session({**_load_session(), "last_draft_id": d.draft_id})
        print(f"\nDraft ID:   {d.draft_id}")
        print(f"To:         {d.recipient}")
        print(f"Subject:    {d.subject}")
        print(f"Attachment: {d.attachment_path or '(none)'}")
        if result.tailored_cv and result.tailored_cv.pdf_path:
            print(f"PDF:        {result.tailored_cv.pdf_path}")
        if d.review_required:
            print(f"\n[REVIEW FLAGS] {'; '.join(d.review_flags)}")
        print(f"\nBody preview:\n{d.body[:300]}...")
        print(f"\nTo approve: python -m src.cli approve-draft --id {d.draft_id} --confirm")
    return 0


def cmd_review_draft(args: argparse.Namespace) -> int:
    session = _load_session()
    last_id = args.id or session.get("last_draft_id", "")
    pair = _get_draft_and_workflow(last_id)
    if not pair:
        print(f"Draft '{last_id}' not found. Re-run the workflow to regenerate it.")
        return 1
    d, _ = pair
    print(f"Draft ID:    {d.draft_id}")
    print(f"Status:      {d.status}")
    print(f"Approved:    {d.approved}")
    print(f"To:          {d.recipient} ({d.recipient_name})")
    print(f"Subject:     {d.subject}")
    print(f"Attachment:  {d.attachment_path}")
    print(f"Review flags:{' | '.join(d.review_flags) if d.review_flags else 'none'}")
    print(f"\n--- BODY ---\n{d.body}\n--- END ---")
    return 0


def cmd_approve_draft(args: argparse.Namespace) -> int:
    if not args.confirm:
        print("ERROR: --confirm is required. Confirm you have reviewed the draft body, recipient, subject, and attachment.")
        return 1
    pair = _get_draft_and_workflow(args.id)
    if not pair:
        print(f"Draft '{args.id}' not found. Re-run the workflow.")
        return 1
    draft, workflow = pair
    approved_draft = workflow.approve_draft(args.id)
    _save_draft_record(approved_draft, workflow, campaign_id=workflow.campaign_id)
    print(f"Draft {args.id} approved.")
    if draft.review_required and draft.review_flags:
        print(f"Note: Approved draft had review flags: {'; '.join(draft.review_flags)}")
    print(f"To send: python -m src.cli send-draft --id {args.id} --confirm")
    return 0


def cmd_send_draft(args: argparse.Namespace) -> int:
    if not args.confirm:
        print("ERROR: --confirm is required before sending.")
        return 1
    pair = _get_draft_and_workflow(args.id)
    if not pair:
        print(f"Draft '{args.id}' not found.")
        return 1
    draft, workflow = pair
    result = workflow.send_draft(args.id)
    _save_draft_record(draft, workflow, campaign_id=workflow.campaign_id)
    if result.success:
        print(f"[OK] Email sent: {result.message}")
    else:
        print(f"[FAILED] Send failed: {result.message}")
        print(f"  Error code: {result.error}")
    return 0 if result.success else 1


def cmd_log(args: argparse.Namespace) -> int:
    from src.outreach.send_log import SendLog
    log = SendLog(db_path="data/send_log.db")
    entries = log.list_all(limit=args.limit)
    if not entries:
        print("Send log is empty.")
        return 0
    print(f"{'Draft ID':<20} {'Recipient':<30} {'Status':<12} {'Updated'}")
    print("-" * 80)
    for e in entries:
        print(f"{e['draft_id']:<20} {e['recipient']:<30} {e['status']:<12} {e['updated_at'][:19]}")
    return 0


def cmd_gmail_setup(args: argparse.Namespace) -> int:
    print("Gmail OAuth Setup")
    print("=" * 60)
    print("""
Step 1: Create a Google Cloud project
  https://console.cloud.google.com/

Step 2: Enable the Gmail API
  APIs & Services → Library → "Gmail API" → Enable

Step 3: Configure OAuth consent screen
  APIs & Services → OAuth consent screen
  Type: External | Scopes: gmail.send

Step 4: Create OAuth credentials (Desktop App)
  APIs & Services → Credentials → Create Credentials → OAuth client ID
  Application type: Desktop App
  Download the JSON as 'credentials.json'

Step 5: Set environment variables
  GMAIL_CREDENTIALS_PATH=<path to credentials.json>
  GMAIL_TOKEN_PATH=<path to token.json>
  GMAIL_SENDER_ADDRESS=<your gmail address>

Step 6: Authorise (run once, opens browser)
  python -c "
  from src.gmail.provider import GmailProvider
  GmailProvider().authorise_interactively()
  "

WARNING: Never commit credentials.json or token.json to source control.
""")
    return 0


def cmd_validate_profile(args: argparse.Namespace) -> int:
    from src.profile.loader import load_profile
    from src.profile.validator import validate_profile
    profile = load_profile("data/master_profile.json")
    errors = validate_profile(profile)
    if errors:
        print(f"Profile validation: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Profile validation: OK (0 errors)")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from src.ui.app import create_app
    app = create_app()
    print("=" * 65)
    print(" SmartMail Local Functional Testing UI running at:")
    print(f" http://{args.host}:{args.port}")
    print(" (Bound to local loopback; dry-run safe mode is default)")
    print("=" * 65)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


def cmd_config_status(args: argparse.Namespace) -> int:
    from src.config import format_config_status_report
    print(format_config_status_report())
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "upload-cv": cmd_upload_cv,
        "run-uploaded": lambda a: cmd_run(a, "uploaded"),
        "run-tailored": lambda a: cmd_run(a, "tailored"),
        "review-draft": cmd_review_draft,
        "approve-draft": cmd_approve_draft,
        "send-draft": cmd_send_draft,
        "log": cmd_log,
        "gmail-setup": cmd_gmail_setup,
        "validate-profile": cmd_validate_profile,
        "config-status": cmd_config_status,
        "ui": cmd_ui,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
