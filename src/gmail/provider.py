"""
src/gmail/provider.py
---------------------
Gmail API provider using OAuth 2.0 (google-auth + google-api-python-client).

IMPORTANT — CONFIGURATION REQUIRED BEFORE LIVE USE
=====================================================
This provider will NOT send real emails until all of the following are in place:

1. Create a Google Cloud project at https://console.cloud.google.com/.
2. Enable the Gmail API (APIs & Services → Library → Gmail API → Enable).
3. Configure an OAuth consent screen (External, scopes: gmail.send).
4. Create OAuth 2.0 credentials (Desktop App) and download the JSON as
   ``credentials.json``.
5. Run the one-time authorisation flow (see ``authorise_interactively()``
   below), which creates a ``token.json`` file.
6. Set environment variables:
       GMAIL_CREDENTIALS_PATH=<path to credentials.json>
       GMAIL_TOKEN_PATH=<path to token.json>
7. Never commit credentials.json or token.json to source control.
   They are included in .gitignore.

Security constraints
--------------------
- Scope is limited to ``https://www.googleapis.com/auth/gmail.send`` only.
- Tokens are stored locally (token.json) and refreshed automatically.
- Tokens and client secrets are never logged or returned in error messages.
- Attachments are base64-encoded RFC 2822 messages; raw body is not logged.
- In ``dry_run`` mode no network call is made.

During development/testing
--------------------------
All tests use ``MockGmailProvider`` (defined at the bottom of this file)
or test against the ``GmailProvider`` with ``dry_run=True``.
No live Gmail calls are made in the test suite.
"""
from __future__ import annotations

import base64
import email as email_lib
import mimetypes
import os
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional Google API imports — degrade gracefully when not installed
# ---------------------------------------------------------------------------
try:
    from google.auth.transport.requests import Request as _GRequest
    from google.oauth2.credentials import Credentials as _Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow as _Flow
    from googleapiclient.discovery import build as _build
    from googleapiclient.errors import HttpError as _HttpError
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False

_GMAIL_SEND_SCOPE = ["https://www.googleapis.com/auth/gmail.send"]

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class GmailProviderError(Exception):
    """Base error for Gmail provider issues."""


class GmailNotConfiguredError(GmailProviderError):
    """Raised when credentials or token paths are not set up."""


class GmailSendError(GmailProviderError):
    """Raised when the Gmail API rejects or fails a send."""


class GmailLibraryMissingError(GmailProviderError):
    """
    Raised when google-auth / google-api-python-client are not installed.
    Install with: pip install google-auth google-auth-oauthlib google-api-python-client
    """


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GmailProvider:
    """
    Gmail API provider implementing the send_email interface expected by
    EmailSender.

    Parameters
    ----------
    credentials_path:
        Path to ``credentials.json`` (OAuth client secrets).
        Defaults to ``GMAIL_CREDENTIALS_PATH`` environment variable.
    token_path:
        Path to ``token.json`` (cached OAuth token).
        Defaults to ``GMAIL_TOKEN_PATH`` environment variable.
    dry_run:
        When True, builds the MIME message and validates it but does NOT
        call the Gmail API.  Safe for testing.
    sender_address:
        Gmail address to send from.  Defaults to ``GMAIL_SENDER_ADDRESS``
        env var, or ``"me"`` (the authenticated user's primary address).
    """

    def __init__(
        self,
        credentials_path: str | Path | None = None,
        token_path: str | Path | None = None,
        dry_run: bool = False,
        sender_address: str = "",
    ) -> None:
        self._creds_path = str(
            credentials_path or os.getenv("GMAIL_CREDENTIALS_PATH", "")
        )
        self._token_path = str(
            token_path or os.getenv("GMAIL_TOKEN_PATH", "")
        )
        self.dry_run = dry_run
        self._sender = sender_address or os.getenv("GMAIL_SENDER_ADDRESS", "me")
        self._service: Any = None  # lazy-initialised

    # ------------------------------------------------------------------
    # Configuration check
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True only when both credential files exist and are readable."""
        if not self._creds_path or not self._token_path:
            return False
        return Path(self._creds_path).exists() and Path(self._token_path).exists()

    def require_configured(self) -> None:
        """Raise GmailNotConfiguredError with a clear setup guide if not configured."""
        if not _GOOGLE_AVAILABLE:
            raise GmailLibraryMissingError(
                "Google API libraries are not installed. "
                "Run: pip install google-auth google-auth-oauthlib google-api-python-client"
            )
        if not self._creds_path:
            raise GmailNotConfiguredError(
                "GMAIL_CREDENTIALS_PATH is not set. "
                "Download OAuth credentials from Google Cloud Console and set the env var."
            )
        if not Path(self._creds_path).exists():
            raise GmailNotConfiguredError(
                f"credentials.json not found at: {self._creds_path}. "
                "Download it from Google Cloud Console → APIs & Services → Credentials."
            )
        if not self._token_path:
            raise GmailNotConfiguredError(
                "GMAIL_TOKEN_PATH is not set. "
                "Run authorise_interactively() once to generate token.json."
            )
        if not Path(self._token_path).exists():
            raise GmailNotConfiguredError(
                f"token.json not found at: {self._token_path}. "
                "Run authorise_interactively() to perform the initial OAuth flow."
            )

    # ------------------------------------------------------------------
    # One-time authorisation (call manually; never in automated flow)
    # ------------------------------------------------------------------

    def authorise_interactively(self) -> None:
        """
        Open the browser-based OAuth consent flow and save the token.

        Call this once from the CLI/REPL.  Do NOT call from tests or
        automated pipelines.
        """
        if not _GOOGLE_AVAILABLE:
            raise GmailLibraryMissingError(str(GmailLibraryMissingError.__doc__))
        if not self._creds_path or not Path(self._creds_path).exists():
            raise GmailNotConfiguredError(
                f"credentials.json not found at: {self._creds_path}"
            )
        flow = _Flow.from_client_secrets_file(self._creds_path, _GMAIL_SEND_SCOPE)
        creds = flow.run_local_server(port=0)
        token_path = Path(self._token_path) if self._token_path else Path("token.json")
        token_path.write_text(creds.to_json())
        # OAuth tokens are credentials: restrict to owner-only access
        # instead of the process default umask.
        try:
            token_path.chmod(0o600)
        except OSError:
            pass
        print(f"Token saved to: {token_path}")

    # ------------------------------------------------------------------
    # Internal: build authenticated service
    # ------------------------------------------------------------------

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service

        self.require_configured()
        creds = _Credentials.from_authorized_user_file(self._token_path, _GMAIL_SEND_SCOPE)

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(_GRequest())
                Path(self._token_path).write_text(creds.to_json())
                try:
                    Path(self._token_path).chmod(0o600)
                except OSError:
                    pass
            else:
                raise GmailNotConfiguredError(
                    "Stored token is invalid and cannot be refreshed. "
                    "Re-run authorise_interactively() to obtain a new token."
                )

        self._service = _build("gmail", "v1", credentials=creds)
        return self._service

    # ------------------------------------------------------------------
    # Public: send_email (interface expected by EmailSender)
    # ------------------------------------------------------------------

    def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        attachment_path: str = "",
    ) -> None:
        """
        Build and send an RFC 2822 email via the Gmail API.

        Raises
        ------
        GmailNotConfiguredError  — when setup is incomplete.
        GmailSendError           — when the API call fails.
        GmailLibraryMissingError — when google libraries are not installed.
        """
        # Build MIME message (always; validates inputs before API call)
        message = _build_mime_message(
            sender=self._sender,
            recipient=recipient,
            subject=subject,
            body=body,
            attachment_path=attachment_path or "",
        )
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
        payload = {"raw": encoded}

        if self.dry_run:
            # Validate message construction but do NOT call API
            return

        # Live send — wrapped in error handler
        try:
            service = self._get_service()
            service.users().messages().send(userId="me", body=payload).execute()
        except Exception as exc:
            # Deliberately do not embed token or body in the error message
            raise GmailSendError(f"Gmail API send failed: {type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# MIME helper (no credentials needed)
# ---------------------------------------------------------------------------

def _build_mime_message(
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    attachment_path: str = "",
) -> MIMEMultipart:
    """Build a MIME multipart message ready for Gmail API submission."""
    msg = MIMEMultipart("mixed")
    msg["to"] = recipient
    msg["from"] = sender
    msg["subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    if attachment_path and Path(attachment_path).exists():
        att_path = Path(attachment_path)
        mime_type, _ = mimetypes.guess_type(str(att_path))
        main_type, sub_type = (mime_type or "application/octet-stream").split("/", 1)

        with att_path.open("rb") as f:
            att_data = f.read()

        part = MIMEApplication(att_data, Name=att_path.name)
        part["Content-Disposition"] = f'attachment; filename="{att_path.name}"'
        msg.attach(part)

    return msg


# ---------------------------------------------------------------------------
# Mock provider for tests — zero network calls, zero real auth
# ---------------------------------------------------------------------------

class MockGmailProvider:
    """
    Drop-in replacement for GmailProvider used exclusively in tests.
    Records all calls without making any network requests or OAuth flows.
    """

    def __init__(self) -> None:
        self.sent_emails: list[dict] = []
        self.should_fail: bool = False
        self.fail_message: str = "Mock send failure"

    def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        attachment_path: str = "",
    ) -> None:
        """Record the call; raise if should_fail is set."""
        if self.should_fail:
            raise GmailSendError(self.fail_message)
        self.sent_emails.append(
            {
                "recipient": recipient,
                "subject": subject,
                # body is intentionally truncated in mock to avoid test logs
                "body_preview": body[:80],
                "attachment_path": attachment_path,
            }
        )

    def is_configured(self) -> bool:
        return True

    @property
    def dry_run(self) -> bool:
        return False
