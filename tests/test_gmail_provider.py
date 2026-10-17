"""
tests/test_gmail_provider.py
-----------------------------
Tests for the Gmail API provider and MIME construction (src/gmail/provider.py).
Zero live network calls or live emails are made.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.gmail.provider import (
    GmailProvider,
    GmailProviderError,
    GmailNotConfiguredError,
    GmailSendError,
    MockGmailProvider,
    _build_mime_message,
)


class TestMIMEConstruction:
    def test_builds_valid_mime_message(self):
        msg = _build_mime_message(
            sender="me@example.com",
            recipient="recruiter@company.com",
            subject="Internship Application",
            body="Dear Recruiter,\nPlease find attached my CV.",
        )
        assert msg["to"] == "recruiter@company.com"
        assert msg["from"] == "me@example.com"
        assert msg["subject"] == "Internship Application"

    def test_builds_mime_message_with_attachment(self, tmp_path: Path):
        fake_cv = tmp_path / "resume.pdf"
        fake_cv.write_bytes(b"%PDF-1.4 simulated pdf content")

        msg = _build_mime_message(
            sender="me@example.com",
            recipient="recruiter@company.com",
            subject="Internship Application",
            body="Please find attached my CV.",
            attachment_path=str(fake_cv),
        )
        parts = msg.get_payload()
        assert len(parts) == 2
        # First part is text
        assert parts[0].get_content_type() == "text/plain"
        # Second part is application attachment
        assert "application" in parts[1].get_content_type()
        assert 'filename="resume.pdf"' in parts[1]["Content-Disposition"]

    def test_builds_mime_message_nonexistent_attachment_ignored_gracefully(self):
        msg = _build_mime_message(
            sender="me@example.com",
            recipient="recruiter@company.com",
            subject="Test",
            body="Test body",
            attachment_path="nonexistent_file_path.pdf",
        )
        parts = msg.get_payload()
        assert len(parts) == 1


class TestGmailProviderConfiguration:
    def test_is_configured_returns_false_when_unconfigured(self):
        provider = GmailProvider(
            credentials_path="nonexistent_creds.json",
            token_path="nonexistent_token.json",
        )
        assert provider.is_configured() is False

    def test_require_configured_raises_descriptive_error(self):
        provider = GmailProvider(
            credentials_path="",
            token_path="",
        )
        with pytest.raises(GmailProviderError) as exc_info:
            provider.require_configured()
        assert len(str(exc_info.value)) > 10

    def test_dry_run_does_not_call_api_or_require_valid_token(self, tmp_path: Path):
        # In dry_run mode, MIME message is built and validated, but no API call is made
        provider = GmailProvider(dry_run=True)
        # Should not raise any network or credential error
        provider.send_email(
            recipient="test@example.com",
            subject="Dry Run Subject",
            body="Dry run body text",
        )


class TestMockGmailProvider:
    def test_mock_records_sent_email(self):
        mock = MockGmailProvider()
        assert mock.is_configured() is True
        mock.send_email(
            recipient="recruiter@tech.com",
            subject="AI Intern Application",
            body="Hello,\nI am applying for the role.",
            attachment_path="/path/to/cv.pdf",
        )
        assert len(mock.sent_emails) == 1
        entry = mock.sent_emails[0]
        assert entry["recipient"] == "recruiter@tech.com"
        assert entry["subject"] == "AI Intern Application"
        assert entry["attachment_path"] == "/path/to/cv.pdf"

    def test_mock_can_simulate_failure(self):
        mock = MockGmailProvider()
        mock.should_fail = True
        mock.fail_message = "Rate limit exceeded"

        with pytest.raises(GmailSendError) as exc_info:
            mock.send_email(
                recipient="test@example.com",
                subject="Test",
                body="Body",
            )
        assert "Rate limit exceeded" in str(exc_info.value)
        assert len(mock.sent_emails) == 0
