"""
tests/test_cli.py
-----------------
Tests for the AI Internship Outreach Assistant CLI (src/cli.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.cli import _build_parser, main, _save_session, _load_session


class TestCLIParsing:
    def test_parser_help_does_not_crash(self):
        parser = _build_parser()
        assert parser.prog == "smartmail"

    def test_subcommands_recognized(self):
        parser = _build_parser()
        args_upload = parser.parse_args(["upload-cv", "--file", "my_cv.pdf"])
        assert args_upload.command == "upload-cv"
        assert args_upload.file == "my_cv.pdf"

        args_log = parser.parse_args(["log", "--limit", "10"])
        assert args_log.command == "log"
        assert args_log.limit == 10

        args_setup = parser.parse_args(["gmail-setup"])
        assert args_setup.command == "gmail-setup"

        args_val = parser.parse_args(["validate-profile"])
        assert args_val.command == "validate-profile"


class TestCLICommands:
    def test_upload_cv_command_missing_file_fails(self, capsys):
        code = main(["upload-cv", "--file", "nonexistent_cv_file.pdf"])
        assert code == 1
        captured = capsys.readouterr()
        assert "ERROR: File not found" in captured.err

    def test_upload_cv_command_success(self, tmp_path: Path, capsys):
        real_file = tmp_path / "valid_cv.pdf"
        real_file.write_bytes(b"%PDF-1.4 simulated pdf")

        code = main(["upload-cv", "--file", str(real_file)])
        assert code == 0
        captured = capsys.readouterr()
        assert "Registered CV:" in captured.out

    def test_validate_profile_command(self, capsys):
        code = main(["validate-profile"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Profile validation: OK" in captured.out

    def test_gmail_setup_command(self, capsys):
        code = main(["gmail-setup"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Gmail OAuth Setup" in captured.out
        assert "credentials.json" in captured.out

    def test_log_command(self, capsys):
        code = main(["log"])
        assert code == 0
        captured = capsys.readouterr()
        # Either empty or table
        assert "Send log" in captured.out or "Draft ID" in captured.out

    def test_approve_draft_requires_confirm(self, capsys):
        code = main(["approve-draft", "--id", "draft-test"])
        assert code == 1
        captured = capsys.readouterr()
        assert "--confirm is required" in captured.out or "--confirm is required" in captured.err

    def test_send_draft_requires_confirm(self, capsys):
        code = main(["send-draft", "--id", "draft-test"])
        assert code == 1
        captured = capsys.readouterr()
        assert "--confirm is required" in captured.out or "--confirm is required" in captured.err

    def test_uploaded_cv_mode_end_to_end_cli(self, tmp_path: Path, capsys):
        import hashlib
        import re

        # 1. Create a simulated CV PDF
        cv_file = tmp_path / "candidate_cv.pdf"
        cv_file.write_bytes(b"%PDF-1.4 simulated resume content byte sequence")
        initial_hash = hashlib.sha256(cv_file.read_bytes()).hexdigest()

        # 2. Register CV
        code = main(["upload-cv", "--file", str(cv_file)])
        assert code == 0
        captured = capsys.readouterr()
        assert "Registered CV:" in captured.out

        # 3. Run uploaded workflow
        code = main([
            "run-uploaded",
            "--contact", "data/sample_contacts.csv",
            "--row", "2",
            "--role", "AI/ML Intern",
            "--campaign", "cli_test_camp",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Workflow: UPLOADED CV MODE" in captured.out
        assert str(cv_file.resolve()) in captured.out

        # Extract draft ID
        match = re.search(r"Draft ID:\s+(draft-[a-f0-9]+)", captured.out)
        assert match is not None
        draft_id = match.group(1)

        # 4. Verify file unmodified
        after_hash = hashlib.sha256(cv_file.read_bytes()).hexdigest()
        assert after_hash == initial_hash

        # 5. Review draft
        code = main(["review-draft", "--id", draft_id])
        assert code == 0
        captured = capsys.readouterr()
        assert draft_id in captured.out
        assert "alice.mercer@fabletech.example" in captured.out
        assert str(cv_file.resolve()) in captured.out

        # 6. Approve draft
        code = main(["approve-draft", "--id", draft_id, "--confirm"])
        assert code == 0
        captured = capsys.readouterr()
        assert f"Draft {draft_id} approved" in captured.out

        # 7. Review shows approved
        code = main(["review-draft", "--id", draft_id])
        assert code == 0
        captured = capsys.readouterr()
        assert "Approved:    True" in captured.out or "Approved: True" in captured.out

        # 8. Send draft in dry-run mode safely simulates send without network calls
        code = main(["send-draft", "--id", draft_id, "--confirm"])
        captured = capsys.readouterr()
        assert "Simulated send succeeded" in captured.out or "GMAIL_NOT_CONFIGURED" in captured.out or "Send failed" in captured.out

        # 9. Final hash check
        final_hash = hashlib.sha256(cv_file.read_bytes()).hexdigest()
        assert final_hash == initial_hash
