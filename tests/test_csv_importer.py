"""
tests/test_csv_importer.py
---------------------------
Tests for src/contacts/csv_importer.py

Covers:
- Valid CSV with all columns
- Missing required column (no 'email' header)
- Missing required value (blank email cell)
- Empty rows skipped
- Duplicate contacts reported, not silently dropped
- Optional fields blank
- Display-name email format
- Subdomain addresses
- Free provider addresses
- Invalid/malformed email value
- File not found
- Row-level errors reported with correct row numbers
"""

import csv
import io
import textwrap
from pathlib import Path

import pytest

from src.contacts.csv_importer import (
    import_contacts_from_csv,
    CSVImportError,
    ContactRecord,
    ImportRowError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path: Path, content: str, filename: str = "contacts.csv") -> Path:
    """Write a CSV string to a temp file (no BOM) and return its path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# File-level error tests
# ---------------------------------------------------------------------------

class TestFileErrors:

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(CSVImportError, match="not found"):
            import_contacts_from_csv(tmp_path / "missing.csv")

    def test_missing_email_column_raises(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,contact_name,notes
            Fable,Alice,some note
        """)
        with pytest.raises(CSVImportError, match="email"):
            import_contacts_from_csv(p)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(CSVImportError):
            import_contacts_from_csv(p)


# ---------------------------------------------------------------------------
# Valid import tests
# ---------------------------------------------------------------------------

class TestValidImport:

    def test_single_valid_row_produces_one_contact(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,contact_name,email,website,role_or_job_url,source_url,notes
            Fable Technologies,Alice,alice@fabletech.example,https://fabletech.example,SWE Intern,,
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 1
        assert len(result.errors) == 0

    def test_contact_fields_populated_correctly(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,contact_name,email,website,role_or_job_url,source_url,notes
            Quill Analytics,Bob,bob@quillanalytics.example,https://quillanalytics.example,ML Intern,https://linkedin.com/,Great team
        """)
        result = import_contacts_from_csv(p)
        c = result.contacts[0]
        assert c.company_name == "Quill Analytics"
        assert c.contact_name == "Bob"
        assert c.email_result.normalized_email == "bob@quillanalytics.example"
        assert c.website == "https://quillanalytics.example"
        assert c.role_or_job_url == "ML Intern"
        assert c.source_url == "https://linkedin.com/"
        assert c.notes == "Great team"

    def test_multiple_valid_rows(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email
            Fable,alice@fabletech.example
            Quill,bob@quillanalytics.example
            Prism,carol@prismrobotics.example
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 3
        assert len(result.errors) == 0

    def test_optional_columns_missing_from_header_ok(self, tmp_path):
        """A CSV with only the email column is valid."""
        p = _write_csv(tmp_path, """
            email
            alice@fabletech.example
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 1
        assert result.contacts[0].company_name == ""

    def test_optional_fields_blank_allowed(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,contact_name,email,website,role_or_job_url,source_url,notes
            ,,alice@fabletech.example,,,,
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 1
        c = result.contacts[0]
        assert c.company_name == ""
        assert c.contact_name == ""
        assert c.notes == ""


# ---------------------------------------------------------------------------
# Row-level error tests
# ---------------------------------------------------------------------------

class TestRowErrors:

    def test_blank_email_cell_produces_row_error(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email
            Fable,
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 0
        assert len(result.errors) == 1
        assert "blank" in result.errors[0].reason.lower() or "required" in result.errors[0].reason.lower()

    def test_malformed_email_produces_row_error(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email
            Fable,not-an-email
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 0
        assert len(result.errors) == 1
        assert "syntax" in result.errors[0].reason.lower() or "valid" in result.errors[0].reason.lower()

    def test_invalid_row_does_not_block_valid_rows(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email
            Good Co,alice@fabletech.example
            Bad Co,not-an-email
            Another Good,bob@quillanalytics.example
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 2
        assert len(result.errors) == 1

    def test_row_error_has_correct_row_number(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email
            Good Co,alice@fabletech.example
            Bad Co,bad-email
        """)
        result = import_contacts_from_csv(p)
        # Row 1 = header, Row 2 = Good Co, Row 3 = Bad Co
        assert result.errors[0].row_number == 3

    def test_row_error_preserves_raw_row(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email
            FableCo,bad-email
        """)
        result = import_contacts_from_csv(p)
        assert result.errors[0].raw_row["company_name"] == "FableCo"


# ---------------------------------------------------------------------------
# Empty-row tests
# ---------------------------------------------------------------------------

class TestEmptyRows:

    def test_empty_row_is_skipped_silently(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email
            Fable,alice@fabletech.example

            Quill,bob@quillanalytics.example
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 2
        assert len(result.errors) == 0

    def test_all_empty_rows_gives_empty_result(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email


        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 0
        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Duplicate tests
# ---------------------------------------------------------------------------

class TestDuplicates:

    def test_duplicate_email_produces_error(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email
            Fable,alice@fabletech.example
            Fable,alice@fabletech.example
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 1
        assert len(result.errors) == 1
        assert "duplicate" in result.errors[0].reason.lower()

    def test_duplicate_detected_case_insensitively(self, tmp_path):
        p = _write_csv(tmp_path, """
            email
            Alice@FableTech.Example
            alice@fabletech.example
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 1
        assert len(result.errors) == 1

    def test_duplicate_error_mentions_first_row(self, tmp_path):
        p = _write_csv(tmp_path, """
            email
            alice@fabletech.example
            alice@fabletech.example
        """)
        result = import_contacts_from_csv(p)
        # First row is data row (header=1, row1=2, row2=3)
        assert "2" in result.errors[0].reason


# ---------------------------------------------------------------------------
# Display-name and normalisation tests
# ---------------------------------------------------------------------------

class TestDisplayNameAndNormalisation:

    def test_display_name_format_imported(self, tmp_path):
        p = _write_csv(tmp_path, """
            company_name,email
            Cascade,"Carol Osei <c.osei@cascadesystems.example>"
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 1
        assert result.contacts[0].email_result.display_name == "Carol Osei"
        assert result.contacts[0].email_result.normalized_email == "c.osei@cascadesystems.example"

    def test_uppercase_email_normalised(self, tmp_path):
        p = _write_csv(tmp_path, """
            email
            ALICE@FABLETECH.EXAMPLE
        """)
        result = import_contacts_from_csv(p)
        assert result.contacts[0].email_result.normalized_email == "alice@fabletech.example"

    def test_whitespace_in_email_cell_stripped(self, tmp_path):
        p = _write_csv(tmp_path, """
            email
              alice@fabletech.example  
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 1


# ---------------------------------------------------------------------------
# Domain-specific tests
# ---------------------------------------------------------------------------

class TestDomainInfo:

    def test_subdomain_address_is_valid(self, tmp_path):
        p = _write_csv(tmp_path, """
            email
            careers@jobs.vertexfintech.example
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 1
        assert result.contacts[0].email_result.is_syntax_valid is True

    def test_gmail_address_is_imported_without_company_claim(self, tmp_path):
        """A Gmail address is a valid contact; we do not infer the employer."""
        p = _write_csv(tmp_path, """
            email
            someuser@gmail.com
        """)
        result = import_contacts_from_csv(p)
        assert len(result.contacts) == 1
        c = result.contacts[0]
        assert c.email_result.is_syntax_valid is True
        # Employer is NOT automatically inferred
        assert c.company_name == ""

    def test_gmail_parse_notes_mention_free_provider(self, tmp_path):
        p = _write_csv(tmp_path, """
            email
            someuser@gmail.com
        """)
        result = import_contacts_from_csv(p)
        notes = result.contacts[0].email_result.parse_notes.lower()
        assert "free" in notes or "provider" in notes


# ---------------------------------------------------------------------------
# Sample CSV smoke test
# ---------------------------------------------------------------------------

class TestSampleCSV:
    """Smoke-test the sample file included in the repo."""

    SAMPLE_PATH = Path("data/sample_contacts.csv")

    def test_sample_csv_loads_without_exception(self):
        if not self.SAMPLE_PATH.exists():
            pytest.skip("sample_contacts.csv not found — skipping smoke test")
        result = import_contacts_from_csv(self.SAMPLE_PATH)
        # At least some contacts should load successfully
        assert len(result.contacts) > 0

    def test_sample_csv_has_no_unexpected_errors(self):
        if not self.SAMPLE_PATH.exists():
            pytest.skip("sample_contacts.csv not found — skipping smoke test")
        result = import_contacts_from_csv(self.SAMPLE_PATH)
        # The sample deliberately has some display-name rows and a free-provider
        # row — all should import successfully (no errors expected)
        error_reasons = [e.reason for e in result.errors]
        assert result.errors == [], (
            f"Unexpected errors in sample CSV: {error_reasons}"
        )
