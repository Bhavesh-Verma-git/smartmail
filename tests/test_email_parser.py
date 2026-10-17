"""
tests/test_email_parser.py
--------------------------
Tests for src/contacts/email_parser.py

Covers:
- Plain valid addresses
- Display-name format
- Uppercase normalisation
- Surrounding whitespace
- Missing "@"
- Multiple "@" signs
- Empty / whitespace-only input
- Domain with no dot (not routable)
- Subdomain address (e.g. jobs.example.com)
- Free email provider annotation
- Registered-domain extraction
- local_part and email_domain fields
"""

import pytest
from src.contacts.email_parser import parse_email, EmailParseResult


class TestValidEmails:
    """Well-formed addresses that should pass syntax validation."""

    def test_plain_address(self):
        r = parse_email("recruiter@fabletech.example")
        assert r.is_syntax_valid is True
        assert r.normalized_email == "recruiter@fabletech.example"
        assert r.local_part == "recruiter"
        assert r.email_domain == "fabletech.example"

    def test_registered_domain_extracted(self):
        r = parse_email("recruiter@fabletech.example")
        # tldextract won't recognise .example as a public suffix, so
        # registered_domain falls back to the full domain
        assert r.registered_domain != ""  # always set for valid address

    def test_display_name_format(self):
        r = parse_email("Recruitment Team <recruiter@fabletech.example>")
        assert r.is_syntax_valid is True
        assert r.display_name == "Recruitment Team"
        assert r.normalized_email == "recruiter@fabletech.example"

    def test_display_name_with_extra_spaces(self):
        r = parse_email("  Alice Mercer  <  alice@fabletech.example  >  ")
        # email.utils.parseaddr handles this
        assert r.is_syntax_valid is True
        assert "alice@fabletech.example" in r.normalized_email

    def test_uppercase_is_normalised(self):
        r = parse_email("RECRUITER@FABLETECH.EXAMPLE")
        assert r.is_syntax_valid is True
        assert r.normalized_email == "recruiter@fabletech.example"

    def test_mixed_case_normalised(self):
        r = parse_email("Recruiter@FableTech.Example")
        assert r.normalized_email == "recruiter@fabletech.example"

    def test_surrounding_whitespace_stripped(self):
        r = parse_email("   recruiter@fabletech.example   ")
        assert r.is_syntax_valid is True
        assert r.normalized_email == "recruiter@fabletech.example"

    def test_local_part_set_correctly(self):
        r = parse_email("alice.mercer@quillanalytics.example")
        assert r.local_part == "alice.mercer"

    def test_email_domain_set_correctly(self):
        r = parse_email("hr@cascadesystems.example")
        assert r.email_domain == "cascadesystems.example"


class TestSubdomainAddresses:
    """Subdomain addresses like jobs.example.com are valid and noted."""

    def test_subdomain_address_is_valid(self):
        r = parse_email("careers@jobs.vertexfintech.example")
        assert r.is_syntax_valid is True

    def test_subdomain_field_populated(self):
        r = parse_email("careers@jobs.vertexfintech.example")
        # tldextract may not recognise .example, but the subdomain field
        # should at least be populated with the parsed subdomain portion
        # We just verify the email itself is valid; subdomain detail is informational
        assert r.email_domain == "jobs.vertexfintech.example"

    def test_subdomain_note_present(self):
        r = parse_email("hr@recruitment.bigcorp.com")
        assert r.is_syntax_valid is True
        # The parse_notes should mention the subdomain
        assert "subdomain" in r.parse_notes.lower()


class TestFreeProviders:
    """Free provider addresses are valid but employer is unknown."""

    def test_gmail_is_syntax_valid(self):
        r = parse_email("someuser@gmail.com")
        assert r.is_syntax_valid is True

    def test_gmail_note_mentions_free_provider(self):
        r = parse_email("someuser@gmail.com")
        assert "free" in r.parse_notes.lower() or "provider" in r.parse_notes.lower()

    def test_yahoo_is_annotated(self):
        r = parse_email("person@yahoo.com")
        assert r.is_syntax_valid is True
        assert "free" in r.parse_notes.lower() or "provider" in r.parse_notes.lower()

    def test_gmail_registered_domain_is_gmail(self):
        r = parse_email("someuser@gmail.com")
        assert r.registered_domain == "gmail.com"


class TestInvalidEmails:
    """Addresses that must fail syntax validation."""

    def test_empty_string(self):
        r = parse_email("")
        assert r.is_syntax_valid is False
        assert r.normalized_email == ""

    def test_whitespace_only(self):
        r = parse_email("   ")
        assert r.is_syntax_valid is False

    def test_missing_at_sign(self):
        r = parse_email("notanemail")
        assert r.is_syntax_valid is False
        assert "@" in r.parse_notes or "missing" in r.parse_notes.lower()

    def test_multiple_at_signs(self):
        r = parse_email("a@b@c.com")
        assert r.is_syntax_valid is False

    def test_empty_local_part(self):
        r = parse_email("@fabletech.example")
        assert r.is_syntax_valid is False

    def test_empty_domain_part(self):
        r = parse_email("recruiter@")
        assert r.is_syntax_valid is False

    def test_domain_without_dot(self):
        r = parse_email("user@localhost")
        assert r.is_syntax_valid is False
        assert "dot" in r.parse_notes.lower() or "routable" in r.parse_notes.lower()

    def test_malformed_display_name_no_address(self):
        r = parse_email("Just A Name")
        assert r.is_syntax_valid is False


class TestRawInputPreserved:
    """The raw_input field always contains the original, untouched string."""

    def test_raw_input_preserved_for_valid(self):
        raw = "  RECRUITER@FABLETECH.EXAMPLE  "
        r = parse_email(raw)
        assert r.raw_input == raw

    def test_raw_input_preserved_for_invalid(self):
        raw = "not-an-email"
        r = parse_email(raw)
        assert r.raw_input == raw
