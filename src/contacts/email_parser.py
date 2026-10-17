"""
src/contacts/email_parser.py
-----------------------------
Parses and normalises email addresses, then extracts domain information.

## How it works

Step 1 — Parse the raw string.
    We use Python's stdlib `email.utils.parseaddr()` which correctly handles
    both plain addresses  ("recruiter@example.com") and RFC-5322 display-name
    format ("Recruitment Team <recruiter@example.com>").  A plain regex would
    miss the display-name case or incorrectly strip angle brackets.

Step 2 — Syntax validation (conservative).
    We check only the structural rules that the stdlib exposes:
      • The address is non-empty after parsing.
      • It contains exactly one "@" sign.
      • The local part (before "@") and domain part (after "@") are both
        non-empty.
      • The domain contains at least one "." (e.g. "user@localhost" is rejected
        as it is not a routable internet address for outreach purposes).
    We do NOT:
      • Check whether the domain actually exists (that requires DNS/network).
      • Verify deliverability (that requires SMTP probing or a paid service).
      • Enforce full RFC-5321 syntax (some valid addresses contain quotes,
        comments, or IP literals; we do not need them for outreach).

Step 3 — Domain extraction.
    We use `tldextract` to split the domain into (subdomain, domain, suffix).
    Example: "jobs.example.com" → subdomain="jobs", domain="example",
    suffix="com", registered_domain="example.com".

    IMPORTANT: the registered domain does NOT identify the employer.
    Examples of cases where it gives no employer signal:
      • "alice@gmail.com"  — free provider; employer unknown
      • "recruit@hire.io"  — could be any company using a hiring platform
      • "hr@jobs.bigcorp.com" — subdomain may be a legitimate company domain

    We record the parsed values without marking the company as "verified".

## Public API

    parse_email(raw: str) -> EmailParseResult

    EmailParseResult is a plain dataclass:
        raw_input       str    original untouched string
        display_name    str    e.g. "Recruitment Team"  (empty if not present)
        normalized_email str   lowercase, stripped     (empty if invalid)
        local_part      str   part before "@"          (empty if invalid)
        email_domain    str   full domain as given      (empty if invalid)
        registered_domain str  e.g. "example.com"      (empty if unavailable)
        subdomain       str   e.g. "jobs"               (empty string if none)
        is_syntax_valid bool  True if structural checks passed
        parse_notes     str   human-readable note about any issue
"""

from __future__ import annotations

import email.utils
from dataclasses import dataclass, field

import tldextract


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EmailParseResult:
    """All information extracted from one raw email string."""
    raw_input: str
    display_name: str = ""
    normalized_email: str = ""
    local_part: str = ""
    email_domain: str = ""
    registered_domain: str = ""
    subdomain: str = ""
    is_syntax_valid: bool = False
    parse_notes: str = ""


# ---------------------------------------------------------------------------
# Free-provider detection (informational only — not a company signal)
# ---------------------------------------------------------------------------

# A small set of well-known free providers.  The list is intentionally short;
# it is used only to annotate parse_notes, never to make company decisions.
_FREE_PROVIDERS: frozenset[str] = frozenset({
    "gmail.com",
    "googlemail.com",
    "yahoo.com",
    "yahoo.co.in",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "protonmail.com",
    "proton.me",
    "icloud.com",
    "me.com",
    "aol.com",
    "zoho.com",
    "yandex.com",
    "mail.com",
    "gmx.com",
})


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_email(raw: str) -> EmailParseResult:
    """
    Parse a raw email string and return an EmailParseResult.

    Handles plain addresses and RFC-5322 display-name format.
    Normalises to lowercase and strips surrounding whitespace.
    Does NOT perform DNS or deliverability checks.

    Args:
        raw: The raw email string, e.g. "Alice <alice@example.com>" or
             "recruiter@EXAMPLE.COM".

    Returns:
        EmailParseResult with all fields populated.
    """
    result = EmailParseResult(raw_input=raw)

    if not raw or not raw.strip():
        result.parse_notes = "Input is empty or whitespace only."
        return result

    # --- Step 1: Parse display-name format -----------------------------------
    display_name, addr = email.utils.parseaddr(raw.strip())
    result.display_name = display_name.strip()

    # parseaddr returns ('', '') when it cannot find an address at all.
    # It also returns the raw string as `addr` for plain "foo@bar.com" input.
    if not addr:
        result.parse_notes = (
            "Could not extract an email address from the input. "
            "Expected formats: 'user@example.com' or 'Name <user@example.com>'."
        )
        return result

    # --- Step 2: Normalise ---------------------------------------------------
    normalised = addr.strip().lower()
    result.normalized_email = normalised

    # --- Step 3: Conservative syntax checks ----------------------------------
    if "@" not in normalised:
        result.parse_notes = "Missing '@' — not a valid email address."
        return result

    parts = normalised.split("@")
    if len(parts) != 2:
        result.parse_notes = "Multiple '@' signs found — not a valid email address."
        return result

    local, domain = parts

    if not local:
        result.parse_notes = "Local part (before '@') is empty."
        return result

    if not domain:
        result.parse_notes = "Domain part (after '@') is empty."
        return result

    if "." not in domain:
        result.parse_notes = (
            f"Domain '{domain}' contains no dot — not a routable internet address."
        )
        return result

    result.local_part = local
    result.email_domain = domain
    result.is_syntax_valid = True

    # --- Step 4: Registered-domain extraction --------------------------------
    extracted = tldextract.extract(domain)
    # tldextract returns empty strings for unknown/unrecognised TLDs.
    if extracted.domain and extracted.suffix:
        result.registered_domain = f"{extracted.domain}.{extracted.suffix}"
    else:
        # e.g. an intranet domain or unknown TLD — record what we have
        result.registered_domain = domain

    result.subdomain = extracted.subdomain  # empty string if no subdomain

    # --- Step 5: Informational notes -----------------------------------------
    notes: list[str] = []
    if result.registered_domain in _FREE_PROVIDERS:
        notes.append(
            f"'{result.registered_domain}' is a free email provider — "
            "employer identity cannot be inferred from this address."
        )
    if result.subdomain:
        notes.append(
            f"Address uses subdomain '{result.subdomain}' — "
            "this may be a legitimate recruitment subdomain."
        )
    result.parse_notes = " ".join(notes)

    return result
