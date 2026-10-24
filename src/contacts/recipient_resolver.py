"""
src/contacts/recipient_resolver.py
----------------------------------
Resolves recipient name, identity status, and mailbox nature from email and spreadsheet inputs.
Distinguishes personal inboxes from functional/generic organizational addresses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.contacts.email_parser import EmailParseResult, parse_email

# Functional/generic organizational mailbox prefixes
GENERIC_LOCAL_PARTS: frozenset[str] = frozenset({
    "hr",
    "careers",
    "jobs",
    "recruitment",
    "recruiter",
    "recruiting",
    "talent",
    "people",
    "hiring",
    "contact",
    "info",
    "hello",
    "team",
    "support",
    "press",
    "media",
    "general",
    "admin",
    "office",
    "help",
    "sales",
    "business",
    "inquiries",
    "enquiries",
    "founder",
    "founders",
    "ceo",
    "cto",
    "coo",
    "leadership",
    "management",
    "intern",
    "interns",
    "internship",
    "internships",
    "manager",
    "staffing",
})

# Status codes for recipient name verification
NAME_VERIFIED = "VERIFIED"          # Confirmed by credible external evidence
NAME_INFERRED = "INFERRED"          # Suggested by local-part or display name, but not verified
NAME_GENERIC_INBOX = "GENERIC_INBOX"# Functional mailbox (e.g. hr@, careers@)
NAME_UNKNOWN = "UNKNOWN"            # No name supplied or inferred

# Convenience aliases
VERIFIED = NAME_VERIFIED
INFERRED = NAME_INFERRED
GENERIC_INBOX = NAME_GENERIC_INBOX
UNKNOWN = NAME_UNKNOWN


@dataclass
class RecipientResolution:
    """Detailed resolution of the recipient's identity and mailbox nature."""
    supplied_name: str              # Exactly as entered in spreadsheet
    inferred_name: str              # Inferred from email username or display name
    verified_name: str              # Verified by external company evidence, if any
    name_verification_status: str   # VERIFIED | INFERRED | GENERIC_INBOX | UNKNOWN
    is_generic_inbox: bool          # True if hr@, careers@, info@, etc.
    inferred_role_or_title: str     # e.g. "Founder" if founder@, "HR Team" if hr@
    explanation: str                # Human-readable justification

    @property
    def status(self) -> str:
        return self.name_verification_status

    @status.setter
    def status(self, val: str) -> None:
        self.name_verification_status = val

    @property
    def is_generic(self) -> bool:
        return self.is_generic_inbox

    @property
    def resolved_name(self) -> str:
        return self.verified_name or self.supplied_name or self.inferred_name


def infer_name_from_local_part(local_part: str) -> str:
    """
    Safely extract a readable candidate name from an email local-part.
    Examples:
        'bhavesh' -> 'Bhavesh'
        'john.doe' -> 'John Doe'
        'jane_smith' -> 'Jane Smith'
        'j.smith' -> 'J Smith'
        'alex-wong' -> 'Alex Wong'
    Returns empty string if purely numeric or functional.
    """
    if not local_part:
        return ""

    cleaned = local_part.strip().lower()
    if cleaned in GENERIC_LOCAL_PARTS:
        return ""

    functional_prefixes = (
        "hr-", "hr_", "hr.", "hrd", "hrindia", "hrteam", "hrdept", "hrops", "hroperations",
        "careers", "talent", "recruit", "hiring", "internship", "internships", "staffing"
    )
    if any(cleaned.startswith(p) for p in functional_prefixes) or any(
        p in cleaned for p in ("recruiting", "recruitment", "internship", "internships", "staffing", "talentacquisition")
    ):
        return ""

    # Split on common separators
    parts = [p for p in re.split(r"[._\-+]+", cleaned) if p and not p.isdigit()]
    if not parts or any(p in GENERIC_LOCAL_PARTS for p in parts):
        return ""

    # Remove trailing numbers like john123 -> john
    parts = [re.sub(r"\d+$", "", p) for p in parts]
    parts = [p.capitalize() for p in parts if len(p) >= 2]

    if not parts:
        return ""

    return " ".join(parts)


def resolve_recipient(
    supplied_name_or_email: str = "",
    email_result: EmailParseResult | str | None = None,
    verified_name_from_evidence: str = "",
    supplied_name: str = "",
) -> RecipientResolution:
    """
    Resolve recipient identity separately from company identity.
    Enforces that email usernames alone NEVER count as verified identity.
    """
    # Handle flexible call patterns
    if "@" in supplied_name_or_email and email_result is None:
        email_result = parse_email(supplied_name_or_email)
        actual_supplied = supplied_name
    else:
        actual_supplied = supplied_name or supplied_name_or_email
        if isinstance(email_result, str):
            email_result = parse_email(email_result)

    supplied = (actual_supplied or "").strip()
    verified = (verified_name_from_evidence or "").strip()

    local_part = ""
    display_name = ""
    if email_result is not None:
        local_part = (email_result.local_part or "").strip().lower()
        display_name = (email_result.display_name or "").strip()

    # 1. Check if generic / functional mailbox
    parts_lower = [p for p in re.split(r"[._\-+]+", local_part) if p]
    is_generic = (
        local_part in GENERIC_LOCAL_PARTS
        or any(p in GENERIC_LOCAL_PARTS for p in parts_lower)
        or any(local_part.startswith(p) for p in (
            "hr-", "hr_", "hr.", "hrd", "hrindia", "hrteam", "hrdept", "hrops", "hroperations",
            "careers", "talent", "recruit", "hiring", "internship", "internships", "staffing"
        ))
        or any(p in local_part for p in ("recruiting", "recruitment", "internship", "internships", "staffing", "talentacquisition"))
    )
    inferred_role = ""
    if is_generic:
        if any(p in ("founder", "founders") for p in parts_lower):
            inferred_role = "Founder"
        elif any(p in ("ceo", "cto", "coo") for p in parts_lower):
            inferred_role = next(p.upper() for p in parts_lower if p in ("ceo", "cto", "coo"))
        elif any(p in ("hr", "careers", "jobs", "recruiting", "recruitment", "talent", "hiring", "intern", "internship", "internships", "manager", "staffing") for p in parts_lower):
            inferred_role = "Talent / Recruiting Team"
        else:
            inferred_role = "General Inquiries Team"

    # 2. Determine inferred name
    inferred = ""
    if not is_generic and local_part:
        inferred = infer_name_from_local_part(local_part)
    elif display_name and display_name.lower() != supplied.lower():
        inferred = display_name
    elif supplied:
        inferred = supplied

    # 3. Determine verification status
    if verified:
        status = NAME_VERIFIED
        explanation = f"Recipient identity confirmed by official evidence as '{verified}'."
    elif is_generic:
        status = NAME_GENERIC_INBOX
        explanation = f"Functional mailbox '{local_part}@' identified ({inferred_role}); generic organization inbox."
    elif supplied:
        status = NAME_INFERRED
        explanation = f"Name '{supplied}' was supplied in spreadsheet; unverified against external public profiles."
    elif inferred:
        status = NAME_INFERRED
        explanation = f"Name '{inferred}' inferred from email username; unverified against external public profiles."
    else:
        status = NAME_UNKNOWN
        explanation = "Recipient name could not be identified from spreadsheet or email address."

    return RecipientResolution(
        supplied_name=supplied,
        inferred_name=inferred,
        verified_name=verified,
        name_verification_status=status,
        is_generic_inbox=is_generic,
        inferred_role_or_title=inferred_role,
        explanation=explanation,
    )
