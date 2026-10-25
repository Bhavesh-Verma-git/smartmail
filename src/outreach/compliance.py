"""
src/outreach/compliance.py
--------------------------
Configurable outreach compliance policy -- NOT a legal verdict.

SmartMail primarily supports legitimate internship/professional outreach
in India, but recipients may be anywhere. This module therefore:

- treats jurisdiction and applicable requirements as CONFIGURATION to be
  reviewed for the relevant jurisdictions, never as an AI-generated claim
  of full legal compliance;
- flags uncertainty (unknown jurisdiction, unknown consent basis) instead
  of asserting compliance;
- enforces the mechanical safeguards that are always required: accurate
  sender/recipient identity, honest subject lines, suppression, and
  duplicate protection (the last two live in the send log).

No function here decides that a send is "legal". It decides whether the
mechanical checks pass and what still needs a human decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompliancePolicy:
    """Configurable policy; review it before any future live sending.

    default_jurisdiction: assumed only when the recipient jurisdiction is
        unknown -- and even then it is recorded as an assumption, not a fact.
    jurisdictions_requiring_consent: jurisdiction codes (e.g. "EU", "CA")
        where the operator has configured an explicit-consent requirement.
        Empty by default: the operator must fill this in after review; an
        empty set does NOT mean "no consent is ever required".
    retention_days: how long contact/draft records are kept before review
        for deletion. This is a retention policy knob, not an automatic
        deletion guarantee (see README limitations).
    """

    default_jurisdiction: str = "IN"
    jurisdictions_requiring_consent: frozenset[str] = frozenset()
    retention_days: int = 365

    def jurisdiction_note(self, recipient_jurisdiction: str | None) -> str:
        if not recipient_jurisdiction:
            return (
                "Recipient jurisdiction unknown; policy default "
                f"({self.default_jurisdiction}) is assumed for flagging only. "
                "Do not treat this as a legal compliance determination."
            )
        if recipient_jurisdiction in self.jurisdictions_requiring_consent:
            return (
                f"Recipient jurisdiction {recipient_jurisdiction} is configured "
                "as requiring explicit consent; human confirmation of the "
                "consent basis is required before any live send."
            )
        return (
            f"Recipient jurisdiction recorded as {recipient_jurisdiction}. "
            "Applicable requirements were not verified by SmartMail."
        )


DEFAULT_POLICY = CompliancePolicy()


# ---------------------------------------------------------------------------
# Future live sending: kill switch, cap, and provider-limit honesty
# ---------------------------------------------------------------------------
# Real transmission is DISABLED by default in this phase. These helpers
# describe how a future deliberate enablement must work; they do not
# enable anything on their own.
#
# Gmail API quota vs account-level limits are DIFFERENT things:
# - The Gmail API has a daily quota measured in quota units per project.
# - A Google Workspace / Gmail account has its own sending limits
#   (recipients per day, etc.) enforced by Google.
# Do NOT hard-code either number from secondary sources: verify the
# current figures on Google's own limits documentation before configuring
# any production cap. The cap below is SmartMail's own conservative
# default (25/day, inside the 25-100 batch context but deliberately at
# the low end), not a Google limit.

def live_sending_enabled() -> bool:
    """True only if live sending was deliberately enabled AND not killed.

    Enablement requires SMARTMAIL_LIVE_SENDING_ENABLED=true.
    SMARTMAIL_SENDING_KILL_SWITCH=1 (or =true) always blocks, even if
    enablement is set. Default: disabled.
    """
    import os
    kill = os.getenv("SMARTMAIL_SENDING_KILL_SWITCH", "").strip().lower()
    if kill in ("1", "true", "yes"):
        return False
    return os.getenv("SMARTMAIL_LIVE_SENDING_ENABLED", "").strip().lower() in ("1", "true", "yes")


def daily_send_cap() -> int:
    """SmartMail's own conservative daily live-send cap (default 25)."""
    import os
    try:
        return max(0, int(os.getenv("SMARTMAIL_DAILY_SEND_CAP", "25")))
    except ValueError:
        return 25

# Subject lines that misrepresent the message (fake reply/forward, fake
# urgency or fake prior relationship) are dishonest by construction.
_MISLEADING_SUBJECT_PREFIXES = ("re:", "fwd:", "fw:")
_MISLEADING_SUBJECT_PHRASES = (
    "urgent action required",
    "your account will be closed",
    "final notice",
)


def check_subject_honest(subject: str) -> list[str]:
    """Return honesty problems with a subject line (empty = honest)."""
    problems: list[str] = []
    s = (subject or "").strip()
    if not s:
        problems.append("Subject line is empty.")
        return problems
    low = s.lower()
    if any(low.startswith(p) for p in _MISLEADING_SUBJECT_PREFIXES):
        problems.append(
            "Subject line mimics a reply/forward (Re:/Fwd:) for a first outreach email."
        )
    if any(p in low for p in _MISLEADING_SUBJECT_PHRASES):
        problems.append("Subject line uses a misleading urgency/notice phrase.")
    return problems


def check_sender_identity(sender_address: str, profile_email: str = "") -> list[str]:
    """Check the sender identity is present and consistent, not invented."""
    problems: list[str] = []
    sender = (sender_address or "").strip()
    if not sender or "@" not in sender:
        problems.append("Sender address is missing or invalid.")
    elif profile_email and sender.lower() != profile_email.strip().lower():
        problems.append(
            "Sender address differs from the candidate profile email; "
            "confirm the sender identity before sending."
        )
    return problems


def pre_send_compliance_check(
    *,
    recipient: str,
    subject: str,
    sender_address: str,
    profile_email: str = "",
    recipient_jurisdiction: str | None = None,
    is_suppressed: bool = False,
    policy: CompliancePolicy = DEFAULT_POLICY,
) -> tuple[bool, list[str]]:
    """Mechanical pre-send check. Returns (allowed, notes).

    A suppressed recipient is always blocked. Everything else produces
    notes for the human reviewer; jurisdiction notes are flags, not verdicts.
    """
    notes: list[str] = []
    if is_suppressed:
        return False, [f"Recipient {recipient} is on the suppression/opt-out list."]
    notes.extend(check_subject_honest(subject))
    notes.extend(check_sender_identity(sender_address, profile_email))
    notes.append(policy.jurisdiction_note(recipient_jurisdiction))
    # Subject/sender problems block; jurisdiction notes alone do not.
    blocking = [n for n in notes if "Subject line" in n or "Sender address" in n]
    return (not blocking), notes
