"""
src/outreach/context.py
-----------------------
Detects the outreach context from contact data and determines how to frame
the email opening.

Three outreach contexts are supported:

    linkedin_follow_up
        The contact notes indicate a confirmed prior LinkedIn conversation
        or connection. Discovery alone (e.g. "found on LinkedIn", "LinkedIn job post")
        is NOT treated as a prior conversation.

    referral
        The contact notes contain an explicit referral made by a named person.
        Extracts only the actual person's name, discarding trailing job titles,
        company descriptions, or clause words. Negative phrases ("no referral")
        are never treated as referrals.

    direct_application
        No confirmed prior conversation or named referral is available.
        Ambiguous notes default to direct application and set review_required = True.

Strict fabrication rules
------------------------
- Never invent a referral name, conversation date, or relationship.
- Use only text actually present in the contact notes field.
- If the notes are ambiguous, fall back to direct_application and set review_required = True.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class OutreachContext(str, Enum):
    """Detected outreach relationship context."""
    LINKEDIN_FOLLOW_UP = "linkedin_follow_up"
    REFERRAL = "referral"
    DIRECT_APPLICATION = "direct_application"


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Explicit non-referral phrases: e.g. "no referral", "not referred by anyone", "without referral"
_NEGATIVE_REFERRAL_PATTERN = re.compile(
    r"\b(?:no\s+referral|not\s+referred|without\s+(?:a\s+)?referral|neither\s+referred|never\s+referred)\b",
    re.IGNORECASE,
)

# Words that should never be accepted as a referrer's personal name
_INVALID_REFERRER_NAMES = frozenset({
    "anyone", "someone", "nobody", "no one", "anybody", "somebody",
    "colleague", "friend", "employee", "alumni", "network", "contact",
    "person", "recruiter", "founder", "manager",
})

# Patterns used to detect referral
_REFERRAL_CAPTURE_PATTERNS = [
    re.compile(r"referred\s+(?:by|from)\s+([A-Za-z\s.'-]+)", re.IGNORECASE),
    re.compile(r"referral\s*[:\-]\s*([A-Za-z\s.'-]+)", re.IGNORECASE),
    re.compile(r"ref(?:erence)?\s*[:\-]\s*([A-Za-z\s.'-]+)", re.IGNORECASE),
    re.compile(r"introduced?\s+(?:by|through|from)\s+([A-Za-z\s.'-]+)", re.IGNORECASE),
    re.compile(r"intro\s+(?:by|from|through)\s+([A-Za-z\s.'-]+)", re.IGNORECASE),
    re.compile(r"recommendation\s+(?:of|from|by)\s+([A-Za-z\s.'-]+)", re.IGNORECASE),
]

# Patterns that indicate discovery/search/viewing on LinkedIn, NOT a prior conversation
_LINKEDIN_DISCOVERY_PATTERNS = [
    re.compile(r"\b(?:found|saw|discovered|sourced|viewed|seen|searched|looking\s+at)\b.{0,40}\b(?:on\s+linkedin|via\s+linkedin|linkedin\b)", re.IGNORECASE),
    re.compile(r"\blinkedin\b.{0,40}\b(?:job|post|posting|listing|ad|search|feed|page|profile|url|link)\b", re.IGNORECASE),
    re.compile(r"\b(?:found|saw|viewed)\s+(?:the\s+)?(?:founder|ceo|cto|profile|company)\b", re.IGNORECASE),
]

# Patterns that indicate a GENUINE prior conversation or connection on LinkedIn
_LINKEDIN_CONVERSATION_PATTERNS = [
    re.compile(r"\b(?:discussed|chatted|talked|spoke|messaged|conversed|conversation|chat|discussion)\b.{0,40}\b(?:on\s+linkedin|via\s+linkedin|over\s+linkedin|linkedin\b)", re.IGNORECASE),
    re.compile(r"\blinkedin\b.{0,40}\b(?:conversation|chat|discussion|exchange|message|thread)\b", re.IGNORECASE),
    re.compile(r"\b(?:connected|in\s+touch)\s+(?:on\s+linkedin|via\s+linkedin|over\s+linkedin)\b", re.IGNORECASE),
    re.compile(r"\b(?:following\s+(?:our|my)\s+(?:linkedin\s+)?(?:conversation|chat|discussion|exchange|message))\b", re.IGNORECASE),
    re.compile(r"\bwe\s+discussed\b.{0,40}\blinkedin\b", re.IGNORECASE),
    re.compile(r"\bdiscussed\b.{0,40}\bon\s+linkedin\b", re.IGNORECASE),
    re.compile(r"\bmet\s+via\s+linkedin\b", re.IGNORECASE),
]


@dataclass
class ContextResult:
    """
    Detected outreach context and any extracted metadata.

    Attributes
    ----------
    context         : Detected outreach context enum value.
    referrer_name   : Extracted referrer name when context is REFERRAL,
                      empty string otherwise.  Never fabricated.
    review_required : True if the detection was ambiguous and human
                      review of the draft is recommended.
    detection_notes : Human-readable explanation of how the context was
                      determined, useful for review.
    """
    context: OutreachContext
    referrer_name: str = ""
    review_required: bool = False
    detection_notes: str = ""


def _clean_referrer_name(raw: str) -> str:
    """
    Extract only the actual person's name from matched referral text,
    stripping trailing clauses, titles, company descriptions, or punctuation.
    E.g. 'Asit Vidyarthi who is the Co-Founder & CEO' -> 'Asit Vidyarthi'
         'Priya Sharma from the alumni network' -> 'Priya Sharma'
    """
    text = raw.strip()
    # Split on clause/title/conjunction delimiters
    parts = re.split(
        r"\b(?:who\s+is|who\s+was|who\s+works|who|whom|which|whose|from|at|for|via|and|is\s+the|was\s+the|co-founder|founder|ceo|cto|cfo|director|manager|lead|vp|head)\b|[,\(\;\:\n\.\-\|\!]",
        text,
        flags=re.IGNORECASE,
    )
    candidate = parts[0].strip()
    # Remove trailing punctuation or whitespace
    candidate = re.sub(r"[.,;:!?'-]+$", "", candidate).strip()

    # Reject if single lowercase word, invalid words, or too long/short
    if not candidate or len(candidate) < 2 or len(candidate) > 50:
        return ""
    words = candidate.split()
    if len(words) > 4:
        return ""
    if any(w.lower() in _INVALID_REFERRER_NAMES for w in words):
        return ""
    # Ensure at least first letter of each word is capital or contains plausible name chars
    if not any(c.isupper() for c in candidate):
        return ""

    return candidate


def detect_outreach_context(notes: str) -> ContextResult:
    """
    Analyse the contact notes field and return the most appropriate
    outreach context.

    Parameters
    ----------
    notes : str
        Free-form notes from the ContactRecord.  May be empty.

    Returns
    -------
    ContextResult with the detected context and any extracted metadata.
    """
    if not notes or not notes.strip():
        return ContextResult(
            context=OutreachContext.DIRECT_APPLICATION,
            detection_notes="No notes supplied; defaulting to direct application.",
        )

    text = notes.strip()

    # 1. Negative referral check (e.g. "Not referred by anyone", "No referral")
    is_negative_referral = bool(_NEGATIVE_REFERRAL_PATTERN.search(text))

    # 2. Check referral patterns if not explicitly negated
    if not is_negative_referral:
        for pattern in _REFERRAL_CAPTURE_PATTERNS:
            m = pattern.search(text)
            if m:
                raw_name = m.group(1).strip()
                clean_name = _clean_referrer_name(raw_name)
                if clean_name:
                    return ContextResult(
                        context=OutreachContext.REFERRAL,
                        referrer_name=clean_name,
                        detection_notes=f"Referral detected from notes. Referrer name: '{clean_name}'.",
                    )
                elif raw_name and not any(w.lower() in _INVALID_REFERRER_NAMES for w in raw_name.split()):
                    # Pattern matched something that looks ambiguous
                    return ContextResult(
                        context=OutreachContext.DIRECT_APPLICATION,
                        review_required=True,
                        detection_notes=(
                            f"Referral phrasing found ('{raw_name}') but a verified personal name "
                            "could not be extracted with certainty. Defaulted to direct application."
                        ),
                    )

    # 3. LinkedIn context analysis
    has_conversation_signal = any(p.search(text) for p in _LINKEDIN_CONVERSATION_PATTERNS)
    has_discovery_signal = any(p.search(text) for p in _LINKEDIN_DISCOVERY_PATTERNS)
    mentions_linkedin = bool(re.search(r"\blinked.?in\b", text, re.IGNORECASE))

    if has_conversation_signal:
        return ContextResult(
            context=OutreachContext.LINKEDIN_FOLLOW_UP,
            detection_notes="Genuine prior LinkedIn conversation/interaction confirmed from notes.",
        )

    if has_discovery_signal:
        return ContextResult(
            context=OutreachContext.DIRECT_APPLICATION,
            detection_notes="Found on LinkedIn or LinkedIn post without confirmed prior conversation; direct application.",
        )

    if mentions_linkedin:
        # Ambiguous mention without conversation or discovery phrasing
        return ContextResult(
            context=OutreachContext.DIRECT_APPLICATION,
            review_required=True,
            detection_notes="Mentioned LinkedIn without confirmed conversation details; defaulted to direct application with review.",
        )

    # 4. Default: direct application
    return ContextResult(
        context=OutreachContext.DIRECT_APPLICATION,
        detection_notes="No LinkedIn conversation or referral signal found; direct application.",
    )
