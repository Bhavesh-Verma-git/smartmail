"""
src/research/models.py
-----------------------
Data classes and status constants for the company research component.

## Status values

company_identity_status:
    SUPPORTED   -- at least one source corroborates the company name/domain.
    REVIEW      -- sources are missing, conflicting, or ambiguous.
    UNKNOWN     -- no relevant sources could be retrieved.

domain_relationship_status:
    SUPPORTED       -- email domain matches independently supported company domain.
    REVIEW          -- mismatch or subdomain that needs human judgement.
    UNKNOWN         -- no website/company domain to compare against.
    NOT_APPLICABLE  -- free email provider; domain cannot identify an employer.

role_status:
    FOUND       -- a supplied source explicitly describes a matching internship opportunity.
    NOT_FOUND   -- sources checked provide no matching opening, or explicitly deny availability.
    UNKNOWN     -- the page is inaccessible, evidence is ambiguous, or current
                   availability cannot be determined from the retrieved content.

application_status:
    APPLICATION_OPEN    -- evidence explicitly supports that applications are currently open.
    APPLICATION_CLOSED  -- evidence explicitly states the listing is closed or unavailable.
    APPLICATION_UNKNOWN -- availability cannot be confirmed from retrieved evidence.
                           A stale or undated listing defaults to this, not OPEN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

# company_identity_status
SUPPORTED = "SUPPORTED"
REVIEW = "REVIEW"
UNKNOWN = "UNKNOWN"

# domain_relationship_status  (shares SUPPORTED / REVIEW / UNKNOWN above)
NOT_APPLICABLE = "NOT_APPLICABLE"

# role_status  (shares UNKNOWN above)
FOUND = "FOUND"
NOT_FOUND = "NOT_FOUND"

# application_status -- separate from role_status.
# Finding a role listing != proving applications are currently open.
APPLICATION_OPEN = "APPLICATION_OPEN"
APPLICATION_CLOSED = "APPLICATION_CLOSED"
APPLICATION_UNKNOWN = "APPLICATION_UNKNOWN"

# retrieval_status values used inside EvidenceItem
RETRIEVAL_OK = "OK"
RETRIEVAL_ERROR = "ERROR"
RETRIEVAL_SKIPPED = "SKIPPED"   # no URL was supplied
RETRIEVAL_EMPTY = "EMPTY"       # page returned 200 but contained 0 readable text
RETRIEVAL_BLOCKED = "BLOCKED"   # 403 Forbidden, Cloudflare, captcha
RETRIEVAL_JS_ONLY = "JS_ONLY"   # page requires client-side JS runtime
RETRIEVAL_NETWORK_ERROR = "NETWORK_ERROR" # DNS failure, connection reset, timeout


# ---------------------------------------------------------------------------
# Evidence item
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """
    One piece of retrieved evidence.

    Every claim in ResearchResult should be traceable to one or more
    EvidenceItems.  Do not store conclusions here -- only facts about
    what was retrieved and what the raw content said.

    Fields
    ------
    source_url        URL that was fetched (or attempted). Empty string if
                      this is manually supplied text rather than a URL.
    source_type       Human label: 'official_website', 'careers_page',
                      'job_listing', 'source_url', 'notes', 'user_supplied',
                      'team_page', 'tech_blog'.
    title             <title> tag from the page, or a short label.
    extracted_text    Up to ~2 000 characters of visible page text.
                      Kept as raw retrieved text so later steps can inspect
                      the original evidence rather than a generated summary.
    retrieved_at      UTC timestamp of retrieval (ISO 8601 string).
    retrieval_status  RETRIEVAL_OK | RETRIEVAL_ERROR | RETRIEVAL_SKIPPED |
                      RETRIEVAL_EMPTY | RETRIEVAL_BLOCKED | RETRIEVAL_JS_ONLY |
                      RETRIEVAL_NETWORK_ERROR.
    retrieval_note    HTTP status code, timeout reason, or "user supplied".
    supports          Which claim this evidence may bear on, e.g.
                      'company_identity', 'domain_relationship',
                      'role_evidence', 'technical_relevance', 'recipient_identity'.
    limitations       Reasons this evidence is incomplete, ambiguous, or
                      should be treated with caution.
    claim_supported   Explanation of what specific claim this item establishes.
    contradictions    Any conflicting statements or evidence found.
    """
    source_url: str
    source_type: str = "web_page"
    title: str = ""
    extracted_text: str = ""
    retrieved_at: str = ""
    retrieval_status: str = RETRIEVAL_OK
    retrieval_note: str = ""
    supports: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    claim_supported: str = ""
    contradictions: list[str] = field(default_factory=list)

    @staticmethod
    def now_utc() -> str:
        """Return the current UTC time as an ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Research input
# ---------------------------------------------------------------------------

@dataclass
class ResearchInput:
    """
    Everything the researcher knows about a contact before research begins.

    Mirrors the fields of ContactRecord but is a separate type so the
    research module does not import the contacts module.

    All string fields default to empty string -- never None -- so downstream
    code can use truthiness checks safely.
    """
    company_name: str = ""
    contact_name: str = ""
    normalized_email: str = ""
    email_domain: str = ""          # full domain, e.g. "jobs.bigcorp.com"
    registered_domain: str = ""     # root domain, e.g. "bigcorp.com"
    is_free_provider: bool = False  # True if email is a free provider
    website: str = ""
    role_or_job_url: str = ""
    source_url: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Research result
# ---------------------------------------------------------------------------

@dataclass
class ResearchResult:
    """
    Structured output from research_contact().

    Status fields describe the *evidence available*, not a verdict on
    legitimacy. Missing information produces REVIEW or UNKNOWN, not a
    fraud label.

    Fields
    ------
    company_identity_status      SUPPORTED | REVIEW | UNKNOWN
    domain_relationship_status   SUPPORTED | REVIEW | UNKNOWN | NOT_APPLICABLE
    role_status                  FOUND | NOT_FOUND | UNKNOWN
    application_status           APPLICATION_OPEN | APPLICATION_CLOSED | APPLICATION_UNKNOWN
    technical_areas              Evidence-backed list of relevant technical
                                 areas e.g. ["AI/ML", "Generative AI"].
    evidence                     All EvidenceItem records collected.
    limitations                  Unresolved questions or access problems.
    recommended_next_action      Short plain-English guidance for the reviewer.
    resolved_company_name        Company name established by evidence.
    recipient_resolution         RecipientResolution object with inferred/verified name.
    ai_ml_evidence_summary       Detailed excerpts of genuine AI/ML products or engineering.
    contradictions               List of contradictions detected across sources.
    """
    company_identity_status: str = UNKNOWN
    domain_relationship_status: str = UNKNOWN
    role_status: str = UNKNOWN
    application_status: str = APPLICATION_UNKNOWN
    technical_areas: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    recommended_next_action: str = ""
    resolved_company_name: str = ""
    company_name: str = ""
    recipient_resolution: Any = None
    ai_ml_evidence_summary: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
