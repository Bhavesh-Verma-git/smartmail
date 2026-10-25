"""
src/research/checks.py
-----------------------
Four independent, testable check functions for company research.

Each function accepts pre-fetched EvidenceItem objects (or a ResearchInput)
and returns a status string + any additional details.  No HTTP requests are
made here -- that is the provider's job.

## Design rules

- All logic is deterministic Python (string matching, URL parsing).
- No LLM calls.  Extracted text is kept available for a later step.
- Every status reflects *available evidence*, never a fraud verdict.
- Missing information -> REVIEW or UNKNOWN, never a fabricated answer.
- Free email providers are treated as NOT_APPLICABLE for domain checks --
  they do not prove or disprove a company relationship.

## Functions

    check_company_identity(inp, evidence) -> (status, limitations)
    check_domain_relationship(inp, evidence) -> (status, limitations)
    check_role_evidence(inp, evidence) -> (status, limitations)
    check_technical_relevance(evidence) -> list[str]  (technical areas found)
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import tldextract

from src.research.models import (
    EvidenceItem,
    ResearchInput,
    SUPPORTED,
    REVIEW,
    UNKNOWN,
    NOT_APPLICABLE,
    FOUND,
    NOT_FOUND,
    RETRIEVAL_OK,
    APPLICATION_OPEN,
    APPLICATION_CLOSED,
    APPLICATION_UNKNOWN,
)

# ---------------------------------------------------------------------------
# Keyword sets for technical relevance check
# ---------------------------------------------------------------------------

# Each entry: (area_label, set_of_keywords_to_search_in_page_text)
# Keywords are matched case-insensitively in the extracted page text.
_TECH_AREAS: list[tuple[str, frozenset[str]]] = [
    # Subdisciplines specified in Stage 3
    ("Generative AI & LLMs", frozenset({
        "generative ai", "genai", "large language model", "llm", "gpt",
        "diffusion model", "text-to-image", "foundation model", "prompt engineering",
        "fine-tuning", "lora",
    })),
    ("RAG & Retrieval Systems", frozenset({
        "rag", "retrieval-augmented generation", "retrieval augmented generation",
        "vector search", "vector database", "embeddings", "semantic search", "hybrid search",
        "chromadb", "pinecone", "qdrant", "weaviate", "milvus",
    })),
    ("AI Agents & Workflows", frozenset({
        "agentic", "agentic ai", "autonomous agent", "multi-agent", "agent framework",
        "tool use", "function calling", "ai agents", "ai agent", "langgraph", "crewai", "autogen",
    })),
    ("Machine Learning & Predictive Analytics", frozenset({
        "machine learning", "predictive model", "predictive modeling", "predictive analytics",
        "scikit-learn", "xgboost", "feature engineering", "supervised learning",
        "unsupervised learning", "time series forecasting",
    })),
    ("Deep Learning & NLP", frozenset({
        "deep learning", "neural network", "neural networks", "natural language processing", "nlp",
        "reinforcement learning", "transformer", "transformers", "bert", "tokenization", "pytorch",
        "tensorflow",
    })),
    ("Computer Vision & Generative Media", frozenset({
        "computer vision", "object detection", "image segmentation", "ocr", "opencv",
        "video generation", "stable diffusion", "image generation",
    })),
    ("AI Infrastructure & Evaluation", frozenset({
        "model serving", "inference optimization", "model evaluation", "evals",
        "mlops", "vllm", "tensorrt", "triton", "model monitoring", "model deployment",
    })),
    ("AI Infrastructure & Serving", frozenset({
        "model serving", "inference optimization", "model evaluation", "evals",
        "mlops", "vllm", "tensorrt", "triton", "model monitoring", "model deployment",
    })),
    # Core categories preserved for backward compatibility
    ("Generative AI", frozenset({
        "generative ai", "genai", "large language model", "llm", "gpt",
        "diffusion model", "text-to-image", "foundation model",
        "rag", "retrieval-augmented generation", "vector search",
    })),
    ("Agentic AI", frozenset({
        "agentic", "agentic ai", "autonomous agent", "multi-agent", "agent framework",
        "tool use", "function calling", "ai agents",
    })),
    ("AI/ML", frozenset({
        "machine learning", "deep learning", "neural network", "ai/ml",
        "artificial intelligence", "model training", "inference",
        "computer vision", "natural language processing", "nlp",
        "reinforcement learning", "predictive analytics",
    })),
    ("Data Science", frozenset({
        "data science", "data scientist", "data analysis", "analytics",
        "statistical", "predictive model", "business intelligence",
        "data pipeline", "feature engineering",
    })),
    ("Software Engineering", frozenset({
        "software engineer", "backend", "frontend", "full stack",
        "api", "microservice", "cloud native", "devops", "platform engineer",
        "distributed system",
    })),
]

# Role-related keyword tiers for internship detection.
#
# Tier 1 — Negative phrases: page explicitly denies an opening.
# If any of these appear in the text, role_status -> NOT_FOUND regardless of
# other keywords present.
_NEGATIVE_PHRASES: frozenset[str] = frozenset({
    "no internship",
    "no internships",
    "internships are not available",
    "internships are currently unavailable",
    "not currently offering intern",
    "not currently accepting intern",
    "not accepting intern",
    "no current intern",
    "internship programme is closed",
    "internship program is closed",
    "position has been filled",
    "this role is no longer available",
    "application period has closed",
    "applications are closed",
    "listing has expired",
    "posting has expired",
})

# Tier 2 — Active opening signals: page explicitly suggests a live opportunity.
# Must appear together with a Tier 3 keyword to count as FOUND + APPLICATION_OPEN.
_ACTIVE_SIGNALS: frozenset[str] = frozenset({
    "apply now",
    "apply today",
    "applications are open",
    "applications now open",
    "applications open",
    "now hiring",
    "we are hiring",
    "we're hiring",
    "submit your application",
    "submit an application",
    "open application",
    "click to apply",
    "how to apply",
    "application deadline",
    "start date",
    "apply before",
    "apply by",
    "accepting applications",
    "open positions",
    "open roles",
    "job opening",
    "job opportunity",
})

# Tier 3 — Internship existence keywords.
# A page must contain at least one of these to be considered at all.
# Alone (without an Active Signal) they produce UNKNOWN, not FOUND.
_INTERN_KEYWORDS: frozenset[str] = frozenset({
    "intern",
    "internship",
    "co-op",
    "coop",
    "placement year",
    "industrial placement",
    "work placement",
    "summer placement",
    "trainee",
    "apprentice",
})


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _registered_domain_from_url(url: str) -> str:
    """
    Extract the registered domain from a URL string.

    Examples:
        "https://jobs.bigcorp.com/role/123" -> "bigcorp.com"
        "https://bigcorp.com"               -> "bigcorp.com"
        ""                                  -> ""
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path  # fallback if scheme is missing
        ext = tldextract.extract(host)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return host
    except Exception:  # pylint: disable=broad-except
        return ""


def _is_http_url(val: str) -> bool:
    """Return True if val is a valid HTTP or HTTPS URL."""
    if not val or not isinstance(val, str):
        return False
    v = val.strip().lower()
    return v.startswith("http://") or v.startswith("https://")


def _text_contains_any(text: str, keywords: frozenset[str]) -> bool:
    """Return True if any keyword appears in text (case-insensitive)."""
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _is_negated_mention(text: str, match_start: int, match_end: int) -> bool:
    """
    Check if a keyword match at [match_start:match_end] in text is negated.

    Checks:
    1. Preceding text within the same clause (split on sentence/clause boundaries).
       Detects negation triggers ('not', 'no', 'never', 'neither', 'nor', 'without',
       'zero', 'aren't', 'isn't', 'don't', 'doesn't', etc.) within 0-5 words before
       the keyword, while preserving affirmative phrases ('not only', 'not just').
    2. Direct hyphenated prefixes like 'non-ai' or 'non-ml'.
    3. Following negation within the same clause (e.g. 'ai/ml is not used', 'ai/ml: no').
    """
    prefix = text[max(0, match_start - 80):match_start]
    clause_parts = re.split(
        r'[\.\;\!\?\n\,]|\b(?:but|however|whereas|although)\b',
        prefix,
        flags=re.IGNORECASE,
    )
    clause_prefix = clause_parts[-1].lower()

    # Direct non- prefix, e.g. "non-ai", "non-ml"
    if re.search(r'\bnon\s*-\s*$', clause_prefix):
        return True

    # Preceding negation words
    neg_match = re.search(
        r'\b(not|no|never|neither|nor|without|zero|aren\'t|isn\'t|wasn\'t|weren\'t|don\'t|doesn\'t|didn\'t|cannot|can\'t|won\'t)\b(?:\s+\w+){0,5}\s*$',
        clause_prefix,
    )
    if neg_match:
        neg_phrase = clause_prefix[neg_match.start():]
        # Affirmative exceptions like "not only", "not just"
        if re.search(r'\bnot\s+(?:only|just)\b', neg_phrase):
            return False
        return True

    # Following negation within the same clause
    suffix = text[match_end:min(len(text), match_end + 40)]
    suffix_clause = re.split(
        r'[\.\;\!\?\n\,]|\b(?:but|however|whereas|although)\b',
        suffix,
        flags=re.IGNORECASE,
    )[0].lower()
    if re.search(r'^\s*(?:is|are|was|were)?\s*(?:not|unsupported|excluded|none)\b', suffix_clause):
        return True
    if re.search(r'^\s*:\s*(?:no|false|none|n/a)\b', suffix_clause):
        return True

    return False


def _keyword_has_positive_mention(text: str, keywords: frozenset[str]) -> bool:
    """
    Return True if any keyword in keywords appears in text in a positive,
    non-negated context (case-insensitive with word boundary checks).
    """
    for kw in keywords:
        pattern = r'(?<!\w)' + re.escape(kw) + r'(?!\w)'
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if not _is_negated_mention(text, m.start(), m.end()):
                return True
    return False


def _ok_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    """Return only successfully retrieved evidence items."""
    return [e for e in evidence if e.retrieval_status == RETRIEVAL_OK]


def _retrieved_external_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    """
    Return only successfully retrieved external evidence items with non-empty content.
    Excludes user-supplied notes, placeholder items, or local candidate inputs.
    User notes, profile data, or unretrieved pages NEVER establish company identity
    or technical relevance.
    """
    return [
        e for e in evidence
        if e.retrieval_status == RETRIEVAL_OK
        and e.source_type not in ("user_supplied", "candidate_profile", "user_notes", "notes")
        and not (e.source_type or "").startswith("user_")
        and bool((e.extracted_text or "").strip() or (e.title or "").strip())
    ]


# ---------------------------------------------------------------------------
# A. Company identity check
# ---------------------------------------------------------------------------

def check_company_identity(
    inp: ResearchInput,
    evidence: list[EvidenceItem],
) -> tuple[str, list[str]]:
    """
    Determine whether available evidence supports the supplied company name.

    Logic
    -----
    1. If no company name was supplied, return REVIEW immediately.
    2. Look for the company name in the extracted text of successfully retrieved pages.
    3. If found in at least one page: SUPPORTED, but record whether the source
       is company-controlled (official_website) or potentially independent
       (source_url, job_listing from a third-party platform).
    4. If pages were retrieved but the name was not found: REVIEW.
    5. If no pages could be retrieved at all: UNKNOWN.

    Important: a company-controlled website trivially contains its own name.
    This is weak corroboration, not independent verification.  The limitations
    list always records the nature of the source so the human reviewer can judge.

    Returns
    -------
    (status, limitations)
        status      : SUPPORTED | REVIEW | UNKNOWN
        limitations : list of strings describing open questions.
    """
    # Source types that are company-controlled (not independent)
    _COMPANY_CONTROLLED: frozenset[str] = frozenset({
        "official_website", "careers_page", "discovered_website",
    })

    limitations: list[str] = []

    if not inp.company_name:
        limitations.append(
            "No company name was supplied -- identity cannot be verified."
        )
        return REVIEW, limitations

    ok_items = _retrieved_external_evidence(evidence)

    if not ok_items:
        # Check if page was retrieved but had empty text
        empty_items = [
            e for e in evidence
            if e.retrieval_status == RETRIEVAL_OK
            and e.source_type not in ("user_supplied", "candidate_profile", "user_notes", "notes")
            and not (e.source_type or "").startswith("user_")
            and not bool((e.extracted_text or "").strip() or (e.title or "").strip())
        ]
        if empty_items:
            limitations.append(
                f"Page retrieved from '{empty_items[0].source_url or 'candidate website'}' contained no visible text; "
                "company identity cannot be confirmed."
            )
        else:
            limitations.append(
                "No external pages could be retrieved; company identity could not be checked."
            )
        return UNKNOWN, limitations

    name_lower = inp.company_name.strip().lower()

    # Separate items where the company name was found
    company_controlled_matches: list[EvidenceItem] = []
    independent_matches: list[EvidenceItem] = []

    for e in ok_items:
        if name_lower in (e.extracted_text + " " + e.title).lower():
            if e.source_type in _COMPANY_CONTROLLED:
                company_controlled_matches.append(e)
            else:
                independent_matches.append(e)

    all_matches = company_controlled_matches + independent_matches

    if all_matches:
        failed = [e for e in evidence if e.retrieval_status != RETRIEVAL_OK]
        if failed:
            limitations.append(
                f"{len(failed)} URL(s) could not be retrieved; "
                "identity check is based on partial evidence only."
            )
        if independent_matches:
            limitations.append(
                "Company name found on at least one potentially independent source "
                f"(e.g. {independent_matches[0].source_type}). "
                "This provides stronger corroboration than a company-controlled page alone."
            )
        elif company_controlled_matches:
            limitations.append(
                "Company name found only on company-controlled page(s) "
                f"({', '.join(e.source_type for e in company_controlled_matches)}). "
                "This is the company's own content and does not constitute independent "
                "verification of identity."
            )
        return SUPPORTED, limitations

    # Pages retrieved but company name not found in any of them
    limitations.append(
        f"Company name '{inp.company_name}' was not found in any retrieved page. "
        "The supplied website may be incorrect, or the page content may not "
        "include the company name prominently."
    )
    return REVIEW, limitations


# ---------------------------------------------------------------------------
# B. Email-domain relationship check
# ---------------------------------------------------------------------------

def check_domain_relationship(
    inp: ResearchInput,
    evidence: list[EvidenceItem],
    identity_status: str | None = None,
) -> tuple[str, list[str]]:
    """
    Compare the contact email domain with the supplied/evidenced company domain.

    Rules (in priority order)
    -----
    1. If the email is from a free provider -> NOT_APPLICABLE.
       Free providers say nothing about employer identity.
    2. If no website was supplied and no evidence was retrieved -> UNKNOWN.
    3. Compare email registered_domain with website registered_domain.
       - Match   -> SUPPORTED (with a note that it is not conclusive proof).
       - Mismatch:
           a. If email_domain is a subdomain of the website domain -> REVIEW
              with a note that subdomains are legitimate.
           b. Otherwise -> REVIEW with a note about the mismatch.
    4. External recruiting platforms (greenhouse.io, lever.co, workday.com,
       smartrecruiters.com, etc.) are noted but not counted as mismatches.

    Returns
    -------
    (status, limitations)
    """
    limitations: list[str] = []
    RECRUITING_PLATFORMS: frozenset[str] = frozenset({
        "greenhouse.io", "lever.co", "workday.com", "smartrecruiters.com",
        "ashbyhq.com", "recruitee.com", "bamboohr.com", "jobvite.com",
        "icims.com", "taleo.net", "successfactors.com", "workable.com",
        "applytojob.com", "teamtailor.com",
    })

    # Rule 1: free email provider
    if inp.is_free_provider:
        limitations.append(
            f"'{inp.registered_domain}' is a free email provider. "
            "The email domain cannot be used to identify an employer."
        )
        return NOT_APPLICABLE, limitations

    # Determine company registered domain from the supplied or discovered website
    website_to_check = inp.website
    is_discovered_website = False
    if not website_to_check:
        for e in evidence:
            if e.source_type in ("official_website", "discovered_website") and e.source_url and e.retrieval_status == RETRIEVAL_OK:
                website_to_check = e.source_url
                is_discovered_website = True
                break

    company_rd = _registered_domain_from_url(website_to_check)

    if not company_rd and not website_to_check:
        limitations.append(
            "No company website was supplied; domain relationship cannot be checked."
        )
        return UNKNOWN, limitations

    if not company_rd:
        limitations.append(
            f"Could not extract a registered domain from website '{website_to_check}'."
        )
        return UNKNOWN, limitations

    email_rd = inp.registered_domain  # already extracted by email_parser

    if not email_rd:
        limitations.append("Email registered domain is unavailable for comparison.")
        return UNKNOWN, limitations

    # Check recruiting platform
    if email_rd in RECRUITING_PLATFORMS:
        limitations.append(
            f"Email domain '{email_rd}' belongs to an external recruiting platform. "
            "This is common and does not indicate a problem, but the email cannot "
            "be confirmed as a direct company address."
        )
        return REVIEW, limitations

    # Subdomain case: check this BEFORE registered-domain equality.
    # If the full email domain (e.g. "jobs.bigcorp.example") is a proper
    # subdomain of the company domain ("bigcorp.example"), flag it for review
    # even though the registered_domain portion matches.
    email_full = inp.email_domain  # e.g. "jobs.bigcorp.example"
    if email_full != company_rd and (
        email_full.endswith(f".{company_rd}") or email_full == company_rd
    ):
        limitations.append(
            f"Email domain '{email_full}' is a subdomain of the company domain "
            f"'{company_rd}'. This is common for recruitment subdomains. "
            "Human review recommended."
        )
        return REVIEW, limitations

    # Direct match on registered domain (no subdomain involved)
    if email_rd == company_rd:
        if is_discovered_website and identity_status != SUPPORTED:
            # Circular reasoning guard: the "website" here was itself
            # guessed from the email domain (https://<domain>), so a
            # bare domain match proves nothing beyond the guess. It stays
            # REVIEW -- UNLESS the retrieved page content itself names the
            # domain label (e.g. the page at enterprise.com says
            # "Enterprise"). That is content corroboration, not circular
            # reasoning: the page could have said anything, and it said
            # the matching name.
            domain_label = (email_rd.split(".")[0] or "").lower()
            content_corroborated = False
            for _e in evidence:
                if _e.source_type == "discovered_website" and _e.retrieval_status == RETRIEVAL_OK:
                    _text = ((_e.title or "") + " " + (_e.extracted_text or "")).lower()
                    if domain_label and domain_label in _text:
                        content_corroborated = True
                        break
            if not content_corroborated:
                limitations.append(
                    f"Email domain '{email_rd}' matches candidate website domain '{company_rd}', "
                    "but that website was discovered from the email domain itself, so the "
                    "match is circumstantial and cannot SUPPORT the relationship on its own. "
                    "Company identity must be validated from page content."
                )
                return REVIEW, limitations
            limitations.append(
                f"Email domain '{email_rd}' matches candidate website domain '{company_rd}'. "
                "The website was discovered from the email domain, but its retrieved "
                "content names the domain label, corroborating the relationship."
            )
            return SUPPORTED, limitations
        if is_discovered_website:
            limitations.append(
                f"Email domain '{email_rd}' matches candidate website domain '{company_rd}'. "
                "The website was discovered from the email domain, but company "
                "identity was independently confirmed from page content."
            )
            return SUPPORTED, limitations
        limitations.append(
            "Domain match is useful evidence but does not confirm that this specific "
            "person is an authorised recruiter."
        )
        return SUPPORTED, limitations

    # Mismatch
    limitations.append(
        f"Email domain '{email_rd}' does not match the company website domain "
        f"'{company_rd}'. This could indicate an external recruiter, a typo, "
        "or a different company. Human review required."
    )
    return REVIEW, limitations


# ---------------------------------------------------------------------------
# C. Internship / role evidence check
# ---------------------------------------------------------------------------

def check_role_evidence(
    inp: ResearchInput,
    evidence: list[EvidenceItem],
) -> tuple[str, list[str], str]:
    """
    Determine whether available evidence describes an active internship opening.

    Returns a three-tuple: (role_status, limitations, application_status)

    Role detection uses three tiers of signals, checked in priority order:

    Tier 1 — Negative phrases (_NEGATIVE_PHRASES)
        If any phrase explicitly denies availability ("no internships",
        "applications are closed", etc.), the function returns NOT_FOUND
        immediately, regardless of other keywords.

    Tier 2 — Active opening signals (_ACTIVE_SIGNALS) + Tier 3 keywords
        If the page contains an internship keyword AND an active signal
        ("apply now", "applications open", "accepting applications", etc.),
        the function returns FOUND with application_status=APPLICATION_OPEN.

    Keyword-only (Tier 3 without Tier 2)
        The page mentions an internship keyword but provides no active signal.
        This covers stale listings, historical mentions ("our former interns"),
        and careers pages that list internships without open applications.
        Returns UNKNOWN + application_status=APPLICATION_UNKNOWN.
        -> A keyword alone is NOT sufficient to mark the role as FOUND.

    No keywords at all
        Pages retrieved but no internship keywords found.
        Returns NOT_FOUND.

    Page inaccessible
        Returns UNKNOWN + application_status=APPLICATION_UNKNOWN.

    Role URL evidence is prioritised: if a role_or_job_url was supplied and
    successfully retrieved, only that page is evaluated.  If it fails,
    UNKNOWN is returned regardless of what other pages say.
    If no role URL was supplied, all successfully retrieved pages are checked.

    Returns
    -------
    (role_status, limitations, application_status)
    """
    limitations: list[str] = []
    ok_items = _retrieved_external_evidence(evidence)
    has_role_url = bool(inp.role_or_job_url and _is_http_url(inp.role_or_job_url))

    # --- Inaccessible page ---------------------------------------------------
    if not ok_items:
        if has_role_url:
            limitations.append(
                f"Role URL '{inp.role_or_job_url}' could not be retrieved; "
                "role evidence is unavailable."
            )
        else:
            limitations.append(
                "No role URL was supplied and no pages could be retrieved."
            )
        return UNKNOWN, limitations, APPLICATION_UNKNOWN

    # Prioritise items from the role URL
    role_items = [
        e for e in ok_items
        if has_role_url and e.source_url == inp.role_or_job_url
    ]

    # If a role URL was given but failed retrieval -> UNKNOWN
    if has_role_url and not role_items:
        limitations.append(
            f"Role URL '{inp.role_or_job_url}' could not be retrieved; "
            "role evidence is unavailable."
        )
        return UNKNOWN, limitations, APPLICATION_UNKNOWN

    check_items = role_items if role_items else ok_items

    for item in check_items:
        combined = (item.extracted_text + " " + item.title).lower()

        # --- Tier 1: explicit denial -----------------------------------------
        if _text_contains_any(combined, _NEGATIVE_PHRASES):
            limitations.append(
                "Retrieved page explicitly indicates no internship is currently "
                f"available (source: {item.source_url or item.source_type})."
            )
            return NOT_FOUND, limitations, APPLICATION_CLOSED

        # --- Tier 2+3: active signal + intern keyword ------------------------
        has_intern_keyword = _text_contains_any(combined, _INTERN_KEYWORDS)
        has_active_signal = _text_contains_any(combined, _ACTIVE_SIGNALS)

        if has_intern_keyword and has_active_signal:
            if not has_role_url:
                limitations.append(
                    "Internship opening signals found on a general page "
                    "(not a dedicated role URL). Verify the specific opening manually."
                )
            return FOUND, limitations, APPLICATION_OPEN

        # --- Keyword only (Tier 3, no Tier 2) --------------------------------
        if has_intern_keyword:
            limitations.append(
                "Page contains internship-related keywords but no active application "
                f"signal (source: {item.source_url or item.source_type}). "
                "This may be a stale listing, historical reference, or a careers page "
                "without open applications. Current availability cannot be confirmed."
            )
            return UNKNOWN, limitations, APPLICATION_UNKNOWN

    # --- No internship keywords found anywhere --------------------------------
    if has_role_url:
        limitations.append(
            f"No internship keywords found on the retrieved page for "
            f"'{inp.role_or_job_url}'. The opening may have closed, moved, "
            "or the page content may require JavaScript to display."
        )
    else:
        limitations.append(
            "No role URL was supplied. No internship keywords were found on the "
            "pages checked. Manually verify whether an opening exists."
        )
    return NOT_FOUND, limitations, APPLICATION_UNKNOWN


# ---------------------------------------------------------------------------
# D. Technical relevance check
# ---------------------------------------------------------------------------

def check_technical_relevance(
    evidence: list[EvidenceItem],
) -> list[str]:
    """
    Identify technical areas mentioned in retrieved page text.

    Returns a deduplicated list of area labels (e.g. ["AI/ML", "Data Science"])
    in the order they appear in _TECH_AREAS.  Returns an empty list if no
    relevant text was found, or if technical keywords appear only in negated
    contexts (e.g. "We are not an AI/ML-focused company.").

    Only RETRIEVAL_OK external items contribute.  User-supplied notes, profile
    data, or unretrieved pages NEVER establish technical relevance.
    """
    external_items = _retrieved_external_evidence(evidence)
    all_text = " ".join(
        (e.extracted_text or "") + " " + (e.title or "")
        for e in external_items
    )

    if not all_text.strip():
        return []

    found: list[str] = []
    for label, keywords in _TECH_AREAS:
        if _keyword_has_positive_mention(all_text, keywords):
            found.append(label)

    return found
