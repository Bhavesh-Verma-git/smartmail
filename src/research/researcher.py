"""
src/research/researcher.py
---------------------------
Top-level research orchestrator.

## Public API

    research_contact(inp, provider=None) -> ResearchResult

    build_research_input(contact) -> ResearchInput
        Convenience helper: converts a ContactRecord (from csv_importer)
        into a ResearchInput without making the research module import
        the contacts module at the top level.

## How it works

1. Inspect the ResearchInput for available URLs (website, role_or_job_url,
   source_url).
2. Fetch each URL with the provider (HttpPageProvider by default).
3. Also record any user-supplied notes as a user_supplied evidence item
   so that human context is preserved alongside retrieved pages.
4. Run the four independent check functions.
5. Assemble and return a ResearchResult.

## Provider injection

Pass a custom provider (e.g. a mock) to research_contact() for testing.
If no provider is supplied, HttpPageProvider is used with default settings.

## What this step does NOT do

- No LLM calls.
- No CV generation.
- No email drafting or sending.
- No ranking or scoring.
"""

from __future__ import annotations

import re

from src.research.checks import (
    _is_http_url,
    check_company_identity,
    check_domain_relationship,
    check_role_evidence,
    check_technical_relevance,
)
from src.research.models import (
    EvidenceItem,
    ResearchInput,
    ResearchResult,
    SUPPORTED,
    REVIEW,
    UNKNOWN,
    NOT_APPLICABLE,
    FOUND,
    NOT_FOUND,
    RETRIEVAL_OK,
    RETRIEVAL_SKIPPED,
    RETRIEVAL_NETWORK_ERROR,
    RETRIEVAL_ERROR,
    APPLICATION_UNKNOWN,
)
from src.research.providers import HttpPageProvider, ResearchProvider

# _FREE_PROVIDERS mirrors the set in email_parser.py so the research module
# stays self-contained (it does not import email_parser).
_FREE_PROVIDERS: frozenset[str] = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in",
    "outlook.com", "hotmail.com", "live.com",
    "protonmail.com", "proton.me", "icloud.com", "me.com",
    "aol.com", "zoho.com", "yandex.com", "mail.com", "gmx.com",
})


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_research_input(contact: object) -> ResearchInput:
    """
    Build a ResearchInput from a ContactRecord produced by csv_importer.

    Uses getattr so this function works with any object that has the right
    attributes -- including simple mocks in tests.

    Parameters
    ----------
    contact : ContactRecord (or duck-typed equivalent)

    Returns
    -------
    ResearchInput
    """
    email_result = getattr(contact, "email_result", None)

    normalized_email = ""
    email_domain = ""
    registered_domain = ""
    is_free_provider = False

    if email_result is not None:
        normalized_email = getattr(email_result, "normalized_email", "")
        email_domain = getattr(email_result, "email_domain", "")
        registered_domain = getattr(email_result, "registered_domain", "")
        is_free_provider = registered_domain in _FREE_PROVIDERS

    company_name = getattr(contact, "company_name", "") or ""
    website = getattr(contact, "website", "") or ""
    notes = getattr(contact, "notes", "") or ""

    # Attempt safe extraction of company name from notes if not explicitly provided
    if not company_name and notes:
        import re
        m_comp = re.search(r"\bcompany\s*[:\-]\s*([A-Za-z0-9&.,\s'-]{2,50})", notes, re.IGNORECASE)
        if m_comp:
            company_name = m_comp.group(1).strip().rstrip(".,;!?")

    # Attempt safe extraction of website from notes if not explicitly provided
    if not website and notes:
        import re
        m_web = re.search(r"https?://[^\s,;]+", notes)
        if m_web:
            website = m_web.group(0).rstrip(".,;!?")

    return ResearchInput(
        company_name=company_name,
        contact_name=getattr(contact, "contact_name", "") or "",
        normalized_email=normalized_email,
        email_domain=email_domain,
        registered_domain=registered_domain,
        is_free_provider=is_free_provider,
        website=website,
        role_or_job_url=getattr(contact, "role_or_job_url", "") if _is_http_url(getattr(contact, "role_or_job_url", "")) else "",
        source_url=getattr(contact, "source_url", "") or "",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Main research function
# ---------------------------------------------------------------------------

def research_contact(
    inp: ResearchInput,
    provider: ResearchProvider | None = None,
) -> ResearchResult:
    """
    Research a contact and return structured evidence and status fields.

    Parameters
    ----------
    inp      : ResearchInput with all known contact details.
    provider : ResearchProvider to use for HTTP fetches.
               Defaults to HttpPageProvider() with a 10-second timeout.

    Returns
    -------
    ResearchResult
    """
    if provider is None:
        from src.research.providers import get_default_research_provider
        provider = get_default_research_provider()

    evidence: list[EvidenceItem] = []

    # --- Tavily search supplement (if TavilySearchProvider is active) ------
    if hasattr(provider, "search_company") and getattr(provider, "is_configured", False):
        try:
            # Query Tavily using only public company/domain keywords (NEVER personal emails)
            tavily_items = provider.search_company(
                company_name=inp.company_name,
                domain=inp.registered_domain if not inp.is_free_provider else "",
                max_results=3,
            )
            for t_item in tavily_items:
                if t_item.retrieval_status == RETRIEVAL_OK:
                    evidence.append(t_item)
        except Exception as exc:
            evidence.append(EvidenceItem(
                source_url="",
                source_type="tavily_search",
                retrieval_status=RETRIEVAL_ERROR,
                retrieval_note="Tavily search skipped on error",
                limitations=[f"Tavily search error: {exc}"],
            ))

    # --- Fetch the company website ----------------------------------------
    if inp.website:
        item = provider.fetch(
            url=inp.website,
            source_type="official_website",
            supports=["company_identity", "domain_relationship",
                      "technical_relevance"],
        )
        evidence.append(item)
    elif inp.registered_domain and not inp.is_free_provider:
        # Safe candidate website discovery for corporate/custom domains
        candidate_urls = [
            f"https://{inp.registered_domain}",
            f"https://www.{inp.registered_domain}",
            f"http://{inp.registered_domain}",
        ]
        discovered_item: EvidenceItem | None = None
        for cand_url in candidate_urls:
            cand = provider.fetch(
                url=cand_url,
                source_type="discovered_website",
                supports=["company_identity", "domain_relationship",
                          "technical_relevance"],
            )
            if cand.retrieval_status == RETRIEVAL_OK and bool((cand.extracted_text or "").strip() or (cand.title or "").strip()):
                cand.limitations.append(
                    f"Candidate website '{cand_url}' discovered from email domain '{inp.registered_domain}'. "
                    "Company identity must be verified from retrieved content."
                )
                discovered_item = cand
                break
            elif cand.retrieval_status == RETRIEVAL_NETWORK_ERROR:
                # If network/DNS fails on the candidate host, stop further attempts on dead domain to avoid stalling
                discovered_item = cand
                break
            elif discovered_item is None:
                discovered_item = cand

        if discovered_item is not None:
            if discovered_item.retrieval_status != RETRIEVAL_OK:
                discovered_item.limitations.append(
                    f"Candidate website discovery failed for domain '{inp.registered_domain}'."
                )
            evidence.append(discovered_item)
        else:
            evidence.append(EvidenceItem(
                source_url="",
                source_type="discovered_website",
                retrieval_status=RETRIEVAL_ERROR,
                retrieval_note="Discovery failed",
                supports=["company_identity", "domain_relationship"],
                limitations=[f"Could not discover a working website for domain '{inp.registered_domain}'."],
            ))
    else:
        evidence.append(EvidenceItem(
            source_url="",
            source_type="official_website",
            retrieval_status=RETRIEVAL_SKIPPED,
            retrieval_note="No website URL supplied.",
            supports=["company_identity"],
            limitations=["No company website was supplied and email is a free provider or lacks domain."],
        ))

    # --- Fetch the role / job URL -----------------------------------------
    if inp.role_or_job_url and _is_http_url(inp.role_or_job_url):
        item = provider.fetch(
            url=inp.role_or_job_url,
            source_type="job_listing",
            supports=["role_evidence", "technical_relevance",
                      "company_identity"],
        )
        evidence.append(item)

    # --- Fetch the source URL (where the user found this contact) ----------
    if inp.source_url:
        item = provider.fetch(
            url=inp.source_url,
            source_type="source_url",
            supports=["company_identity", "role_evidence"],
        )
        evidence.append(item)

    # --- Record user-supplied notes as an evidence item -------------------
    if inp.notes:
        evidence.append(EvidenceItem(
            source_url="",
            source_type="user_supplied",
            title="User notes",
            extracted_text=inp.notes,
            retrieved_at=EvidenceItem.now_utc(),
            retrieval_status=RETRIEVAL_OK,
            retrieval_note="user supplied",
            supports=["company_identity", "role_evidence"],
            limitations=["This is user-supplied text, not independently verified."],
        ))

    # --- Run the four checks ---------------------------------------------
    identity_status, identity_limits = check_company_identity(inp, evidence)
    domain_status, domain_limits = check_domain_relationship(inp, evidence)
    role_status, role_limits, app_status = check_role_evidence(inp, evidence)
    tech_areas = check_technical_relevance(evidence)

    # --- Contradictions & Conflict Detection ------------------------------
    contradictions: list[str] = []
    if inp.company_name and inp.registered_domain and not inp.is_free_provider:
        # Check if supplied company name blatantly conflicts with known corporate domain
        comp_clean = re.sub(r"[^\w]", "", inp.company_name.lower())
        dom_root = inp.registered_domain.split(".")[0].lower()
        # If both are established distinct brands (e.g. google vs microsoft)
        well_known_brands = {"google", "microsoft", "meta", "apple", "amazon", "netflix", "openai", "anthropic"}
        if (comp_clean in well_known_brands or dom_root in well_known_brands) and (comp_clean != dom_root and comp_clean not in dom_root and dom_root not in comp_clean):
            conflict_msg = f"Supplied company name '{inp.company_name}' directly conflicts with email domain '{inp.registered_domain}'."
            contradictions.append(conflict_msg)
            identity_limits.append(conflict_msg)

    # --- Recipient Identity Resolution -----------------------------------
    from src.contacts.email_parser import EmailParseResult
    from src.contacts.recipient_resolver import resolve_recipient

    local_p = inp.normalized_email.split("@")[0] if "@" in inp.normalized_email else ""
    dummy_ep = EmailParseResult(
        raw_input=inp.normalized_email,
        normalized_email=inp.normalized_email,
        local_part=local_p,
        email_domain=inp.email_domain,
        registered_domain=inp.registered_domain,
    )

    # Look for candidate name in external evidence for verified name check
    verified_name = ""
    candidate_to_check = inp.contact_name or (local_p.capitalize() if len(local_p) > 2 and not inp.is_free_provider else "")
    if candidate_to_check and len(candidate_to_check) > 3:
        for item in evidence:
            if item.retrieval_status == RETRIEVAL_OK and item.source_type in ("official_website", "team_page", "careers_page"):
                text_lower = (item.extracted_text or "").lower()
                if candidate_to_check.lower() in text_lower:
                    # Found in external content
                    if any(term in text_lower for term in ("team", "founder", "lead", "engineer", "director", "manager", "head of", "recruiter", "about")):
                        verified_name = candidate_to_check
                        break

    recipient_res = resolve_recipient(
        supplied_name=inp.contact_name,
        email_result=dummy_ep,
        verified_name_from_evidence=verified_name,
    )

    # --- Resolved Company Name --------------------------------------------
    resolved_company = inp.company_name
    if not resolved_company and inp.registered_domain and not inp.is_free_provider:
        resolved_company = inp.registered_domain.split(".")[0].capitalize()

    # --- Substantive AI/ML Evidence Summary --------------------------------
    ai_evidence_summary: list[dict[str, Any]] = []
    ok_ext = [e for e in evidence if e.retrieval_status == RETRIEVAL_OK and e.source_type not in ("user_supplied", "notes")]
    for item in ok_ext:
        text = item.extracted_text or ""
        # Look for sentences containing key AI/ML concepts
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 25 and len(s_clean) < 300:
                s_lower = s_clean.lower()
                for area in ("generative ai", "llm", "rag", "agent", "machine learning", "deep learning", "computer vision", "nlp", "model"):
                    if area in s_lower:
                        ai_evidence_summary.append({
                            "keyword": area,
                            "excerpt": s_clean,
                            "source_url": item.source_url,
                        })
                        break
            if len(ai_evidence_summary) >= 5:
                break

    # Optional LLM structured analysis of public company evidence
    try:
        from src.llm.router import LLMRouter
        router = LLMRouter()
        if router.has_any_llm and ok_ext:
            llm_analysis = router.extract_company_ai_evidence(
                company_name=inp.company_name or resolved_company,
                evidence_texts=[e.extracted_text for e in ok_ext if e.extracted_text],
            )
            if llm_analysis and isinstance(llm_analysis, dict):
                extracted_areas = llm_analysis.get("technical_areas", [])
                if isinstance(extracted_areas, list):
                    for ea in extracted_areas:
                        if isinstance(ea, str) and ea not in tech_areas:
                            # Grounding check: verify at least one word from the area appears in retrieved text
                            ea_words = [w.lower() for w in re.split(r"\W+", ea) if len(w) > 3]
                            full_evidence_corpus = " ".join(e.extracted_text.lower() for e in ok_ext)
                            if any(w in full_evidence_corpus for w in ea_words):
                                tech_areas.append(ea)
                summary_snippet = llm_analysis.get("evidence_summary", "")
                if summary_snippet and isinstance(summary_snippet, str):
                    ai_evidence_summary.insert(0, {
                        "keyword": "llm_analysis",
                        "excerpt": summary_snippet[:300],
                        "source_url": ok_ext[0].source_url if ok_ext else "",
                    })
    except Exception as exc:
        all_limitations.append(f"Optional LLM extraction skipped: {type(exc).__name__}")

    # --- Aggregate limitations -------------------------------------------
    all_limitations: list[str] = (
        identity_limits + domain_limits + role_limits
    )
    for item in evidence:
        all_limitations.extend(item.limitations)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_limitations: list[str] = []
    for lim in all_limitations:
        if lim not in seen:
            seen.add(lim)
            unique_limitations.append(lim)

    # --- Recommended next action -----------------------------------------
    next_action = _recommend_next_action(
        identity_status, domain_status, role_status, unique_limitations
    )

    return ResearchResult(
        company_identity_status=identity_status,
        domain_relationship_status=domain_status,
        role_status=role_status,
        application_status=app_status,
        technical_areas=tech_areas,
        evidence=evidence,
        limitations=unique_limitations,
        recommended_next_action=next_action,
        resolved_company_name=resolved_company,
        recipient_resolution=recipient_res,
        ai_ml_evidence_summary=ai_evidence_summary,
        contradictions=contradictions,
    )


# ---------------------------------------------------------------------------
# Next-action recommendation
# ---------------------------------------------------------------------------

def _recommend_next_action(
    identity: str,
    domain: str,
    role: str,
    limitations: list[str],
) -> str:
    """
    Return a short plain-English guidance string for the human reviewer.

    This is purely advisory -- it does not trigger any automated action.
    """
    if identity == UNKNOWN and role == UNKNOWN:
        return (
            "No pages could be retrieved. Verify the website and URLs manually "
            "before proceeding."
        )

    if identity == REVIEW and domain == REVIEW:
        return (
            "Company identity and domain relationship both need human review. "
            "Verify the company website and email address manually."
        )

    if domain == NOT_APPLICABLE:
        return (
            "Email is from a free provider -- verify the company identity "
            "through other means (website, LinkedIn) before outreach."
        )

    if role == FOUND and identity == SUPPORTED and domain == SUPPORTED:
        return (
            "Evidence supports the company, domain, and role. "
            "Ready for profile tailoring and outreach drafting."
        )

    if role == FOUND and identity == SUPPORTED:
        return (
            "Company and role appear supported. Review domain relationship "
            "before drafting outreach."
        )

    if role == NOT_FOUND:
        return (
            "No role opening was found in the sources checked. "
            "Manually confirm whether an opportunity exists before outreach."
        )

    return (
        "Review the limitations listed above and verify any REVIEW-status "
        "fields manually before proceeding."
    )
