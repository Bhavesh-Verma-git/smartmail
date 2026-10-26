"""
src/outreach/quality_gate.py
----------------------------
Stage 5: Personalization & Grounding Quality Gate.

Evaluates every generated outreach draft against:
- Collected company research evidence
- Candidate master profile & CV
- Repetition, taxonomy dumps, and unsupported claims

Quality Gate Rules:
1. Has supported company anchor: Contains at least one meaningful company-specific
   observation supported by evidence (or if evidence is absent, draft was honestly held for review).
2. Grounded candidate claims: Candidate claims match verified profile facts.
3. Relevant connection: Explains why candidate's experience is relevant to company's focus.
4. Distinguishable from same-sector peers: Uses specific technical initiatives, not universal boilerplates.
5. No unsupported vacancy claims: Avoids claiming confirmed vacancies when none were found.
6. No taxonomy dumps: Forbids dumping long lists of comma-separated buzzwords.
7. Concise and professional: 100-300 words, proper salutation, proper signature, no template placeholders.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.outreach.models import EmailDraft
from src.outreach.opportunity_analyzer import CompanyOpportunity, DOMAIN_DIFFUSION_VISION
from src.outreach.experience_matcher import CandidateExperienceMatch

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Evaluation outcome from the Personalization Quality Gate."""
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    review_flags_to_add: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# Canonical definition and human-readable metadata for the nine quality-gate checks
QUALITY_GATE_CHECKS_METADATA: list[tuple[str, str, str]] = [
    ("has_supported_anchor", "Supported Company Anchor", "Verifies email reflects concrete company research findings rather than generic claims"),
    ("relevant_candidate_connection", "Relevant Candidate Project", "Cites specific, defensible CV project matched to company technical domain"),
    ("grounded_candidate_claims", "Grounded Candidate Claims", "Candidate technical skills and achievements exist in master profile"),
    ("no_unsupported_vacancy_claims", "No False Vacancy Claims", "Refrains from asserting an advertised job posting when none confirmed"),
    ("distinguishable_anchor", "Distinguishable Peer Anchor", "Uses differentiated technical initiatives; avoids identical boilerplate reused across peers"),
    ("no_taxonomy_dump", "No Taxonomy Dumping", "Avoids long comma-separated laundry lists of generic buzzwords"),
    ("no_malformed_fragments", "Clean Extraction Text", "Free of HTML tags, raw Markdown headers, or web scraping artifacts"),
    ("contact_consistency", "Authoritative Contact Consistency", "Signature email matches authoritative candidate email and profile"),
    ("concise_and_professional", "Professional Formatting", "Standard salutation, valid signature, 60–350 words, no unfilled template placeholders"),
]


def evaluate_personalization_quality(
    draft: EmailDraft,
    opportunity: CompanyOpportunity,
    candidate_match: CandidateExperienceMatch,
    profile: dict[str, Any],
) -> QualityGateResult:
    """
    Run the multi-point personalization quality gate on a generated EmailDraft.
    """
    body = draft.body or ""
    body_lower = body.lower()
    comp_clean = opportunity.company_name.strip() if opportunity.company_name else ""
    checks: dict[str, bool] = {}
    failures: list[str] = []
    review_flags: list[str] = []
    notes: list[str] = []

    # 1. Check: No taxonomy dumps
    # Detects patterns like: "focus on A, B, C, D, E" with 4+ comma-separated categories or 4+ broad buckets
    buzzwords = ("generative ai", "agentic ai", "ai/ml", "data science", "software engineering", "machine learning & predictive analytics")
    buzzword_count = sum(1 for b in buzzwords if b in body_lower)
    focus_match = re.search(r"focus\s+on\s+([^.]+?)(?:,\s*i\s+|\.\s*)", body, re.IGNORECASE)
    comma_count = focus_match.group(1).count(",") if focus_match else 0

    has_taxonomy_dump = (comma_count >= 4 and buzzword_count >= 3) or buzzword_count >= 4
    checks["no_taxonomy_dump"] = not has_taxonomy_dump
    if has_taxonomy_dump:
        failures.append("Taxonomy dump detected: email contains a laundry list of broad tech areas.")
        review_flags.append("Generic technology dump detected in body copy.")

    # 2. Check: Supported company anchor
    # If company has strong/moderate evidence, the email body MUST mention the company anchor or finding
    if opportunity.evidence_strength in ("STRONG", "MODERATE"):
        # Look for anchor keywords from concrete_finding
        finding_words = [
            w.lower() for w in re.split(r"\W+", opportunity.concrete_finding)
            if len(w) > 3 and w.lower() not in ("with", "from", "that", "this", "your", "their", "have", "been", "into", "building", "developing")
        ]
        matches = sum(1 for w in finding_words if w in body_lower)
        has_anchor = matches >= 2 or (comp_clean.lower() in body_lower and matches >= 1)
        # Check company fact provenance if structured facts exist
        if opportunity.company_facts and not any(f.validation_status == "VALIDATED" for f in opportunity.company_facts):
            has_anchor = False
            failures.append("Company claim provenance failure: no validated company facts found in retrieved source.")
            review_flags.append("Company evidence provenance failed span validation.")

        checks["has_supported_anchor"] = has_anchor
        if not has_anchor:
            failures.append("Missing company-specific anchor: email does not reflect verified research findings.")
            review_flags.append("Company-specific research anchor missing from body text.")
    else:
        # If low/no evidence, the draft must NOT pretend to be deeply personalized
        checks["has_supported_anchor"] = True
        if not draft.review_required:
            notes.append("Low company evidence; human review required.")

    # 3. Check: Relevant candidate connection
    # Must mention the selected candidate project/experience or organization
    proj_name = candidate_match.organization_or_project.lower()
    has_proj_mention = (
        proj_name in body_lower
        or any(tech.lower() in body_lower for tech in candidate_match.relevant_technologies)
        or "controlplane" in body_lower
        or "red panda" in body_lower
        or "iiser" in body_lower
    )
    checks["relevant_candidate_connection"] = has_proj_mention
    if not has_proj_mention and candidate_match.is_specific_match:
        failures.append("Candidate experience not cited: selected project not referenced in email.")
        review_flags.append("Selected candidate experience missing from email body.")

    # 4. Check: Avoid unsupported vacancy claims
    # If no advertised internship was found, email must NOT claim an open role was advertised
    unsupported_role_claims = (
        "saw your open position for",
        "applying for your advertised",
        "your job posting for",
        "saw your posting on",
        "the advertised role of",
    )
    has_unsupported_role_claim = (
        not opportunity.has_advertised_internship
        and any(claim in body_lower for claim in unsupported_role_claims)
    )
    checks["no_unsupported_vacancy_claims"] = not has_unsupported_role_claim
    if has_unsupported_role_claim:
        failures.append("Unsupported vacancy claim: claimed open listing without verified evidence.")
        review_flags.append("Unsupported vacancy claim detected without verified job listing.")

    # 5. Check: Grounded candidate claims
    # Ensure no fabricated technologies or companies appear in candidate claims
    # (Verify candidate claims against profile)
    profile_text = str(profile).lower()
    known_candidate_tools = (
        "langgraph", "faiss", "bm25", "litellm", "neo4j", "xgboost", "streamlit", "langsmith",
        "flux", "stable diffusion", "controlnet", "trellis", "aws", "celery", "redis",
        "deberta", "bert", "t5", "lstm", "gru", "scikit-learn", "mlflow", "dagshub", "docker"
    )
    candidate_claims_grounded = True
    for tool in known_candidate_tools:
        if tool in body_lower and tool not in profile_text:
            candidate_claims_grounded = False
            break

    # Candidate Invariant: ControlPlane.ai is a personal project, not an employer
    if re.search(r"\b(?:during my time at|intern(?:ship)? at|employed at|worked as an employee at)\s+controlplane\b", body_lower):
        candidate_claims_grounded = False
        failures.append("Invalid candidate claim: ControlPlane.ai is a personal project, not an employer.")
        review_flags.append("ControlPlane.ai mischaracterized as past employer rather than personal project.")

    # Candidate Invariant: E-Commerce Price Prediction is transferable NLP, not ATS
    if "price prediction" in body_lower and any(w in body_lower for w in ("resume-scoring", "ats system project", "resume scoring project")):
        candidate_claims_grounded = False
        failures.append("Invalid candidate claim: Price Prediction is transferable NLP, not a resume-scoring project.")
        review_flags.append("Price Prediction project mischaracterized as resume scoring system.")

    # Candidate Invariant: No unsupported production deployment claim
    if "built production rag systems with citation-aware responses" in body_lower:
        candidate_claims_grounded = False
        failures.append("Ungrounded candidate claim: 'built production RAG systems with citation-aware responses' not in master profile.")
        review_flags.append("Ungrounded RAG production deployment claim detected.")

    # Candidate Invariant: No bloated/unsupported state-machine or classifier claims
    bloated_claim_patterns = (
        r"\b(?:9-node|nine-node)\s+state\s+machine\b",
        r"\breciprocal\s+rank\s+fusion\b",
        r"\bneo4j\s+compliance\b",
        r"\bxgboost\s+hallucination\b",
        r"\bragas\s+judge\b",
    )
    for pat in bloated_claim_patterns:
        if re.search(pat, body_lower):
            candidate_claims_grounded = False
            failures.append(f"Bloated/unsupported claim detected in draft: matches '{pat}'.")
            review_flags.append("Unsupported or bloated technical claim detected in body copy.")
            break

    checks["grounded_candidate_claims"] = candidate_claims_grounded
    if not candidate_claims_grounded and not any("Candidate claim" in f or "Invalid candidate claim" in f or "Ungrounded candidate claim" in f or "Bloated" in f for f in failures):
        failures.append("Ungrounded candidate claim: tool or claim mentioned is not supported by master profile.")
        review_flags.append("Candidate claim contains ungrounded technical skills or relationships.")

    # 6. Check: No malformed fragments or research contamination
    # Disallows '[...]', '([...])', raw HTML, raw Markdown headers, or web navigation residue
    malformed_patterns = [
        r"\[\s*\.\.\.\s*\]",
        r"\(\s*\[\s*\.\.\.\s*\]\s*\)",
        r"<[^>]+>",
        r"&[a-zA-Z0-9#]+;",
        r"\bSkip to content\b",
        r"\bCookie Policy\b",
        r"\bPrivacy Policy\b",
        r"\bLaunch offer\b",
        r"\bAll rights reserved\b",
        r"###|\*\*[^*]+\*\*",
    ]
    has_malformed = any(re.search(pat, body, re.IGNORECASE) for pat in malformed_patterns)
    checks["no_malformed_fragments"] = not has_malformed
    if has_malformed:
        failures.append("Malformed research fragment or extraction artifact detected in body copy.")
        review_flags.append("Research contamination or extraction artifact detected in draft.")

    # 7. Check: Candidate contact consistency
    from src.profile.contact_resolver import resolve_candidate_contact
    contact = resolve_candidate_contact(profile)
    authoritative_email = contact.email
    has_contact_consistency = authoritative_email in body
    checks["contact_consistency"] = has_contact_consistency
    if not has_contact_consistency:
        failures.append(f"Contact email inconsistency: signature does not match authoritative email ({authoritative_email}).")
        review_flags.append(f"Signature email does not match authoritative candidate email {authoritative_email}.")

    if contact.has_unresolved_conflict:
        failures.append("Unresolved candidate contact discrepancy between profile and CV.")
        review_flags.append("Unresolved candidate contact discrepancy requires human confirmation.")

    # 8. Check: Concise and professional formatting
    words = body.split()
    word_count = len(words)
    is_concise = 60 <= word_count <= 350
    has_salutation = body.strip().startswith("Dear ")
    has_signature = "Bhavesh Verma" in body and "IIT Kharagpur" in body
    has_placeholders = bool(re.search(r"\[[A-Za-z0-9_\s]{2,30}\]|\{[a-zA-Z0-9_]+\}", body))

    is_formatting_valid = is_concise and has_salutation and has_signature and not has_placeholders and not has_malformed
    checks["concise_and_professional"] = is_formatting_valid
    if not is_formatting_valid:
        if not is_concise:
            failures.append(f"Word count out of bounds ({word_count} words; expected 60-350).")
        if has_placeholders:
            failures.append("Unfilled template placeholders detected in body.")
            review_flags.append("Template placeholders detected in email text.")
        if not has_salutation:
            failures.append("Missing standard professional salutation.")
        if not has_signature:
            failures.append("Missing candidate signature.")

    # 9. Check: Distinguishable company anchor
    is_differentiated = True
    if opportunity.evidence_strength in ("STRONG", "MODERATE"):
        generic_match = "during my internship at red panda games, i developed generative ai pipelines and diffusion models" in body_lower and "additionally, i have practical experience building context-aware ai middleware" in body_lower
        if generic_match and opportunity.technical_domain != DOMAIN_DIFFUSION_VISION:
            is_differentiated = False
            failures.append("Duplicate paragraph detected: diffusion models and middleware dumped together without domain alignment.")
            review_flags.append("Repetitive cross-domain paragraph detected.")

        if "combining faiss and bm25 with reciprocal rank fusion" in body_lower or "nine-node state machine with parallel auditing" in body_lower:
            is_differentiated = False
            failures.append("Duplicate paragraph detected: hardcoded repetitive ControlPlane.ai boilerplate reused.")
            review_flags.append("Repetitive ControlPlane.ai boilerplate detected.")

    checks["distinguishable_anchor"] = is_differentiated

    passed = len(failures) == 0

    return QualityGateResult(
        passed=passed,
        checks=checks,
        failure_reasons=failures,
        review_flags_to_add=review_flags,
        notes=notes,
    )
