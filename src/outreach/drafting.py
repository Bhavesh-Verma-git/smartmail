"""
src/outreach/drafting.py
------------------------
Outreach email drafting logic for both Uploaded CV and Tailored CV modes.

Personalization is grounded exclusively in:
  - Verified company research (matched_areas, evidence text)
  - The candidate profile stored in data/master_profile.json
  - Outreach context detected from contact notes (linkedin_follow_up,
    referral, direct_application)

Strict rules
------------
- Never invent a referral name, prior conversation, company product,
  hiring status, skill, project, or achievement.
- If required information is absent, produce a truthful neutral draft
  and record specific review flags.
- Signature format: Best regards, name, institution, phone, email.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from src.outreach.context import OutreachContext, detect_outreach_context
from src.outreach.experience_matcher import match_candidate_experience
from src.outreach.models import EmailDraft, GenerationStatus, PersonalizationRecord, WorkflowMode
from src.outreach.opportunity_analyzer import analyze_company_opportunity
from src.outreach.personalized_generator import generate_personalized_email_body
from src.outreach.quality_gate import evaluate_personalization_quality
from src.profile.contact_resolver import resolve_candidate_contact
from src.verification.models import PipelineResult, RELEVANT, SPECIFIC_INTERNSHIP


# ---------------------------------------------------------------------------
# Internal profile helpers
# ---------------------------------------------------------------------------

def _candidate_name(profile: dict[str, Any], cv_text: str | None = None) -> str:
    contact = resolve_candidate_contact(profile, cv_text)
    return contact.full_name


def _preferred_name(profile: dict[str, Any]) -> str:
    return profile.get("personal_info", {}).get("preferred_name", "") or _candidate_name(profile)


def _institution(profile: dict[str, Any], cv_text: str | None = None) -> str:
    contact = resolve_candidate_contact(profile, cv_text)
    return contact.institution


def _phone(profile: dict[str, Any], cv_text: str | None = None) -> str:
    contact = resolve_candidate_contact(profile, cv_text)
    return contact.phone


def _email_addr(profile: dict[str, Any], cv_text: str | None = None) -> str:
    contact = resolve_candidate_contact(profile, cv_text)
    return contact.email


def _availability(profile: dict[str, Any]) -> str:
    prefs = profile.get("preferences", {})
    val = prefs.get("availability_duration_months")
    return str(val).strip() if val else ""


def _format_internship_duration(raw: str) -> str:
    """
    Ensure correct English grammar for internship duration.
    e.g. '3-6 months' -> 'a 3–6 month internship'
         '6 months'   -> 'a six-month internship'
    NEVER produces 'a 3-6 months internship'.
    """
    s = (raw or "").strip()
    if not s:
        return "an internship"

    # Normalize hyphens and whitespace
    s = re.sub(r"\s*[-–]\s*", "–", s)

    # Range like '3–6 months' -> 'a 3–6 month internship'
    m_range = re.match(r"^(\d+)–(\d+)\s*months?$", s, re.IGNORECASE)
    if m_range:
        return f"a {m_range.group(1)}–{m_range.group(2)} month internship"

    # Single number like '6 months' -> 'a six-month internship'
    m_single = re.match(r"^(\d+)\s*months?$", s, re.IGNORECASE)
    if m_single:
        num = int(m_single.group(1))
        word_map = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 12: "twelve"}
        word = word_map.get(num, str(num))
        return f"a {word}-month internship"

    # Fallback cleanup
    s = re.sub(r"\bmonths\b", "month", s, flags=re.IGNORECASE)
    if not s.lower().endswith("internship"):
        s = f"{s} internship"
    if not s.lower().startswith("a "):
        s = f"a {s}"
    return s


# ---------------------------------------------------------------------------
# Subject line builder
# ---------------------------------------------------------------------------

def build_subject_line(
    candidate_name: str,
    target_role: str,
    company_name: str,
    case_type: str = "",
    context: OutreachContext = OutreachContext.DIRECT_APPLICATION,
    referrer_name: str = "",
) -> str:
    """
    Generate a professional subject line matched to the outreach context.
    Does not automatically append candidate name to every direct application subject.
    Never invents a company name if missing.
    """
    role = target_role or "AI/ML Internship"
    clean_company = company_name.strip() if company_name and company_name != "Unknown Company" else ""

    if context == OutreachContext.LINKEDIN_FOLLOW_UP:
        return f"{role} Opportunity — Following Our LinkedIn Conversation"

    if context == OutreachContext.REFERRAL and referrer_name:
        return f"{role} Application — Referred by {referrer_name}"

    if clean_company:
        if case_type == SPECIFIC_INTERNSHIP:
            return f"Application for {role} — {clean_company}"
        if not target_role:
            return f"{role} Inquiry — {clean_company}"
        return f"{role} Application — {clean_company}"

    # Company is unknown or missing
    if case_type == SPECIFIC_INTERNSHIP:
        return f"Application for {role}"
    return f"Inquiry regarding {role} Opportunities"


# ---------------------------------------------------------------------------
# Body section builders
# ---------------------------------------------------------------------------

def _build_salutation(
    contact_name: str,
    company_name: str,
    is_generic: bool = False,
) -> str:
    """Return a warm, professional salutation respecting recipient identity."""
    clean_company = company_name.strip() if company_name and company_name != "Unknown Company" else ""
    name = (contact_name or "").strip()

    generic_keywords = {
        "hr", "careers", "jobs", "info", "contact", "support", "admin",
        "recruiting", "talent", "team", "founder", "founders", "ceo",
        "inquiries", "inquiry", "office", "hello", "hi", "sales", "hiring",
        "intern", "interns", "internship", "internships", "manager", "staffing",
        "general", "inbox", "lead", "head", "hrindia", "hrdept", "hrops", "hrd",
    }
    name_words = {w.lower() for w in re.findall(r"[A-Za-z]+", name)}
    name_lower = name.lower()
    is_generic_name = (
        is_generic
        or not name
        or name_lower in generic_keywords
        or bool(name_words & generic_keywords)
        or any(name_lower.startswith(p) for p in (
            "hr", "career", "talent", "recruit", "jobs", "job", "hiring", "intern", "staffing", "info", "contact", "support", "admin", "office"
        ))
        or any(w.startswith(("hr", "career", "talent", "recruit", "job", "hiring", "intern", "staffing")) for w in name_words)
    )

    if is_generic_name:
        if clean_company:
            return f"Dear Hiring Team at {clean_company},"
        return "Dear Hiring Team,"

    if name:
        first = name.split()[0].capitalize()
        return f"Dear {first},"
    if clean_company:
        return f"Dear Hiring Team at {clean_company},"
    return "Dear Hiring Team,"



def _build_opening_paragraph(
    context: OutreachContext,
    referrer_name: str,
    role_mention: str,
    company_name: str,
    profile: dict[str, Any],
    case_type: str,
) -> str:
    """
    Build the opening paragraph tailored to context and candidate identity.
    """
    institution = _institution(profile)
    clean_company = company_name.strip() if company_name and company_name != "Unknown Company" else ""
    at_company = f" at {clean_company}" if clean_company else ""

    if context == OutreachContext.LINKEDIN_FOLLOW_UP:
        return (
            f"Following our recent conversation on LinkedIn, I am writing to formally express my interest "
            f"in {role_mention} opportunities{at_company}. "
            f"I am a fourth-year Dual Degree student at {institution} with a focus on Applied AI and Machine Learning."
        )

    if context == OutreachContext.REFERRAL and referrer_name:
        return (
            f"I am writing to apply for an {role_mention}{at_company}, on the recommendation of {referrer_name}, "
            f"who suggested I reach out to your team. "
            f"I am a fourth-year Dual Degree student at {institution} with a focus on Applied AI and Machine Learning."
        )

    if case_type == SPECIFIC_INTERNSHIP:
        return (
            f"I am writing to express my strong interest in the {role_mention} role{at_company}. "
            f"I am a fourth-year Dual Degree student at {institution} with a focus on Applied AI and Machine Learning."
        )

    return (
        f"I am writing to inquire about potential {role_mention} opportunities{at_company}. "
        f"I am a fourth-year Dual Degree student at {institution} with a strong foundation in Applied AI and Machine Learning."
    )



def _build_closing(availability: str, seed_key: str = "") -> str:
    """Return a concise, natural closing paragraph with grammatically correct availability if known."""
    clean_avail = (availability or "").strip()
    avail_phrase = f" I am available for {_format_internship_duration(clean_avail)}." if clean_avail else ""

    if seed_key:
        seed = sum(ord(c) for c in seed_key) % 3
        if seed == 1:
            base = (
                "I have attached my CV detailing my technical projects and coursework, and would be glad "
                "to discuss how I could support your engineering goals."
            )
            return f"{base}{avail_phrase}"
        elif seed == 2:
            base = (
                "My CV is attached for your review. I would welcome the chance to speak with your team "
                "about potential technical contributions."
            )
            return f"{base}{avail_phrase}"

    base = (
        "I have attached my CV for your review and would welcome the opportunity to discuss "
        "how my technical background can contribute to your team."
    )
    return f"{base}{avail_phrase}"



def _build_signature(profile: dict[str, Any], cv_text: str | None = None) -> str:
    """
    Build clean signature using authoritative candidate contact details.
    Sign-off: Best regards, Name (IIT Kharagpur), phone, email.
    """
    contact = resolve_candidate_contact(profile, cv_text)
    name = contact.full_name
    institution = contact.institution
    phone = contact.phone
    email = contact.email

    lines = ["Best regards,", "", f"{name} ({institution})"]
    if phone:
        lines.append(phone)
    if email:
        lines.append(email)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CV & Profile Technical Connections
# ---------------------------------------------------------------------------

def _determine_cv_connections(
    matched_areas: list[str],
    profile: dict[str, Any],
    cv_text: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract relevant verified skills and projects from candidate profile/CV
    that directly align with the verified company technical areas.
    Never invents projects or returns ungrounded fallbacks when matched_areas is empty or unrelated.
    """
    if not matched_areas:
        return [], []

    skills: list[str] = []
    projects: list[str] = []

    areas_lower = [a.lower() for a in matched_areas]

    # Inspect candidate profile
    profile_projects = profile.get("projects", [])
    profile_exp = profile.get("experience", [])

    def has_project_matching(keywords: tuple[str, ...]) -> dict[str, Any] | None:
        for p in profile_projects:
            name_desc = (p.get("name", "") + " " + p.get("short_description", "") + " " + " ".join(p.get("technologies", []))).lower()
            if any(k.lower() in name_desc for k in keywords):
                return p
        return None

    def has_experience_matching(keywords: tuple[str, ...]) -> dict[str, Any] | None:
        for exp in profile_exp:
            exp_text = (exp.get("organization", "") + " " + exp.get("role", "") + " " + " ".join(exp.get("responsibilities", [])) + " " + " ".join(exp.get("technologies_used", []))).lower()
            if any(k.lower() in exp_text for k in keywords):
                return exp
        return None

    # Check GenAI / LLM / Agent / RAG areas
    if any(any(k in a for k in ("generative", "llm", "agent", "rag", "retrieval", "knowledge")) for a in areas_lower):
        cp_proj = has_project_matching(("controlplane", "langgraph", "rag"))
        rp_exp = has_experience_matching(("red panda", "diffusion"))
        if cp_proj:
            projects.append(cp_proj.get("name", "ControlPlane.ai (AI Governance & RAG Middleware)"))
            skills.extend(["LangGraph", "Hybrid RAG (FAISS + BM25)", "LiteLLM"])
        if rp_exp:
            projects.append(f"{rp_exp.get('organization', 'Red Panda Games')} (Generative AI Diffusion Pipelines)")
            skills.append("Diffusion Pipelines (Flux, SD1.5, ControlNet)")

    # Check Computer Vision
    if any(any(k in a for k in ("vision", "media", "diffusion", "image")) for a in areas_lower):
        rp_exp = has_experience_matching(("red panda", "diffusion", "mesh"))
        if rp_exp:
            skills.extend(["Diffusion Models (Flux, SD1.5, ControlNet)", "3D Mesh Processing"])
            proj_title = f"{rp_exp.get('organization', 'Red Panda Games')} (Generative AI Diffusion Pipelines)"
            if proj_title not in projects:
                projects.append(proj_title)
        cv_proj = has_project_matching(("face", "vision", "image"))
        if cv_proj:
            projects.append(cv_proj.get("name", ""))
            skills.append("Computer Vision")

    # Check NLP
    if any(any(k in a for k in ("nlp", "language", "embedding", "text")) for a in areas_lower):
        ecom_proj = has_project_matching(("price prediction", "deberta", "nlp"))
        if ecom_proj:
            projects.append(ecom_proj.get("name", "E-Commerce Price Prediction (NLP Embeddings)"))
            skills.extend(["DeBERTa-v3 Embeddings", "NLP Feature Engineering"])
        char_proj = has_project_matching(("character", "backstory", "nli"))
        if char_proj:
            projects.append(char_proj.get("name", ""))
        iiser_exp = has_experience_matching(("iiser", "computational linguistics", "pos tagging"))
        if iiser_exp:
            projects.append(f"{iiser_exp.get('organization', 'IISER Bhopal')} (Sequence Models for Text Classification)")
            skills.append("Sequence Modeling (RNN/LSTM/GRU)")

    # Check Infrastructure & Serving / Evaluation
    if any(any(k in a for k in ("infrastructure", "serving", "evaluation", "latency", "gpu")) for a in areas_lower):
        rp_exp = has_experience_matching(("red panda", "gpu", "celery", "redis"))
        if rp_exp:
            skills.extend(["GPU Inference Optimization", "Memory Offloading", "Celery & Redis Microservices"])
            proj_title = f"{rp_exp.get('organization', 'Red Panda Games')} (GPU Inference Optimization)"
            if proj_title not in projects:
                projects.append(proj_title)
        cp_proj = has_project_matching(("controlplane", "langsmith", "auditing"))
        if cp_proj:
            skills.append("LangSmith Auditing")
            proj_title = f"{cp_proj.get('name', 'ControlPlane.ai')} (Parallel Auditing & Evaluation)"
            if proj_title not in projects:
                projects.append(proj_title)

    # Check ML / Predictive Analytics / Data Science
    if any(any(k in a for k in ("predictive", "machine learning", "data science", "analytics")) for a in areas_lower):
        sec_proj = has_project_matching(("security", "phishing", "etl"))
        if sec_proj:
            projects.append(sec_proj.get("name", "Network Security System (Automated ETL & Classification)"))
            skills.extend(["XGBoost", "Scikit-learn", "MLflow & DagsHub", "Docker CI/CD"])
        prof_proj = has_project_matching(("profitability", "lightgbm"))
        if prof_proj:
            projects.append(prof_proj.get("name", ""))

    return list(dict.fromkeys(skills)), list(dict.fromkeys([p for p in projects if p]))


# ---------------------------------------------------------------------------
# Primary drafting entry point
# ---------------------------------------------------------------------------

def draft_outreach_email(
    profile: dict[str, Any],
    recipient_email: str,
    company_name: str,
    contact_name: str = "",
    target_role: str = "",
    job_description: str = "",
    verification_pipeline_result: PipelineResult | None = None,
    mode: WorkflowMode = WorkflowMode.UPLOADED_CV,
    attachment_path: str = "",
    contact_notes: str = "",
    classification: Any = None,
    recipient_resolution: Any = None,
    research_result: Any = None,
    cv_text: str | None = None,
) -> EmailDraft:
    """
    Draft a personalized outreach email adhering strictly to verified profile
    data and research findings.
    """
    candidate_name = _candidate_name(profile)
    availability = _availability(profile)

    # ---- 1. Recipient resolution & generic inbox check ----------------------
    is_generic = False
    recipient_type = "individual"
    if recipient_resolution is not None:
        is_generic = getattr(recipient_resolution, "is_generic", False)
        res_name = getattr(recipient_resolution, "resolved_name", "")
        if not contact_name and res_name:
            contact_name = res_name
        if is_generic:
            recipient_type = "generic_inbox"
        elif contact_name:
            recipient_type = "individual"
        else:
            recipient_type = "unknown"

    # ---- 2. Verification output & specific review flags ---------------------
    case_type = ""
    matched_areas: list[str] = []
    review_flags: list[str] = []
    review_required = False

    if not verification_pipeline_result:
        review_required = True
        review_flags.append("Missing verification result: company verification was not performed.")
    else:
        verif = verification_pipeline_result.verification
        match = verification_pipeline_result.match
        case_type = match.case_type
        matched_areas = match.matched_areas

        # Pass through specific verification flags
        for flag in verif.review_flags:
            if flag not in review_flags:
                review_flags.append(flag)

        if verif.overall_status != "PROCEED":
            review_required = True

        if verif.company_status != "VERIFIED":
            review_required = True
            review_flags.append("Company identity unverified: evidence does not independently verify company identity.")

        if match.relevance_status != RELEVANT:
            review_required = True
            if not matched_areas:
                review_flags.append("Technical relevance unverified: no verified AI/ML technical focus found in external evidence.")
            else:
                review_flags.append("Technical relevance uncertain: company technical focus requires human review.")

        if not verif.evidence_sources:
            review_flags.append("No reliable company-specific evidence retrieved.")

        if case_type != SPECIFIC_INTERNSHIP:
            review_flags.append("Internship availability unknown: drafted as general inquiry without confirmed open listing.")

    # Check for missing company name
    clean_company = company_name.strip() if company_name and company_name != "Unknown Company" else ""
    if not clean_company:
        review_required = True
        review_flags.append("Company name missing: outreach drafted with neutral salutation.")

    # ---- 3. Stage 3 Classification integration ------------------------------
    if classification is not None:
        clf_outcome = getattr(classification, "primary_outcome", "")
        reason_code = getattr(classification, "reason_code", "")
        contradictions = getattr(classification, "contradictions", [])
        for c in contradictions:
            flag = f"Contradiction: {c}"
            if flag not in review_flags:
                review_flags.append(flag)
        if clf_outcome == "NOT_AI_RELEVANT":
            review_required = True
            flag = "Company evaluated as NOT_AI_RELEVANT: personalized technical claims omitted."
            if flag not in review_flags:
                review_flags.append(flag)
        elif clf_outcome in ("REVIEW_NAME", "ELIMINATED", "REVIEW"):
            review_required = True
        if reason_code in ("DOMAIN_MISMATCH", "COMPANY_UNVERIFIED", "RESEARCH_FAILED"):
            review_required = True
            flag = f"Company research warning: {reason_code}."
            if flag not in review_flags:
                review_flags.append(flag)

    if is_generic:
        flag = f"Generic inbox ({recipient_email}): addressed to team without inventing a personal name."
        if flag not in review_flags:
            review_flags.append(flag)

    if not availability:
        review_required = True
        review_flags.append("Candidate availability unknown: duration claim omitted and requires human review.")

    # ---- 4. Context detection -----------------------------------------------
    ctx_result = detect_outreach_context(contact_notes)
    context = ctx_result.context
    referrer_name = ctx_result.referrer_name

    if context == OutreachContext.REFERRAL:
        if not referrer_name:
            review_required = True
            review_flags.append("Referral not verified: referral mentioned in notes but no referrer name confirmed.")
            context = OutreachContext.DIRECT_APPLICATION
    elif ctx_result.review_required:
        review_required = True
        review_flags.append(f"Outreach context ambiguous: {ctx_result.detection_notes}")

    role_mention = target_role or "AI/ML Internship"

    # ---- 4b. Authoritative Candidate Contact Resolution ---------------------
    resolved_contact = resolve_candidate_contact(profile, cv_text)
    if resolved_contact.has_unresolved_conflict:
        review_required = True
        conflict_flag = "Conflicting candidate contact email detected across profile and CV; human review required before sending."
        if conflict_flag not in review_flags:
            review_flags.append(conflict_flag)

    # ---- 5. Company Opportunity Analysis & Experience Matching -------------
    domain_from_email = recipient_email.split("@")[1] if "@" in recipient_email else ""
    opportunity = analyze_company_opportunity(
        research_result=research_result,
        company_name=company_name,
        domain=domain_from_email,
        matched_areas=matched_areas,
    )
    candidate_match = match_candidate_experience(
        opportunity=opportunity,
        profile=profile,
        cv_text=cv_text,
    )

    # If company has no evidence, record uncertainty flag
    if opportunity.evidence_strength == "NONE" and not matched_areas:
        flag = "No reliable company-specific evidence retrieved."
        if flag not in review_flags:
            review_flags.append(flag)

    # ---- 6. Evidence-Grounded Email Generation ------------------------------
    from src.llm.router import LLMRouter
    router = LLMRouter()

    subject, body, gen_meta = generate_personalized_email_body(
        opportunity=opportunity,
        candidate_match=candidate_match,
        profile=profile,
        company_name=company_name,
        contact_name=contact_name,
        target_role=target_role,
        case_type=case_type,
        context=context,
        referrer_name=referrer_name,
        router=router,
        matched_areas=matched_areas,
    )

    draft_id = f"draft-{uuid.uuid4().hex[:8]}"

    # ---- 7. Preliminary Draft for Quality Gate Evaluation -------------------
    prelim_draft = EmailDraft(
        draft_id=draft_id,
        recipient=recipient_email,
        recipient_name=contact_name,
        company_name=company_name,
        subject=subject,
        body=body,
        attachment_path=attachment_path,
        mode=mode,
        review_required=review_required,
        review_flags=review_flags,
    )

    # ---- 8. Personalization Quality Gate ------------------------------------
    gate_res = evaluate_personalization_quality(
        draft=prelim_draft,
        opportunity=opportunity,
        candidate_match=candidate_match,
        profile=profile,
    )

    if not gate_res.passed:
        review_required = True
        for rf in gate_res.review_flags_to_add:
            if rf not in review_flags:
                review_flags.append(rf)

    # ---- 9. Generation Status -----------------------------------------------
    reason_code = getattr(classification, "reason_code", "") if classification else ""
    if reason_code == "INVALID_EMAIL" or not recipient_email or "@" not in recipient_email:
        generation_status = GenerationStatus.GENERATION_FAILED.value
    elif review_required or not gate_res.passed:
        generation_status = GenerationStatus.NEEDS_REVIEW.value
    else:
        generation_status = GenerationStatus.DRAFT_GENERATED.value

    # Determine matched skills and highlighted projects
    legacy_skills, legacy_projects = _determine_cv_connections(matched_areas, profile, cv_text)
    matched_skills = list(dict.fromkeys(candidate_match.relevant_technologies + legacy_skills))
    matched_projects = list(dict.fromkeys([candidate_match.primary_experience_name] + legacy_projects))
    matched_projects = [p for p in matched_projects if p and p != "Applied Machine Learning & Engineering"]

    evidence_summary = getattr(research_result, "ai_ml_evidence_summary", "") if research_result else ""

    # ---- 10. Personalization Record Audit -----------------------------------
    pers_record = PersonalizationRecord(
        company_finding=opportunity.concrete_finding,
        evidence_source_url=opportunity.primary_source_url,
        evidence_source_title=opportunity.primary_source_title,
        source_excerpt=opportunity.source_excerpt,
        candidate_experience=candidate_match.primary_experience_name,
        connection_reasoning=candidate_match.connection_rationale,
        technical_domain=opportunity.technical_domain,
        supporting_cv_evidence=candidate_match.supporting_cv_evidence,
        limitations_or_assumptions=candidate_match.limitations_or_assumptions,
        contact_discrepancies=[d.to_dict() for d in resolved_contact.discrepancies],
        unresolved_caveats=opportunity.uncertainties,
        quality_gate_passed=gate_res.passed,
        quality_gate_checks=gate_res.checks,
        quality_gate_notes=gate_res.failure_reasons,
    )

    if generation_status == GenerationStatus.DRAFT_GENERATED.value:
        target_disp = opportunity.concrete_finding or (', '.join(matched_areas) if matched_areas else "verified work")
        if matched_areas and ', '.join(matched_areas) not in target_disp:
            target_disp = f"{target_disp} ({', '.join(matched_areas)})"
        explanation_parts = [f"Personalized for {clean_company or 'the company'} based on verified focus in {target_disp}."]
        if matched_projects:
            explanation_parts.append(f"Connected to candidate's background in {', '.join(matched_projects)}.")
        explanation_parts.append(f"Addressed to {contact_name or 'the hiring team'}.")
        personalization_explanation = " ".join(explanation_parts)
    elif generation_status == GenerationStatus.NEEDS_REVIEW.value:
        flags_text = "; ".join(review_flags) if review_flags else "Company or recipient evidence requires review"
        personalization_explanation = (
            f"Draft held for human review ({flags_text}). "
            f"Personalized claims restrained to prevent unsupported statements."
        )
    else:
        personalization_explanation = (
            f"Generation failed: {'; '.join(review_flags) or 'Contact cannot be addressed safely'}."
        )

    return EmailDraft(
        draft_id=draft_id,
        recipient=recipient_email,
        recipient_name=contact_name,
        company_name=company_name,
        subject=subject,
        body=body,
        attachment_path=attachment_path,
        mode=mode,
        review_required=review_required,
        review_flags=review_flags,
        generation_status=generation_status,
        personalization_explanation=personalization_explanation,
        evidence_summary=evidence_summary,
        matched_skills=matched_skills,
        matched_projects=matched_projects,
        recipient_type=recipient_type,
        personalization_record=pers_record,
    )

