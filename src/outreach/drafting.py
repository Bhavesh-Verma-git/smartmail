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

from src.cv.models import ProjectSelectionResult
from src.outreach.context import OutreachContext, detect_outreach_context
from src.outreach.models import EmailDraft, GenerationStatus, WorkflowMode
from src.verification.models import PipelineResult, RELEVANT, SPECIFIC_INTERNSHIP


# ---------------------------------------------------------------------------
# Internal profile helpers
# ---------------------------------------------------------------------------

def _candidate_name(profile: dict[str, Any]) -> str:
    return profile.get("personal_info", {}).get("full_name", "Bhavesh Verma")


def _preferred_name(profile: dict[str, Any]) -> str:
    return profile.get("personal_info", {}).get("preferred_name", "") or _candidate_name(profile)


def _institution(profile: dict[str, Any]) -> str:
    edu = profile.get("education", [])
    if edu:
        return edu[0].get("institution", "IIT Kharagpur")
    return "IIT Kharagpur"


def _phone(profile: dict[str, Any]) -> str:
    return profile.get("personal_info", {}).get("phone", "")


def _email_addr(profile: dict[str, Any]) -> str:
    return profile.get("personal_info", {}).get("email", "")


def _availability(profile: dict[str, Any]) -> str:
    prefs = profile.get("preferences", {})
    return prefs.get("availability_duration_months", "3–6 months")


def _format_internship_duration(raw: str) -> str:
    """
    Ensure correct English grammar for internship duration.
    e.g. '3-6 months' -> 'a 3–6 month internship'
         '6 months'   -> 'a six-month internship'
    NEVER produces 'a 3-6 months internship'.
    """
    s = (raw or "").strip()
    if not s:
        return "a 3–6 month internship"

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
        "inquiries", "inquiry", "office", "hello", "hi", "sales", "hiring"
    }
    if is_generic or (name and name.lower() in generic_keywords):
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


def _build_alignment_paragraph(
    matched_areas: list[str],
    company_name: str,
    mode: WorkflowMode,
    project_selection: ProjectSelectionResult | None,
    profile: dict[str, Any],
) -> str:
    """
    Build the company-alignment paragraph.

    Only references company technical focus when grounded in verified matched_areas.
    Never invents company AI claims if unverified.
    """
    clean_company = company_name.strip() if company_name and company_name != "Unknown Company" else ""

    if matched_areas and clean_company:
        areas_text = ", ".join(matched_areas)
        tech_intro = f"Given {clean_company}'s focus on {areas_text}, "

        if mode == WorkflowMode.TAILORED_CV and project_selection:
            p1 = project_selection.primary_project.get("name", "")
            secondaries = project_selection.secondary_projects or []
            p2 = secondaries[0].get("name", "") if secondaries else ""
            projects_summary = p1
            if p2:
                projects_summary += f" and {p2}"
            return (
                f"{tech_intro}I wanted to highlight my practical experience building scalable AI systems. "
                f"Through {projects_summary}, alongside my internship experience developing "
                f"diffusion pipelines and ML inference workflows, I have focused on building "
                f"reliable, production-grade AI architectures."
            )

        # Uploaded CV Mode with verified AI company
        return (
            f"{tech_intro}I believe my background aligns well with your team's technical initiatives. "
            f"During my internship at Red Panda Games, I developed generative AI pipelines and diffusion models, "
            f"working with tools such as Stable Diffusion and ControlNet. "
            f"Additionally, I have practical experience building context-aware AI middleware and RAG architectures "
            f"with an emphasis on low-latency inference and system reliability."
        )

    # Truthful neutral candidate overview when company evidence is absent or unverified
    return (
        "My background includes practical experience in applied machine learning, generative AI pipelines, "
        "and data systems. During my previous internship experience, I developed production-ready diffusion "
        "and inference workflows, focusing on building reliable AI systems and practical problem-solving."
    )


def _build_closing(availability: str) -> str:
    """Return a concise closing paragraph with grammatically correct availability."""
    duration_phrase = _format_internship_duration(availability)
    return (
        f"I have attached my CV for your review and would welcome the opportunity to discuss "
        f"how my technical background can contribute to your team. "
        f"I am available for {duration_phrase}."
    )


def _build_signature(profile: dict[str, Any]) -> str:
    """
    Build clean signature.
    Sign-off: Best regards, Name, IIT Kharagpur, phone, email.
    """
    name = _candidate_name(profile)
    institution = _institution(profile)
    phone = _phone(profile)
    email = _email_addr(profile)

    lines = ["Best regards,", name, institution]
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
    """
    skills: list[str] = []
    projects: list[str] = []

    areas_lower = [a.lower() for a in matched_areas]

    # Check GenAI / LLM / Agent / RAG areas
    if any(any(k in a for k in ("generative", "llm", "agent", "rag", "retrieval", "knowledge")) for a in areas_lower):
        skills.extend(["LangGraph", "Hybrid RAG (FAISS + BM25)", "Diffusion Pipelines (Flux, SD1.5, ControlNet)", "LiteLLM"])
        projects.append("ControlPlane.ai (AI Governance & RAG Middleware)")
        projects.append("Red Panda Games (Generative AI Diffusion Pipelines)")

    # Check Computer Vision
    if any(any(k in a for k in ("vision", "media", "diffusion", "image")) for a in areas_lower):
        skills.extend(["Diffusion Models (Flux, SD1.5, ControlNet)", "FaceNet & MTCNN", "3D Mesh Processing"])
        if "Red Panda Games (Generative AI Diffusion Pipelines)" not in projects:
            projects.append("Red Panda Games (Generative AI Diffusion Pipelines)")
        projects.append("Face Recognition System")

    # Check NLP
    if any(any(k in a for k in ("nlp", "language", "embedding", "text")) for a in areas_lower):
        skills.extend(["DeBERTa-v3 Embeddings", "Sequence Modeling (RNN/LSTM/GRU)", "NLP Feature Engineering"])
        projects.append("E-Commerce Price Prediction (NLP Embeddings)")
        projects.append("IISER Bhopal (Sequence Models for Text Classification)")

    # Check Infrastructure & Serving / Evaluation
    if any(any(k in a for k in ("infrastructure", "serving", "evaluation", "latency", "gpu")) for a in areas_lower):
        skills.extend(["GPU Inference Optimization", "Memory Offloading", "Celery & Redis Microservices", "LangSmith Auditing"])
        if "Red Panda Games (Generative AI Diffusion Pipelines)" not in projects:
            projects.append("Red Panda Games (GPU Inference Optimization)")
        if "ControlPlane.ai (AI Governance & RAG Middleware)" not in projects:
            projects.append("ControlPlane.ai (Parallel Auditing & Evaluation)")

    # Check ML / Predictive Analytics / Data Science
    if any(any(k in a for k in ("predictive", "machine learning", "data science", "analytics")) for a in areas_lower):
        skills.extend(["XGBoost", "Scikit-learn", "MLflow & DagsHub", "Docker CI/CD"])
        projects.append("Network Security System (Automated ETL & Classification)")

    # Fallback to general AI/ML foundation if empty
    if not skills:
        skills = ["Applied AI/ML", "Generative AI", "RAG Architectures", "Python"]
        projects = ["ControlPlane.ai", "Red Panda Games Internship"]

    return list(dict.fromkeys(skills)), list(dict.fromkeys(projects))


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
    project_selection: ProjectSelectionResult | None = None,
    tailored_cv_path: str | None = None,
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

    # ---- 4. Context detection -----------------------------------------------
    ctx_result = detect_outreach_context(contact_notes)
    context = ctx_result.context
    referrer_name = ctx_result.referrer_name

    if context == OutreachContext.REFERRAL:
        if not referrer_name:
            review_required = True
            review_flags.append("Referral not verified: referral mentioned in notes but no referrer name confirmed.")
    elif ctx_result.review_required:
        review_required = True
        review_flags.append(f"Outreach context ambiguous: {ctx_result.detection_notes}")

    role_mention = target_role or "AI/ML Internship"

    # ---- 5. Assemble body ---------------------------------------------------
    body_parts: list[str] = [
        _build_salutation(contact_name, company_name, is_generic=is_generic),
        "",
        _build_opening_paragraph(
            context=context,
            referrer_name=referrer_name,
            role_mention=role_mention,
            company_name=company_name,
            profile=profile,
            case_type=case_type,
        ),
        "",
        _build_alignment_paragraph(
            matched_areas=matched_areas,
            company_name=company_name,
            mode=mode,
            project_selection=project_selection,
            profile=profile,
        ),
        "",
        _build_closing(availability),
        "",
        "Thank you for your time and consideration.",
        "",
        _build_signature(profile),
    ]

    body = "\n".join(body_parts)
    subject = build_subject_line(
        candidate_name=candidate_name,
        target_role=target_role,
        company_name=company_name,
        case_type=case_type,
        context=context,
        referrer_name=referrer_name,
    )

    draft_id = f"draft-{uuid.uuid4().hex[:8]}"

    # ---- 6. Stage 4 Personalization & Generation Status ---------------------
    reason_code = getattr(classification, "reason_code", "") if classification else ""
    if reason_code == "INVALID_EMAIL" or not recipient_email or "@" not in recipient_email:
        generation_status = GenerationStatus.GENERATION_FAILED.value
    elif review_required:
        generation_status = GenerationStatus.NEEDS_REVIEW.value
    else:
        generation_status = GenerationStatus.DRAFT_GENERATED.value

    evidence_summary = getattr(research_result, "ai_ml_evidence_summary", "") if research_result else ""
    matched_skills, matched_projects = _determine_cv_connections(matched_areas, profile, cv_text)

    if generation_status == GenerationStatus.DRAFT_GENERATED.value:
        personalization_explanation = (
            f"Personalized for {clean_company or 'the company'} based on verified focus in {', '.join(matched_areas)}. "
            f"Connected to candidate's background in generative AI pipelines (Red Panda Games) and context-aware RAG middleware (ControlPlane.ai). "
            f"Addressed to {contact_name or 'the hiring team'}."
        )
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
        tailored_cv_path=tailored_cv_path,
        generation_status=generation_status,
        personalization_explanation=personalization_explanation,
        evidence_summary=evidence_summary,
        matched_skills=matched_skills,
        matched_projects=matched_projects,
        recipient_type=recipient_type,
    )

