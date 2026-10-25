"""
src/outreach/personalized_generator.py
--------------------------------------
Stage D: Evidence-grounded outreach email generation.

Generates genuinely company-specific outreach emails using:
1. LLMRouter (Gemini with Groq fallback) when configured, adhering strictly to
   grounded evidence prompts.
2. Grounded Deterministic Composer when LLM is offline or unconfigured, ensuring
   high-quality, varied, company-specific outreach without generic boilerplate.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.llm.router import LLMRouter
from src.outreach.context import OutreachContext
from src.outreach.experience_matcher import (
    CandidateExperienceMatch,
    DOMAIN_GOVERNANCE_SAFETY,
    DOMAIN_RAG_AGENTS,
    DOMAIN_DIFFUSION_VISION,
    DOMAIN_NLP_LINGUISTICS,
    DOMAIN_ML_DATA_SYSTEMS,
)
from src.outreach.opportunity_analyzer import CompanyOpportunity, sanitize_evidence_text
from src.verification.models import SPECIFIC_INTERNSHIP

logger = logging.getLogger(__name__)


def strip_markdown_formatting(text: str) -> str:
    """
    Remove literal Markdown formatting such as **bold**, *italic*, __bold__, _italic_, `code`
    to ensure generated emails are stored and rendered as clean, natural text.
    """
    if not text:
        return ""
    # Strip bold markdown (**text** or __text__)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    # Strip italic markdown (*text* or _text_)
    cleaned = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", cleaned)
    # Strip inline backtick code snippets (`code`)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    return cleaned


def generate_personalized_email_body(
    opportunity: CompanyOpportunity,
    candidate_match: CandidateExperienceMatch,
    profile: dict[str, Any],
    company_name: str,
    contact_name: str = "",
    target_role: str = "",
    case_type: str = "",
    context: OutreachContext = OutreachContext.DIRECT_APPLICATION,
    referrer_name: str = "",
    router: LLMRouter | None = None,
    matched_areas: list[str] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """
    Generate tailored email subject, body, and metadata grounded in verified opportunity
    and candidate experience.

    Returns
    -------
    (subject, body, generation_metadata)
    """
    clean_company = company_name.strip() if company_name and company_name != "Unknown Company" else ""
    role_mention = target_role or "AI/ML Internship"
    edu_inst = profile.get("education", [{}])[0].get("institution", "IIT Kharagpur") if profile.get("education") else "IIT Kharagpur"

    # Availability formatting
    prefs = profile.get("preferences", {})
    val = prefs.get("availability_duration_months")
    clean_avail = str(val).strip() if val else ""
    from src.outreach.drafting import _format_internship_duration, _build_salutation, _build_closing, _build_signature, build_subject_line
    duration_phrase = _format_internship_duration(clean_avail) if clean_avail else ""

    subject = build_subject_line(
        candidate_name=profile.get("personal_info", {}).get("full_name", "Bhavesh Verma"),
        target_role=target_role,
        company_name=company_name,
        case_type=case_type,
        context=context,
        referrer_name=referrer_name,
    )

    # 1. Try LLM Generation if configured and company has substantive evidence for direct applications
    if (
        router
        and router.has_any_llm
        and opportunity.evidence_strength in ("STRONG", "MODERATE")
        and context == OutreachContext.DIRECT_APPLICATION
    ):
        try:
            llm_result = _generate_with_llm(
                router=router,
                opportunity=opportunity,
                candidate_match=candidate_match,
                profile=profile,
                clean_company=clean_company,
                contact_name=contact_name,
                role_mention=role_mention,
                context=context,
                referrer_name=referrer_name,
                duration_phrase=duration_phrase,
            )
            if llm_result:
                return (
                    strip_markdown_formatting(llm_result[0]),
                    strip_markdown_formatting(llm_result[1]),
                    {"method": "llm", "provider": router.primary.provider_name},
                )
        except Exception as exc:
            logger.warning(f"LLM draft generation failed: {exc}. Using grounded deterministic composer.")

    # 2. Grounded Deterministic Composer
    body = _compose_grounded_email(
        opportunity=opportunity,
        candidate_match=candidate_match,
        profile=profile,
        clean_company=clean_company,
        contact_name=contact_name,
        role_mention=role_mention,
        case_type=case_type,
        context=context,
        referrer_name=referrer_name,
        duration_phrase=duration_phrase,
        matched_areas=matched_areas,
    )

    return strip_markdown_formatting(subject), strip_markdown_formatting(body), {"method": "grounded_deterministic"}


def _compose_grounded_email(
    opportunity: CompanyOpportunity,
    candidate_match: CandidateExperienceMatch,
    profile: dict[str, Any],
    clean_company: str,
    contact_name: str,
    role_mention: str,
    case_type: str,
    context: OutreachContext,
    referrer_name: str,
    duration_phrase: str,
    matched_areas: list[str] | None = None,
) -> str:
    """Compose natural, company-specific email body without universal templates."""
    from src.outreach.drafting import _build_salutation, _build_signature

    salutation = _build_salutation(contact_name, clean_company)
    edu_inst = profile.get("education", [{}])[0].get("institution", "IIT Kharagpur") if profile.get("education") else "IIT Kharagpur"

    # Opening Paragraph
    at_company = f" at {clean_company}" if clean_company else ""

    if context == OutreachContext.LINKEDIN_FOLLOW_UP:
        opening = (
            f"Following our recent conversation on LinkedIn, I am writing to formally express my interest "
            f"in {role_mention} opportunities{at_company}. "
            f"I am a fourth-year Dual Degree student at {edu_inst} with a focus on Applied AI and Machine Learning."
        )
    elif context == OutreachContext.REFERRAL and referrer_name:
        opening = (
            f"I am writing to apply for an {role_mention}{at_company}, on the recommendation of {referrer_name}, "
            f"who suggested I reach out to your team. "
            f"I am a fourth-year Dual Degree student at {edu_inst} with a focus on Applied AI and Machine Learning."
        )
    elif case_type == SPECIFIC_INTERNSHIP:
        opening = (
            f"I am writing to express my strong interest in the {role_mention} role{at_company}. "
            f"I am a fourth-year Dual Degree student at {edu_inst} with a focus on Applied AI and Machine Learning."
        )
    elif opportunity.concrete_finding and opportunity.evidence_strength in ("STRONG", "MODERATE") and clean_company:
        seed = sum(ord(c) for c in clean_company) % 3
        if seed == 1:
            opening = (
                f"I am reaching out to inquire whether your team at {clean_company} has an opening for an {role_mention}. "
                f"I am currently a fourth-year Dual Degree student at {edu_inst} focusing on Applied AI and Machine Learning."
            )
        elif seed == 2:
            opening = (
                f"I am writing to explore potential {role_mention} opportunities with {clean_company}. "
                f"I am a fourth-year Dual Degree student at {edu_inst} with practical experience in Applied AI and Machine Learning."
            )
        else:
            opening = (
                f"I wanted to reach out regarding potential {role_mention} roles at {clean_company}. "
                f"I am a fourth-year Dual Degree student at {edu_inst} with a strong foundation in Applied AI and Machine Learning."
            )
    else:
        opening = (
            f"I am writing to inquire about potential {role_mention} opportunities{at_company}. "
            f"I am a fourth-year Dual Degree student at {edu_inst} with a strong foundation in Applied AI and Machine Learning."
        )

    # Substantive Alignment Paragraph
    # Uploaded-CV mode with STRONG/MODERATE company evidence
    # Build alignment using the company-specific concrete_finding and the
    # candidate's domain-matched concrete_candidate_work.
    # We deliberately do NOT use candidate_match.connection_rationale here
    # because that field is still domain-keyed (same phrasing for all
    # DOMAIN_RAG_AGENTS companies, for example).  Instead we anchor the
    # sentence to opportunity.concrete_finding, which IS company-specific.
    if opportunity.concrete_finding and opportunity.evidence_strength in ("STRONG", "MODERATE") and clean_company:
        work = candidate_match.concrete_candidate_work
        finding = sanitize_evidence_text(opportunity.concrete_finding)

        # Determine a concise lead-in verb phrase from the finding text.
        # If the finding already starts with an action verb (e.g. "engineering...",
        # "developing..."), prefix with "on"; otherwise use "in".
        finding_lower = finding.lower().strip()
        action_starters = (
            "engineering", "developing", "building", "designing", "deploying",
            "architecting", "creating", "delivering", "advancing", "powering",
        )
        if any(finding_lower.startswith(v) for v in action_starters):
            company_clause = f"Given {clean_company}'s work {finding}"
        else:
            company_clause = f"Given {clean_company}'s focus on {finding}"

        alignment = (
            f"{company_clause}, I believe my experience is directly relevant. "
            f"I {work}."
        )

    # Backward-compatible fallback when only matched_areas is available (legacy unit tests)
    elif matched_areas and clean_company:
        areas_text = ", ".join(matched_areas)
        tech_intro = f"Given {clean_company}'s focus on {areas_text}, "

        areas_lower = [a.lower() for a in matched_areas]
        has_genai = any(any(k in a for k in ("generative", "diffusion", "llm")) for a in areas_lower)
        has_rag = any(any(k in a for k in ("rag", "retrieval", "agent")) for a in areas_lower)
        has_vision = any(any(k in a for k in ("vision", "media", "image")) for a in areas_lower)
        has_nlp = any(any(k in a for k in ("nlp", "language", "embedding", "text")) for a in areas_lower)
        has_ml = any(any(k in a for k in ("predictive", "machine learning", "data science", "analytics")) for a in areas_lower)

        profile_exp = profile.get("experience", [])
        profile_projects = profile.get("projects", [])
        has_red_panda = any("red panda" in exp.get("organization", "").lower() for exp in profile_exp)
        has_controlplane = any("controlplane" in p.get("name", "").lower() for p in profile_projects)
        has_nlp_in_profile = any("nlp" in str(exp).lower() or "iiser" in exp.get("organization", "").lower() for exp in profile_exp)
        has_mlops_in_profile = any("security" in str(p).lower() or "etl" in str(p).lower() for p in profile_projects) or bool(profile_exp)

        highlight_sentences = []
        if (has_genai or has_vision or has_rag) and (has_red_panda or has_controlplane):
            if has_red_panda and (has_genai or has_vision):
                highlight_sentences.append(
                    "During my internship at Red Panda Games, I developed generative AI pipelines and diffusion models, "
                    "working with tools such as Stable Diffusion and ControlNet."
                )
            if has_controlplane and (has_rag or not has_red_panda):
                prefix = "Additionally, I" if highlight_sentences else "I"
                highlight_sentences.append(
                    f"{prefix} have practical experience building context-aware AI middleware and RAG architectures "
                    "with an emphasis on low-latency inference and system reliability."
                )
        elif has_nlp and has_nlp_in_profile:
            highlight_sentences.append(
                "I have practical experience developing NLP pipelines, language embeddings with DeBERTa-v3, "
                "and sequence models for text classification and attribute extraction."
            )
        elif has_ml and has_mlops_in_profile:
            highlight_sentences.append(
                "I have practical experience developing machine learning workflows, automated ETL pipelines, "
                "and classification systems using Scikit-learn, XGBoost, and containerized architectures."
            )
        else:
            highlight_sentences.append(
                "I believe my background in applied machine learning, neural architectures, "
                "and production pipelines aligns well with your team's technical initiatives."
            )

        highlights_text = " ".join(highlight_sentences)
        alignment = f"{tech_intro}I believe my background aligns well with your team's technical initiatives. {highlights_text}"

    # Truthful neutral candidate overview when company evidence is absent or unverified
    else:
        alignment = (
            "My background includes practical experience in applied machine learning, data systems, "
            "and software engineering. During my previous internship experience, I worked on production ML workflows "
            "and data pipelines, focusing on building reliable systems and practical problem-solving."
        )

    # Closing Paragraph
    from src.outreach.drafting import _build_closing
    closing = _build_closing(duration_phrase, seed_key=clean_company)

    signature = _build_signature(profile)

    body_parts = [
        salutation,
        "",
        opening,
        "",
        alignment,
        "",
        closing,
        "",
        "Thank you for your time and consideration.",
        "",
        signature,
    ]

    return "\n".join(body_parts)


def _generate_with_llm(
    router: LLMRouter,
    opportunity: CompanyOpportunity,
    candidate_match: CandidateExperienceMatch,
    profile: dict[str, Any],
    clean_company: str,
    contact_name: str,
    role_mention: str,
    context: OutreachContext,
    referrer_name: str,
    duration_phrase: str,
) -> tuple[str, str] | None:
    """Invoke LLM to generate structured outreach email with grounded constraints."""
    from src.outreach.drafting import _build_salutation, _build_signature, build_subject_line

    salutation = _build_salutation(contact_name, clean_company)
    signature = _build_signature(profile)
    edu_inst = profile.get("education", [{}])[0].get("institution", "IIT Kharagpur") if profile.get("education") else "IIT Kharagpur"

    subject = build_subject_line(
        candidate_name=profile.get("personal_info", {}).get("full_name", "Bhavesh Verma"),
        target_role=role_mention,
        company_name=clean_company,
        context=context,
        referrer_name=referrer_name,
    )

    system_instruction = (
        "You are an expert technical assistant drafting a concise, highly tailored cold outreach email for an "
        "engineering internship inquiry. The candidate is a 4th-year Dual Degree student at IIT Kharagpur.\n"
        "STRICT GROUNDING CONSTRAINTS:\n"
        "1. Mention ONLY the company technical focus provided in the prompt. Never invent products or initiatives.\n"
        "2. Connect ONLY the specified candidate experience provided in the prompt. Never invent past employers or tools.\n"
        "3. Do NOT use boilerplate buzzword dumps like 'focus on Generative AI, Agentic AI, AI/ML, Data Science, Software Engineering'.\n"
        "4. Tone: direct, humble, engineering-focused, natural, concise (under 200 words total).\n"
        "5. Output valid JSON with keys: 'opening_paragraph', 'alignment_paragraph', 'closing_paragraph'."
    )

    prompt = (
        f"Company Name: {clean_company}\n"
        f"Company Technical Focus: {opportunity.concrete_finding}\n"
        f"Candidate Education: 4th-year Dual Degree student at {edu_inst}\n"
        f"Selected Candidate Experience: {candidate_match.organization_or_project}\n"
        f"Candidate Concrete Work: {candidate_match.concrete_candidate_work}\n"
        f"Connection Rationale: {candidate_match.connection_rationale}\n"
        f"Availability: {duration_phrase}\n"
        f"Target Role: {role_mention}\n\n"
        "Write the opening, alignment, and closing paragraphs for the outreach email. "
        "Return a valid JSON object with keys: 'opening_paragraph', 'alignment_paragraph', 'closing_paragraph'."
    )

    resp = router.generate_json_with_fallback(prompt=prompt, system_instruction=system_instruction)
    if resp.success and resp.parsed_json:
        pj = resp.parsed_json
        opening = sanitize_evidence_text(pj.get("opening_paragraph", "").strip())
        alignment = sanitize_evidence_text(pj.get("alignment_paragraph", "").strip())
        closing = sanitize_evidence_text(pj.get("closing_paragraph", "").strip())

        # Clean any accidental leading salutation in opening paragraph
        opening = re.sub(r"^(Dear|Hi|Hello)\s+[^,\n]+,\s*", "", opening, flags=re.IGNORECASE).strip()
        # Clean any accidental sign-off in closing paragraph
        closing = re.sub(r"(Best regards|Sincerely|Thanks|Thank you for your time).*$", "", closing, flags=re.IGNORECASE).strip()

        # Replace any candidate placeholders
        cand_name = profile.get("personal_info", {}).get("preferred_name", "") or profile.get("personal_info", {}).get("full_name", "Bhavesh")
        for ph in ("[Name]", "[Candidate Name]", "[Your Name]", "[Candidate]", "[My Name]"):
            opening = opening.replace(ph, cand_name)
            alignment = alignment.replace(ph, cand_name)
            closing = closing.replace(ph, cand_name)

        if opening and alignment and closing:
            closing_parts = [closing]
            if not any(k in closing.lower() for k in ("thank you", "thanks", "consideration")):
                closing_parts.extend(["", "Thank you for your time and consideration."])

            body_parts = [
                salutation,
                "",
                opening,
                "",
                alignment,
                "",
                *closing_parts,
                "",
                signature,
            ]
            return strip_markdown_formatting(subject), strip_markdown_formatting("\n".join(body_parts))

    return None
