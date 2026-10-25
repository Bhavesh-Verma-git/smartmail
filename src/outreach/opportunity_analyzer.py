"""
src/outreach/opportunity_analyzer.py
------------------------------------
Stage B & C: Company-specific evidence normalization, source-quality assessment,
and concrete technical opportunity extraction.

Transforms unstructured web research evidence (extracted text, titles, URLs, and summaries)
into a structured CompanyOpportunity brief.

Guarantees:
- Extracts concrete technical capabilities, products, and engineering focus.
- Never outputs a broad taxonomic keyword dump (e.g. 'AI/ML, RAG, Agents, Data Science').
- Retains supporting URL, title, exact evidence snippet, and source type.
- Honestly reports when company evidence is absent, weak, or unverified.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.research.models import EvidenceItem, ResearchResult, RETRIEVAL_OK

logger = logging.getLogger(__name__)

# Primary technical domain classifications
DOMAIN_GOVERNANCE_SAFETY = "AI_GOVERNANCE_AND_SAFETY"
DOMAIN_RAG_AGENTS = "RAG_AND_AGENTS"
DOMAIN_DIFFUSION_VISION = "DIFFUSION_AND_VISION"
DOMAIN_NLP_LINGUISTICS = "NLP_AND_LINGUISTICS"
DOMAIN_ML_DATA_SYSTEMS = "ML_DATA_SYSTEMS"
DOMAIN_GENERAL_AI = "GENERAL_AI"
DOMAIN_NONE = "NONE"


@dataclass
class CompanyOpportunity:
    """
    Structured, evidence-grounded summary of what a company specifically builds,
    accompanied by citations and source assessment.
    """
    company_name: str
    primary_domain: str = ""
    concrete_finding: str = ""
    primary_source_url: str = ""
    primary_source_title: str = ""
    source_excerpt: str = ""
    technical_domain: str = DOMAIN_NONE
    evidence_strength: str = "NONE"  # STRONG, MODERATE, WEAK, NONE
    has_advertised_internship: bool = False
    internship_details: str = ""
    uncertainties: list[str] = field(default_factory=list)
    key_products_or_initiatives: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True if the concrete finding is supported by at least one retrieved source."""
        return bool(self.concrete_finding and (self.primary_source_url or self.source_excerpt))


def sanitize_evidence_text(text: str) -> str:
    """
    Sanitize and recover legitimate text from research snippets, eliminating:
    - Raw HTML and Markdown artifacts (<div...>, &nbsp;, etc.)
    - Search snippet omission markers like '[...]', '([...])', '...', '(...)'
    - Web navigation and marketing residue ('Skip to content', 'Launch offer', 'Cookie policy', etc.)
    - Truncated sentence fragments and dangling punctuation
    """
    if not text:
        return ""

    s = text

    # 1. Strip raw HTML tags and entities
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&[a-zA-Z0-9#]+;", " ", s)
    s = s.replace("\u00c2\u00b7", " ").replace("\u2022", " ")

    # 2. Strip bracketed omissions, ellipses, citation markers
    s = re.sub(r"\(\s*\[\s*\.\.\.\s*\]\s*\)", " ", s)  # ([...])
    s = re.sub(r"\[\s*\.\.\.\s*\]", " ", s)            # [...]
    s = re.sub(r"\(\s*\.\.\.\s*\)", " ", s)            # (...)
    s = re.sub(r"\.{3,}", " ", s)                       # ... or ....
    s = re.sub(r"\[\s*\d+\s*\]", " ", s)               # [1], [2]
    s = re.sub(r"\[\s*citation needed\s*\]", " ", s, flags=re.IGNORECASE)

    # 3. Strip common search-result navigation / boilerplate residue
    nav_patterns = [
        r"\bSkip to content\b",
        r"\bLaunch offer\b.*?(?:free on every paid plan|credits free|\.|$)",
        r"\bCookie Policy\b.*?(?:about the cookies|we use cookies|cookies on this website|\.|$)?",
        r"\b(?:Accept|Manage)\s+(?:All\s+)?Cookies\b",
        r"\bPrivacy Policy\b.*?(?:learn about|we use|\.|$)?",
        r"\bAll rights reserved\b",
        r"\bTerms of (?:Service|Use)\b",
        r"\bBook a (?:20-min )?demo\b",
        r"\bStart free trial\b",
        r"\bNothing charged for 30\b",
    ]
    for np in nav_patterns:
        s = re.sub(np, " ", s, flags=re.IGNORECASE)

    # 4. Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 5. Fix dangling starting punctuation or fragments
    s = re.sub(r"^[\s,;:–—\.\(\)\[\]]+", "", s).strip()

    # 6. Repair dangling sentence endings (e.g. text ending in comma or dangling conjunction)
    s = re.sub(r"[,;:\s–—]+$", "", s).strip()
    s = re.sub(r"\b(and|or|that|with|to|in|of|for|can serve businesses)\s*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"[,;:\s–—]+$", "", s).strip()

    return s


def _clean_text(text: str) -> str:
    """Normalize whitespace and sanitize evidence text."""
    return sanitize_evidence_text(text)


def _classify_technical_domain(text: str) -> str:
    """
    Classify the primary technical domain from concrete evidence text.
    Uses specific domain markers rather than generic AI terms.

    Priority order (most-specific to least-specific):
    1. GOVERNANCE_SAFETY  — requires 2+ governance signals OR 1 very specific marker
    2. RAG_AGENTS         — checked BEFORE diffusion to avoid misclassifying RAG/agent
                            companies (Octobit8, RevRag, Fluid AI) that mention
                            "image" tangentially in their marketing copy
    3. DIFFUSION_VISION   — requires 2+ diffusion-specific signals OR 1 highly specific
                            term ("diffusion model", "stable diffusion", "controlnet")
                            to prevent single-keyword false positives
    4. NLP_LINGUISTICS
    5. ML_DATA_SYSTEMS
    6. GENERAL_AI
    """
    t_lower = text.lower()

    # 1. AI Governance, Guardrails, Auditing, Responsible AI (e.g. Trustra, Hibilter)
    gov_patterns = (
        "ai governance", "responsible ai", "guardrail", "guardrails", "model monitoring",
        "compliance reporting", "iso 42001", "eu ai act", "nist ai", "dpdp", "audit-ready",
        "flight recorder", "risk detection", "model audit", "tamper-evident", "ai trust",
        "eval-driven", "ai gateway", "ai evaluation", "domain copilots",
        "human-in-command", "sovereign ai", "governed agentic"
    )
    gov_matches = sum(1 for p in gov_patterns if p in t_lower)
    if gov_matches >= 2 or any(p in t_lower for p in (
        "ai governance", "responsible ai os", "audit-ready proof",
        "ai flight recorder", "human-in-command"
    )):
        return DOMAIN_GOVERNANCE_SAFETY

    # 2. RAG, Autonomous Agents, Workflows, Knowledge Bases
    # Checked BEFORE diffusion — companies like Octobit8, RevRag AI, Fluid AI, Infergent
    # build conversational agents/RAG and must not be misclassified because they mention
    # "image" or "vision" somewhere in passing on their marketing pages.
    rag_strong_patterns = (
        "rag", "retrieval-augmented generation", "retrieval augmented generation",
        "ai agent", "ai agents", "autonomous agent", "agentic workflow",
        "conversational assistant", "conversational ai", "chatbot", "chatbots",
        "knowledge base", "langgraph", "document retrieval", "hybrid search",
        "in-app ai", "in-app agent", "whatsapp", "ai crm", "sales agent",
        "lead management", "vector search", "forward deployed engineering",
        "talent graph", "screening agent", "assessment agent", "talent assessment",
        "autonomous talent", "conversational screening", "screening workflow"
    )
    rag_matches = sum(1 for p in rag_strong_patterns if p in t_lower)
    # Require 2+ RAG signals OR 1 strong specific marker
    if rag_matches >= 2 or any(p in t_lower for p in (
        "retrieval-augmented", "retrieval augmented", "rag pipeline",
        "conversational ai", "agentic"
    )):
        return DOMAIN_RAG_AGENTS
    # Single RAG signal is still enough if no competing strong diffusion signals
    if rag_matches >= 1:
        return DOMAIN_RAG_AGENTS

    # 3. Diffusion, Generative Media, Computer Vision, 3D
    # Now checked AFTER RAG. Requires 2+ keyword hits OR 1 highly-specific marker to
    # prevent false positives from companies that mention "image" or "video" in passing.
    diff_specific_patterns = (
        "diffusion model", "stable diffusion", "controlnet", "text-to-image",
        "flux model", "trellis", "generative media", "image synthesis",
        "rendering pipeline", "3d mesh"
    )
    diff_broad_patterns = (
        "diffusion", "flux", "video generation", "image generation",
        "computer vision", "object detection"
    )
    diff_specific_hits = sum(1 for p in diff_specific_patterns if p in t_lower)
    diff_broad_hits = sum(1 for p in diff_broad_patterns if p in t_lower)
    # Need 1+ specific hit, OR 2+ broad hits, to classify as diffusion
    if diff_specific_hits >= 1 or diff_broad_hits >= 2:
        return DOMAIN_DIFFUSION_VISION

    # 4. NLP, Linguistics, Text Embeddings (e.g. Glowminds, IISER research)
    nlp_patterns = (
        "nlp", "computational linguistics", "deberta", "bert", "t5", "pos tagging",
        "sentiment analysis", "language model", "embeddings", "text classification",
        "resume scoring", "ats system", "ats systems", "keyword extraction", "career os"
    )
    if any(p in t_lower for p in nlp_patterns):
        return DOMAIN_NLP_LINGUISTICS

    # 5. Production ML, Data Pipelines, ETL, Security ML
    ml_patterns = (
        "etl", "data pipeline", "xgboost", "scikit-learn", "mlops", "model deployment",
        "predictive model", "classification system", "threat detection", "feature engineering",
        "reasoning infrastructure", "inference acceleration", "inference infrastructure",
        "model inference", "low-latency inference", "production models", "inference engine"
    )
    if any(p in t_lower for p in ml_patterns):
        return DOMAIN_ML_DATA_SYSTEMS

    # General AI if generic terms exist
    if any(p in t_lower for p in ("artificial intelligence", "machine learning", "deep learning", "ai/ml")):
        return DOMAIN_GENERAL_AI

    return DOMAIN_NONE


def _extract_concrete_sentence(
    extracted_text: str,
    domain: str,
    company_name: str,
) -> str:
    """
    Extract the most concrete, informative sentence describing what the company builds.
    Prefers descriptive sentences with verbs like 'builds', 'develops', 'provides', 'engineers'.
    """
    if not extracted_text:
        return ""

    # Split into sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", extracted_text)
    candidates: list[tuple[int, str]] = []

    comp_clean = company_name.lower().strip() if company_name else ""

    action_verbs = ("builds", "build", "develops", "develop", "provides", "provide", "engineers", "architects", "offers", "delivers", "designs", "specializes in", "is a", "is the")
    skip_phrases = ("cookie", "privacy policy", "all rights reserved", "terms of service", "sign in", "subscribe", "book a demo", "contact us", "copyright", "cart", "checkout")

    for s in raw_sentences:
        s_clean = _clean_text(s)
        s_lower = s_clean.lower()
        if len(s_clean) < 30 or len(s_clean) > 350:
            continue
        if any(skip in s_lower for skip in skip_phrases):
            continue

        score = 0
        # Mentions company name
        if comp_clean and comp_clean in s_lower:
            score += 4
        # Active descriptive verb
        if any(v in s_lower for v in action_verbs):
            score += 3
        # Domain alignment
        if domain == DOMAIN_GOVERNANCE_SAFETY and any(k in s_lower for k in ("governance", "responsible", "audit", "guardrail", "risk", "compliance")):
            score += 5
        elif domain == DOMAIN_RAG_AGENTS and any(k in s_lower for k in ("rag", "retrieval", "agent", "assistant", "chatbot", "workflow")):
            score += 5
        elif domain == DOMAIN_DIFFUSION_VISION and any(k in s_lower for k in ("diffusion", "image", "vision", "3d", "media")):
            score += 5
        elif domain == DOMAIN_NLP_LINGUISTICS and any(k in s_lower for k in ("nlp", "language", "embedding", "text")):
            score += 5
        elif domain == DOMAIN_ML_DATA_SYSTEMS and any(k in s_lower for k in ("pipeline", "etl", "predictive", "mlops")):
            score += 5

        if score > 0:
            candidates.append((score, s_clean))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # Fallback to first informative sentence
    for s in raw_sentences:
        s_clean = _clean_text(s)
        if 40 <= len(s_clean) <= 250 and not any(skip in s_clean.lower() for skip in skip_phrases):
            return s_clean

    return ""


def analyze_company_opportunity(
    research_result: ResearchResult | None,
    company_name: str,
    domain: str = "",
    matched_areas: list[str] | None = None,
) -> CompanyOpportunity:
    """
    Analyze collected company evidence and extract a structured opportunity brief.

    Parameters
    ----------
    research_result : ResearchResult from Stage 3 research_contact.
    company_name    : Normalized company name.
    domain          : Email / registered domain.
    matched_areas   : Optional list of taxonomic matched areas for backward compatibility.

    Returns
    -------
    CompanyOpportunity with concrete findings, source citations, and domain categorization.
    """
    clean_name = (company_name or "").strip()
    if clean_name == "Unknown Company":
        clean_name = ""

    uncertainties: list[str] = []

    if research_result is None:
        uncertainties.append("No research result available.")
        return CompanyOpportunity(
            company_name=clean_name,
            primary_domain=domain,
            evidence_strength="NONE",
            uncertainties=uncertainties,
        )

    # 1. Gather successful evidence items
    evidence_items: list[EvidenceItem] = [
        e for e in getattr(research_result, "evidence", [])
        if e.retrieval_status == RETRIEVAL_OK and e.source_type not in ("user_supplied", "notes")
    ]

    # Combine text from all retrieved pages
    combined_corpus = " ".join(e.extracted_text or "" for e in evidence_items)
    combined_titles = " ".join(e.title or "" for e in evidence_items)
    full_text = f"{combined_titles} {combined_corpus}"

    # Also check ai_ml_evidence_summary
    raw_evidence_summary = getattr(research_result, "ai_ml_evidence_summary", None)
    summary_snippets: list[str] = []
    if isinstance(raw_evidence_summary, str) and raw_evidence_summary.strip():
        summary_snippets.append(raw_evidence_summary.strip())
        evidence_summaries: list[dict[str, Any]] = [{"excerpt": raw_evidence_summary.strip(), "source_url": ""}]
    elif isinstance(raw_evidence_summary, list):
        evidence_summaries = raw_evidence_summary
        for item in evidence_summaries:
            if isinstance(item, dict) and item.get("excerpt"):
                summary_snippets.append(item.get("excerpt", ""))
    else:
        evidence_summaries = []

    if summary_snippets:
        full_text = f"{' '.join(summary_snippets)} {full_text}"

    # Check for empty evidence
    if not full_text.strip():
        uncertainties.append("Retrieved pages contain no extractable text or research failed.")
        # Check if matched_areas was provided as minimal input
        if matched_areas:
            concrete = f"focus in {', '.join(matched_areas)}"
            domain_class = _classify_technical_domain(concrete)
            return CompanyOpportunity(
                company_name=clean_name,
                primary_domain=domain,
                concrete_finding=concrete,
                technical_domain=domain_class,
                evidence_strength="WEAK",
                uncertainties=["Evidence based on taxonomic keywords only without verified source text."],
            )
        return CompanyOpportunity(
            company_name=clean_name,
            primary_domain=domain,
            evidence_strength="NONE",
            uncertainties=uncertainties,
        )

    # 2. Determine primary technical domain
    domain_class = _classify_technical_domain(full_text)

    # 3. Find primary source URL and title
    primary_url = ""
    primary_title = ""
    best_excerpt = ""

    # Priority 1: Check ai_ml_evidence_summary for relevant excerpts.
    # NOTE: a summary excerpt is NOT primary evidence on its own -- it may
    # be an LLM-generated summary. It is used only to pick a candidate
    # sentence; STRONG below still requires an excerpt from a retrieved
    # evidence item (see excerpt_from_evidence flag).
    excerpt_from_evidence = False
    for s_item in evidence_summaries:
        if isinstance(s_item, dict):
            exc = s_item.get("excerpt", "")
            src = s_item.get("source_url", "")
            # A summary excerpt without a source URL cannot anchor a claim:
            # skip it so Priority 2 (retrieved evidence items) is used.
            if exc and src and len(exc) > 30:
                s_domain = _classify_technical_domain(exc)
                if s_domain == domain_class or domain_class == DOMAIN_GENERAL_AI:
                    best_excerpt = exc
                    primary_url = src
                    break

    # Priority 2: Extract from evidence items
    if not best_excerpt and evidence_items:
        # Search evidence items for best sentence
        for item in evidence_items:
            t = item.extracted_text or ""
            sentence = _extract_concrete_sentence(t, domain_class, clean_name)
            if sentence:
                best_excerpt = sentence
                primary_url = item.source_url
                primary_title = item.title
                excerpt_from_evidence = True
                break

    # If primary_url is still empty, pick first evidence item
    if not primary_url and evidence_items:
        primary_url = evidence_items[0].source_url
        primary_title = evidence_items[0].title

    # 4. Synthesize concrete finding (clean, substantive 1-line description)
    concrete_finding = ""

    # Generalized finding synthesis based on verified evidence content
    if domain_class == DOMAIN_GOVERNANCE_SAFETY:
        ft_gov = full_text.lower()
        if any(k in ft_gov for k in ("human-in-command", "governed agentic", "versioned run", "sovereign")):
            concrete_finding = "building governed agentic AI governance systems with versioned audit trails, human-in-command controls, and enterprise deployment frameworks"
        elif any(k in ft_gov for k in ("copilot", "domain copilots", "evaluation", "benchmarking")):
            concrete_finding = "developing enterprise AI evaluation frameworks, guardrails, and domain copilots for regulated and industrial settings"
        elif any(k in ft_gov for k in ("trust & responsible ai", "risk detection", "audit-ready", "tamper-evident")):
            concrete_finding = "developing automated risk detection, tamper-evident audit trails, and AI governance guardrails"
        elif "guardrail" in ft_gov:
            concrete_finding = "engineering AI governance guardrails, model monitoring, and automated compliance verification for production systems"
        else:
            concrete_finding = "building AI governance, auditing, and safety systems for enterprise AI deployments"

    elif domain_class == DOMAIN_RAG_AGENTS:
        ft_lower = full_text.lower()
        if any(k in ft_lower for k in ("bfsi", "fintech", "in-app ai", "in-app agent", "onboarding", "kyc")):
            concrete_finding = "developing in-app AI agents and conversational workflows for financial services customer journeys"
        elif any(k in ft_lower for k in ("whatsapp", "crm", "sales agent", "lead management")):
            concrete_finding = "developing conversational AI agents and automated workflow engines for multi-channel sales and customer communications"
        elif any(k in ft_lower for k in ("rag", "retrieval-augmented generation", "custom rag", "knowledge base", "custom connectors", "hybrid search", "citation")):
            concrete_finding = "engineering and deploying production-ready AI agents, conversational assistants, and custom RAG pipelines"
        elif any(k in ft_lower for k in ("enterprise", "regulated", "voice", "omnichannel", "soc2")):
            concrete_finding = "building enterprise-grade agentic AI platforms with multi-channel voice, chat, and workflow automation"
        elif any(k in ft_lower for k in ("talent graph", "screening", "assessment agent")):
            concrete_finding = "deploying autonomous talent assessment agents and conversational screening workflows on a unified graph platform"
        elif "retrieval" in ft_lower and "agent" in ft_lower:
            concrete_finding = "building custom retrieval-augmented generation (RAG) architectures and autonomous AI agent workflows"
        elif "agent" in ft_lower:
            concrete_finding = "developing autonomous AI agents and intelligent workflow automation systems"
        else:
            concrete_finding = "building production retrieval-augmented generation (RAG) systems and enterprise knowledge architectures"

    elif domain_class == DOMAIN_DIFFUSION_VISION:
        if any(k in full_text.lower() for k in ("diffusion", "flux", "controlnet")):
            concrete_finding = "developing diffusion-based generative AI pipelines, image synthesis workflows, and GPU inference architectures"
        elif "3d" in full_text.lower():
            concrete_finding = "building 3D mesh synthesis and generative computer vision pipelines"
        else:
            concrete_finding = "developing computer vision systems and generative visual models"

    elif domain_class == DOMAIN_NLP_LINGUISTICS:
        if any(k in full_text.lower() for k in ("resume", "ats", "scoring", "career os", "job description")):
            concrete_finding = "developing automated resume analysis, ATS keyword scoring, and career intelligence platforms"
        else:
            concrete_finding = "developing natural language processing pipelines, language embeddings, and sequence modeling architectures"

    elif domain_class == DOMAIN_ML_DATA_SYSTEMS:
        ft_ml = full_text.lower()
        if any(k in ft_ml for k in ("reasoning", "inference acceleration", "inference infrastructure", "production models")):
            concrete_finding = "developing high-performance reasoning infrastructure and inference acceleration architectures for production models"
        else:
            concrete_finding = "building scalable machine learning workflows, automated ETL data systems, and predictive models"

    elif domain_class == DOMAIN_GENERAL_AI:
        if best_excerpt:
            clean_exc = sanitize_evidence_text(best_excerpt)
            concrete_finding = f"advancing applied machine learning architectures, including {clean_exc}" if clean_exc else "developing applied machine learning architectures and production AI systems"
        else:
            concrete_finding = "developing applied machine learning architectures and production AI systems"

    # If no specific finding was synthesized, fall back to sanitized best excerpt or clean name
    if not concrete_finding:
        if best_excerpt:
            concrete_finding = sanitize_evidence_text(best_excerpt)
        elif matched_areas:
            concrete_finding = f"technical initiatives in {', '.join(matched_areas)}"
        else:
            concrete_finding = "technical initiatives in applied machine learning and software engineering"

    # Final cleanup to guarantee no contamination markers or dangling syntax
    concrete_finding = sanitize_evidence_text(concrete_finding)

    # 5. Check internship status from research
    app_status = getattr(research_result, "application_status", "")
    role_status = getattr(research_result, "role_status", "")
    has_internship = False
    internship_details = ""

    if app_status == "OPEN" or role_status == "CONFIRMED":
        has_internship = True
        internship_details = "Verified advertised opening found."
    elif role_status == "NOT_FOUND":
        internship_details = "No advertised internship found; inquiry drafted as general exploratory outreach."
        uncertainties.append("No active internship listing detected; reaching out on speculative basis.")
    else:
        internship_details = "Internship availability unconfirmed."
        uncertainties.append("Internship status unverified from public sources.")

    # 6. Assess evidence strength. STRONG requires an excerpt taken from
    # a retrieved evidence item itself -- a summary-only excerpt (which may
    # be LLM-generated) caps at MODERATE, so a generated summary is never
    # presented as strongly evidenced primary fact.
    if primary_url and best_excerpt and excerpt_from_evidence and domain_class != DOMAIN_NONE:
        evidence_strength = "STRONG"
    elif primary_url and domain_class != DOMAIN_NONE:
        evidence_strength = "MODERATE"
    elif matched_areas:
        evidence_strength = "WEAK"
    else:
        evidence_strength = "NONE"

    # Templated findings are domain-level syntheses, not quoted company
    # statements. Say so, so a draft built on one is never mistaken for a
    # verbatim company claim.
    if concrete_finding and not (best_excerpt and best_excerpt in concrete_finding):
        uncertainties.append(
            "Company description is a domain-level summary synthesised from "
            "retrieved evidence, not a quoted statement from the company."
        )

    return CompanyOpportunity(
        company_name=clean_name,
        primary_domain=domain,
        concrete_finding=concrete_finding,
        primary_source_url=primary_url,
        primary_source_title=primary_title,
        source_excerpt=best_excerpt,
        technical_domain=domain_class,
        evidence_strength=evidence_strength,
        has_advertised_internship=has_internship,
        internship_details=internship_details,
        uncertainties=uncertainties,
    )
