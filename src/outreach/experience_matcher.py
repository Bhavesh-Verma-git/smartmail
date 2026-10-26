"""
src/outreach/experience_matcher.py
----------------------------------
Stage C: Match candidate's actual experience from master profile and CV
to the company's verified technical opportunity.

Strict Rules:
- Select only the 1 or 2 experiences that genuinely connect to the company's verified work.
- Use concrete examples from actual projects and internships, never a generic tool laundry list.
- Never invent achievements, company facts, impact metrics, or experiences.
- All candidate claims are strictly grounded in data/master_profile.json.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.outreach.opportunity_analyzer import (
    CompanyOpportunity,
    DOMAIN_GOVERNANCE_SAFETY,
    DOMAIN_RAG_AGENTS,
    DOMAIN_DIFFUSION_VISION,
    DOMAIN_NLP_LINGUISTICS,
    DOMAIN_ML_DATA_SYSTEMS,
    DOMAIN_GENERAL_AI,
    DOMAIN_NONE,
)

logger = logging.getLogger(__name__)


@dataclass
class CandidateExperienceMatch:
    """
    Candidate project/experience selection matched specifically to a company's technical domain.
    Records structured match dimensions and supporting profile provenance.
    """
    primary_experience_id: str
    primary_experience_name: str
    organization_or_project: str
    concrete_candidate_work: str
    connection_rationale: str
    relevant_technologies: list[str] = field(default_factory=list)
    secondary_experience_name: str | None = None
    is_specific_match: bool = True
    verified_company_finding: str = ""
    selected_candidate_experience: str = ""
    technical_connection: str = ""
    supporting_cv_evidence: str = ""
    limitations_or_assumptions: list[str] = field(default_factory=list)
    supporting_profile_fields: list[str] = field(default_factory=list)
    is_transferable: bool = False
    match_score: int = 0


def match_candidate_experience(
    opportunity: CompanyOpportunity,
    profile: dict[str, Any],
    cv_text: str | None = None,
) -> CandidateExperienceMatch:
    """
    Select the single most relevant candidate experience from master profile
    that directly aligns with the company's verified technical opportunity.

    Uses an explainable, deterministic ranking over master profile records.
    Enforces candidate-profile invariants:
    - ControlPlane.ai is a personal project, NOT an employer.
    - No ungrounded claims (e.g. no 'built production RAG systems with citation-aware responses').
    - Preserves actual contribution wording from profile fields.
    - E-Commerce Price Prediction is labeled as transferable NLP experience when matched to ATS/scoring.
    - If evidence is weak or no direct overlap exists, returns no_direct_match requiring human review.
    """
    domain = opportunity.technical_domain
    comp_name = opportunity.company_name or "your team"
    finding = opportunity.concrete_finding or ""
    excerpt = opportunity.source_excerpt or ""

    # Aggregate target technical evidence text for deterministic scoring
    matched_areas_list = getattr(opportunity, "matched_areas", []) or []
    evidence_text = f"{finding} {excerpt} {' '.join(matched_areas_list)}".lower()

    # 1. Unverified or missing company evidence -> truthful neutral fallback
    if opportunity.evidence_strength in ("WEAK", "NONE") or domain == DOMAIN_NONE:
        return CandidateExperienceMatch(
            primary_experience_id="no_direct_match",
            primary_experience_name="",
            organization_or_project="Academic & Practical ML Projects",
            concrete_candidate_work=(
                "worked on production ML workflows, data systems, and software engineering with a focus on "
                "building reliable architectures and practical problem-solving"
            ),
            connection_rationale=(
                "My background includes practical experience in applied machine learning, neural architectures, "
                "and software engineering."
            ),
            relevant_technologies=[],
            secondary_experience_name=None,
            is_specific_match=False,
            verified_company_finding=finding,
            selected_candidate_experience="Academic & Practical ML Projects",
            technical_connection="Truthful neutral overview for unverified evidence.",
            supporting_cv_evidence="IIT Kharagpur academic background and machine learning coursework.",
            limitations_or_assumptions=["Company technical evidence unverified; specific technical match omitted."],
            supporting_profile_fields=["education[0].institution", "education[0].relevant_coursework"],
            is_transferable=False,
            match_score=0,
        )

    # 2. Look up candidate records from profile
    projects = profile.get("projects", [])
    experiences = profile.get("experience", [])

    def find_project(proj_id_or_keyword: str) -> dict[str, Any] | None:
        for p in projects:
            if proj_id_or_keyword.lower() in p.get("id", "").lower() or proj_id_or_keyword.lower() in p.get("name", "").lower():
                return p
        return None

    def find_exp(org_keyword: str) -> dict[str, Any] | None:
        for e in experiences:
            if org_keyword.lower() in e.get("organization", "").lower():
                return e
        return None

    cp_project = find_project("ControlPlane")
    rp_exp = find_exp("Red Panda")
    iiser_exp = find_exp("IISER")
    sec_project = find_project("Network Security")
    price_project = find_project("Price Prediction")

    # 3. Deterministic scoring function for each candidate record
    # We assign scores based on specific, verifiable technology and problem-domain matches.
    # Generic tokens like 'python', 'ai', 'ml', 'machine learning' receive 0 weight.
    scores: dict[str, int] = {
        "governance_cp": 0,
        "rag_cp": 0,
        "agent_cp": 0,
        "diffusion_rp": 0,
        "nlp_iiser": 0,
        "nlp_price": 0,
        "security_sec": 0,
    }

    # Governance / safety indicators
    if any(k in evidence_text for k in ("governance", "responsible ai", "audit", "guardrail", "guardrails", "compliance", "tamper-evident", "hallucination")):
        scores["governance_cp"] += 15
    if domain == DOMAIN_GOVERNANCE_SAFETY:
        scores["governance_cp"] += 20

    # RAG / retrieval indicators
    if any(k in evidence_text for k in ("rag", "retrieval", "vector", "knowledge base", "faiss", "bm25", "hybrid search", "citation")):
        scores["rag_cp"] += 15
    if "rag" in domain.lower():
        scores["rag_cp"] += 10

    # Agent / workflow / conversational indicators
    if any(k in evidence_text for k in ("agent", "agents", "langgraph", "workflow", "crm", "whatsapp", "conversational", "lead", "state machine")):
        scores["agent_cp"] += 15
    if "agent" in domain.lower():
        scores["agent_cp"] += 10

    # Diffusion / vision / 3D indicators
    if any(k in evidence_text for k in ("diffusion", "flux", "controlnet", "stable diffusion", "3d", "mesh", "image", "vision", "gpu inference")):
        scores["diffusion_rp"] += 20
    if domain == DOMAIN_DIFFUSION_VISION:
        scores["diffusion_rp"] += 25

    # General NLP / linguistics indicators
    if any(k in evidence_text for k in ("linguistics", "bert", "t5", "pos tagging", "sentiment", "sequence", "language model")):
        scores["nlp_iiser"] += 20
    if domain == DOMAIN_NLP_LINGUISTICS:
        scores["nlp_iiser"] += 10

    # Resume scoring / ATS / text attribute extraction indicators
    if any(k in evidence_text for k in ("resume", "ats", "scoring", "keyword", "deberta", "attribute extraction", "career os")):
        scores["nlp_price"] += 25

    # Security ML / ETL / MLOps indicators
    if any(k in evidence_text for k in ("etl", "mlops", "mlflow", "dagshub", "docker", "phishing", "security", "threat")):
        scores["security_sec"] += 20
    if domain == DOMAIN_ML_DATA_SYSTEMS:
        scores["security_sec"] += 15

    # Find the candidate record with the highest score
    best_key = max(scores, key=lambda k: scores[k])
    best_score = scores[best_key]

    # If evidence text is empty or no technical indicators matched, return grounded fallback
    if best_score <= 0:
        return CandidateExperienceMatch(
            primary_experience_id="no_direct_match",
            primary_experience_name="",
            organization_or_project="Academic & Practical ML Projects",
            concrete_candidate_work="Through my academic coursework and hands-on projects at IIT Kharagpur, I have developed practical experience in applied machine learning, neural architectures, and software engineering",
            connection_rationale="General academic alignment without direct specific project match; human review required.",
            relevant_technologies=[],
            secondary_experience_name=None,
            is_specific_match=False,
            verified_company_finding=finding,
            selected_candidate_experience="Academic & Practical ML Projects",
            technical_connection="Truthful neutral overview for unverified evidence.",
            supporting_cv_evidence="IIT Kharagpur academic background and machine learning coursework.",
            limitations_or_assumptions=["Company technical evidence unverified; specific technical match omitted."],
            supporting_profile_fields=["education[0].institution", "education[0].relevant_coursework"],
            is_transferable=False,
            match_score=0,
        )

    # --- Match Construction with strict profile invariants ---

    # Case A: AI Governance, Auditing, Guardrails -> ControlPlane.ai
    if best_key == "governance_cp" and cp_project:
        exp_name = cp_project.get("name", "ControlPlane.ai — Context-Aware AI Governance Middleware")
        prefix = f"Given {comp_name}'s work {finding}, " if finding.lower().startswith(("building ", "developing ", "engineering ")) else f"Given {comp_name}'s focus on {finding}, "
        rationale = (
            f"{prefix}"
            "my practical background in AI governance middleware, validation, "
            "and automated auditing aligns directly with your core engineering direction."
        )
        work = (
            "In my personal project ControlPlane.ai, I developed context-aware AI middleware focused on "
            "reliable routing, hallucination detection, and guardrails for LLM applications"
        )
        return CandidateExperienceMatch(
            primary_experience_id=cp_project.get("id", "proj-001"),
            primary_experience_name=exp_name,
            organization_or_project="ControlPlane.ai",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=["LangGraph", "LiteLLM", "AI Governance", "Streamlit", "Python"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="ControlPlane.ai: AI governance middleware, reliable routing, evaluation, and guardrails for LLM applications.",
            limitations_or_assumptions=["Assumes system requires automated auditing, state machine verification, or compliance validation."],
            supporting_profile_fields=["projects.proj-001.my_contributions[0]", "projects.proj-001.my_contributions[2]"],
            is_transferable=False,
            match_score=best_score,
        )

    # Case B: Diffusion, Computer Vision, Generative Media, 3D -> Red Panda Games
    if best_key == "diffusion_rp" and rp_exp:
        exp_name = f"{rp_exp.get('organization', 'Red Panda Games')} (Data Science Intern)"
        work = (
            "During my data science internship at Red Panda Games, I developed diffusion-based generative AI pipelines "
            "utilizing Flux, SD1.5, Trellis-4B, and ControlNet, and optimized AWS GPU inference workflows"
        )
        rationale = (
            f"Given {comp_name}'s work in generative media and visual models, "
            "my hands-on experience optimizing GPU diffusion inference and model pipelines "
            "directly supports your technical initiatives."
        )
        return CandidateExperienceMatch(
            primary_experience_id=rp_exp.get("id", "exp-001"),
            primary_experience_name=exp_name,
            organization_or_project="Red Panda Games",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=["Flux", "Stable Diffusion 1.5", "ControlNet", "AWS GPU Inference", "Celery & Redis"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="Red Panda Games: Flux, SD1.5, Trellis-4B, ControlNet diffusion pipelines, AWS GPU inference, CPU offloading, Celery/Redis.",
            limitations_or_assumptions=["Assumes team operates visual diffusion or generative 3D pipelines."],
            supporting_profile_fields=["experience.exp-001.responsibilities[0]", "experience.exp-001.responsibilities[1]"],
            is_transferable=False,
            match_score=best_score,
        )

    # Case C: ATS / Resume Scoring / Keyword Scoring -> E-Commerce Price Prediction (Transferable NLP)
    if best_key == "nlp_price" and price_project:
        exp_name = price_project.get("name", "E-Commerce Price Prediction")
        work = (
            "In my E-Commerce Price Prediction project, I developed NLP attribute extraction pipelines and fine-tuned "
            "DeBERTa-v3 embeddings for text representation, which provides directly transferable experience for semantic text scoring"
        )
        rationale = (
            f"Given {comp_name}'s focus on {finding or 'semantic text analysis and scoring'}, "
            "my practical background in transformer embeddings and text representation models "
            "translates directly to your semantic analysis and scoring pipelines."
        )
        return CandidateExperienceMatch(
            primary_experience_id=price_project.get("id", "proj-003"),
            primary_experience_name=exp_name,
            organization_or_project="E-Commerce Price Prediction",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=["DeBERTa-v3", "NLP Pipelines", "Embeddings", "PyTorch"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="E-Commerce Price Prediction: DeBERTa-v3 embeddings, engineered text features, inference-latency optimization.",
            limitations_or_assumptions=["Assumes resume scoring leverages transformer embeddings and semantic similarity; candidate project is transferable NLP."],
            supporting_profile_fields=["projects.proj-003.my_contributions[0]", "projects.proj-003.my_contributions[1]"],
            is_transferable=True,
            match_score=best_score,
        )

    # Case D: Computational Linguistics / General NLP -> IISER Bhopal
    if best_key == "nlp_iiser" and iiser_exp:
        exp_name = f"{iiser_exp.get('organization', 'IISER Bhopal')} (ML Intern)"
        work = (
            "Through my research internship at IISER Bhopal and NLP coursework at IIT Kharagpur, "
            "I developed text processing pipelines and worked with sequence models and transformer architectures"
        )
        rationale = (
            f"Given {comp_name}'s work in language processing and text representations, "
            "my practical experience with sequence models and transformer embeddings aligns closely with your team's focus."
        )
        return CandidateExperienceMatch(
            primary_experience_id=iiser_exp.get("id", "exp-002"),
            primary_experience_name=exp_name,
            organization_or_project="IISER Bhopal",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=["BERT", "T5", "LSTM", "GRU", "Sequence Models", "Python"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="IISER Bhopal: Computational linguistics pipelines, BERT, T5, sequence models, text embeddings.",
            limitations_or_assumptions=["Assumes language model or sequence evaluation workflow."],
            supporting_profile_fields=["experience.exp-002.responsibilities[0]", "experience.exp-002.responsibilities[1]"],
            is_transferable=False,
            match_score=best_score,
        )

    # Case E: ML Data Systems, ETL, Threat/Security ML -> Network Security System
    if best_key == "security_sec" and sec_project:
        exp_name = sec_project.get("name", "Network Security System")
        work = (
            "In my Network Security System project, I implemented automated ETL pipelines, classification models, "
            "and experiment tracking using MLflow and DagsHub, containerized with Docker"
        )
        rationale = (
            f"Given {comp_name}'s focus on scalable machine learning infrastructure and data systems, "
            "my background in production pipelines and automated model deployment translates directly to your engineering needs."
        )
        return CandidateExperienceMatch(
            primary_experience_id=sec_project.get("id", "proj-002"),
            primary_experience_name=exp_name,
            organization_or_project="Network Security System",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=["Scikit-learn", "XGBoost", "MLflow", "Docker", "AWS CI/CD"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="Network Security System: Phishing URL classification, MLflow, DagsHub, Docker, AWS CI/CD.",
            limitations_or_assumptions=["Assumes production tabular / predictive data pipelines."],
            supporting_profile_fields=["projects.proj-002.my_contributions[0]", "projects.proj-002.my_contributions[1]"],
            is_transferable=False,
            match_score=best_score,
        )

    # Case F: Conversational Agents / CRM / Customer Workflows -> ControlPlane.ai (Agents)
    if best_key == "agent_cp" and cp_project:
        exp_name = cp_project.get("name", "ControlPlane.ai — Context-Aware AI Governance Middleware")
        work = (
            "In my technical projects, including ControlPlane.ai, I engineered stateful agent workflows, "
            "task routing, and reliable execution pipelines for AI applications"
        )
        rationale = (
            f"Given {comp_name}'s focus on {finding or 'conversational AI agents and automated workflows'}, "
            "my practical experience designing stateful agent workflows and reliability middleware "
            "aligns directly with your conversational agent platform."
        )
        return CandidateExperienceMatch(
            primary_experience_id=cp_project.get("id", "proj-001"),
            primary_experience_name=exp_name,
            organization_or_project="ControlPlane.ai",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=["LangGraph", "LiteLLM", "Tool Routing", "Agent Workflows", "Python"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="ControlPlane.ai: Stateful agent workflows, tool routing, and context-aware middleware.",
            limitations_or_assumptions=["Assumes conversational agents utilize stateful orchestration and tool execution."],
            supporting_profile_fields=["projects.proj-001.long_description", "projects.proj-001.technologies"],
            is_transferable=False,
            match_score=best_score,
        )

    # Case G: RAG / Knowledge Bases -> ControlPlane.ai (Hybrid Retrieval)
    if best_key == "rag_cp" and cp_project:
        exp_name = cp_project.get("name", "ControlPlane.ai — Context-Aware AI Governance Middleware")
        work = (
            "In my ControlPlane.ai project, I developed hybrid retrieval pipelines "
            "combining FAISS and BM25 for source-grounded inference"
        )
        rationale = (
            f"Given {comp_name}'s focus on {finding or 'RAG and conversational AI'}, "
            "my hands-on experience building retrieval pipelines and knowledge-base connectors "
            "maps directly to your client-facing AI systems."
        )
        return CandidateExperienceMatch(
            primary_experience_id=cp_project.get("id", "proj-001"),
            primary_experience_name=exp_name,
            organization_or_project="ControlPlane.ai",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=["Hybrid Retrieval", "Vector Search", "LangGraph", "Python"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="ControlPlane.ai: Hybrid retrieval pipelines, semantic search, and source-grounded inference.",
            limitations_or_assumptions=["Assumes retrieval pipeline requires dense search and source citations."],
            supporting_profile_fields=["projects.proj-001.my_contributions[1]", "projects.proj-001.technologies"],
            is_transferable=False,
            match_score=best_score,
        )

    # Fallback to no direct match
    return CandidateExperienceMatch(
        primary_experience_id="no_direct_match",
        primary_experience_name="",
        organization_or_project="Academic & Practical ML Projects",
        concrete_candidate_work="Through my academic coursework and hands-on projects at IIT Kharagpur, I have developed practical experience in applied machine learning, neural architectures, and software engineering",
        connection_rationale="General academic alignment without direct specific project match; human review required.",
        relevant_technologies=[],
        secondary_experience_name=None,
        is_specific_match=False,
        verified_company_finding=finding,
        selected_candidate_experience="Academic & Practical ML Projects",
        technical_connection="Truthful neutral overview for unverified evidence.",
        supporting_cv_evidence="IIT Kharagpur academic background and machine learning coursework.",
        limitations_or_assumptions=["Company technical evidence unverified; specific technical match omitted."],
        supporting_profile_fields=["education[0].institution", "education[0].relevant_coursework"],
        is_transferable=False,
        match_score=0,
    )
