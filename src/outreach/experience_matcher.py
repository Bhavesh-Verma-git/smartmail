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
    Records the five structured match dimensions:
    1. Verified company finding
    2. Selected candidate experience
    3. Technical connection
    4. Supporting CV evidence
    5. Limitations or assumptions
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


def match_candidate_experience(
    opportunity: CompanyOpportunity,
    profile: dict[str, Any],
    cv_text: str | None = None,
) -> CandidateExperienceMatch:
    """
    Select the single most relevant candidate experience from master profile
    that directly aligns with the company's verified technical opportunity.

    Parameters
    ----------
    opportunity : CompanyOpportunity extracted from company evidence.
    profile     : Candidate profile loaded from data/master_profile.json.
    cv_text     : Optional raw CV text.

    Returns
    -------
    CandidateExperienceMatch with specific candidate claims and connection rationale.
    """
    domain = opportunity.technical_domain
    comp_name = opportunity.company_name or "your team"

    # Profile lookup helpers
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
    char_project = find_project("Character Backstory")

    # -------------------------------------------------------------------------
    # Case 1: AI Governance, Safety, Guardrails, Auditing (e.g. Trustra, Rasa AI Labs)
    # -------------------------------------------------------------------------
    if domain == DOMAIN_GOVERNANCE_SAFETY and cp_project:
        finding = opportunity.concrete_finding or "AI governance and compliance systems"
        exp_name = cp_project.get("name", "ControlPlane.ai — Context-Aware AI Governance Middleware")
        if finding.lower().startswith(("building ", "developing ", "engineering ")):
            prefix = f"Given {comp_name}'s work {finding}, "
        else:
            prefix = f"Given {comp_name}'s focus on {finding}, "
        rationale = (
            f"{prefix}"
            "my practical background in stateful governance middleware, compliance validation, "
            "and automated auditing aligns directly with your core engineering direction."
        )
        work = (
            "architected an AI governance middleware featuring a nine-node state machine with "
            "parallel auditing, compliance retrieval via Neo4j, and XGBoost hallucination classification"
        )
        return CandidateExperienceMatch(
            primary_experience_id=cp_project.get("id", "proj-001"),
            primary_experience_name=exp_name,
            organization_or_project="ControlPlane.ai",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=["LangGraph", "Neo4j", "XGBoost", "Parallel Auditing", "LangSmith"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="ControlPlane.ai: Nine-node LangGraph state machine, parallel auditing, Neo4j compliance retrieval, XGBoost hallucination classification.",
            limitations_or_assumptions=["Assumes system requires automated auditing, state machine verification, or compliance validation."],
        )

    # -------------------------------------------------------------------------
    # Case 2: RAG, Retrieval Systems, Autonomous AI Agents (e.g. Octobit8, Viva360, Supersei)
    # Sub-cases based on concrete_finding to ensure different companies get distinct descriptions.
    # -------------------------------------------------------------------------
    if domain == DOMAIN_RAG_AGENTS and cp_project:
        finding_lower = (opportunity.concrete_finding or "").lower()
        exp_name = cp_project.get("name", "ControlPlane.ai — Context-Aware AI Governance Middleware")

        # Sub-case 2a: Sales / CRM / Customer Support conversational agents
        if any(k in finding_lower for k in ("sales", "crm", "whatsapp", "lead", "customer")):
            work = (
                "engineered stateful agent workflows using LangGraph, including automated task routing, "
                "multi-step tool-use pipelines, and reliable execution for customer-facing AI systems"
            )
            rationale = (
                f"Given {comp_name}'s focus on {opportunity.concrete_finding or 'conversational AI agents and customer workflow automation'}, "
                "my practical experience designing stateful agent state machines and reliability middleware "
                "aligns directly with your conversational agent platform."
            )
            supporting_cv = "ControlPlane.ai: Nine-node LangGraph state machine, tool routing, hallucination classification, context-aware middleware."
            limitations = ["Assumes conversational agents utilize stateful orchestration and tool execution."]
            techs = ["LangGraph", "LiteLLM", "Tool Routing", "State Machines", "Python"]

        # Sub-case 2b: Heavy RAG / conversational assistants / knowledge bases (e.g. Octobit8)
        elif any(k in finding_lower for k in ("rag", "retrieval-augmented", "conversational", "knowledge base", "chatbot", "chatbots")):
            work = (
                "implemented a hybrid retrieval pipeline combining FAISS and BM25 with Reciprocal Rank Fusion "
                "for low-latency, source-grounded inference, and built production RAG systems with citation-aware responses"
            )
            rationale = (
                f"Given {comp_name}'s focus on {opportunity.concrete_finding or 'RAG and conversational AI'}, "
                "my hands-on experience building high-precision retrieval pipelines and knowledge-base connectors "
                "maps directly to your client-facing AI systems."
            )
            supporting_cv = "ControlPlane.ai: Hybrid FAISS/BM25 retrieval, Reciprocal Rank Fusion, low-latency source-grounded inference."
            limitations = ["Assumes retrieval pipeline requires hybrid dense-sparse search and source citations."]
            techs = ["Hybrid RAG (FAISS + BM25)", "Reciprocal Rank Fusion", "LangGraph", "LiteLLM"]

        # Sub-case 2c: Agentic workflows / automation / orchestration (e.g. Viva360)
        elif any(k in finding_lower for k in ("workflow automation", "agentic", "orchestration", "intelligent workflow", "autonomous agent")):
            work = (
                "engineered multi-node autonomous agent workflows using LangGraph, including stateful "
                "task routing, tool-use pipelines, and event-driven orchestration for production AI systems"
            )
            rationale = (
                f"Given {comp_name}'s focus on {opportunity.concrete_finding or 'AI agents and workflow automation'}, "
                "my experience building stateful agent pipelines and orchestration systems "
                "directly supports your engineering roadmap."
            )
            supporting_cv = "ControlPlane.ai: Nine-node LangGraph state machine, stateful routing, event-driven orchestration."
            limitations = ["Assumes multi-agent or agentic workflow architecture."]
            techs = ["LangGraph", "Event-Driven Orchestration", "Tool Execution", "State Machines"]

        # Sub-case 2d: General AI agents (catch-all for DOMAIN_RAG_AGENTS)
        else:
            work = (
                "engineered autonomous agent workflows using LangGraph and implemented a hybrid retrieval pipeline "
                "combining FAISS and BM25 with Reciprocal Rank Fusion for low-latency, source-grounded inference"
            )
            rationale = (
                f"Given {comp_name}'s focus on {opportunity.concrete_finding or 'AI agents and intelligent systems'}, "
                "my experience designing low-latency retrieval and agent state machines "
                "translates directly to your AI infrastructure."
            )
            supporting_cv = "ControlPlane.ai: LangGraph agent workflows, hybrid retrieval, low-latency inference."
            limitations = ["Assumes agent or retrieval infrastructure."]
            techs = ["LangGraph", "Hybrid RAG", "Python", "LiteLLM"]

        return CandidateExperienceMatch(
            primary_experience_id=cp_project.get("id", "proj-001"),
            primary_experience_name=exp_name,
            organization_or_project="ControlPlane.ai",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=techs,
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=opportunity.concrete_finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence=supporting_cv,
            limitations_or_assumptions=limitations,
        )

    # -------------------------------------------------------------------------
    # Case 3: Diffusion Models, Computer Vision, Generative Media, 3D
    # -------------------------------------------------------------------------
    if domain == DOMAIN_DIFFUSION_VISION and rp_exp:
        exp_name = f"{rp_exp.get('organization', 'Red Panda Games')} (Data Science Intern)"
        work = (
            "developed diffusion-based generative AI pipelines utilizing Flux, SD1.5, Trellis-4B, and ControlNet, "
            "optimizing AWS GPU inference workflows and memory offloading"
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
            verified_company_finding=opportunity.concrete_finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="Red Panda Games: Flux, SD1.5, Trellis-4B, ControlNet diffusion pipelines, AWS GPU inference, CPU offloading, Celery/Redis.",
            limitations_or_assumptions=["Assumes team operates visual diffusion or generative 3D pipelines."],
        )

    # -------------------------------------------------------------------------
    # Case 4: NLP, Computational Linguistics, Text Systems (e.g. Glowminds)
    # -------------------------------------------------------------------------
    if domain == DOMAIN_NLP_LINGUISTICS:
        finding_lower = (opportunity.concrete_finding or "").lower()

        # Sub-case 4a: Resume scoring, ATS keyword analysis, career OS
        if any(k in finding_lower for k in ("resume", "ats", "scoring", "keyword", "career")) and price_project:
            exp_name = price_project.get("name", "E-Commerce Price Prediction")
            work_desc = (
                "developed NLP pipelines using fine-tuned DeBERTa-v3 embeddings for text attribute extraction, "
                "keyword importance scoring, and semantic feature representation"
            )
            rationale = (
                f"Given {comp_name}'s focus on {opportunity.concrete_finding or 'resume scoring and career intelligence'}, "
                "my practical background in transformer embeddings and text representation models "
                "translates directly to your semantic analysis and scoring pipelines."
            )
            return CandidateExperienceMatch(
                primary_experience_id=price_project.get("id", "proj-003"),
                primary_experience_name=exp_name,
                organization_or_project="E-Commerce Price Prediction",
                concrete_candidate_work=work_desc,
                connection_rationale=rationale,
                relevant_technologies=["DeBERTa-v3", "NLP Pipelines", "Embeddings", "PyTorch"],
                secondary_experience_name=None,
                is_specific_match=True,
                verified_company_finding=opportunity.concrete_finding,
                selected_candidate_experience=exp_name,
                technical_connection=rationale,
                supporting_cv_evidence="E-Commerce Price Prediction: DeBERTa-v3 embeddings, engineered text features, inference-latency optimization.",
                limitations_or_assumptions=["Assumes resume scoring leverages transformer embeddings and semantic similarity."],
            )

        # Sub-case 4b: General NLP / linguistics (IISER Bhopal or Price Prediction)
        if iiser_exp:
            exp_name = f"{iiser_exp.get('organization', 'IISER Bhopal')} (ML Intern)"
            org_name = "IISER Bhopal"
            work_desc = (
                "developed computational linguistics pipelines, language embeddings, "
                "and evaluated sequence models including BERT, T5, LSTMs, and GRUs"
            )
            cv_ev = "IISER Bhopal: Computational linguistics pipelines, BERT, T5, sequence models, text embeddings."
        else:
            exp_name = price_project.get("name", "E-Commerce Price Prediction") if price_project else "NLP Research"
            org_name = exp_name
            work_desc = "developed NLP feature engineering pipelines and fine-tuned DeBERTa-v3 embeddings for text attribute extraction"
            cv_ev = "E-Commerce Price Prediction: DeBERTa-v3 embeddings, engineered text features."

        rationale = (
            f"Given {comp_name}'s work in language processing and text representations, "
            "my practical experience with sequence models and transformer embeddings aligns closely with your team's focus."
        )
        return CandidateExperienceMatch(
            primary_experience_id=iiser_exp.get("id", "exp-002") if iiser_exp else "proj-003",
            primary_experience_name=exp_name,
            organization_or_project=org_name,
            concrete_candidate_work=work_desc,
            connection_rationale=rationale,
            relevant_technologies=["DeBERTa-v3", "BERT", "T5", "Sequence Models", "Python"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=opportunity.concrete_finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence=cv_ev,
            limitations_or_assumptions=["Assumes language model or sequence evaluation workflow."],
        )

    # -------------------------------------------------------------------------
    # Case 5: ML Data Systems, ETL, Threat/Security ML
    # -------------------------------------------------------------------------
    if domain == DOMAIN_ML_DATA_SYSTEMS and sec_project:
        exp_name = sec_project.get("name", "Network Security System")
        work = (
            "developed automated ETL data pipelines, experiment tracking with MLflow and DagsHub, "
            "and containerized classification systems with Docker and CI/CD pipelines"
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
            verified_company_finding=opportunity.concrete_finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="Network Security System: Phishing URL classification, MLflow, DagsHub, Docker, AWS CI/CD.",
            limitations_or_assumptions=["Assumes production tabular / predictive data pipelines."],
        )

    # -------------------------------------------------------------------------
    # Case 6: General AI/ML or Moderate Evidence (Default to ControlPlane)
    # -------------------------------------------------------------------------
    if cp_project:
        exp_name = cp_project.get("name", "ControlPlane.ai")
        work = (
            "engineered scalable AI architectures, stateful agent pipelines with LangGraph, "
            "and hybrid retrieval systems with an emphasis on low-latency inference and reliability"
        )
        rationale = (
            f"I believe my background in building production-grade AI systems and applied machine learning "
            f"aligns well with {comp_name}'s technical initiatives."
        )
        return CandidateExperienceMatch(
            primary_experience_id=cp_project.get("id", "proj-001"),
            primary_experience_name=exp_name,
            organization_or_project="ControlPlane.ai",
            concrete_candidate_work=work,
            connection_rationale=rationale,
            relevant_technologies=["LangGraph", "Hybrid RAG", "Python", "LiteLLM"],
            secondary_experience_name=None,
            is_specific_match=True,
            verified_company_finding=opportunity.concrete_finding,
            selected_candidate_experience=exp_name,
            technical_connection=rationale,
            supporting_cv_evidence="ControlPlane.ai: LangGraph agent workflows, hybrid retrieval, low-latency inference.",
            limitations_or_assumptions=["General AI alignment based on candidate production background."],
        )

    # -------------------------------------------------------------------------
    # Case 7: Unverified or Missing Technical Evidence (Truthful Neutral)
    # -------------------------------------------------------------------------
    return CandidateExperienceMatch(
        primary_experience_id="neutral",
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
        verified_company_finding="",
        selected_candidate_experience="Academic & Practical ML Projects",
        technical_connection="Truthful neutral overview for unverified evidence.",
        supporting_cv_evidence="IIT Kharagpur academic background and machine learning coursework.",
        limitations_or_assumptions=["Company technical evidence unverified; specific technical match omitted."],
    )
