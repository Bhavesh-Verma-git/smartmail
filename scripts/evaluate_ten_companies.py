"""
scripts/evaluate_ten_companies.py
---------------------------------
Evaluates the 10 evaluation companies required by Prompt 10967:
1. Octobit8
2. Trustra
3. Hibilter
4. Glowminds
5. DivyaNetra AI
6. Fluid AI
7. RevRag AI
8. Infergent Labs
9. VeloceAI
10. Indus AI

Runs deterministically offline with verified saved evidence.
"""
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.llm.router import LLMRouter
# Enforce offline deterministic generation for benchmark
LLMRouter.has_any_llm = False

from src.contacts.email_parser import parse_email
from src.contacts.recipient_resolver import resolve_recipient
from src.outreach.drafting import draft_outreach_email
from src.outreach.experience_matcher import (
    DOMAIN_DIFFUSION_VISION,
    DOMAIN_GOVERNANCE_SAFETY,
    DOMAIN_ML_DATA_SYSTEMS,
    DOMAIN_NLP_LINGUISTICS,
    DOMAIN_RAG_AGENTS,
)
from src.outreach.models import EmailDraft, GenerationStatus, WorkflowMode
from src.outreach.opportunity_analyzer import analyze_company_opportunity
from src.outreach.quality_gate import evaluate_personalization_quality
from src.profile.loader import load_profile
from src.research.models import EvidenceItem, ResearchResult, RETRIEVAL_OK, RETRIEVAL_ERROR
from src.verification.models import (
    GENERAL_INQUIRY,
    MatchResult,
    PipelineResult,
    PROCEED,
    RELEVANT,
    VerificationResult,
)


def run_ten_company_evaluation():
    profile = load_profile(str(root_dir / "data" / "master_profile.json"))

    # Define the 10 evaluation test cases with verified evidence
    cases = [
        {
            "id": "1",
            "company_name": "Octobit8",
            "domain": "octobit8.com",
            "email": "careers@octobit8.com",
            "source_url": "https://octobit8.com/",
            "evidence_text": (
                "Octobit8 is a full-stack AI engineering company: AI agents, RAG pipelines, "
                "conversational AI, custom models and cloud AI. We build custom retrieval-augmented "
                "generation (RAG) systems that let AI answer from your documents and databases citing sources."
            ),
            "evidence_status": RETRIEVAL_OK,
            "supports": ["company_identity", "technical_relevance"],
            "technical_areas": ["RAG & Retrieval Systems", "Autonomous AI Agents"],
            "role_status": "SPECULATIVE",
            "notes": "No public open internship listing found on official site; outreach is speculative inquiry.",
        },
        {
            "id": "2",
            "company_name": "Trustra",
            "domain": "trustra.ai",
            "email": "contact@trustra.ai",
            "source_url": "https://trustra.ai/",
            "evidence_text": (
                "Trustra.ai | The Trust & Responsible AI OS. Prove your AI to anyone. "
                "AI Flight Recorder reads chatbot logs into a tamper-evident record. "
                "Automated risk detection, tamper-evident audit trails, and AI governance guardrails."
            ),
            "evidence_status": RETRIEVAL_OK,
            "supports": ["company_identity", "technical_relevance"],
            "technical_areas": ["AI Governance & Safety"],
            "role_status": "SPECULATIVE",
            "notes": "Early-stage startup; speculative outreach regarding governance engineering.",
        },
        {
            "id": "3",
            "company_name": "Hibilter",
            "domain": "hibilter.com",
            "email": "info@hibilter.com",
            "source_url": "https://hibilter.com/",
            "evidence_text": (
                "Hibilter (Hibilter IT Systems Pvt. Ltd.) builds governed, auditable, agentic AI systems, "
                "GPU intelligence, and sovereign technology solutions designed with a 'human-in-command' "
                "framework for high-risk actions. Versioned runs, complete audit trails, and strict human oversight."
            ),
            "evidence_status": RETRIEVAL_OK,
            "supports": ["company_identity", "technical_relevance"],
            "technical_areas": ["AI Governance & Safety", "Autonomous AI Agents"],
            "role_status": "SPECULATIVE",
            "notes": "Domain hibilter.ai was unreachable; verified official entity is hibilter.com / hibilter.in.",
        },
        {
            "id": "4",
            "company_name": "Glowminds",
            "domain": "glowminds.in",
            "email": "hr@glowminds.in",
            "source_url": "https://glowminds.in/",
            "evidence_text": (
                "Glowminds AI Technologies Private Limited builds digital tools, AI voice agents, "
                "custom CRMs, and web automation tools for businesses, alongside the Glowminds Career OS platform."
            ),
            "evidence_status": RETRIEVAL_OK,
            "supports": ["company_identity", "technical_relevance"],
            "technical_areas": ["Autonomous AI Agents", "Web Automation"],
            "role_status": "SPECULATIVE",
            "notes": "Domain glowminds.ai does not resolve; resolved to glowminds.in (Glowminds AI Technologies Pvt Ltd).",
        },
        {
            "id": "5",
            "company_name": "DivyaNetra AI",
            "domain": "divyanetra.ai",
            "email": "contact@divyanetra.ai",
            "source_url": "https://divyanetra.ai/",
            "evidence_text": (
                "DivyaNetra AI Private Limited is an artificial intelligence startup in Greater Noida. "
                "Develops investigation & forensics AI, legal tech 'Law OS', document intelligence, "
                "and multi-agent workflows."
            ),
            "evidence_status": RETRIEVAL_ERROR,
            "supports": [],
            "technical_areas": ["Document Intelligence", "Forensics AI"],
            "role_status": "UNRESOLVED",
            "notes": "Official domain divyanetra.ai timed out/unreachable during live fetch. Unverified via direct web source.",
        },
        {
            "id": "6",
            "company_name": "Fluid AI",
            "domain": "fluid.ai",
            "email": "careers@fluid.ai",
            "source_url": "https://fluid.ai/",
            "evidence_text": (
                "Fluid AI | Enterprise Agentic AI Platform for regulated industries. Deploy AI agents "
                "across voice, chat, email & WhatsApp. SOC2 certified, on-premise ready, conversational workflow engines."
            ),
            "evidence_status": RETRIEVAL_OK,
            "supports": ["company_identity", "technical_relevance"],
            "technical_areas": ["Autonomous AI Agents", "Conversational AI"],
            "role_status": "SPECULATIVE",
            "notes": "Enterprise agent platform; verified site accessibility and product claims.",
        },
        {
            "id": "7",
            "company_name": "RevRag AI",
            "domain": "revrag.ai",
            "email": "hello@revrag.ai",
            "source_url": "https://www.revrag.ai/",
            "evidence_text": (
                "RevRag AI develops autonomous AI sales agents ('Emma') and workflow automation engines "
                "for multi-channel sales, lead qualification, and CRM communications."
            ),
            "evidence_status": RETRIEVAL_OK,
            "supports": ["company_identity", "technical_relevance"],
            "technical_areas": ["Autonomous AI Agents", "Conversational AI"],
            "role_status": "SPECULATIVE",
            "notes": "Verified B2B AI agent platform.",
        },
        {
            "id": "8",
            "company_name": "Infergent Labs",
            "domain": "infergent.com",
            "email": "founders@infergent.com",
            "source_url": "https://infergent.com/",
            "evidence_text": (
                "Infergent Labs engineers in-app AI agents and conversational workflows for financial "
                "services (BFSI), customer onboarding, and digital customer journeys."
            ),
            "evidence_status": RETRIEVAL_OK,
            "supports": ["company_identity", "technical_relevance"],
            "technical_areas": ["Autonomous AI Agents", "FinTech AI"],
            "role_status": "SPECULATIVE",
            "notes": "In-app conversational AI startup.",
        },
        {
            "id": "9",
            "company_name": "VeloceAI",
            "domain": "veloceai.in",
            "email": "team@veloceai.in",
            "source_url": "https://veloceai.in/",
            "evidence_text": (
                "VeloceAI - The Reasoning Infrastructure for Enterprise AI. Providing high-performance reasoning "
                "and inference acceleration architectures for production models."
            ),
            "evidence_status": RETRIEVAL_OK,
            "supports": ["company_identity", "technical_relevance"],
            "technical_areas": ["ML Infrastructure & Acceleration"],
            "role_status": "SPECULATIVE",
            "notes": "AI inference and reasoning systems startup.",
        },
        {
            "id": "10",
            "company_name": "Indus AI",
            "domain": "indusai.app",
            "email": "connect@indusai.app",
            "source_url": "https://indusai.app/",
            "evidence_text": (
                "Indus AI deploys autonomous talent assessment agents and conversational screening workflows "
                "on a unified talent graph platform."
            ),
            "evidence_status": RETRIEVAL_OK,
            "supports": ["company_identity", "technical_relevance"],
            "technical_areas": ["Autonomous AI Agents", "Talent Assessment"],
            "role_status": "SPECULATIVE",
            "notes": "Agentic talent screening platform.",
        },
    ]

    results = []

    print(f"{'#':<3} | {'Company':<15} | {'Domain':<15} | {'Selected Project':<28} | {'Quality Gate':<12} | {'Status'}")
    print("-" * 95)

    for c in cases:
        c_id = c["id"]
        c_name = c["company_name"]
        domain = c["domain"]
        email = c["email"]
        src_url = c["source_url"]
        ev_status = c["evidence_status"]

        if ev_status == RETRIEVAL_ERROR:
            # Unresolved case (e.g. DivyaNetra domain unreachable)
            ev_item = EvidenceItem(
                source_url=src_url,
                source_type="official_website",
                title=f"{c_name} Website",
                extracted_text="",
                retrieval_status=RETRIEVAL_ERROR,
                supports=[],
                claim_supported="Website unreachable",
            )
            res = ResearchResult(
                company_name=c_name,
                evidence=[ev_item],
                technical_areas=[],
                ai_ml_evidence_summary=[],
            )
            pipe = PipelineResult(
                verification=VerificationResult(
                    company_status="UNVERIFIED",
                    contact_status="UNVERIFIED",
                    overall_status="REVIEW",
                    review_flags=[f"Primary website unreachable: {src_url}"],
                    evidence_sources=[],
                ),
                match=MatchResult(
                    relevance_status="UNVERIFIED",
                    case_type=GENERAL_INQUIRY,
                    matched_areas=[],
                ),
            )
        else:
            ev_item = EvidenceItem(
                source_url=src_url,
                source_type="official_website",
                title=f"{c_name} Official",
                extracted_text=c["evidence_text"],
                retrieval_status=RETRIEVAL_OK,
                supports=c["supports"],
                claim_supported="Extracted core capabilities",
            )
            res = ResearchResult(
                company_name=c_name,
                evidence=[ev_item],
                technical_areas=c["technical_areas"],
                ai_ml_evidence_summary=[{"excerpt": c["evidence_text"], "source_url": src_url}],
            )
            pipe = PipelineResult(
                verification=VerificationResult(
                    company_status="VERIFIED",
                    contact_status="VERIFIED",
                    overall_status=PROCEED,
                    review_flags=[],
                    evidence_sources=[src_url],
                ),
                match=MatchResult(
                    relevance_status=RELEVANT,
                    case_type=GENERAL_INQUIRY,
                    matched_areas=c["technical_areas"],
                ),
            )

        draft = draft_outreach_email(
            profile=profile,
            recipient_email=email,
            company_name=c_name,
            verification_pipeline_result=pipe,
            research_result=res,
        )

        pr = draft.personalization_record
        selected_proj = (draft.matched_projects[0] if draft.matched_projects else (pr.candidate_experience if pr else "None"))
        qg_passed = pr.quality_gate_passed if pr else False
        flags = draft.review_flags

        # Determine verdict:
        # If domain was unreachable or identity unverified -> UNRESOLVED
        # If quality gate passed with defensible project -> PASS
        # Otherwise -> FAIL
        if ev_status == RETRIEVAL_ERROR or pipe.verification.company_status != "VERIFIED":
            status = "UNRESOLVED"
        elif qg_passed and selected_proj != "None":
            status = "PASS"
        else:
            status = "FAIL"

        short_proj = (selected_proj[:25] + "...") if len(selected_proj) > 28 else selected_proj
        qg_label = "PASSED" if qg_passed else "FLAGGED"

        print(f"{c_id:<3} | {c_name:<15} | {domain:<15} | {short_proj:<28} | {qg_label:<12} | {status}")

        results.append({
            "id": c_id,
            "company_name": c_name,
            "domain": domain,
            "email": email,
            "source_url": src_url,
            "evidence_text": c["evidence_text"],
            "role_status": c["role_status"],
            "notes": c["notes"],
            "company_finding": pr.company_finding if pr else "",
            "technical_domain": pr.technical_domain if pr else "",
            "selected_project": selected_proj,
            "alignment_rationale": pr.connection_reasoning if pr else "",
            "subject": draft.subject,
            "body": draft.body,
            "quality_gate_passed": qg_passed,
            "quality_gate_checks": pr.quality_gate_checks if pr else {},
            "review_flags": flags,
            "status": status,
        })

    # Save detailed evaluation to scratch/ten_company_evaluation.json
    out_path = root_dir / "scratch" / "ten_company_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nDetailed evaluation saved to:", out_path)


if __name__ == "__main__":
    run_ten_company_evaluation()
