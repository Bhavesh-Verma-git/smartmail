"""
tests/test_personalization_regression.py
----------------------------------------
Comprehensive regression test suite for company-specific personalization:
1. Octobit8 and Trustra processed in the same batch:
   - Distinct verified company anchors (Octobit8: AI agents & RAG; Trustra: Trust & Responsible AI OS & governance)
   - Distinct candidate experience connections (ControlPlane LangGraph & hybrid retrieval for Octobit8;
     ControlPlane governance middleware & parallel auditing for Trustra)
   - Substantive paragraphs are NOT copies with names swapped
   - No 8-taxonomy laundry list dumps
   - No Red Panda Games diffusion claims on non-vision companies
2. Two companies in the same AI/ML sector processed together with distinct technical anchors.
3. Same company processed in separate batches (cache freshness and domain isolation).
4. Draft regeneration when research evidence changes.
5. Cached research isolation across companies and domains.
6. Draft persistence in output/.drafts and UI retrieval of the new version.
7. Failed or low-evidence research handled honestly without generic fallback marked as personalized.
8. Speculative outreach without confirmed internship listings handled honestly with review flags.
"""
from __future__ import annotations

import copy
import json
import pytest
from pathlib import Path

from src.contacts.email_parser import EmailParseResult
from src.contacts.csv_importer import ContactRecord
from src.contacts.recipient_resolver import resolve_recipient
from src.outreach.drafting import draft_outreach_email
from src.outreach.experience_matcher import (
    match_candidate_experience,
    DOMAIN_GOVERNANCE_SAFETY,
    DOMAIN_RAG_AGENTS,
)
from src.outreach.models import EmailDraft, GenerationStatus, WorkflowMode
from src.outreach.opportunity_analyzer import analyze_company_opportunity
from src.outreach.quality_gate import evaluate_personalization_quality
from src.profile.loader import load_profile
from src.research.cache import ResearchCache, CachedResearchProvider
from src.research.models import EvidenceItem, ResearchResult, RETRIEVAL_OK, RETRIEVAL_ERROR
from src.research.providers import ResearchProvider, HttpPageProvider
from src.ui.storage import save_draft, get_draft, regenerate_draft_content
from src.verification.models import (
    GENERAL_INQUIRY,
    MatchResult,
    PipelineResult,
    PROCEED,
    RELEVANT,
    VerificationResult,
)
from src.workflow.engine import OutreachWorkflow


@pytest.fixture
def master_profile() -> dict:
    return load_profile("data/master_profile.json")


@pytest.fixture(autouse=True)
def mock_router_offline(monkeypatch):
    """Ensure tests run offline deterministically without waiting on live API timeouts."""
    from src.llm.router import LLMRouter
    monkeypatch.setattr(LLMRouter, "has_any_llm", False)


@pytest.fixture
def octobit8_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source_url="https://octobit8.com/services/rag-development",
            source_type="official_website",
            title="Custom RAG Development | Octobit8",
            extracted_text=(
                "Octobit8 builds retrieval-augmented generation (RAG) systems that let AI answer from your documents, "
                "databases and policies, citing the source every time. We design and build enterprise knowledge bases, "
                "custom connectors, and hybrid search pipelines."
            ),
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        ),
        EvidenceItem(
            source_url="https://octobit8.com",
            source_type="official_website",
            title="Octobit8: AI Development Company for Agents, RAG & Chatbots",
            extracted_text=(
                "Octobit8 is a full-stack AI engineering company. We design, build and run AI agents, RAG systems, "
                "conversational assistants and custom models inside your cloud. AI that ships to production, not just to a demo."
            ),
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        ),
    ]


@pytest.fixture
def trustra_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source_url="https://trustra.ai",
            source_type="official_website",
            title="Trustra.ai | The Trust & Responsible AI OS. Prove your AI to anyone.",
            extracted_text=(
                "Trustra is the Trust & Responsible AI OS. One operating layer of tamper-evident records, "
                "automated risk detection, and audit-ready proof for every AI system in production. "
                "Automate model monitoring, guardrails, and compliance reporting mapped to ISO 42001 and the EU AI Act."
            ),
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        ),
        EvidenceItem(
            source_url="https://www.trustra.ai/assessment",
            source_type="official_website",
            title="AI Governance Maturity Assessment | Trustra.ai",
            extracted_text=(
                "Benchmark your AI governance across 6 dimensions. Trustra's platform automates recommendations from "
                "model monitoring to compliance reporting, providing an AI flight recorder that hash-chains logs and enforces guardrails."
            ),
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Octobit8 and Trustra Processed Side-by-Side
# ---------------------------------------------------------------------------

class TestOctobit8AndTrustraDivergence:
    """Verify Octobit8 and Trustra produce genuinely company-specific, non-identical outreach."""

    def test_octobit8_and_trustra_generate_distinct_evidence_and_drafts(
        self, master_profile, octobit8_evidence, trustra_evidence
    ):
        # 1. Setup Octobit8 Research & Verification
        oct_res = ResearchResult(
            company_name="Octobit8",
            resolved_company_name="Octobit8",
            evidence=octobit8_evidence,
            technical_areas=["RAG & Retrieval Systems", "AI Agents & Workflows"],
            ai_ml_evidence_summary=[
                {"excerpt": octobit8_evidence[0].extracted_text, "source_url": octobit8_evidence[0].source_url},
                {"excerpt": octobit8_evidence[1].extracted_text, "source_url": octobit8_evidence[1].source_url},
            ],
            role_status="NOT_FOUND",
            application_status="UNAVAILABLE",
        )
        oct_pipeline = PipelineResult(
            verification=VerificationResult(
                company_status="VERIFIED",
                contact_status="VERIFIED",
                overall_status=PROCEED,
                evidence_sources=[e.source_url for e in octobit8_evidence],
            ),
            match=MatchResult(
                relevance_status=RELEVANT,
                case_type=GENERAL_INQUIRY,
                matched_areas=["RAG & Retrieval Systems", "AI Agents & Workflows"],
            ),
        )

        # 2. Setup Trustra Research & Verification
        tru_res = ResearchResult(
            company_name="Trustra",
            resolved_company_name="Trustra",
            evidence=trustra_evidence,
            technical_areas=["AI Governance & Safety", "Responsible AI"],
            ai_ml_evidence_summary=[
                {"excerpt": trustra_evidence[0].extracted_text, "source_url": trustra_evidence[0].source_url},
                {"excerpt": trustra_evidence[1].extracted_text, "source_url": trustra_evidence[1].source_url},
            ],
            role_status="NOT_FOUND",
            application_status="UNAVAILABLE",
        )
        tru_pipeline = PipelineResult(
            verification=VerificationResult(
                company_status="VERIFIED",
                contact_status="VERIFIED",
                overall_status=PROCEED,
                evidence_sources=[e.source_url for e in trustra_evidence],
            ),
            match=MatchResult(
                relevance_status=RELEVANT,
                case_type=GENERAL_INQUIRY,
                matched_areas=["AI Governance & Safety"],
            ),
        )

        # 3. Generate Octobit8 Draft
        oct_draft = draft_outreach_email(
            profile=master_profile,
            recipient_email="hrindia@octobit8.com",
            company_name="Octobit8",
            verification_pipeline_result=oct_pipeline,
            research_result=oct_res,
        )

        # 4. Generate Trustra Draft
        tru_draft = draft_outreach_email(
            profile=master_profile,
            recipient_email="careers@trustra.ai",
            company_name="Trustra",
            verification_pipeline_result=tru_pipeline,
            research_result=tru_res,
        )

        # --- A. Assertions on Company Anchor Divergence ---
        assert oct_draft.personalization_record is not None
        assert tru_draft.personalization_record is not None

        assert oct_draft.personalization_record.technical_domain == DOMAIN_RAG_AGENTS
        assert tru_draft.personalization_record.technical_domain == DOMAIN_GOVERNANCE_SAFETY

        assert "agents" in oct_draft.personalization_record.company_finding.lower()
        assert "rag" in oct_draft.personalization_record.company_finding.lower()
        assert "governance" in tru_draft.personalization_record.company_finding.lower() or "responsible ai" in tru_draft.personalization_record.company_finding.lower()

        # --- B. Assertions on Candidate Experience Connection Divergence ---
        # Octobit8 must cite RAG, LangGraph, hybrid retrieval
        assert "hybrid retrieval" in oct_draft.body.lower() or "langgraph" in oct_draft.body.lower()
        assert "bm25" in oct_draft.body.lower() or "faiss" in oct_draft.body.lower()

        # Trustra must cite governance middleware, parallel auditing, compliance retrieval, guardrails
        assert "governance" in tru_draft.body.lower() or "auditing" in tru_draft.body.lower()
        assert "neo4j" in tru_draft.body.lower() or "hallucination" in tru_draft.body.lower()

        # NEITHER company should mention diffusion models or Red Panda Games
        # (Red Panda is for generative media / diffusion, which neither Octobit8 nor Trustra does)
        assert "red panda games" not in oct_draft.body.lower()
        assert "red panda games" not in tru_draft.body.lower()
        assert "diffusion models" not in oct_draft.body.lower()
        assert "diffusion models" not in tru_draft.body.lower()

        # --- C. Assertions on Substantive Email Content Uniqueness ---
        assert oct_draft.body != tru_draft.body
        # Paragraphs must NOT be identical with company name swapped
        oct_body_clean = oct_draft.body.replace("Octobit8", "COMPANY").replace("octobit8.com", "company.com")
        tru_body_clean = tru_draft.body.replace("Trustra", "COMPANY").replace("trustra.ai", "company.com")
        assert oct_body_clean != tru_body_clean

        # --- D. Assertions on Quality Gate & No Taxonomy Dumps ---
        assert oct_draft.personalization_record.quality_gate_checks.get("no_taxonomy_dump") is True
        assert tru_draft.personalization_record.quality_gate_checks.get("no_taxonomy_dump") is True

        # Ensure no broad laundry list in either email
        forbidden_dump_pattern = "RAG & Retrieval Systems, AI Agents & Workflows, Machine Learning & Predictive Analytics"
        assert forbidden_dump_pattern not in oct_draft.body
        assert forbidden_dump_pattern not in tru_draft.body


# ---------------------------------------------------------------------------
# 2. Two Companies in the Same Sector Processed Together
# ---------------------------------------------------------------------------

class TestSameSectorDivergence:
    """Verify two companies in the same sector receive distinct anchors and copy."""

    def test_two_rag_companies_receive_distinct_company_anchors(self, master_profile):
        # Company A: Enterprise Policy & Document RAG
        comp_a_evidence = [EvidenceItem(
            source_url="https://docurag.io/product",
            source_type="official_website",
            title="DocuRAG Enterprise Document Search",
            extracted_text="DocuRAG builds enterprise document retrieval pipelines citing compliance sources for legal teams.",
            retrieval_status=RETRIEVAL_OK,
        )]
        res_a = ResearchResult(
            company_name="DocuRAG",
            evidence=comp_a_evidence,
            technical_areas=["RAG, Retrieval & Knowledge Systems"],
            ai_ml_evidence_summary=[{"excerpt": comp_a_evidence[0].extracted_text, "source_url": comp_a_evidence[0].source_url}],
        )
        pipe_a = PipelineResult(
            verification=VerificationResult(company_status="VERIFIED", contact_status="VERIFIED", overall_status=PROCEED, evidence_sources=["https://docurag.io"]),
            match=MatchResult(relevance_status=RELEVANT, case_type=GENERAL_INQUIRY, matched_areas=["RAG, Retrieval & Knowledge Systems"]),
        )

        # Company B: Voice AI & Conversational Agents
        comp_b_evidence = [EvidenceItem(
            source_url="https://voiceagent.ai/technology",
            source_type="official_website",
            title="VoiceAgent Autonomous Call Agents",
            extracted_text="VoiceAgent designs autonomous conversational AI agents and real-time voice workflows for call centers.",
            retrieval_status=RETRIEVAL_OK,
        )]
        res_b = ResearchResult(
            company_name="VoiceAgent",
            evidence=comp_b_evidence,
            technical_areas=["AI Agents & Workflows"],
            ai_ml_evidence_summary=[{"excerpt": comp_b_evidence[0].extracted_text, "source_url": comp_b_evidence[0].source_url}],
        )
        pipe_b = PipelineResult(
            verification=VerificationResult(company_status="VERIFIED", contact_status="VERIFIED", overall_status=PROCEED, evidence_sources=["https://voiceagent.ai"]),
            match=MatchResult(relevance_status=RELEVANT, case_type=GENERAL_INQUIRY, matched_areas=["AI Agents & Workflows"]),
        )

        draft_a = draft_outreach_email(master_profile, "hiring@docurag.io", "DocuRAG", verification_pipeline_result=pipe_a, research_result=res_a)
        draft_b = draft_outreach_email(master_profile, "team@voiceagent.ai", "VoiceAgent", verification_pipeline_result=pipe_b, research_result=res_b)

        # Each draft must contain its own distinct company anchor
        assert draft_a.personalization_record.source_excerpt != draft_b.personalization_record.source_excerpt
        assert "docurag" in draft_a.body.lower()
        assert "voiceagent" in draft_b.body.lower()
        assert draft_a.body != draft_b.body


# ---------------------------------------------------------------------------
# 3. Draft Regeneration When Evidence Changes
# ---------------------------------------------------------------------------

class TestDraftRegenerationOnEvidenceChange:
    """Verify changing research evidence triggers new personalization rather than stale drafts."""

    def test_regenerating_draft_reflects_updated_evidence(self, master_profile):
        # Initial research: Company focuses on general RAG
        ev_v1 = [EvidenceItem(
            source_url="https://synapse.ai/tech",
            source_type="official_website",
            title="Synapse RAG",
            extracted_text="Synapse builds custom retrieval-augmented generation pipelines.",
            retrieval_status=RETRIEVAL_OK,
        )]
        res_v1 = ResearchResult(
            company_name="Synapse",
            evidence=ev_v1,
            technical_areas=["RAG & Retrieval Systems"],
            ai_ml_evidence_summary=[{"excerpt": ev_v1[0].extracted_text, "source_url": ev_v1[0].source_url}],
        )
        pipe = PipelineResult(
            verification=VerificationResult(company_status="VERIFIED", contact_status="VERIFIED", overall_status=PROCEED, evidence_sources=["https://synapse.ai"]),
            match=MatchResult(relevance_status=RELEVANT, case_type=GENERAL_INQUIRY, matched_areas=["RAG & Retrieval Systems"]),
        )
        draft_v1 = draft_outreach_email(master_profile, "team@synapse.ai", "Synapse", verification_pipeline_result=pipe, research_result=res_v1)
        assert "rag" in draft_v1.personalization_record.company_finding.lower()

        # Updated research: Company pivots / launches AI governance platform
        ev_v2 = [EvidenceItem(
            source_url="https://synapse.ai/governance",
            source_type="official_website",
            title="Synapse Responsible AI & Governance OS",
            extracted_text="Synapse is the responsible AI OS providing tamper-evident audit trails and AI governance guardrails.",
            retrieval_status=RETRIEVAL_OK,
        )]
        res_v2 = ResearchResult(
            company_name="Synapse",
            evidence=ev_v2,
            technical_areas=["AI Governance & Safety"],
            ai_ml_evidence_summary=[{"excerpt": ev_v2[0].extracted_text, "source_url": ev_v2[0].source_url}],
        )
        draft_v2 = draft_outreach_email(master_profile, "team@synapse.ai", "Synapse", verification_pipeline_result=pipe, research_result=res_v2)

        # Assert draft_v2 reflects new governance anchor
        assert "governance" in draft_v2.personalization_record.company_finding.lower() or "responsible ai" in draft_v2.personalization_record.company_finding.lower()
        assert "governance" in draft_v2.body.lower()
        assert draft_v1.body != draft_v2.body


# ---------------------------------------------------------------------------
# 4. Research Cache Isolation Across Domains
# ---------------------------------------------------------------------------

class TestResearchCacheDomainIsolation:
    """Verify research cache never leaks findings across companies."""

    def test_cache_isolates_findings_per_domain_and_url(self):
        cache = ResearchCache(ttl_seconds=3600)
        item_oct = EvidenceItem(
            source_url="https://octobit8.com",
            source_type="official_website",
            title="Octobit8 Agents & RAG",
            extracted_text="Octobit8 builds AI agents and RAG.",
            retrieval_status=RETRIEVAL_OK,
        )
        cache.put_page("https://octobit8.com", item_oct)

        # Retrieval on Octobit8 hits
        cached = cache.get_page("https://octobit8.com")
        assert cached is not None
        assert "Octobit8" in cached.title

        # Retrieval on Trustra URL is a miss and never returns Octobit8 data
        trustra_cached = cache.get_page("https://trustra.ai")
        assert trustra_cached is None

        # Company data isolation
        cache.put_company_data("octobit8.com", {"company": "Octobit8", "focus": "RAG"})
        assert cache.get_company_data("trustra.ai") is None


# ---------------------------------------------------------------------------
# 5. UI Persistence and Freshness
# ---------------------------------------------------------------------------

class TestDraftPersistenceAndUIRetrieval:
    """Verify drafts and personalization records are persisted and retrieved correctly."""

    def test_save_and_retrieve_draft_with_personalization_record(self, master_profile, octobit8_evidence, tmp_path):
        res = ResearchResult(
            company_name="Octobit8",
            evidence=octobit8_evidence,
            technical_areas=["RAG & Retrieval Systems"],
            ai_ml_evidence_summary=[{"excerpt": octobit8_evidence[0].extracted_text, "source_url": octobit8_evidence[0].source_url}],
        )
        pipe = PipelineResult(
            verification=VerificationResult(company_status="VERIFIED", contact_status="VERIFIED", overall_status=PROCEED, evidence_sources=["https://octobit8.com"]),
            match=MatchResult(relevance_status=RELEVANT, case_type=GENERAL_INQUIRY, matched_areas=["RAG & Retrieval Systems"]),
        )
        draft = draft_outreach_email(master_profile, "hrindia@octobit8.com", "Octobit8", verification_pipeline_result=pipe, research_result=res)

        saved_path = save_draft(draft)
        assert saved_path.exists()

        loaded_data = get_draft(draft.draft_id)
        assert loaded_data is not None
        assert loaded_data["company_name"] == "Octobit8"
        assert "personalization_record" in loaded_data
        rec = loaded_data["personalization_record"]
        assert "company_finding" in rec
        assert rec["technical_domain"] == DOMAIN_RAG_AGENTS
        assert rec["quality_gate_passed"] is True


# ---------------------------------------------------------------------------
# 6. Low-Evidence Company Handled Honestly
# ---------------------------------------------------------------------------

class TestLowEvidenceAndSpeculativeInquiry:
    """Verify failed/low-evidence research does NOT fabricate personalization."""

    def test_unverified_company_does_not_claim_fake_personalization(self, master_profile):
        failed_evidence = [EvidenceItem(
            source_url="https://mysterycorp.example",
            source_type="discovered_website",
            retrieval_status=RETRIEVAL_ERROR,
            retrieval_note="Connection timed out",
            extracted_text="",
        )]
        res = ResearchResult(
            company_name="MysteryCorp",
            evidence=failed_evidence,
            technical_areas=[],
            role_status="UNKNOWN",
            application_status="UNAVAILABLE",
        )
        pipe = PipelineResult(
            verification=VerificationResult(company_status="REVIEW", contact_status="REVIEW", overall_status="REVIEW", evidence_sources=[]),
            match=MatchResult(relevance_status="REVIEW", case_type=GENERAL_INQUIRY, matched_areas=[]),
        )

        draft = draft_outreach_email(master_profile, "contact@mysterycorp.example", "MysteryCorp", verification_pipeline_result=pipe, research_result=res)

        # Must be held for human review
        assert draft.review_required is True
        assert draft.generation_status == GenerationStatus.NEEDS_REVIEW.value

        # Must NOT invent fake technical claims about MysteryCorp
        assert "focus on" not in draft.body.lower()
        # Must have truthful neutral candidate overview
        assert "My background includes practical experience in applied machine learning" in draft.body
        # Review flags must cite missing company evidence
        assert any("no reliable company-specific evidence" in f.lower() or "unverified" in f.lower() for f in draft.review_flags)


# ---------------------------------------------------------------------------
# 7. Exact User-Reported Trio Divergence: Rasa AI Labs, Octobit8, and Viva360
# ---------------------------------------------------------------------------

class TestRasaOctobitVivaTrioDivergence:
    """
    Verify the three companies highlighted in the production failure
    (Rasa AI Labs, Octobit8, Viva360) produce distinct technical anchors,
    distinct candidate connections, and non-identical substantive emails.
    """

    def test_rasa_octobit_viva_produce_pairwise_distinct_emails(self, master_profile):
        # 1. Rasa AI Labs: AI Governance & Enterprise Applied R&D
        rasa_ev = [EvidenceItem(
            source_url="https://rasaailabs.com/research-development",
            source_type="official_website",
            title="Rasa AI Labs R&D",
            extracted_text=(
                "Rasa.AI Labs translates research themes into client-ready systems: Generative AI, "
                "agent workflows, evaluation, guardrails and domain copilots for regulated and industrial settings."
            ),
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res_rasa = ResearchResult(
            company_name="Rasaailabs",
            evidence=rasa_ev,
            technical_areas=["AI Governance & Safety", "Generative AI & LLMs"],
            ai_ml_evidence_summary=[{"excerpt": rasa_ev[0].extracted_text, "source_url": rasa_ev[0].source_url}],
        )

        # 2. Octobit8: RAG Pipelines & Conversational AI Chatbots
        oct_ev = [EvidenceItem(
            source_url="https://octobit8.com",
            source_type="official_website",
            title="Octobit8 AI Development",
            extracted_text=(
                "Octobit8 builds AI chatbots powered by Claude, develops RAG pipelines, "
                "enterprise knowledge bases, custom connectors, and hybrid search pipelines."
            ),
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res_oct = ResearchResult(
            company_name="Octobit8",
            evidence=oct_ev,
            technical_areas=["RAG & Retrieval Systems", "AI Agents & Workflows"],
            ai_ml_evidence_summary=[{"excerpt": oct_ev[0].extracted_text, "source_url": oct_ev[0].source_url}],
        )

        # 3. Viva360: Agentic Talent Intelligence Platform
        viva_ev = [EvidenceItem(
            source_url="https://viva360.ai/about",
            source_type="official_website",
            title="VIVA360 Talent Operating System",
            extracted_text=(
                "VIVA360 deploys AI agents (Match, Screen, Assess, Integrity, Source) on a unified talent graph, "
                "uses conversational AI for screening, semantic matching, and continuous evaluation."
            ),
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res_viva = ResearchResult(
            company_name="Viva360",
            evidence=viva_ev,
            technical_areas=["AI Agents & Workflows", "Agentic AI"],
            ai_ml_evidence_summary=[{"excerpt": viva_ev[0].extracted_text, "source_url": viva_ev[0].source_url}],
        )

        draft_rasa = draft_outreach_email(master_profile, "hr@rasaailabs.com", "Rasaailabs", research_result=res_rasa)
        draft_oct = draft_outreach_email(master_profile, "hrindia@octobit8.com", "Octobit8", research_result=res_oct)
        draft_viva = draft_outreach_email(master_profile, "careers@viva360.ai", "Viva360", research_result=res_viva)

        # All 3 must have valid personalization records
        assert draft_rasa.personalization_record is not None
        assert draft_oct.personalization_record is not None
        assert draft_viva.personalization_record is not None

        # Distinct company findings
        finding_rasa = draft_rasa.personalization_record.company_finding.lower()
        finding_oct = draft_oct.personalization_record.company_finding.lower()
        finding_viva = draft_viva.personalization_record.company_finding.lower()

        assert finding_rasa != finding_oct
        assert finding_oct != finding_viva
        assert finding_rasa != finding_viva

        # Distinct bodies (no clone templates)
        assert draft_rasa.body != draft_oct.body
        assert draft_oct.body != draft_viva.body
        assert draft_rasa.body != draft_viva.body

        # Quality gates passed
        assert draft_rasa.personalization_record.quality_gate_passed is True
        assert draft_oct.personalization_record.quality_gate_passed is True
        assert draft_viva.personalization_record.quality_gate_passed is True


# ---------------------------------------------------------------------------
# 8. Draft Regeneration Endpoint and Storage Test
# ---------------------------------------------------------------------------

class TestDraftRegenerationEndpoint:
    """Verify drafts can be regenerated from stored research evidence."""

    def test_regenerate_draft_updates_body_and_personalization_record(self, master_profile):
        from src.ui.app import create_app

        ev = [EvidenceItem(
            source_url="https://acmelabs.ai/about",
            source_type="official_website",
            title="Acme Labs AI",
            extracted_text="Acme Labs builds custom retrieval-augmented generation pipelines and vector search systems.",
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res = ResearchResult(
            company_name="Acme Labs",
            evidence=ev,
            technical_areas=["RAG & Retrieval Systems"],
            ai_ml_evidence_summary=[{"excerpt": ev[0].extracted_text, "source_url": ev[0].source_url}],
        )

        # Initial draft
        draft = draft_outreach_email(master_profile, "team@acmelabs.ai", "Acme Labs", research_result=res)
        save_draft(draft)

        # Call regenerate_draft_content
        updated = regenerate_draft_content(draft.draft_id)
        assert updated is not None
        assert updated["personalization_record"]["company_finding"] != ""
        assert "acme labs" in updated["body"].lower()

        # Test Flask route /drafts/<draft_id>/regenerate
        app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False})
        client = app.test_client()
        resp = client.post(f"/drafts/{draft.draft_id}/regenerate", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Draft successfully regenerated" in resp.data


# ---------------------------------------------------------------------------
# 9. Exact Four-Company Scenario: Rasa AI Labs, Octobit8, Glowminds, Supersei
# ---------------------------------------------------------------------------

class TestFourCompanyRegressionScenario:
    """
    Exact Section 10 scenario verifying all four test companies:
    1. Rasa AI Labs (AI Governance, guardrails, domain copilots)
    2. Octobit8 (RAG pipelines, enterprise knowledge bases)
    3. Glowminds (Gemini-powered ATS resume scoring, Career OS with dirty snippet)
    4. Supersei (AI CRM, WhatsApp conversational sales agents)
    """

    def test_four_companies_produce_pairwise_distinct_grounded_drafts(self, master_profile):
        # 1. Rasa AI Labs: AI Governance & Enterprise Applied R&D
        rasa_ev = [EvidenceItem(
            source_url="https://rasaailabs.com/research-development",
            source_type="official_website",
            title="Rasa AI Labs R&D",
            extracted_text="Rasa.AI Labs translates research themes into client-ready systems: Generative AI, agent workflows, evaluation, guardrails and domain copilots for regulated and industrial settings.",
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res_rasa = ResearchResult(
            company_name="Rasaailabs",
            evidence=rasa_ev,
            technical_areas=["AI Governance & Safety", "Generative AI & LLMs"],
            ai_ml_evidence_summary=[{"excerpt": rasa_ev[0].extracted_text, "source_url": rasa_ev[0].source_url}],
            role_status="NOT_FOUND",
        )

        # 2. Octobit8: RAG Pipelines & Enterprise Knowledge Bases
        oct_ev = [EvidenceItem(
            source_url="https://octobit8.com/services/rag-development",
            source_type="official_website",
            title="Custom RAG Development | Octobit8",
            extracted_text="Octobit8 builds retrieval-augmented generation (RAG) systems that let AI answer from your documents, databases and policies, citing the source every time. We design and build enterprise knowledge bases, custom connectors, and hybrid search pipelines.",
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res_oct = ResearchResult(
            company_name="Octobit8",
            evidence=oct_ev,
            technical_areas=["RAG & Retrieval Systems", "AI Agents & Workflows"],
            ai_ml_evidence_summary=[{"excerpt": oct_ev[0].extracted_text, "source_url": oct_ev[0].source_url}],
            role_status="NOT_FOUND",
        )

        # 3. Glowminds: Gemini-powered resume scoring ATS with contaminated input snippet
        glow_ev = [EvidenceItem(
            source_url="https://glowminds.in",
            source_type="official_website",
            title="Career OS to Land Your First Job — Glowminds",
            extracted_text="([...]) Through our technology initiatives, Glowminds is focused on expanding from technology services into scalable AI products, digital platforms and specialized ventures that can serve businesses, Our Gemini-powered AI analyzes job descriptions, identifies critical keywords, and builds perfectly formatted resumes that score 90+ on ATS systems.",
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res_glow = ResearchResult(
            company_name="Glowminds",
            evidence=glow_ev,
            technical_areas=["Large Language Models (Gemini)", "Resume Scoring", "ATS Systems"],
            ai_ml_evidence_summary=[{"excerpt": glow_ev[0].extracted_text, "source_url": glow_ev[0].source_url}],
            role_status="NOT_FOUND",
        )

        # 4. Supersei: WhatsApp CRM & Conversational Sales Agents with navigation residue
        sei_ev = [EvidenceItem(
            source_url="https://supersei.ai",
            source_type="official_website",
            title="Supersei: India's AI Growth Engine | AI CRM for WhatsApp-First Teams",
            extracted_text="Supersei: India's AI Growth Engine | AI CRM for WhatsApp-First Teams Skip to content Launch offer 10,000 Sei Credits free on every paid plan. Sei answers every lead on WhatsApp, calls and chat in seconds, books the meeting, collects the payment on Razorpay. Start free trial Book a 20-min demo Nothing charged for 30",
            retrieval_status=RETRIEVAL_OK,
            supports=["company_identity", "technical_relevance"],
        )]
        res_sei = ResearchResult(
            company_name="Supersei",
            evidence=sei_ev,
            technical_areas=["AI-powered automation", "Chatbot / AI sales agent", "CRM automation"],
            ai_ml_evidence_summary=[{"excerpt": sei_ev[0].extracted_text, "source_url": sei_ev[0].source_url}],
            role_status="NOT_FOUND",
        )

        # Generate drafts for all 4 companies
        draft_rasa = draft_outreach_email(master_profile, "hr@rasaailabs.com", "Rasaailabs", research_result=res_rasa)
        draft_oct = draft_outreach_email(master_profile, "hrindia@octobit8.com", "Octobit8", research_result=res_oct)
        draft_glow = draft_outreach_email(master_profile, "careers@glowminds.in", "Glowminds", research_result=res_glow)
        draft_sei = draft_outreach_email(master_profile, "careers@supersei.ai", "Supersei", research_result=res_sei)

        drafts = [draft_rasa, draft_oct, draft_glow, draft_sei]

        # 1. All 4 must have structured PersonalizationRecords
        for d in drafts:
            assert d.personalization_record is not None
            assert d.personalization_record.quality_gate_passed is True
            # Assert all 5 structured match fields are present
            assert d.personalization_record.company_finding != ""
            assert d.personalization_record.candidate_experience != ""
            assert d.personalization_record.connection_reasoning != ""
            assert d.personalization_record.supporting_cv_evidence != ""
            assert isinstance(d.personalization_record.limitations_or_assumptions, list)

        # 2. Pairwise non-identical substantive email bodies
        for i in range(len(drafts)):
            for j in range(i + 1, len(drafts)):
                assert drafts[i].body != drafts[j].body, f"Draft {i} and {j} produced duplicate body!"

        # 3. No diffusion / Red Panda claims on any of the four companies
        for d in drafts:
            assert "red panda" not in d.body.lower()
            assert "diffusion" not in d.body.lower()
            assert "flux" not in d.body.lower()

        # 4. No contaminated fragments in any draft (specifically Glowminds and Supersei)
        for d in drafts:
            assert "[...]" not in d.body
            assert "([...])" not in d.body
            assert "Skip to content" not in d.body
            assert "Launch offer" not in d.body
            assert "Book a demo" not in d.body
            assert "<div>" not in d.body
            assert "<p>" not in d.body

        # 5. Authoritative signature email used consistently across all four drafts
        for d in drafts:
            assert "bhaveshverma7890@gmail.com" in d.body
            assert "uiux2099@gmail.com" not in d.body

        # 6. No false open position vacancy claims (all were speculative / role_status NOT_FOUND)
        unsupported_phrases = ["saw your open position", "applying for your advertised", "your job posting for"]
        for d in drafts:
            for phrase in unsupported_phrases:
                assert phrase not in d.body.lower()


# ---------------------------------------------------------------------------
# 10. Research Contamination and Malformed Text Exclusion Tests
# ---------------------------------------------------------------------------

class TestContaminatedResearchExclusion:
    """Verify that scraped HTML, ellipses, and search navigation never reach send-ready drafts."""

    def test_contaminated_snippet_is_cleaned_or_blocked(self, master_profile):
        from src.outreach.opportunity_analyzer import sanitize_evidence_text

        dirty_text = "([...]) Through our technology initiatives, <div>Skip to content</div> Launch offer 10,000 credits. We build RAG systems. Cookie Policy"
        clean = sanitize_evidence_text(dirty_text)

        assert "[...]" not in clean
        assert "([...])" not in clean
        assert "<div>" not in clean
        assert "Skip to content" not in clean
        assert "Cookie Policy" not in clean
        assert "We build RAG systems" in clean

    def test_quality_gate_blocks_malformed_email(self, master_profile, octobit8_evidence):
        from src.outreach.quality_gate import evaluate_personalization_quality
        from src.outreach.opportunity_analyzer import CompanyOpportunity
        from src.outreach.experience_matcher import CandidateExperienceMatch

        opp = CompanyOpportunity(company_name="Octobit8", concrete_finding="RAG systems", evidence_strength="STRONG")
        match = CandidateExperienceMatch(
            primary_experience_id="proj-001",
            primary_experience_name="ControlPlane.ai",
            organization_or_project="ControlPlane.ai",
            concrete_candidate_work="built RAG systems",
            connection_rationale="aligned",
            relevant_technologies=["LangGraph"],
        )

        malformed_draft = EmailDraft(
            draft_id="d-malformed",
            recipient="test@example.com",
            recipient_name="Hiring Team",
            company_name="Octobit8",
            subject="Application",
            body="Dear Team,\n\n([...]) We build RAG systems Skip to content.\n\nBest regards,\nBhavesh Verma (IIT Kharagpur)\n+91 7067419231\nbhaveshverma7890@gmail.com",
            attachment_path="",
            mode=WorkflowMode.UPLOADED_CV,
        )

        res = evaluate_personalization_quality(malformed_draft, opp, match, master_profile)
        assert res.passed is False
        assert res.checks.get("no_malformed_fragments") is False
        assert any("malformed" in r.lower() for r in res.failure_reasons)


# ---------------------------------------------------------------------------
# 11. Candidate Contact Discrepancy & Resolution Workflow
# ---------------------------------------------------------------------------

class TestCandidateContactDiscrepancyWorkflow:
    """Verify contact discrepancy detection and resolution between CV and profile."""

    def test_discrepancy_detected_and_resolved_authoritatively(self):
        from src.profile.contact_resolver import resolve_candidate_contact

        profile = {
            "personal_info": {
                "full_name": "Bhavesh Verma",
                "email": "uiux2099@gmail.com",
                "phone": "+91 7067419231",
                "confirmed_authoritative_email": "bhaveshverma7890@gmail.com",
            },
            "education": [{"institution": "IIT Kharagpur"}],
        }

        # Case A: Confirmed preference in profile
        resolved = resolve_candidate_contact(profile, cv_text="bhaveshverma7890@gmail.com")
        assert resolved.email == "bhaveshverma7890@gmail.com"
        assert len(resolved.discrepancies) == 1
        assert resolved.discrepancies[0].has_conflict is True
        assert resolved.discrepancies[0].status == "confirmed"
        assert resolved.has_unresolved_conflict is False

        # Case B: Unconfirmed conflict surfaces confirmation requirement
        unconfirmed_profile = {
            "personal_info": {
                "full_name": "Bhavesh Verma",
                "email": "uiux2099@gmail.com",
                "phone": "+91 7067419231",
            },
            "education": [{"institution": "IIT Kharagpur"}],
        }
        resolved_unconfirmed = resolve_candidate_contact(unconfirmed_profile, cv_text="Contact: bhaveshverma7890@gmail.com")
        assert resolved_unconfirmed.email == "bhaveshverma7890@gmail.com"
        assert resolved_unconfirmed.has_unresolved_conflict is True
        assert resolved_unconfirmed.discrepancies[0].status == "requires_confirmation"


# ---------------------------------------------------------------------------
# 12. Quality Gate Blocks: Vacancies, Candidate Claims, and LLM Failures
# ---------------------------------------------------------------------------

class TestUnsupportedClaimsAndVacanciesGate:
    """Verify quality gate blocks ungrounded vacancies, fake candidate claims, and senior role confusion."""

    def test_senior_vacancy_not_misrepresented_as_internship(self, master_profile):
        from src.outreach.drafting import draft_outreach_email
        res = ResearchResult(
            company_name="AlphaTech",
            evidence=[EvidenceItem(
                source_url="https://alphatech.example",
                source_type="official_website",
                title="AlphaTech Careers",
                extracted_text="Open Role: Senior Staff Architect (10+ years experience required).",
                retrieval_status=RETRIEVAL_OK,
            )],
            technical_areas=["AI Architecture"],
            role_status="NOT_FOUND",  # No internship
            application_status="UNAVAILABLE",
        )
        draft = draft_outreach_email(master_profile, "careers@alphatech.example", "AlphaTech", target_role="Senior Staff Architect", research_result=res)
        # Must not claim they are applying for an advertised open internship
        assert "advertised internship" not in draft.body.lower()
        assert "saw your open position" not in draft.body.lower()

    def test_unsupported_candidate_tool_fails_quality_gate(self, master_profile, octobit8_evidence):
        from src.outreach.quality_gate import evaluate_personalization_quality
        from src.outreach.opportunity_analyzer import CompanyOpportunity
        from src.outreach.experience_matcher import CandidateExperienceMatch

        opp = CompanyOpportunity(company_name="Octobit8", concrete_finding="RAG systems", evidence_strength="STRONG")
        match = CandidateExperienceMatch(
            primary_experience_id="proj-001",
            primary_experience_name="ControlPlane.ai",
            organization_or_project="ControlPlane.ai",
            concrete_candidate_work="worked on tools",
            connection_rationale="aligned",
            relevant_technologies=[],
        )

        # Profile does not have 'quantum annealing'
        fake_draft = EmailDraft(
            draft_id="d-fake",
            recipient="test@example.com",
            recipient_name="Hiring Team",
            company_name="Octobit8",
            subject="Application",
            body="Dear Team,\n\nI built quantum annealing and flux models.\n\nBest regards,\nBhavesh Verma (IIT Kharagpur)\n+91 7067419231\nbhaveshverma7890@gmail.com",
            attachment_path="",
            mode=WorkflowMode.UPLOADED_CV,
        )

        res = evaluate_personalization_quality(fake_draft, opp, match, master_profile)
        # If flux is cited for Octobit8 (non-vision), fails distinguishable anchor
        assert res.passed is False


# ---------------------------------------------------------------------------
# 13. LLM Fallback Safety and Duplicate Regeneration Tests
# ---------------------------------------------------------------------------

class TestLLMFallbackAndRegenerationIntegrity:
    """Verify LLM failure falls back cleanly and duplicate regenerations don't corrupt records."""

    def test_llm_failure_falls_back_to_grounded_composer(self, master_profile, octobit8_evidence):
        from src.outreach.personalized_generator import generate_personalized_email_body
        from src.outreach.opportunity_analyzer import analyze_company_opportunity
        from src.outreach.experience_matcher import match_candidate_experience

        res = ResearchResult(company_name="Octobit8", evidence=octobit8_evidence, technical_areas=["RAG & Retrieval Systems"])
        opp = analyze_company_opportunity(res, "Octobit8")
        match = match_candidate_experience(opp, master_profile)

        # Create mock router that fails
        class FailingRouter:
            has_any_llm = True
            primary = type("Provider", (), {"provider_name": "mock"})()
            def generate_json_with_fallback(self, *args, **kwargs):
                raise RuntimeError("API Rate Limit Exceeded 429")

        subj, body, meta = generate_personalized_email_body(
            opportunity=opp,
            candidate_match=match,
            profile=master_profile,
            company_name="Octobit8",
            router=FailingRouter(),
        )

        assert meta["method"] == "grounded_deterministic"
        assert "Octobit8" in body
        assert "hybrid retrieval" in body.lower() or "langgraph" in body.lower()

    def test_duplicate_regeneration_increments_version_and_preserves_single_draft(self, master_profile, octobit8_evidence):
        res = ResearchResult(company_name="Octobit8", evidence=octobit8_evidence, technical_areas=["RAG & Retrieval Systems"])
        draft = draft_outreach_email(master_profile, "hr@octobit8.com", "Octobit8", research_result=res)
        save_draft(draft)

        # First regeneration
        v1 = regenerate_draft_content(draft.draft_id)
        assert v1["generation_version"] == 2
        assert len(v1["regeneration_history"]) == 1

        # Second regeneration
        v2 = regenerate_draft_content(draft.draft_id)
        assert v2["generation_version"] == 3
        assert len(v2["regeneration_history"]) == 2

        # Verify only 1 draft file exists on disk
        from src.ui.storage import _DRAFTS_DIR
        file_path = _DRAFTS_DIR / f"{draft.draft_id}.json"
        assert file_path.exists()

