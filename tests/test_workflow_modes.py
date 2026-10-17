"""
tests/test_workflow_modes.py
----------------------------
Tests for workflow mode selection and Uploaded CV Mode safeguards.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.contacts.csv_importer import ContactRecord
from src.contacts.email_parser import parse_email
from src.outreach.models import WorkflowMode
from src.profile.loader import load_profile
from src.research.models import (
    APPLICATION_OPEN,
    EvidenceItem,
    FOUND,
    ResearchResult,
    SUPPORTED,
)
from src.workflow.engine import OutreachWorkflow


class MockResearchProvider:
    """Mock research provider returning verified company and contact evidence."""
    def research(self, inp):
        return ResearchResult(
            company_identity_status=SUPPORTED,
            domain_relationship_status=SUPPORTED,
            role_status=FOUND,
            application_status=APPLICATION_OPEN,
            technical_areas=["AI/ML", "Generative AI"],
            evidence=[
                EvidenceItem(
                    source_url="https://example.com",
                    source_type="official_website",
                    title="Apex AI Labs",
                    extracted_text="Apex AI Labs builds cutting-edge Generative AI and ML solutions.",
                )
            ],
        )


@pytest.fixture
def sample_contact():
    return ContactRecord(
        row_number=2,
        company_name="Apex AI Labs",
        contact_name="Dr. Jane Smith",
        email_result=parse_email("jane.smith@example.com"),
        website="https://example.com",
        role_or_job_url="AI/ML Intern",
        source_url="https://example.com/careers",
        notes="Met at career symposium",
    )


@pytest.fixture
def dummy_uploaded_cv(tmp_path):
    cv_file = tmp_path / "original_resume.pdf"
    content = b"%PDF-1.4\n% DUMMY CV CONTENT FOR TESTING\nBhavesh Verma IIT Kharagpur AI/ML"
    cv_file.write_bytes(content)
    return cv_file


class TestModeSelection:
    """Tests for mode selection defaults and switching."""

    def test_default_mode_is_uploaded_cv(self):
        workflow = OutreachWorkflow()
        assert workflow.mode == WorkflowMode.UPLOADED_CV

    def test_switch_to_tailored_cv_mode(self):
        workflow = OutreachWorkflow()
        workflow.set_mode(WorkflowMode.TAILORED_CV)
        assert workflow.mode == WorkflowMode.TAILORED_CV

    def test_switch_back_to_uploaded_cv_mode(self):
        workflow = OutreachWorkflow(default_mode=WorkflowMode.TAILORED_CV)
        assert workflow.mode == WorkflowMode.TAILORED_CV
        workflow.set_mode(WorkflowMode.UPLOADED_CV)
        assert workflow.mode == WorkflowMode.UPLOADED_CV


class TestUploadedCVModeSafeguards:
    """Tests for Uploaded CV Mode: byte preservation and isolation."""

    def test_uploaded_cv_remains_byte_for_byte_unchanged(
        self, dummy_uploaded_cv, sample_contact
    ):
        original_bytes = dummy_uploaded_cv.read_bytes()
        original_hash = hashlib.sha256(original_bytes).hexdigest()

        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_uploaded_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )

        result = workflow.process_contact(
            contact=sample_contact,
            job_description="Looking for an AI/ML intern experienced in PyTorch and Generative AI.",
            research_provider=MockResearchProvider(),
        )

        # Assert workflow succeeded
        assert result.status == "PROCEED"
        assert result.draft is not None

        # Assert file is byte-for-byte identical
        current_bytes = dummy_uploaded_cv.read_bytes()
        current_hash = hashlib.sha256(current_bytes).hexdigest()
        assert current_bytes == original_bytes
        assert current_hash == original_hash
        assert workflow.verify_uploaded_cv_unmodified() is True

    def test_uploaded_cv_is_selected_as_attachment(
        self, dummy_uploaded_cv, sample_contact
    ):
        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_uploaded_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )

        result = workflow.process_contact(
            contact=sample_contact,
            job_description="AI Intern position",
            research_provider=MockResearchProvider(),
        )

        assert result.draft is not None
        assert result.draft.attachment_path == str(dummy_uploaded_cv.resolve())
        assert result.draft.tailored_cv_path is None

    def test_cv_generation_and_project_selection_not_invoked_in_uploaded_mode(
        self, dummy_uploaded_cv, sample_contact
    ):
        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_uploaded_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )

        result = workflow.process_contact(
            contact=sample_contact,
            job_description="AI Intern",
            research_provider=MockResearchProvider(),
        )

        # Verify neither project selection nor CV generation was invoked
        assert workflow.project_selection_invoked is False
        assert workflow.cv_generation_invoked is False
        assert result.selection_result is None
        assert result.tailored_cv is None

    def test_uploaded_mode_requires_existing_file(self, sample_contact):
        workflow = OutreachWorkflow(default_mode=WorkflowMode.UPLOADED_CV)
        with pytest.raises(RuntimeError, match="requires an existing CV"):
            workflow.process_contact(
                contact=sample_contact,
                research_provider=MockResearchProvider(),
            )


class TestTailoredCVWorkflowIntegration:
    """Tests for running the complete workflow in Tailored CV Mode."""

    def test_tailored_mode_runs_project_selection_and_cv_generation(
        self, sample_contact, tmp_path
    ):
        workflow = OutreachWorkflow(default_mode=WorkflowMode.TAILORED_CV)

        result = workflow.process_contact(
            contact=sample_contact,
            job_description="Seeking a machine learning engineer with NLP, DeBERTa, and price prediction experience.",
            research_provider=MockResearchProvider(),
            custom_output_dir=tmp_path,
        )

        assert result.status == "PROCEED"
        assert workflow.project_selection_invoked is True
        assert workflow.cv_generation_invoked is True
        assert result.selection_result is not None
        assert result.tailored_cv is not None

        # Verify selected projects: ControlPlane.ai first, E-Commerce Price Prediction second
        selected = result.selection_result.selected_projects
        assert len(selected) == 3
        assert selected[0]["name"].startswith("ControlPlane.ai")
        assert selected[1]["name"] == "E-Commerce Price Prediction"
        assert selected[2]["name"] == "Building a Network Security System"

        # Verify draft attachment is the tailored CV file
        assert result.draft is not None
        assert result.draft.attachment_path == result.tailored_cv.file_path
        assert Path(result.draft.attachment_path).exists()
        assert Path(result.draft.attachment_path).parent == tmp_path

        # Verify email mentions the relevant projects
        assert "ControlPlane.ai" in result.draft.body
        assert "E-Commerce Price Prediction" in result.draft.body

    def test_switching_modes_changes_workflow_behavior(
        self, dummy_uploaded_cv, sample_contact, tmp_path
    ):
        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_uploaded_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )

        # 1. Run in Uploaded CV Mode
        res_uploaded = workflow.process_contact(
            contact=sample_contact,
            job_description="AI Intern",
            research_provider=MockResearchProvider(),
        )
        assert workflow.project_selection_invoked is False
        assert workflow.cv_generation_invoked is False
        assert res_uploaded.draft.attachment_path == str(dummy_uploaded_cv.resolve())

        # 2. Switch to Tailored CV Mode
        workflow.set_mode(WorkflowMode.TAILORED_CV)
        res_tailored = workflow.process_contact(
            contact=sample_contact,
            job_description="AI Intern",
            research_provider=MockResearchProvider(),
            custom_output_dir=tmp_path,
        )
        assert workflow.project_selection_invoked is True
        assert workflow.cv_generation_invoked is True
        assert res_tailored.draft.attachment_path != str(dummy_uploaded_cv.resolve())
        assert res_tailored.draft.attachment_path == res_tailored.tailored_cv.file_path
