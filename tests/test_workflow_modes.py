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

    def test_tailored_mode_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Tailored-CV mode was removed"):
            OutreachWorkflow(default_mode=WorkflowMode.TAILORED_CV)

    def test_set_mode_rejects_tailored(self):
        workflow = OutreachWorkflow()
        with pytest.raises(ValueError, match="Tailored-CV mode was removed"):
            workflow.set_mode(WorkflowMode.TAILORED_CV)
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

    def test_uploaded_mode_attachment_is_the_uploaded_file_and_no_generated_file(
        self, dummy_uploaded_cv, sample_contact, tmp_path
    ):
        workflow = OutreachWorkflow(
            uploaded_cv_path=dummy_uploaded_cv,
            default_mode=WorkflowMode.UPLOADED_CV,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = workflow.process_contact(
            contact=sample_contact,
            job_description="AI Intern",
            research_provider=MockResearchProvider(),
            custom_output_dir=out_dir,
        )
        assert result.draft is not None
        assert result.draft.attachment_path == str(dummy_uploaded_cv.resolve())
        # No generated CV file may appear in the output dir
        assert list(out_dir.glob("*.pdf")) == []

    def test_uploaded_mode_requires_existing_file(self, sample_contact):
        workflow = OutreachWorkflow(default_mode=WorkflowMode.UPLOADED_CV)
        with pytest.raises(RuntimeError, match="requires an existing CV"):
            workflow.process_contact(
                contact=sample_contact,
                research_provider=MockResearchProvider(),
            )
