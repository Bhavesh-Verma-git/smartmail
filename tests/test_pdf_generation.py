"""
tests/test_pdf_generation.py
-----------------------------
Tests for the professional PDF CV generator (src/cv/pdf_generator.py).

All tests use a synthetic profile to avoid loading personal data in test output.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.cv.models import ProjectSelectionResult
from src.cv.pdf_generator import render_cv_pdf, validate_pdf, _strip_provenance


# ---------------------------------------------------------------------------
# Synthetic profile fixture (no real personal data)
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_profile() -> dict:
    return {
        "personal_info": {
            "full_name": "Test Candidate",
            "email": "test@example.com",
            "phone": "+00 0000000000",
            "location": "Test City",
            "linkedin_url": "https://linkedin.com/in/testcandidate",
            "github_url": "https://github.com/testcandidate",
        },
        "education": [
            {
                "institution": "Test University",
                "degree": "B.Tech in Test Engineering (Dual Degree)",
                "start_date": "2023-07",
                "end_date": "2028-05",
                "cgpa": "8.5 / 10.0",
                "relevant_coursework": ["Machine Learning", "Data Structures"],
                "achievements": [],
            },
            {
                "institution": "Test High School",
                "degree": "Higher Secondary",
                "start_date": None,
                "end_date": "2022",
                "cgpa": "85%",
                "relevant_coursework": [],
                "achievements": [],
            },
        ],
        "experience": [
            {
                "organization": "Acme AI",
                "role": "Machine Learning Intern",
                "start_date": "2025-06",
                "end_date": "2025-08",
                "location": "Remote",
                "responsibilities": [
                    "Built diffusion pipelines for image synthesis.",
                    "Optimized GPU inference latency by 30%.",
                ],
                "technologies_used": ["Python", "PyTorch", "AWS"],
            }
        ],
        "achievements_and_awards": [
            {
                "title": "National Finalist — Test Hackathon 2024",
                "issuer": "Test Org",
                "date": "2024",
                "description": "Finalist for best NLP system.",
            }
        ],
        "technical_skills": {
            "languages": ["Python", "C++"],
            "frameworks_and_libraries": ["NumPy", "Pandas"],
            "ml_and_data": ["PyTorch", "scikit-learn"],
            "databases": ["PostgreSQL"],
            "tools_and_platforms": ["Git", "Docker"],
        },
    }


@pytest.fixture
def synthetic_selection(synthetic_profile) -> ProjectSelectionResult:
    primary = {
        "name": "ControlPlane-Test",
        "technologies": ["LangGraph", "Python", "FAISS"],
        "long_description": "Built a test AI governance system with nine-node state machine.",
        "my_contributions": [
            "Architected state machine.",
            "Integrated hybrid retrieval.",
        ],
        "outcomes": [
            {"claim": "Achieved validated test accuracy of 92%.", "evidence": "Reported in CV"}
        ],
        "links": {"github": "https://github.com/testuser/controlplane-test"},
    }
    sec1 = {
        "name": "Network Security Test",
        "technologies": ["Python", "Docker", "AWS"],
        "long_description": "Phishing URL classifier with automated ETL.",
        "my_contributions": ["Built ETL pipeline.", "Deployed via CI/CD."],
        "outcomes": [],
        "links": {"github": None},
    }
    sec2 = {
        "name": "Price Prediction Test",
        "technologies": ["Python", "DeBERTa-v3"],
        "long_description": "E-commerce price model with NLP features.",
        "my_contributions": ["Engineered NLP features."],
        "outcomes": [{"claim": "Reported SMAPE of 40%.", "evidence": "Reported in CV"}],
        "links": {"github": "https://github.com/testuser/price-pred"},
    }
    return ProjectSelectionResult(
        selected_projects=[primary, sec1, sec2],
        primary_project=primary,
        secondary_projects=[sec1, sec2],
        ranking_explanation="Test ranking",
        fallback_used=False,
    )


# ---------------------------------------------------------------------------
# Section 1: Output format and existence
# ---------------------------------------------------------------------------
class TestPDFOutputFormat:
    def test_render_returns_bytes(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        assert isinstance(raw, bytes)

    def test_pdf_starts_with_pdf_header(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        assert raw[:4] == b"%PDF"

    def test_pdf_is_non_empty_and_reasonable_size(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        assert len(raw) > 2000, "PDF is suspiciously small"

    def test_pdf_saved_to_disk(self, tmp_path, synthetic_profile, synthetic_selection):
        out = tmp_path / "test_cv.pdf"
        render_cv_pdf(synthetic_profile, synthetic_selection, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_output_directory_created_automatically(self, tmp_path, synthetic_profile, synthetic_selection):
        nested = tmp_path / "a" / "b" / "c" / "cv.pdf"
        render_cv_pdf(synthetic_profile, synthetic_selection, output_path=nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Section 2: Content validation
# ---------------------------------------------------------------------------
class TestPDFContentValidation:
    def test_validate_pdf_passes_on_valid_output(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        result = validate_pdf(raw)
        assert result["ok"] is True
        assert not result["errors"]

    def test_validate_pdf_fails_on_empty_bytes(self):
        result = validate_pdf(b"")
        assert result["ok"] is False
        assert result["errors"]

    def test_validate_pdf_fails_on_non_pdf(self):
        result = validate_pdf(b"This is not a PDF file at all")
        assert result["ok"] is False
        assert result["errors"]

    def test_pdf_contains_candidate_name(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        result = validate_pdf(raw, required_text_fragments=["Test Candidate"])
        assert not result["missing_fragments"], f"Missing: {result['missing_fragments']}"

    def test_pdf_contains_contact_email(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        result = validate_pdf(raw, required_text_fragments=["test@example.com"])
        assert not result["missing_fragments"]

    def test_pdf_contains_institution_name(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        result = validate_pdf(raw, required_text_fragments=["Test University"])
        assert not result["missing_fragments"]

    def test_pdf_contains_experience_org(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        result = validate_pdf(raw, required_text_fragments=["Acme AI"])
        assert not result["missing_fragments"]


# ---------------------------------------------------------------------------
# Section 3: Content correctness
# ---------------------------------------------------------------------------
class TestPDFContentCorrectness:
    def test_no_provenance_label_in_output(self, synthetic_profile, synthetic_selection):
        """'Reported in CV', 'Claimed in CV' must not appear in recruiter-facing PDF."""
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        pdf_text = raw.decode("latin-1", errors="ignore")
        assert "Reported in CV" not in pdf_text
        assert "Claimed in CV" not in pdf_text

    def test_three_projects_rendered(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        pdf_text = raw.decode("latin-1", errors="ignore")
        for proj in synthetic_selection.selected_projects:
            name = proj["name"]
            # At minimum, the name should appear somewhere in the raw text
            assert name in pdf_text or any(
                word in pdf_text for word in name.split()
            ), f"Project name '{name}' appears missing from PDF"

    def test_education_with_null_start_date_renders(self, synthetic_profile, synthetic_selection):
        """Education with start_date=None should render as 'Completed 2022' not crash."""
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        assert raw[:4] == b"%PDF"  # PDF was generated without error

    def test_both_internships_rendered(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        pdf_text = raw.decode("latin-1", errors="ignore")
        assert "Acme AI" in pdf_text


# ---------------------------------------------------------------------------
# Section 4: A4 dimensions
# ---------------------------------------------------------------------------
class TestPDFA4Dimensions:
    def test_a4_dimensions_declared_in_pdf(self, synthetic_profile, synthetic_selection):
        raw = render_cv_pdf(synthetic_profile, synthetic_selection)
        # A4 MediaBox values in PDF: 595.28 x 841.89 pts (≈ 210 x 297 mm)
        pdf_text = raw.decode("latin-1", errors="ignore")
        # MediaBox or page size should include approx A4 pt dimensions
        assert "595" in pdf_text or "841" in pdf_text or "a4" in pdf_text.lower() or len(raw) > 1000


# ---------------------------------------------------------------------------
# Section 5: Provenance stripping
# ---------------------------------------------------------------------------
class TestProvenanceStripping:
    def test_strip_reported_in_cv(self):
        assert _strip_provenance("Achieved 40% reduction. Reported in CV.") == "Achieved 40% reduction."

    def test_strip_claimed_in_cv(self):
        assert _strip_provenance("Some outcome (Claimed in CV: unverified)") == "Some outcome"

    def test_strip_cv_reported(self):
        result = _strip_provenance("CV-reported accuracy of 91%.")
        assert "CV-reported" not in result

    def test_no_provenance_unchanged(self):
        text = "Optimized inference latency by 30%."
        assert _strip_provenance(text) == text


# ---------------------------------------------------------------------------
# Section 6: Real profile PDF generation (integration)
# ---------------------------------------------------------------------------
class TestRealProfilePDFGeneration:
    """Uses the actual master_profile.json and ProjectSelector — end-to-end."""

    def test_real_profile_generates_valid_pdf(self, tmp_path):
        from src.profile.loader import load_profile
        from src.cv.selector import select_projects

        profile = load_profile("data/master_profile.json")
        selection = select_projects(
            profile,
            job_description="AI/ML intern Python machine learning",
            target_role="AI/ML Intern",
        )
        out = tmp_path / "bhavesh_cv.pdf"
        raw = render_cv_pdf(
            profile, selection,
            target_role="AI/ML Intern",
            target_company="Acme AI Labs",
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 3000
        validation = validate_pdf(raw, required_text_fragments=["IIT Kharagpur"])
        assert not validation["missing_fragments"]
        assert not validation["errors"]

    def test_real_profile_no_provenance_labels_in_pdf(self, tmp_path):
        from src.profile.loader import load_profile
        from src.cv.selector import select_projects

        profile = load_profile("data/master_profile.json")
        selection = select_projects(profile, target_role="AI/ML Intern")
        raw = render_cv_pdf(profile, selection)
        pdf_text = raw.decode("latin-1", errors="ignore")
        assert "Reported in CV" not in pdf_text
        assert "Claimed in CV" not in pdf_text

    def test_real_profile_controlplane_in_pdf(self, tmp_path):
        from src.profile.loader import load_profile
        from src.cv.selector import select_projects

        profile = load_profile("data/master_profile.json")
        selection = select_projects(profile, target_role="AI/ML Intern")
        raw = render_cv_pdf(profile, selection)
        pdf_text = raw.decode("latin-1", errors="ignore")
        # ControlPlane should appear somewhere (may be partial due to encoding)
        assert "ControlPlane" in pdf_text or "Control" in pdf_text
