"""
tests/test_cv_tailoring.py
--------------------------
Tests for project selection and Tailored CV generation rules.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cv.generator import generate_tailored_cv
from src.cv.selector import (
    ELIGIBLE_SECONDARY_PROJECT_NAMES,
    EXCLUDED_COMPETITION_PROJECT_NAMES,
    MANDATORY_PRIMARY_PROJECT_NAME,
    select_projects,
)
from src.profile.loader import load_profile

MASTER_PROFILE_PATH = Path("data/master_profile.json")


@pytest.fixture
def profile():
    return load_profile(MASTER_PROFILE_PATH)


class TestProjectSelectionRules:
    """Tests enforcing the exact project selection rules."""

    def test_controlplane_always_selected_first(self, profile):
        result = select_projects(
            profile=profile,
            job_description="Cybersecurity and cloud infrastructure role",
        )
        assert len(result.selected_projects) == 3
        first_project = result.selected_projects[0]
        assert first_project["name"].startswith(MANDATORY_PRIMARY_PROJECT_NAME)
        assert result.primary_project["name"].startswith(MANDATORY_PRIMARY_PROJECT_NAME)

    def test_exactly_three_projects_selected(self, profile):
        result = select_projects(
            profile=profile,
            job_description="General machine learning internship",
        )
        assert len(result.selected_projects) == 3

    def test_secondary_slots_use_only_eligible_projects(self, profile):
        result = select_projects(
            profile=profile,
            job_description="General machine learning role",
        )
        second_name = result.selected_projects[1]["name"]
        third_name = result.selected_projects[2]["name"]

        assert second_name in ELIGIBLE_SECONDARY_PROJECT_NAMES
        assert third_name in ELIGIBLE_SECONDARY_PROJECT_NAMES
        assert second_name != third_name

    def test_competitions_never_selected_as_projects(self, profile):
        # Even with a job description mentioning competitions/hackathons
        result = select_projects(
            profile=profile,
            job_description="Hackathon finalist, backstory verification, profitability scoring",
        )
        selected_names = [p["name"] for p in result.selected_projects]
        for excluded in EXCLUDED_COMPETITION_PROJECT_NAMES:
            assert excluded not in selected_names, f"Excluded competition '{excluded}' was selected as project!"

    def test_project_ranking_responds_to_job_relevance_security(self, profile):
        job_desc = "Seeking an intern for network security, phishing URL detection, Docker, ETL, and AWS CI/CD pipelines."
        result = select_projects(
            profile=profile,
            job_description=job_desc,
        )
        assert result.selected_projects[0]["name"].startswith("ControlPlane.ai")
        assert result.selected_projects[1]["name"] == "Building a Network Security System"
        assert result.selected_projects[2]["name"] == "E-Commerce Price Prediction"
        assert result.fallback_used is False
        assert "ranked ahead" in result.ranking_explanation

    def test_project_ranking_responds_to_job_relevance_ecommerce(self, profile):
        job_desc = "Looking for an NLP intern with experience in DeBERTa, regex, retail e-commerce price prediction, and tabular data."
        result = select_projects(
            profile=profile,
            job_description=job_desc,
        )
        assert result.selected_projects[0]["name"].startswith("ControlPlane.ai")
        assert result.selected_projects[1]["name"] == "E-Commerce Price Prediction"
        assert result.selected_projects[2]["name"] == "Building a Network Security System"
        assert result.fallback_used is False
        assert "ranked ahead" in result.ranking_explanation

    def test_insufficient_job_info_triggers_deterministic_fallback(self, profile):
        result = select_projects(
            profile=profile,
            job_description="",  # Empty / insufficient context
            target_role="",
        )
        assert result.selected_projects[0]["name"].startswith("ControlPlane.ai")
        assert result.selected_projects[1]["name"] == "Building a Network Security System"
        assert result.selected_projects[2]["name"] == "E-Commerce Price Prediction"
        assert result.fallback_used is True
        assert len(result.uncertainty_notes) > 0
        assert "fallback" in result.uncertainty_notes[0].lower()


class TestTailoredCVGeneration:
    """Tests for generating the role-specific CV file."""

    def test_generated_cv_is_saved_separately(self, profile, tmp_path):
        res = generate_tailored_cv(
            profile=profile,
            job_description="AI / NLP Intern",
            target_company="Acme AI",
            target_role="Generative AI Intern",
            output_dir=tmp_path,
        )
        cv_path = Path(res.file_path)
        assert cv_path.exists()
        assert cv_path != MASTER_PROFILE_PATH
        assert cv_path.parent == tmp_path

    def test_generated_cv_preserves_both_internships(self, profile, tmp_path):
        res = generate_tailored_cv(
            profile=profile,
            job_description="AI/ML Intern",
            output_dir=tmp_path,
        )
        content = res.content
        assert "Red Panda Games" in content
        assert "Data Science Intern" in content
        assert "IISER Bhopal" in content
        assert "Machine Learning Intern" in content

    def test_generated_cv_preserves_competitions_in_awards_section(self, profile, tmp_path):
        res = generate_tailored_cv(
            profile=profile,
            job_description="AI/ML Intern",
            output_dir=tmp_path,
        )
        content = res.content
        assert "COMPETITIONS & AWARDS" in content
        assert "Kharagpur Data Science Hackathon 2026" in content
        assert "American Express Campus Challenge 2026" in content

    def test_master_profile_json_not_modified_by_cv_generation(self, profile, tmp_path):
        original_mtime = MASTER_PROFILE_PATH.stat().st_mtime
        original_content = MASTER_PROFILE_PATH.read_text(encoding="utf-8")

        generate_tailored_cv(
            profile=profile,
            job_description="AI/ML Intern",
            output_dir=tmp_path,
        )

        current_content = MASTER_PROFILE_PATH.read_text(encoding="utf-8")
        assert current_content == original_content
        assert MASTER_PROFILE_PATH.stat().st_mtime == original_mtime
