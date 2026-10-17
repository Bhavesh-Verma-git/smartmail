"""
tests/test_profile_invalid.py
------------------------------
Tests for error-detection behaviour: missing fields, empty values, bad JSON.
"""

import json
import pytest
from pathlib import Path

from src.profile.loader import load_profile, ProfileLoadError
from src.profile.validator import validate_profile, assert_valid_profile

FIXTURE_DIR = Path(__file__).parent / "fixtures"
INVALID_PROFILE_PATH = FIXTURE_DIR / "invalid_profile.json"


# ---------------------------------------------------------------------------
# Loader error tests
# ---------------------------------------------------------------------------

class TestLoaderErrors:
    """load_profile() raises ProfileLoadError for bad inputs."""

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(ProfileLoadError, match="not found"):
            load_profile(missing)

    def test_malformed_json_raises(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ this is not valid JSON }", encoding="utf-8")
        with pytest.raises(ProfileLoadError, match="invalid JSON"):
            load_profile(bad_json)

    def test_json_array_at_top_level_raises(self, tmp_path):
        array_json = tmp_path / "array.json"
        array_json.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ProfileLoadError, match="object"):
            load_profile(array_json)

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ProfileLoadError, match="invalid JSON"):
            load_profile(empty)


# ---------------------------------------------------------------------------
# Validator error tests — using invalid_profile.json fixture
# ---------------------------------------------------------------------------

class TestValidateInvalidProfile:
    """validate_profile() catches all problems in the invalid fixture."""

    def test_invalid_fixture_loads_without_error(self):
        """The file is valid JSON but semantically incomplete."""
        data = load_profile(INVALID_PROFILE_PATH)
        assert isinstance(data, dict)

    def test_missing_sections_are_reported(self):
        data = load_profile(INVALID_PROFILE_PATH)
        errors = validate_profile(data)
        # Missing top-level keys: experience, technical_skills
        error_text = " ".join(errors)
        assert "experience" in error_text or "technical_skills" in error_text

    def test_empty_full_name_is_reported(self):
        data = load_profile(INVALID_PROFILE_PATH)
        errors = validate_profile(data)
        error_text = " ".join(errors)
        assert "full_name" in error_text

    def test_empty_education_list_is_reported(self):
        data = load_profile(INVALID_PROFILE_PATH)
        errors = validate_profile(data)
        error_text = " ".join(errors)
        assert "education" in error_text

    def test_empty_projects_list_is_reported(self):
        data = load_profile(INVALID_PROFILE_PATH)
        errors = validate_profile(data)
        error_text = " ".join(errors)
        assert "projects" in error_text

    def test_returns_multiple_errors(self):
        data = load_profile(INVALID_PROFILE_PATH)
        errors = validate_profile(data)
        assert len(errors) >= 3, (
            f"Expected at least 3 errors for the invalid fixture, got {len(errors)}: {errors}"
        )

    def test_assert_raises_value_error(self):
        data = load_profile(INVALID_PROFILE_PATH)
        with pytest.raises(ValueError, match="validation failed"):
            assert_valid_profile(data)


# ---------------------------------------------------------------------------
# Validator error tests — inline dicts (fast, no file I/O)
# ---------------------------------------------------------------------------

class TestValidateInlineErrors:
    """Quick inline dict tests — no fixture files needed."""

    def test_completely_empty_dict_reports_all_sections(self):
        errors = validate_profile({})
        error_text = " ".join(errors)
        for section in ("personal_info", "education", "experience", "projects", "technical_skills"):
            assert section in error_text, f"Expected '{section}' to be reported"

    def test_non_dict_profile_is_rejected(self):
        errors = validate_profile("not a dict")  # type: ignore[arg-type]
        assert len(errors) == 1
        assert "dict" in errors[0]

    def test_missing_email_is_reported(self):
        profile = {
            "personal_info": {"full_name": "Test User"},
            "education": [{"institution": "Uni", "degree": "BSc", "start_date": "2020-09", "end_date": "2024-06"}],
            "experience": [],
            "projects": [{"id": "p1", "name": "X", "short_description": "Y",
                          "technologies": ["Python"], "my_contributions": ["wrote it"]}],
            "technical_skills": {"languages": ["Python"]},
        }
        errors = validate_profile(profile)
        error_text = " ".join(errors)
        assert "email" in error_text

    def test_project_missing_id_is_reported(self):
        profile = {
            "personal_info": {"full_name": "A", "email": "a@b.com"},
            "education": [{"institution": "U", "degree": "B", "start_date": "2020-01", "end_date": "2024-01"}],
            "experience": [],
            "projects": [{"name": "No ID Project", "short_description": "X",
                          "technologies": ["Go"], "my_contributions": ["did it"]}],
            "technical_skills": {"languages": ["Go"]},
        }
        errors = validate_profile(profile)
        error_text = " ".join(errors)
        assert "id" in error_text

    def test_experience_missing_role_is_reported(self):
        profile = {
            "personal_info": {"full_name": "A", "email": "a@b.com"},
            "education": [{"institution": "U", "degree": "B", "start_date": "2020-01", "end_date": "2024-01"}],
            "experience": [
                {"id": "e1", "organization": "Corp", "start_date": "2023-01",
                 "end_date": "2023-06", "responsibilities": ["did stuff"]}
                # missing: role
            ],
            "projects": [{"id": "p1", "name": "X", "short_description": "Y",
                          "technologies": ["Python"], "my_contributions": ["wrote it"]}],
            "technical_skills": {"languages": ["Python"]},
        }
        errors = validate_profile(profile)
        error_text = " ".join(errors)
        assert "role" in error_text

    def test_education_empty_string_start_date_is_reported(self):
        profile = {
            "personal_info": {"full_name": "A", "email": "a@b.com"},
            "education": [{"institution": "U", "degree": "B", "start_date": "", "end_date": "2024-01"}],
            "experience": [],
            "projects": [{"id": "p1", "name": "X", "short_description": "Y",
                          "technologies": ["Python"], "my_contributions": ["wrote it"]}],
            "technical_skills": {"languages": ["Python"]},
        }
        errors = validate_profile(profile)
        error_text = " ".join(errors)
        assert "start_date" in error_text

    def test_education_missing_institution_is_reported(self):
        profile = {
            "personal_info": {"full_name": "A", "email": "a@b.com"},
            "education": [{"degree": "B", "end_date": "2024-01"}],
            "experience": [],
            "projects": [{"id": "p1", "name": "X", "short_description": "Y",
                          "technologies": ["Python"], "my_contributions": ["wrote it"]}],
            "technical_skills": {"languages": ["Python"]},
        }
        errors = validate_profile(profile)
        error_text = " ".join(errors)
        assert "institution" in error_text

    def test_education_missing_degree_is_reported(self):
        profile = {
            "personal_info": {"full_name": "A", "email": "a@b.com"},
            "education": [{"institution": "U", "end_date": "2024-01"}],
            "experience": [],
            "projects": [{"id": "p1", "name": "X", "short_description": "Y",
                          "technologies": ["Python"], "my_contributions": ["wrote it"]}],
            "technical_skills": {"languages": ["Python"]},
        }
        errors = validate_profile(profile)
        error_text = " ".join(errors)
        assert "degree" in error_text

    def test_education_missing_end_date_is_reported(self):
        profile = {
            "personal_info": {"full_name": "A", "email": "a@b.com"},
            "education": [{"institution": "U", "degree": "B"}],
            "experience": [],
            "projects": [{"id": "p1", "name": "X", "short_description": "Y",
                          "technologies": ["Python"], "my_contributions": ["wrote it"]}],
            "technical_skills": {"languages": ["Python"]},
        }
        errors = validate_profile(profile)
        error_text = " ".join(errors)
        assert "end_date" in error_text

