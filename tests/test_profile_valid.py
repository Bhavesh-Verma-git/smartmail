"""
tests/test_profile_valid.py
---------------------------
Tests for happy-path behaviour: loading and validating a well-formed profile.
"""

import json
import pytest
from pathlib import Path

from src.profile.loader import load_profile
from src.profile.validator import validate_profile, assert_valid_profile

FIXTURE_DIR = Path(__file__).parent / "fixtures"
VALID_PROFILE_PATH = FIXTURE_DIR / "valid_profile.json"


class TestLoadValidProfile:
    """loader.load_profile() with a valid file."""

    def test_returns_dict(self):
        data = load_profile(VALID_PROFILE_PATH)
        assert isinstance(data, dict)

    def test_has_required_top_level_keys(self):
        data = load_profile(VALID_PROFILE_PATH)
        for key in ("personal_info", "education", "experience", "projects", "technical_skills"):
            assert key in data, f"Expected top-level key '{key}' not found"

    def test_personal_info_fields(self):
        data = load_profile(VALID_PROFILE_PATH)
        assert data["personal_info"]["full_name"] == "Jane Doe"
        assert data["personal_info"]["email"] == "jane.doe@example.com"

    def test_education_is_list_with_one_entry(self):
        data = load_profile(VALID_PROFILE_PATH)
        assert isinstance(data["education"], list)
        assert len(data["education"]) == 1

    def test_experience_entry_has_id(self):
        data = load_profile(VALID_PROFILE_PATH)
        assert data["experience"][0]["id"] == "exp-001"

    def test_project_entry_has_stable_id_and_name(self):
        data = load_profile(VALID_PROFILE_PATH)
        proj = data["projects"][0]
        assert proj["id"] == "proj-001"
        assert proj["name"] == "Awesome CLI Tool"

    def test_project_has_technologies_list(self):
        data = load_profile(VALID_PROFILE_PATH)
        assert isinstance(data["projects"][0]["technologies"], list)
        assert len(data["projects"][0]["technologies"]) > 0

    def test_skills_languages_is_list(self):
        data = load_profile(VALID_PROFILE_PATH)
        assert isinstance(data["technical_skills"]["languages"], list)


class TestValidateValidProfile:
    """validator.validate_profile() with a valid profile."""

    def test_no_errors_on_valid_profile(self):
        data = load_profile(VALID_PROFILE_PATH)
        errors = validate_profile(data)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_assert_valid_profile_does_not_raise(self):
        data = load_profile(VALID_PROFILE_PATH)
        # Should complete without raising
        assert_valid_profile(data)

    def test_accepts_path_object(self):
        data = load_profile(Path(VALID_PROFILE_PATH))
        assert isinstance(data, dict)

    def test_accepts_string_path(self):
        data = load_profile(str(VALID_PROFILE_PATH))
        assert isinstance(data, dict)

    def test_education_with_null_start_date_is_valid(self):
        data = load_profile(VALID_PROFILE_PATH)
        data["education"].append({
            "institution": "Shri Shankaracharya Vidyalaya",
            "degree": "Higher Secondary (CBSE)",
            "start_date": None,
            "end_date": "2022",
        })
        errors = validate_profile(data)
        assert errors == [], f"Expected no errors for null start_date, got: {errors}"

    def test_education_with_omitted_start_date_is_valid(self):
        data = load_profile(VALID_PROFILE_PATH)
        data["education"].append({
            "institution": "Sun Shine Hr. Sec. School",
            "degree": "Secondary (CGBSE)",
            "end_date": "2020",
        })
        errors = validate_profile(data)
        assert errors == [], f"Expected no errors for omitted start_date, got: {errors}"

