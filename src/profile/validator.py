"""
src/profile/validator.py
------------------------
Validates a loaded profile dict against a set of required-field rules.

Design notes:
- Pure functions only — no side effects, easy to test.
- Returns a list of human-readable error strings so the caller decides how to
  surface them (print, log, raise, etc.).
- Each section (personal_info, education, experience, projects) is validated
  separately so you get ALL errors at once, not just the first one.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Required field definitions
# ---------------------------------------------------------------------------

# Top-level sections that must exist and be non-empty lists or dicts
_REQUIRED_TOP_LEVEL_KEYS: list[str] = [
    "personal_info",
    "education",
    "experience",
    "projects",
    "technical_skills",
]

_REQUIRED_PERSONAL_INFO_KEYS: list[str] = [
    "full_name",
    "email",
]

_REQUIRED_EDUCATION_KEYS: list[str] = [
    "institution",
    "degree",
    "end_date",
]

_REQUIRED_EXPERIENCE_KEYS: list[str] = [
    "id",
    "organization",
    "role",
    "start_date",
    "end_date",
    "responsibilities",
]

_REQUIRED_PROJECT_KEYS: list[str] = [
    "id",
    "name",
    "short_description",
    "technologies",
    "my_contributions",
]

_REQUIRED_SKILLS_KEYS: list[str] = [
    "languages",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _check_keys(
    data: dict,
    required_keys: list[str],
    context: str,
) -> list[str]:
    """
    Return a list of error messages for any required key that is missing
    or whose value is an empty string / empty list.
    """
    errors: list[str] = []
    for key in required_keys:
        if key not in data:
            errors.append(f"[{context}] Missing required field: '{key}'")
        elif data[key] in ("", [], None):
            errors.append(
                f"[{context}] Field '{key}' is present but empty — "
                "please provide a value."
            )
    return errors


# ---------------------------------------------------------------------------
# Section validators
# ---------------------------------------------------------------------------

def _validate_personal_info(profile: dict) -> list[str]:
    info = profile.get("personal_info", {})
    if not isinstance(info, dict):
        return ["[personal_info] Must be an object (dict)."]
    return _check_keys(info, _REQUIRED_PERSONAL_INFO_KEYS, "personal_info")


def _validate_education(profile: dict) -> list[str]:
    edu_list = profile.get("education", [])
    if not isinstance(edu_list, list):
        return ["[education] Must be a list."]
    if len(edu_list) == 0:
        return ["[education] Must contain at least one entry."]

    errors: list[str] = []
    for i, entry in enumerate(edu_list):
        if not isinstance(entry, dict):
            errors.append(f"[education[{i}]] Each entry must be an object.")
            continue
        errors.extend(
            _check_keys(entry, _REQUIRED_EDUCATION_KEYS, f"education[{i}]")
        )
        if "start_date" in entry:
            start_date = entry["start_date"]
            if start_date is not None:
                if not isinstance(start_date, str) or start_date.strip() == "":
                    errors.append(
                        f"[education[{i}]] Field 'start_date' is present but empty — "
                        "please provide a valid date string or null if unknown."
                    )
    return errors


def _validate_experience(profile: dict) -> list[str]:
    exp_list = profile.get("experience", [])
    if not isinstance(exp_list, list):
        return ["[experience] Must be a list."]
    # Experience may legitimately be empty for a fresh graduate; warn, not error.
    errors: list[str] = []
    for i, entry in enumerate(exp_list):
        if not isinstance(entry, dict):
            errors.append(f"[experience[{i}]] Each entry must be an object.")
            continue
        errors.extend(
            _check_keys(entry, _REQUIRED_EXPERIENCE_KEYS, f"experience[{i}]")
        )
    return errors


def _validate_projects(profile: dict) -> list[str]:
    proj_list = profile.get("projects", [])
    if not isinstance(proj_list, list):
        return ["[projects] Must be a list."]
    if len(proj_list) == 0:
        return ["[projects] Must contain at least one entry."]

    errors: list[str] = []
    for i, entry in enumerate(proj_list):
        if not isinstance(entry, dict):
            errors.append(f"[projects[{i}]] Each entry must be an object.")
            continue
        errors.extend(
            _check_keys(entry, _REQUIRED_PROJECT_KEYS, f"projects[{i}]")
        )
    return errors


def _validate_technical_skills(profile: dict) -> list[str]:
    skills = profile.get("technical_skills", {})
    if not isinstance(skills, dict):
        return ["[technical_skills] Must be an object (dict)."]
    return _check_keys(skills, _REQUIRED_SKILLS_KEYS, "technical_skills")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_profile(profile: dict) -> list[str]:
    """
    Validate a loaded profile dict and return all error messages found.

    Args:
        profile: A dict returned by loader.load_profile().

    Returns:
        list[str]: Empty list if valid; otherwise one message per problem.

    Example:
        errors = validate_profile(data)
        if errors:
            for e in errors:
                print(e)
    """
    if not isinstance(profile, dict):
        return [f"Profile must be a dict, got {type(profile).__name__}."]

    errors: list[str] = []

    # Check top-level required sections exist
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in profile:
            errors.append(f"[top-level] Missing required section: '{key}'")

    # Only validate sub-sections if the top-level key is actually present
    if "personal_info" in profile:
        errors.extend(_validate_personal_info(profile))
    if "education" in profile:
        errors.extend(_validate_education(profile))
    if "experience" in profile:
        errors.extend(_validate_experience(profile))
    if "projects" in profile:
        errors.extend(_validate_projects(profile))
    if "technical_skills" in profile:
        errors.extend(_validate_technical_skills(profile))

    return errors


def assert_valid_profile(profile: dict) -> None:
    """
    Convenience wrapper: validates and raises ValueError if any errors found.

    Args:
        profile: A dict returned by loader.load_profile().

    Raises:
        ValueError: With all error messages joined into one string.
    """
    errors = validate_profile(profile)
    if errors:
        formatted = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"Profile validation failed:\n{formatted}")
