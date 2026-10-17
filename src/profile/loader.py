"""
src/profile/loader.py
---------------------
Loads and parses master_profile.json (or any profile JSON file).

Design notes:
- Uses only the Python standard library (json, pathlib).
- Returns a plain dict so callers can work with familiar Python data structures.
- Raises clear, typed exceptions so problems are easy to diagnose.
"""

import json
from pathlib import Path


class ProfileLoadError(Exception):
    """Raised when the profile file cannot be read or parsed."""


def load_profile(profile_path: str | Path) -> dict:
    """
    Load a profile JSON file and return its contents as a Python dict.

    Args:
        profile_path: Path to the JSON file (str or Path object).

    Returns:
        dict: The parsed profile data.

    Raises:
        ProfileLoadError: If the file is missing, unreadable, or not valid JSON.
        TypeError: If profile_path is not a str or Path.
    """
    path = Path(profile_path)

    if not path.exists():
        raise ProfileLoadError(
            f"Profile file not found: {path.resolve()}\n"
            "Hint: check that the path is correct and the file exists."
        )

    if not path.is_file():
        raise ProfileLoadError(
            f"Expected a file but found a directory: {path.resolve()}"
        )

    try:
        raw_text = path.read_text(encoding="utf-8-sig")  # handles BOM written by Windows tools
    except OSError as exc:
        raise ProfileLoadError(
            f"Could not read profile file: {path.resolve()}\nReason: {exc}"
        ) from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProfileLoadError(
            f"Profile file contains invalid JSON: {path.resolve()}\n"
            f"JSON error at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ProfileLoadError(
            f"Profile JSON must be an object (dict) at the top level, "
            f"got: {type(data).__name__}"
        )

    return data
