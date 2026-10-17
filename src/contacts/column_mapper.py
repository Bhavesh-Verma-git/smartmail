"""
src/contacts/column_mapper.py
-----------------------------
Intelligent column mapping for contact spreadsheets (.csv, .xlsx, .xls).
Maps various spreadsheet header names to standard SmartMail contact fields.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

# Standard internal contact fields
STANDARD_FIELDS: tuple[str, ...] = (
    "email",
    "contact_name",
    "company_name",
    "website",
    "role_or_job_url",
    "source_url",
    "notes",
)

FIELD_DESCRIPTIONS: dict[str, str] = {
    "email": "Contact Email Address (Required)",
    "contact_name": "Contact / Recruiter Name (Optional)",
    "company_name": "Company Name (Optional)",
    "website": "Company Website URL / Domain (Optional)",
    "role_or_job_url": "Role Title or Job Posting URL (Optional)",
    "source_url": "Source Link / Profile URL (Optional)",
    "notes": "Outreach Context Notes (Optional)",
}

# Synonyms and aliases for automatic matching
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "email": (
        "email",
        "email address",
        "e-mail",
        "e_mail",
        "contact email",
        "contact_email",
        "recipient email",
        "recipient_email",
        "work email",
        "work_email",
        "mail",
        "email_id",
        "email address*",
    ),
    "contact_name": (
        "name",
        "contact name",
        "contact_name",
        "person",
        "recipient",
        "full name",
        "full_name",
        "recruiter name",
        "recruiter",
        "contact",
        "first name",
    ),
    "company_name": (
        "company",
        "company name",
        "company_name",
        "organization",
        "organisation",
        "employer",
        "firm",
        "business",
        "account",
    ),
    "website": (
        "website",
        "company website",
        "company_website",
        "domain",
        "company domain",
        "url",
        "web",
        "homepage",
        "site",
    ),
    "role_or_job_url": (
        "role",
        "position",
        "job url",
        "job_url",
        "job link",
        "job_link",
        "role or job url",
        "role_or_job_url",
        "internship",
        "role title",
        "job title",
        "title",
        "posting",
    ),
    "source_url": (
        "source url",
        "source_url",
        "profile url",
        "profile_url",
        "source",
        "source link",
        "link",
        "reference",
    ),
    "notes": (
        "notes",
        "comments",
        "additional information",
        "additional info",
        "remarks",
        "details",
        "context",
    ),
}


def _normalize_header(header: str) -> str:
    """Normalize a header string for comparison: lowercase, strip punctuation/extra spaces."""
    if not header:
        return ""
    text = header.strip().lower()
    # Replace underscores and hyphens with spaces
    text = re.sub(r"[_\-]+", " ", text)
    # Remove special punctuation except spaces and alphanumeric
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class ColumnMappingResult:
    """Result of automatic column mapping."""
    # target_field -> source_header_name (or None if unmapped)
    mapping: dict[str, str | None]
    # Header name detected for email, if any
    email_column: str | None
    # Whether email was confidently detected
    email_confident: bool
    # List of all headers in the source file
    source_headers: list[str]
    # Explanation notes
    mapping_notes: list[str] = field(default_factory=list)


def auto_map_columns(headers: list[str]) -> ColumnMappingResult:
    """
    Given a list of column headers from a spreadsheet, determine the best
    mapping to SmartMail standard fields.
    """
    cleaned_headers = [str(h).strip() for h in headers if h is not None and str(h).strip()]
    norm_to_orig: dict[str, str] = {_normalize_header(h): h for h in cleaned_headers}

    mapping: dict[str, str | None] = {f: None for f in STANDARD_FIELDS}
    used_headers: set[str] = set()
    notes: list[str] = []

    # Pass 1: Exact or direct alias matching across all standard fields
    for target_field in STANDARD_FIELDS:
        aliases = COLUMN_ALIASES[target_field]
        matched_orig: str | None = None

        for alias in aliases:
            norm_alias = _normalize_header(alias)
            if norm_alias in norm_to_orig:
                candidate = norm_to_orig[norm_alias]
                if candidate not in used_headers:
                    matched_orig = candidate
                    break

        if matched_orig:
            mapping[target_field] = matched_orig
            used_headers.add(matched_orig)
            notes.append(f"Mapped '{matched_orig}' -> {target_field} (exact match)")

    # Pass 2: Substring / word boundary fallback only for unmapped fields
    for target_field in STANDARD_FIELDS:
        if mapping[target_field] is not None:
            continue

        aliases = COLUMN_ALIASES[target_field]
        matched_orig: str | None = None

        for alias in aliases:
            norm_alias = _normalize_header(alias)
            # Avoid single-letter or very short partial matches
            if len(norm_alias) < 3:
                continue
            for norm_h, orig_h in norm_to_orig.items():
                if orig_h not in used_headers:
                    if re.search(r"\b" + re.escape(norm_alias) + r"\b", norm_h):
                        matched_orig = orig_h
                        break
            if matched_orig:
                break

        if matched_orig:
            mapping[target_field] = matched_orig
            used_headers.add(matched_orig)
            notes.append(f"Mapped '{matched_orig}' -> {target_field} (fuzzy match)")
        else:
            notes.append(f"Field '{target_field}' remains unmapped (optional)")

    email_col = mapping.get("email")
    email_confident = bool(email_col)

    return ColumnMappingResult(
        mapping=mapping,
        email_column=email_col,
        email_confident=email_confident,
        source_headers=cleaned_headers,
        mapping_notes=notes,
    )


def validate_user_mapping(
    user_mapping: Mapping[str, str | None],
    available_headers: list[str],
) -> tuple[bool, str | None, dict[str, str]]:
    """
    Validate a user-submitted mapping.
    Ensures:
    1. 'email' field is mapped to a valid header.
    2. No duplicate mappings (two fields mapped to the same column, except None).
    3. All mapped headers exist in available_headers.

    Returns:
        (is_valid, error_message, clean_mapping)
    """
    clean: dict[str, str] = {}
    used_cols: dict[str, str] = {}  # col -> target_field

    email_col = user_mapping.get("email")
    if not email_col or str(email_col).strip() == "" or email_col not in available_headers:
        return False, "An Email column must be selected from the spreadsheet headers.", {}

    for target_field in STANDARD_FIELDS:
        source_col = user_mapping.get(target_field)
        if source_col and str(source_col).strip():
            col_str = str(source_col).strip()
            if col_str not in available_headers:
                return False, f"Selected column '{col_str}' is not in the spreadsheet headers.", {}
            if col_str in used_cols:
                return False, f"Column '{col_str}' cannot be mapped to both '{used_cols[col_str]}' and '{target_field}'.", {}
            used_cols[col_str] = target_field
            clean[target_field] = col_str

    return True, None, clean
