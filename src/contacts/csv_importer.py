"""
src/contacts/csv_importer.py
-----------------------------
Imports internship contacts from a CSV file.

## Expected CSV columns

Required:
    email          — contact email address (plain or display-name format)

Optional (missing values become empty strings, not errors):
    company_name   — name of the company
    contact_name   — name of the recruiter / person
    website        — company website URL
    role_or_job_url — specific role title or link to the job posting
    source_url     — where you found this contact
    notes          — your own free-form notes

## Design rules

- If the 'email' column is missing from the header row entirely, the whole
  file is rejected with a clear error.
- Each row is validated independently.  Invalid rows produce an ImportRowError
  in the errors list; they are NOT silently dropped.
- Duplicates (same normalised email seen more than once) are reported and the
  second occurrence is skipped.
- Whitespace is stripped from every cell.
- Empty rows (all cells blank) are skipped without an error.
- The function is deterministic: same CSV → same output every time.
- No data is written to disk.  The caller decides what to do with the results.

## Public API

    import_contacts_from_csv(csv_path) -> ImportResult

    ImportResult:
        contacts  list[ContactRecord]  — valid, de-duplicated contacts
        errors    list[ImportRowError] — one entry per invalid / duplicate row

    ContactRecord:
        row_number      int    — 1-based row number in the CSV (header = row 1)
        company_name    str
        contact_name    str
        email_result    EmailParseResult
        website         str
        role_or_job_url str
        source_url      str
        notes           str

    ImportRowError:
        row_number  int
        raw_row     dict   — the original cell values for debugging
        reason      str    — human-readable explanation
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from src.contacts.email_parser import EmailParseResult, parse_email


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset({"email"})
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "company_name",
    "contact_name",
    "website",
    "role_or_job_url",
    "source_url",
    "notes",
)
ALL_SUPPORTED_COLUMNS: tuple[str, ...] = ("email",) + OPTIONAL_COLUMNS


@dataclass
class ContactRecord:
    """One valid, parsed contact from the CSV."""
    row_number: int
    company_name: str
    contact_name: str
    email_result: EmailParseResult
    website: str
    role_or_job_url: str
    source_url: str
    notes: str
    raw_values: dict[str, str] = field(default_factory=dict)

    @property
    def email(self) -> str:
        """Normalized email address."""
        return self.email_result.normalized_email


@dataclass
class ImportRowError:
    """Describes a row that could not be imported."""
    row_number: int
    raw_row: dict
    reason: str


@dataclass
class ImportResult:
    """Returned by import_contacts_from_csv()."""
    contacts: list[ContactRecord] = field(default_factory=list)
    errors: list[ImportRowError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CSV importer
# ---------------------------------------------------------------------------

class CSVImportError(Exception):
    """Raised for file-level problems (missing file, missing required column)."""


def import_contacts_from_csv(csv_path: str | Path) -> ImportResult:
    """
    Parse a CSV file and return validated ContactRecord objects.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        ImportResult with contacts (valid rows) and errors (invalid rows).

    Raises:
        CSVImportError: If the file is missing or lacks the required 'email' column.
    """
    path = Path(csv_path)

    if not path.exists():
        raise CSVImportError(
            f"CSV file not found: {path.resolve()}\n"
            "Hint: check that the path is correct."
        )
    if not path.is_file():
        raise CSVImportError(f"Expected a file, got a directory: {path.resolve()}")

    try:
        raw_text = path.read_text(encoding="utf-8-sig")  # handles Windows BOM
    except OSError as exc:
        raise CSVImportError(f"Could not read CSV file: {exc}") from exc

    reader = csv.DictReader(raw_text.splitlines())

    # --- Validate that the file has the required column ----------------------
    if reader.fieldnames is None:
        raise CSVImportError("CSV file is empty — no header row found.")

    # Normalise header names: strip whitespace and lowercase for comparison
    normalised_headers = {h.strip().lower() for h in reader.fieldnames if h}
    missing_required = REQUIRED_COLUMNS - normalised_headers
    if missing_required:
        raise CSVImportError(
            f"CSV is missing required column(s): {sorted(missing_required)}.\n"
            f"Columns found: {sorted(normalised_headers)}.\n"
            f"Required: {sorted(REQUIRED_COLUMNS)}."
        )

    result = ImportResult()
    seen_emails: dict[str, int] = {}  # normalised_email -> first row_number

    for raw_row in reader:
        # reader.line_num counts lines including the header, so data row 1
        # is line_num 2.  We expose a 1-based data-row number to the user.
        row_number = reader.line_num

        # Strip whitespace from every cell; treat missing optional columns as ""
        row: dict[str, str] = {
            col: (raw_row.get(col) or "").strip()
            for col in ALL_SUPPORTED_COLUMNS
        }

        # Skip completely empty rows (all values blank)
        if all(v == "" for v in row.values()):
            continue

        # --- Validate email --------------------------------------------------
        raw_email = row["email"]
        if not raw_email:
            result.errors.append(ImportRowError(
                row_number=row_number,
                raw_row=dict(raw_row),
                reason="'email' field is blank — this field is required.",
            ))
            continue

        email_result = parse_email(raw_email)
        if not email_result.is_syntax_valid:
            result.errors.append(ImportRowError(
                row_number=row_number,
                raw_row=dict(raw_row),
                reason=(
                    f"Email '{raw_email}' failed syntax validation: "
                    f"{email_result.parse_notes}"
                ),
            ))
            continue

        # --- Duplicate check -------------------------------------------------
        norm = email_result.normalized_email
        if norm in seen_emails:
            result.errors.append(ImportRowError(
                row_number=row_number,
                raw_row=dict(raw_row),
                reason=(
                    f"Duplicate email '{norm}' — first seen at row "
                    f"{seen_emails[norm]}. This row is skipped."
                ),
            ))
            continue

        seen_emails[norm] = row_number

        # --- Build the contact record ----------------------------------------
        contact = ContactRecord(
            row_number=row_number,
            company_name=row["company_name"],
            contact_name=row["contact_name"],
            email_result=email_result,
            website=row["website"],
            role_or_job_url=row["role_or_job_url"],
            source_url=row["source_url"],
            notes=row["notes"],
            raw_values=dict(raw_row),
        )
        result.contacts.append(contact)

    return result
