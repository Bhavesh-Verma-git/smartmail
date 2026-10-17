"""
src/ui/cv_validator.py
----------------------
Validates uploaded CV files for safety, format integrity, and size limits.
Enforces that uploaded files are valid, readable, non-empty PDF documents.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pypdf

# Configurable limit: 10 MB
MAX_CV_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB


@dataclass
class CVValidationResult:
    """Result of CV file validation."""
    is_valid: bool
    filename: str
    size_bytes: int
    size_formatted: str
    page_count: int = 0
    error: str | None = None
    sha256: str | None = None


def format_file_size(size_bytes: int) -> str:
    """Format bytes into readable KB/MB string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def validate_cv_stream(
    stream: BinaryIO | bytes,
    filename: str,
    max_size_bytes: int = MAX_CV_SIZE_BYTES,
) -> CVValidationResult:
    """
    Validate a CV file stream or raw bytes.

    Checks:
    1. Filename has a supported extension (.pdf).
    2. File is not empty (size > 0).
    3. File does not exceed max_size_bytes.
    4. File header starts with valid PDF magic bytes (%PDF-).
    5. File is parseable by pypdf without corruption.
    6. PDF is not encrypted / password-protected.
    7. PDF has at least one valid page.
    """
    if isinstance(stream, bytes):
        raw_bytes = stream
    else:
        raw_bytes = stream.read()

    size_bytes = len(raw_bytes)
    size_formatted = format_file_size(size_bytes)

    # 1. Check extension
    suffix = Path(filename).suffix.lower()
    if suffix != ".pdf":
        return CVValidationResult(
            is_valid=False,
            filename=filename,
            size_bytes=size_bytes,
            size_formatted=size_formatted,
            error=f"Unsupported file format '{suffix}'. Only standard PDF (.pdf) documents are accepted.",
        )

    # 2. Check empty file
    if size_bytes == 0:
        return CVValidationResult(
            is_valid=False,
            filename=filename,
            size_bytes=0,
            size_formatted="0 B",
            error="The uploaded file is completely empty (0 bytes). Please upload a valid CV document.",
        )

    # 3. Check maximum size limit
    if size_bytes > max_size_bytes:
        max_formatted = format_file_size(max_size_bytes)
        return CVValidationResult(
            is_valid=False,
            filename=filename,
            size_bytes=size_bytes,
            size_formatted=size_formatted,
            error=f"File exceeds maximum allowed size ({max_formatted}). Your file is {size_formatted}.",
        )

    # 4. Check PDF magic bytes
    if not raw_bytes.startswith(b"%PDF-"):
        return CVValidationResult(
            is_valid=False,
            filename=filename,
            size_bytes=size_bytes,
            size_formatted=size_formatted,
            error="File does not have a valid PDF header (%PDF-). The file may be corrupted or renamed.",
        )

    # 5. Parse structure with pypdf
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
    except Exception as exc:
        return CVValidationResult(
            is_valid=False,
            filename=filename,
            size_bytes=size_bytes,
            size_formatted=size_formatted,
            error=f"Failed to read PDF document structure: {exc}",
        )

    # 6. Check encryption
    if reader.is_encrypted:
        return CVValidationResult(
            is_valid=False,
            filename=filename,
            size_bytes=size_bytes,
            size_formatted=size_formatted,
            error="The PDF file is encrypted or password-protected. Please upload an unprotected CV.",
        )

    # 7. Check page count
    page_count = len(reader.pages)
    if page_count < 1:
        return CVValidationResult(
            is_valid=False,
            filename=filename,
            size_bytes=size_bytes,
            size_formatted=size_formatted,
            error="The PDF document does not contain any readable pages.",
        )

    # Compute SHA-256 for audit integrity
    import hashlib
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    return CVValidationResult(
        is_valid=True,
        filename=filename,
        size_bytes=size_bytes,
        size_formatted=size_formatted,
        page_count=page_count,
        sha256=sha256,
        error=None,
    )


def validate_cv_file(
    file_path: str | Path,
    max_size_bytes: int = MAX_CV_SIZE_BYTES,
) -> CVValidationResult:
    """Validate a CV file given its filesystem path."""
    path = Path(file_path)
    if not path.exists():
        return CVValidationResult(
            is_valid=False,
            filename=path.name,
            size_bytes=0,
            size_formatted="0 B",
            error=f"File not found: {path.name}",
        )
    return validate_cv_stream(path.read_bytes(), filename=path.name, max_size_bytes=max_size_bytes)
