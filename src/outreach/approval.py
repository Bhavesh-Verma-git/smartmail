"""
src/outreach/approval.py
------------------------
Approval binding for a single draft.

An approval is bound to the EXACT draft that was reviewed: recipient,
subject, body, and attachment identity (path + SHA-256 content hash when
the file exists). Any material edit changes the hash, so the old approval
no longer matches and the draft must be re-approved.

This is a safety binding, not a cryptographic signature: it detects
edits, it does not authenticate the approver.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def attachment_identity(attachment_path: str) -> tuple[str, str]:
    """Return (path, sha256 hex or '') for an attachment file."""
    p = Path(attachment_path) if attachment_path else None
    if p is None or not p.is_file():
        return attachment_path or "", ""
    digest = hashlib.sha256()
    try:
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return attachment_path or "", ""
    return str(p), digest.hexdigest()


def approval_hash_for(
    recipient: str,
    subject: str,
    body: str,
    attachment_path: str,
) -> str:
    """Hash binding an approval to one exact draft version."""
    _path, sha = attachment_identity(attachment_path)
    material = "\n".join([
        (recipient or "").strip().lower(),
        (subject or "").strip(),
        (body or "").strip(),
        _path,
        sha,
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def approval_matches(
    stored_hash: str,
    recipient: str,
    subject: str,
    body: str,
    attachment_path: str,
) -> bool:
    """True only if the stored hash matches the draft's current content."""
    if not stored_hash:
        return False
    return stored_hash == approval_hash_for(recipient, subject, body, attachment_path)
