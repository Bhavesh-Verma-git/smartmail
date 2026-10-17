"""
src/cv/models.py
----------------
Data models for project selection and CV tailoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectSelectionResult:
    """
    Result of selecting and ranking projects for a tailored CV.

    Attributes:
        selected_projects: Exactly 3 project dictionaries, with ControlPlane.ai always first.
        primary_project: The mandatory primary project (ControlPlane.ai).
        secondary_projects: The 2 eligible projects, ranked by job relevance.
        ranking_explanation: Plain-English explanation of why projects were ranked this way.
        fallback_used: True if job description lacked differentiating keywords, triggering fallback.
        uncertainty_notes: List of any notes regarding ambiguity or missing job info.
    """
    selected_projects: list[dict]
    primary_project: dict
    secondary_projects: list[dict]
    ranking_explanation: str
    fallback_used: bool = False
    uncertainty_notes: list[str] = field(default_factory=list)


@dataclass
class TailoredCVResult:
    """
    Result of generating a role-specific CV.

    Attributes:
        file_path: Primary generated CV file path (PDF when generated, Markdown otherwise).
        content: The Markdown content of the generated CV.
        pdf_path: Absolute path to the rendered PDF file, or empty string if failed.
        md_path: Absolute path to the Markdown CV file.
        selection_result: The ProjectSelectionResult used during generation.
        candidate_name: Candidate full name.
        target_role: The role targeted for tailoring.
        target_company: The company targeted for outreach.
        pdf_error: Human-readable description of PDF generation failure, if any.
    """
    file_path: str
    content: str
    selection_result: ProjectSelectionResult
    candidate_name: str
    target_role: str
    target_company: str
    pdf_path: str = ""
    md_path: str = ""
    pdf_error: str = ""
