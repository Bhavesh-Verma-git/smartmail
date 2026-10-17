"""
src/cv/selector.py
------------------
Project selection logic for Tailored CV Mode.

Rules:
1. ControlPlane.ai MUST ALWAYS be included as the first project (slot 1).
2. Exactly two additional projects are selected from the eligible pool:
   - "Building a Network Security System"
   - "E-Commerce Price Prediction"
   Therefore, selected projects always contain exactly 3 projects.
3. Competitions and achievements are NEVER selected as projects:
   - "Character Backstory Verification System"
   - "Predictive Profitability Framework"
   These belong strictly in the competitions/achievements section.
4. The two eligible projects are ranked by relevance to the job requirements.
5. If job information is insufficient or tied, use a documented deterministic fallback:
   Order: "Building a Network Security System", then "E-Commerce Price Prediction".
   Clearly record fallback_used=True and report uncertainty notes.
"""
from __future__ import annotations

import re
from typing import Any

from src.cv.models import ProjectSelectionResult


MANDATORY_PRIMARY_PROJECT_NAME = "ControlPlane.ai"
ELIGIBLE_SECONDARY_PROJECT_NAMES = [
    "Building a Network Security System",
    "E-Commerce Price Prediction",
]
EXCLUDED_COMPETITION_PROJECT_NAMES = frozenset({
    "Character Backstory Verification System",
    "Predictive Profitability Framework",
})

# Keyword indicators for ranking the two eligible projects against job requirements
_SECURITY_KEYWORDS = frozenset({
    "security", "cybersecurity", "network security", "phishing", "url",
    "etl", "docker", "aws", "ci/cd", "mlflow", "dagshub", "infrastructure",
    "devops", "pipeline", "classification", "deployment", "backend",
    "sysadmin", "cloud", "container", "automation",
})

_ECOMMERCE_NLP_KEYWORDS = frozenset({
    "ecommerce", "e-commerce", "price", "pricing", "prediction", "retail",
    "deberta", "deberta-v3", "nlp", "natural language processing", "text",
    "feature engineering", "regex", "smape", "regression", "tabular",
    "latency", "embeddings", "sentiment", "product", "sales", "finance",
})


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _score_project_relevance(project_name: str, job_text: str) -> int:
    """Calculate relevance keyword match count for an eligible project against job text."""
    lowered = job_text.lower()
    score = 0
    keywords = (
        _SECURITY_KEYWORDS
        if "security" in project_name.lower()
        else _ECOMMERCE_NLP_KEYWORDS
    )
    for kw in keywords:
        # Match whole words or phrases
        pattern = r"\b" + re.escape(kw) + r"\b"
        matches = len(re.findall(pattern, lowered))
        score += matches
    return score


class ProjectSelector:
    """
    Selects and ranks projects for the tailored CV workflow following strict rules.
    """

    @classmethod
    def select(
        cls,
        profile: dict[str, Any],
        job_description: str = "",
        target_role: str = "",
        technical_areas: list[str] | None = None,
    ) -> ProjectSelectionResult:
        """
        Select exactly 3 projects from the profile:
        - ControlPlane.ai (mandatory first)
        - 2 eligible projects ranked by job relevance (or deterministic fallback)
        """
        all_projects = profile.get("projects", [])
        project_map = {p.get("name", "").strip(): p for p in all_projects}

        # 1. Identify mandatory primary project (ControlPlane.ai)
        primary_project = None
        for name, proj in project_map.items():
            if _normalize_name(name) == _normalize_name(MANDATORY_PRIMARY_PROJECT_NAME):
                primary_project = proj
                break

        if not primary_project:
            # Fallback if named slightly differently
            for proj in all_projects:
                if "controlplane" in proj.get("name", "").lower():
                    primary_project = proj
                    break

        if not primary_project:
            raise ValueError(
                f"Mandatory project '{MANDATORY_PRIMARY_PROJECT_NAME}' not found in profile."
            )

        # 2. Extract eligible secondary projects, strictly excluding competitions
        eligible_candidates = []
        for name in ELIGIBLE_SECONDARY_PROJECT_NAMES:
            for proj in all_projects:
                p_name = proj.get("name", "").strip()
                if _normalize_name(p_name) == _normalize_name(name):
                    # Ensure it is not in the excluded competition set
                    if p_name not in EXCLUDED_COMPETITION_PROJECT_NAMES:
                        eligible_candidates.append(proj)
                        break

        if len(eligible_candidates) < 2:
            raise ValueError(
                f"Expected exactly 2 eligible secondary projects, found {len(eligible_candidates)}."
            )

        # 3. Combine job context
        areas_text = " ".join(technical_areas or [])
        combined_job_text = f"{job_description} {target_role} {areas_text}".strip()

        # 4. Rank eligible secondary projects
        proj_a = eligible_candidates[0]  # Building a Network Security System
        proj_b = eligible_candidates[1]  # E-Commerce Price Prediction

        name_a = proj_a.get("name", "")
        name_b = proj_b.get("name", "")

        score_a = _score_project_relevance(name_a, combined_job_text)
        score_b = _score_project_relevance(name_b, combined_job_text)

        fallback_used = False
        uncertainty_notes = []

        if score_a > score_b:
            secondary_ranked = [proj_a, proj_b]
            ranking_explanation = (
                f"'{name_a}' ranked ahead of '{name_b}' based on job keyword matches "
                f"({score_a} vs {score_b})."
            )
        elif score_b > score_a:
            secondary_ranked = [proj_b, proj_a]
            ranking_explanation = (
                f"'{name_b}' ranked ahead of '{name_a}' based on job keyword matches "
                f"({score_b} vs {score_a})."
            )
        else:
            # Tied or insufficient job information -> documented deterministic fallback
            fallback_used = True
            secondary_ranked = [proj_a, proj_b]  # Standard deterministic order
            ranking_explanation = (
                f"Job context did not contain differentiating keywords between '{name_a}' and '{name_b}' "
                f"(scores: {score_a} vs {score_b}). Used documented deterministic fallback order."
            )
            uncertainty_notes.append(
                f"Insufficient job information to differentiate secondary projects ({name_a} vs {name_b}). "
                f"Applied deterministic fallback ordering."
            )

        # Selected projects: ControlPlane.ai ALWAYS first, followed by the two secondary projects
        selected_projects = [primary_project, secondary_ranked[0], secondary_ranked[1]]

        return ProjectSelectionResult(
            selected_projects=selected_projects,
            primary_project=primary_project,
            secondary_projects=secondary_ranked,
            ranking_explanation=ranking_explanation,
            fallback_used=fallback_used,
            uncertainty_notes=uncertainty_notes,
        )


def select_projects(
    profile: dict[str, Any],
    job_description: str = "",
    target_role: str = "",
    technical_areas: list[str] | None = None,
) -> ProjectSelectionResult:
    """Convenience wrapper for ProjectSelector.select()."""
    return ProjectSelector.select(
        profile=profile,
        job_description=job_description,
        target_role=target_role,
        technical_areas=technical_areas,
    )
