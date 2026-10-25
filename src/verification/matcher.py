"""
src/verification/matcher.py
----------------------------
Profile relevance matching.

## Public API

    match_profile(research_result, profile) -> MatchResult

## What this does

Compares the company's evidence-backed technical_areas (from Stage 3) against
the candidate's master profile to determine whether the company is likely
relevant to an AI/ML internship.

Profile is the dict loaded from master_profile.json.  The matcher reads the
actual skills, projects, and role preferences from the profile — it does not
rely on a hardcoded summary.

## How matching works

1. Extract profile skill keywords from:
   - technical_skills.ml_and_data
   - technical_skills.frameworks_and_libraries
   - preferences.target_roles
   - project technologies and highlight_for_roles

2. Compare against research_result.technical_areas (e.g. ["AI/ML", "Generative AI"])
   using a map from area labels to profile keyword groups.

3. Determine case type:
   - SPECIFIC_INTERNSHIP: role_status == FOUND
   - GENERAL_INQUIRY: everything else

4. Determine relevance_status:
   - RELEVANT    : at least one technical area from evidence matches a profile area
   - NOT_A_MATCH : evidence retrieved but no relevant areas found
   - REVIEW      : no technical evidence retrieved, or profile has no relevant skills

5. Build explanation from actual profile facts, not generic text.

## What this does NOT do

- Does not score or rank companies numerically.
- Does not invent skills or project experience not in the profile.
- Does not override a BLOCKED verification status.
- Does not claim relevance when evidence is missing.
"""

from __future__ import annotations

from src.research.checks import _TECH_AREAS
from src.research.models import (
    ResearchResult,
    FOUND,
    UNKNOWN as RES_UNKNOWN,
    NOT_FOUND,
    RETRIEVAL_OK,
)
from src.verification.models import (
    MatchResult,
    RELEVANT,
    NOT_A_MATCH,
    SPECIFIC_INTERNSHIP,
    GENERAL_INQUIRY,
)

_REVIEW = "REVIEW"

# ---------------------------------------------------------------------------
# Mapping: research area label -> profile skill categories to check
# ---------------------------------------------------------------------------
# Each area label (from check_technical_relevance) maps to a tuple of
# (human-friendly label, profile field path as a tuple of dict keys).
#
# We look in these profile fields for any non-placeholder keyword overlap.

_AREA_TO_PROFILE_FIELDS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "Generative AI": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    "Agentic AI": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    "AI/ML": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Target roles",        ("preferences", "target_roles")),
        ("Languages",           ("technical_skills", "languages")),
    ],
    "Data Science": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    "Software Engineering": [
        ("Languages",           ("technical_skills", "languages")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Tools",               ("technical_skills", "tools_and_platforms")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    # Specific Stage 3 subdisciplines
    "Machine Learning & Predictive Analytics": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    "Generative AI & LLMs": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    "AI Agents & Workflows": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    "RAG & Retrieval Systems": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    "NLP & Deep Learning": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    "Computer Vision & Generative Media": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
    "AI Infrastructure & Serving": [
        ("ML/Data skills",      ("technical_skills", "ml_and_data")),
        ("Frameworks",          ("technical_skills", "frameworks_and_libraries")),
        ("Tools",               ("technical_skills", "tools_and_platforms")),
        ("Target roles",        ("preferences", "target_roles")),
    ],
}

# Labels used by checks.py that differ from the matcher labels. Normalising
# them here (instead of silently missing) fixes the "Deep Learning & NLP" /
# "AI Infrastructure & Evaluation" mismatches that previously never matched.
_AREA_LABEL_ALIASES: dict[str, str] = {
    "Deep Learning & NLP": "NLP & Deep Learning",
    "AI Infrastructure & Evaluation": "AI Infrastructure & Serving",
}

# A single generic indicator (e.g. "analytics") must not match every AI/ML
# area. These specific indicators are distinctive enough to match alone.
_STRONG_AIML_INDICATORS: frozenset[str] = frozenset({
    "pytorch", "tensorflow", "langgraph", "langchain", "diffusion",
    "stable diffusion", "transformers", "huggingface", "faiss",
    "computer vision", "opencv", "rag", "generative ai", "llm",
    "natural language processing", "neural network", "deep learning",
    "machine learning", "xgboost", "vector database",
})

def _aiml_indicator_strength(profile_keywords: frozenset[str]) -> tuple[int, bool]:
    # Count distinct AI/ML indicators present; note any strong indicator.
    found = {
        ind for ind in _AIML_INDICATORS
        if any(ind in kw.lower() for kw in profile_keywords)
        or ind in profile_keywords
    }
    return len(found), bool(found & _STRONG_AIML_INDICATORS)

def _aiml_matches_profile(profile_keywords: frozenset[str]) -> bool:
    # True only with >=2 indicators, or 1 strong indicator.
    count, strong = _aiml_indicator_strength(profile_keywords)
    return count >= 2 or strong

# Keywords in area labels that are considered inherently AI/ML relevant.
# If the profile contains ANY ML/data skills, these areas automatically match.
_AIML_AREAS: frozenset[str] = frozenset({
    "Generative AI", "Agentic AI", "AI/ML", "Data Science",
    "Machine Learning & Predictive Analytics",
    "Generative AI & LLMs",
    "AI Agents & Workflows",
    "RAG & Retrieval Systems",
    "NLP & Deep Learning",
    "Computer Vision & Generative Media",
    "AI Infrastructure & Serving",
})

# Specific adjacent ML tools/frameworks
_AIML_INDICATORS: frozenset[str] = frozenset({
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "llm", "generative ai", "rag",
    "agentic", "pytorch", "tensorflow", "scikit-learn", "transformers",
    "huggingface", "langchain", "openai", "gemini", "ml intern",
    "ai intern", "machine learning intern", "ai/ml",
    "data science", "analytics",
    # Specific adjacent ML tools/frameworks
    "jax", "keras", "xgboost", "lightgbm", "spacy", "cuda", "onnx",
    "langgraph", "vector database", "chromadb", "pinecone", "qdrant", "weaviate",
    "faiss", "computer vision", "opencv", "yolo", "diffusion", "stable diffusion",
    "vllm", "triton", "tensorrt", "ollama", "lora", "fine-tuning", "prompt engineering",
})

# Placeholder sentinel — profile values starting with "[PLACEHOLDER" have not
# been filled in and should not contribute to matching.
_PLACEHOLDER_PREFIX = "[PLACEHOLDER"


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def match_profile(
    research_result: ResearchResult,
    profile: dict,
    cv_text: str | None = None,
) -> MatchResult:
    """
    Compare research evidence against the candidate's master profile and optional CV text.

    Parameters
    ----------
    research_result : ResearchResult from Stage 3.
    profile         : dict loaded from master_profile.json.
    cv_text         : Optional raw text extracted from uploaded CV.

    Returns
    -------
    MatchResult
    """
    role_status = research_result.role_status
    tech_areas = research_result.technical_areas  # evidence-backed, may be []

    # Determine case type
    case_type = SPECIFIC_INTERNSHIP if role_status == FOUND else GENERAL_INQUIRY

    # Extract profile keywords
    profile_keywords = _extract_profile_keywords(profile, cv_text=cv_text)
    profile_evidence_items = _extract_profile_evidence_items(profile, cv_text=cv_text)

    match_notes: list[str] = []

    # If no technical areas were retrieved, we cannot make a reliable match
    if not tech_areas:
        match_notes.append(
            "No technical areas were identified from the retrieved evidence. "
            "This may be because the company website was inaccessible, contained "
            "no relevant keywords, or the company's focus is not yet determined."
        )
        if not profile_keywords:
            match_notes.append(
                "The master profile does not yet contain filled-in skills. "
                "Fill in technical_skills and preferences to enable matching."
            )
        return MatchResult(
            relevance_status=_REVIEW,
            case_type=case_type,
            matched_areas=[],
            profile_evidence=[],
            explanation=(
                "Relevance cannot be determined: no technical areas were "
                "found in the retrieved evidence."
            ),
            match_notes=match_notes,
            evidence_sources=[],
        )

    # Profile has no real skills filled in
    if not profile_keywords:
        match_notes.append(
            "The master profile does not contain filled-in skills or preferences. "
            "Update technical_skills and preferences.target_roles to enable matching."
        )
        return MatchResult(
            relevance_status=_REVIEW,
            case_type=case_type,
            matched_areas=[],
            profile_evidence=[],
            explanation=(
                "Relevance cannot be determined: master profile skills are not yet filled in."
            ),
            match_notes=match_notes,
            evidence_sources=[],
        )

    # Match technical areas against profile
    matched_areas: list[str] = []
    matched_profile_items: list[str] = []

    for area in tech_areas:
        if _area_matches_profile(area, profile_keywords):
            matched_areas.append(area)
            # Find which profile items contributed
            items = _contributing_profile_items(area, profile, profile_evidence_items)
            for item in items:
                if item not in matched_profile_items:
                    matched_profile_items.append(item)

    matched_aiml_areas = [a for a in matched_areas if _AREA_LABEL_ALIASES.get(a, a) in _AIML_AREAS]
    candidate_aiml = _is_aiml_focused(profile, profile_keywords)

    if matched_aiml_areas:
        # At least one evidence-backed AI/ML area matched the candidate's skills
        relevance_status = RELEVANT
        explanation = _build_relevant_explanation(
            matched_areas, matched_profile_items, case_type, profile
        )
    elif matched_areas == ["Software Engineering"] or (matched_areas and not matched_aiml_areas):
        # Only generic Software Engineering matched — no AI/ML areas
        if candidate_aiml:
            if _has_software_engineering_target(profile):
                # Candidate explicitly targets SWE roles alongside AI/ML, but AI relevance is uncertain
                relevance_status = _REVIEW
                explanation = (
                    "The company has software engineering evidence (e.g. backend, APIs), "
                    "but no specific AI/ML technical focus was found. AI/ML relevance is uncertain; "
                    "human review is required before outreach."
                )
                match_notes.append(
                    "Generic software-engineering keywords (e.g. Python, APIs) alone do not "
                    "establish AI/ML relevance. Review whether an AI/ML opportunity exists."
                )
            else:
                # Candidate is strictly AI/ML focused — generic SWE alone does not match
                relevance_status = NOT_A_MATCH
                explanation = (
                    f"The company's technical focus ({', '.join(tech_areas)}) does not overlap "
                    "with the candidate's target AI/ML roles. Python or general software skills "
                    "alone do not establish AI/ML relevance."
                )
                match_notes.append(
                    "Candidate is focused on AI/ML. Generic software engineering without AI/ML "
                    "is not considered a match."
                )
        else:
            # Candidate is not AI/ML focused and matched SWE
            relevance_status = RELEVANT
            explanation = _build_relevant_explanation(
                matched_areas, matched_profile_items, case_type, profile
            )
    elif matched_areas:
        relevance_status = RELEVANT
        explanation = _build_relevant_explanation(
            matched_areas, matched_profile_items, case_type, profile
        )
    else:
        # Evidence retrieved but no areas match the profile
        relevance_status = NOT_A_MATCH
        explanation = (
            f"The company's technical areas ({', '.join(tech_areas)}) "
            "do not overlap with the candidate's profile skills and target roles. "
            "This may not be a suitable AI/ML internship target."
        )
        match_notes.append(
            "Consider whether to proceed as a general software engineering inquiry."
        )

    # Extract source URLs from research evidence that actually support this match
    match_sources = _extract_match_sources(
        research_result, matched_areas, relevance_status, case_type
    )

    return MatchResult(
        relevance_status=relevance_status,
        case_type=case_type,
        matched_areas=matched_areas,
        profile_evidence=matched_profile_items,
        explanation=explanation,
        match_notes=match_notes,
        evidence_sources=match_sources,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_real_value(value: str) -> bool:
    """Return True if a profile string is filled in (not a placeholder)."""
    return bool(value) and not value.startswith(_PLACEHOLDER_PREFIX)


def _get_nested(profile: dict, *keys: str) -> list[str]:
    """
    Navigate nested dicts by keys and return a list of real string values.

    Returns [] if any key is missing or the value is not a list.
    """
    node = profile
    for k in keys:
        if not isinstance(node, dict):
            return []
        node = node.get(k, [])
    if not isinstance(node, list):
        return []
    return [v for v in node if isinstance(v, str) and _is_real_value(v)]


def _extract_profile_keywords(profile: dict, cv_text: str | None = None) -> frozenset[str]:
    """
    Return a frozenset of all real (non-placeholder) lowercase keywords
    extracted from the profile's skill and preference fields, plus any skills
    found in the provided CV text.
    """
    keywords: list[str] = []
    keywords += _get_nested(profile, "technical_skills", "ml_and_data")
    keywords += _get_nested(profile, "technical_skills", "frameworks_and_libraries")
    keywords += _get_nested(profile, "technical_skills", "languages")
    keywords += _get_nested(profile, "technical_skills", "tools_and_platforms")
    keywords += _get_nested(profile, "preferences", "target_roles")

    # Project technologies and highlight_for_roles
    for proj in profile.get("projects", []):
        if not isinstance(proj, dict):
            continue
        keywords += [
            v for v in proj.get("technologies", [])
            if isinstance(v, str) and _is_real_value(v)
        ]
        keywords += [
            v for v in proj.get("highlight_for_roles", [])
            if isinstance(v, str) and _is_real_value(v)
        ]

    # Incorporate candidate skills from uploaded CV text
    if cv_text:
        cv_lower = cv_text.lower()
        for ind in _AIML_INDICATORS:
            if ind in cv_lower:
                keywords.append(ind)

    return frozenset(kw.lower() for kw in keywords)


def _extract_profile_evidence_items(profile: dict, cv_text: str | None = None) -> list[str]:
    """
    Return a list of human-readable profile items (project names, skill
    groups, CV skills) that can be cited in the explanation.
    """
    items: list[str] = []

    ml_skills = _get_nested(profile, "technical_skills", "ml_and_data")
    if ml_skills:
        items.append(f"ML/Data skills: {', '.join(ml_skills[:4])}")

    frameworks = _get_nested(profile, "technical_skills", "frameworks_and_libraries")
    if frameworks:
        items.append(f"Frameworks: {', '.join(frameworks[:4])}")

    target_roles = _get_nested(profile, "preferences", "target_roles")
    if target_roles:
        items.append(f"Target roles: {', '.join(target_roles)}")

    for proj in profile.get("projects", []):
        if not isinstance(proj, dict):
            continue
        name = proj.get("name", "")
        if _is_real_value(name):
            items.append(f"Project: {name}")

    if cv_text:
        cv_lower = cv_text.lower()
        found_cv_skills = [ind for ind in _AIML_INDICATORS if ind in cv_lower][:5]
        if found_cv_skills:
            items.append(f"Uploaded CV skills: {', '.join(found_cv_skills)}")

    return items


def _is_aiml_focused(profile: dict, profile_keywords: frozenset[str]) -> bool:
    """Return True if candidate's target roles or skills are AI/ML focused.

    A target role counts only with >=2 indicators or 1 strong indicator,
    so a role mentioning only a generic word like "analytics" does not
    mark the whole profile AI/ML-focused.
    """
    target_roles = [r.lower() for r in _get_nested(profile, "preferences", "target_roles")]
    for role in target_roles:
        role_kws = frozenset([role])
        if _aiml_matches_profile(role_kws):
            return True
    ml_skills = _get_nested(profile, "technical_skills", "ml_and_data")
    if ml_skills:
        return True
    return _aiml_matches_profile(profile_keywords)


def _has_software_engineering_target(profile: dict) -> bool:
    """Return True if candidate explicitly targets software engineering roles."""
    target_roles = [r.lower() for r in _get_nested(profile, "preferences", "target_roles")]
    se_roles = {"software engineer", "software engineering", "swe", "backend", "frontend", "full stack", "web developer"}
    return any(any(se in r for se in se_roles) for r in target_roles)


def _extract_match_sources(
    research_result: ResearchResult,
    matched_areas: list[str],
    relevance_status: str,
    case_type: str,
) -> list[str]:
    """
    Extract relevant research source URLs that actually establish or support
    the matching decision. Excludes empty URLs and duplicates while preserving order.
    Does not claim a source supports relevance if relevance was not established.
    """
    if relevance_status == NOT_A_MATCH or not matched_areas:
        return []

    sources: list[str] = []
    seen: set[str] = set()

    for e in getattr(research_result, "evidence", []):
        url = (getattr(e, "source_url", "") or "").strip()
        if not url:
            continue
        retrieval_status = getattr(e, "retrieval_status", "")
        if retrieval_status != RETRIEVAL_OK:
            continue

        supports_match = False
        text = (
            (getattr(e, "extracted_text", "") or "") + " " + (getattr(e, "title", "") or "")
        ).lower()

        # Check if this evidence contained keywords for any matched area
        for area in matched_areas:
            for label, kws in _TECH_AREAS:
                if label == area and any(kw in text for kw in kws):
                    supports_match = True
                    break
            if supports_match:
                break

        # Job listing supporting a specific internship
        if case_type == SPECIFIC_INTERNSHIP:
            if getattr(e, "source_type", "") == "job_listing" or "role_evidence" in getattr(e, "supports", []):
                supports_match = True

        # Explicit technical_relevance support declaration
        if not supports_match and "technical_relevance" in getattr(e, "supports", []):
            if matched_areas:
                supports_match = True

        if supports_match and url not in seen:
            seen.add(url)
            sources.append(url)

    return sources


def _area_matches_profile(area: str, profile_keywords: frozenset[str]) -> bool:
    """
    Return True if the research area is covered by the profile's keywords.

    For AI/ML areas, we check whether the profile has *any* ML/data skill.
    For Software Engineering, we check languages and frameworks.
    """
    area = _AREA_LABEL_ALIASES.get(area, area)
    if area in _AIML_AREAS:
        return _aiml_matches_profile(profile_keywords)

    if area == "Software Engineering":
        sw_indicators = {
            "python", "javascript", "typescript", "java", "go", "rust",
            "software engineer", "backend", "frontend", "full stack",
            "api", "fastapi", "django", "flask", "react", "node",
            "software engineering intern",
        }
        return any(ind in kw.lower() for kw in profile_keywords for ind in sw_indicators)

    return False


def _contributing_profile_items(
    area: str,
    profile: dict,
    evidence_items: list[str],
) -> list[str]:
    """
    Return the subset of evidence_items that are relevant to the given area.
    Limits to first 3 items to keep explanations concise.
    """
    # For AI/ML areas, ML skills and projects are most relevant
    if area in _AIML_AREAS:
        return [
            item for item in evidence_items
            if item.startswith(("ML/Data", "Project", "Target"))
        ][:3]

    # For Software Engineering, languages and frameworks
    return [
        item for item in evidence_items
        if item.startswith(("Frameworks", "Target", "Project"))
    ][:3]


def _build_relevant_explanation(
    matched_areas: list[str],
    profile_items: list[str],
    case_type: str,
    profile: dict,
) -> str:
    """
    Build a 1–2 sentence explanation grounded in actual profile facts.
    """
    area_str = ", ".join(matched_areas)
    item_str = "; ".join(profile_items[:3]) if profile_items else "profile skills"

    if case_type == SPECIFIC_INTERNSHIP:
        return (
            f"The company's technical focus ({area_str}) aligns with the "
            f"candidate's profile ({item_str}). "
            "A specific internship listing was found — this is a strong candidate "
            "for tailored outreach."
        )
    else:
        return (
            f"The company's technical focus ({area_str}) overlaps with the "
            f"candidate's profile ({item_str}). "
            "No specific listing was found, so outreach should be a general "
            "AI/ML internship inquiry."
        )
