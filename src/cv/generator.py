"""
src/cv/generator.py
-------------------
CV generator for Tailored CV Mode.

Formats a role-specific CV using profile data and the selected projects.
Saves the generated CV as an independent file and NEVER modifies the
master profile or any uploaded CV file.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from src.cv.models import ProjectSelectionResult, TailoredCVResult
from src.cv.selector import select_projects
from src.cv import pdf_generator as _pdf_gen


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-_\. ]", "_", name).strip()


def render_cv_markdown(
    profile: dict[str, Any],
    selection_result: ProjectSelectionResult,
    target_company: str = "",
    target_role: str = "",
) -> str:
    """
    Render a clean, standard Markdown CV from profile data and selected projects.
    Preserves all verified profile details without inventing any claims.
    """
    personal = profile.get("personal_info", {})
    name = personal.get("full_name", "Candidate")
    email = personal.get("email", "")
    phone = personal.get("phone", "")
    location = personal.get("location", "")
    linkedin = personal.get("linkedin_url", "")
    github = personal.get("github_url", "")

    header_parts = [p for p in [email, phone, location] if p]
    header_contact = " | ".join(header_parts)
    links_parts = []
    if linkedin:
        links_parts.append(f"[LinkedIn]({linkedin})")
    if github:
        links_parts.append(f"[GitHub]({github})")
    header_links = " | ".join(links_parts)

    lines: list[str] = []
    lines.append(f"# {name}")
    if header_contact:
        lines.append(header_contact)
    if header_links:
        lines.append(header_links)
    lines.append("")

    # Targeted subtitle if company or role is specified
    if target_role or target_company:
        role_label = target_role or "AI/ML Intern"
        company_label = f" — Tailored for {target_company}" if target_company else ""
        lines.append(f"**Target Role:** {role_label}{company_label}")
        lines.append("")

    # --- EDUCATION ---
    lines.append("## EDUCATION")
    lines.append("")
    education_entries = profile.get("education", [])
    for edu in education_entries:
        inst = edu.get("institution", "")
        deg = edu.get("degree", "")
        start = edu.get("start_date")
        end = edu.get("end_date", "")
        cgpa = edu.get("cgpa")
        coursework = edu.get("relevant_coursework", [])
        achievements = edu.get("achievements", [])

        if start:
            date_str = f"{start} – {end}"
        else:
            date_str = f"Completed {end}" if end else ""

        line_heading = f"**{inst}**"
        if date_str:
            line_heading += f" ({date_str})"
        lines.append(line_heading)

        sub_parts = [deg]
        if cgpa:
            sub_parts.append(f"Score / CGPA: {cgpa}")
        lines.append(" | ".join(sub_parts))

        if coursework:
            lines.append(f"- *Relevant Coursework:* {', '.join(coursework)}")
        if achievements:
            for ach in achievements:
                lines.append(f"- *Achievement:* {ach}")
        lines.append("")

    # --- EXPERIENCE (Preserve both internships) ---
    lines.append("## EXPERIENCE")
    lines.append("")
    experience_entries = profile.get("experience", [])
    for exp in experience_entries:
        org = exp.get("organization", "")
        role = exp.get("role", "")
        start = exp.get("start_date", "")
        end = exp.get("end_date", "")
        loc = exp.get("location", "")
        resps = exp.get("responsibilities", [])
        tech = exp.get("technologies_used", [])

        exp_heading = f"**{org}** — *{role}*"
        date_loc_parts = []
        if start or end:
            date_loc_parts.append(f"{start} – {end}")
        if loc:
            date_loc_parts.append(loc)
        if date_loc_parts:
            exp_heading += f" ({', '.join(date_loc_parts)})"
        lines.append(exp_heading)

        for resp in resps:
            lines.append(f"- {resp}")
        if tech:
            lines.append(f"- *Technologies:* {', '.join(tech)}")
        lines.append("")

    # --- SELECTED PROJECTS (Exactly the 3 selected projects) ---
    lines.append("## SELECTED PROJECTS")
    lines.append("")
    for proj in selection_result.selected_projects:
        p_name = proj.get("name", "")
        techs = proj.get("technologies", [])
        short_desc = proj.get("short_description", "")
        long_desc = proj.get("long_description", "")
        contributions = proj.get("my_contributions", [])
        outcomes = proj.get("outcomes", [])
        links = proj.get("links", {}) or {}

        p_heading = f"**{p_name}**"
        if techs:
            p_heading += f" | *{', '.join(techs)}*"
        lines.append(p_heading)

        desc = long_desc or short_desc
        if desc:
            lines.append(f"- {desc}")
        for c in contributions:
            lines.append(f"- *Contribution:* {c}")
        for o in outcomes:
            claim = o.get("claim") if isinstance(o, dict) else str(o)
            if claim:
                lines.append(f"- *Outcome (Claimed in CV):* {claim}")
        
        repo_link = links.get("github")
        if repo_link:
            lines.append(f"- *Repository:* [{repo_link}]({repo_link})")
        lines.append("")

    # --- COMPETITIONS & AWARDS ---
    lines.append("## COMPETITIONS & AWARDS")
    lines.append("")
    awards = profile.get("achievements_and_awards", [])
    for aw in awards:
        title = aw.get("title", "")
        issuer = aw.get("issuer", "")
        date = aw.get("date", "")
        desc = aw.get("description", "")
        
        aw_line = f"- **{title}**"
        info = [p for p in [issuer, date] if p]
        if info:
            aw_line += f" ({', '.join(info)})"
        if desc:
            aw_line += f": {desc}"
        lines.append(aw_line)
    lines.append("")

    # --- TECHNICAL SKILLS ---
    lines.append("## TECHNICAL SKILLS")
    lines.append("")
    skills = profile.get("technical_skills", {})
    order = [
        ("Languages", "languages"),
        ("Frameworks & Libraries", "frameworks_and_libraries"),
        ("Machine Learning & Data", "ml_and_data"),
        ("Databases", "databases"),
        ("Tools & Platforms", "tools_and_platforms"),
    ]
    for label, key in order:
        items = skills.get(key, [])
        if items:
            lines.append(f"- **{label}:** {', '.join(items)}")
    lines.append("")

    return "\n".join(lines)


class CVGenerator:
    """
    Generates and saves a role-specific tailored CV.
    """

    @classmethod
    def generate(
        cls,
        profile: dict[str, Any],
        selection_result: ProjectSelectionResult | None = None,
        job_description: str = "",
        target_role: str = "",
        target_company: str = "",
        technical_areas: list[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> TailoredCVResult:
        """
        Generate a tailored CV and save it to a separate output file.
        """
        # Select projects if not already provided
        if selection_result is None:
            selection_result = select_projects(
                profile=profile,
                job_description=job_description,
                target_role=target_role,
                technical_areas=technical_areas,
            )

        content = render_cv_markdown(
            profile=profile,
            selection_result=selection_result,
            target_company=target_company,
            target_role=target_role,
        )

        candidate_name = profile.get("personal_info", {}).get("full_name", "Candidate")
        sanitized_company = _sanitize_filename(target_company or "General")
        sanitized_role = _sanitize_filename(target_role or "AI_ML_Intern")

        if output_dir is None:
            output_dir = Path("output") / "tailored_cvs"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"CV_{_sanitize_filename(candidate_name)}_{sanitized_company}_{sanitized_role}.md"
        out_path = output_dir / filename

        out_path.write_text(content, encoding="utf-8")

        # --- PDF generation -----------------------------------------------
        pdf_path = ""
        pdf_error = ""
        pdf_filename = f"CV_{_sanitize_filename(candidate_name)}_{sanitized_company}_{sanitized_role}.pdf"
        pdf_out_path = output_dir / pdf_filename
        try:
            _pdf_gen.render_cv_pdf(
                profile=profile,
                selection_result=selection_result,
                target_role=target_role,
                target_company=target_company,
                output_path=pdf_out_path,
            )
            pdf_path = str(pdf_out_path.resolve())
        except Exception as exc:  # pragma: no cover
            pdf_error = f"PDF generation failed: {exc}"

        primary_path = pdf_path if pdf_path else str(out_path.resolve())
        return TailoredCVResult(
            file_path=primary_path,
            content=content,
            selection_result=selection_result,
            candidate_name=candidate_name,
            target_role=target_role,
            target_company=target_company,
            pdf_path=pdf_path,
            md_path=str(out_path.resolve()),
            pdf_error=pdf_error,
        )


def generate_tailored_cv(
    profile: dict[str, Any],
    selection_result: ProjectSelectionResult | None = None,
    job_description: str = "",
    target_role: str = "",
    target_company: str = "",
    technical_areas: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> TailoredCVResult:
    """Convenience wrapper for CVGenerator.generate()."""
    return CVGenerator.generate(
        profile=profile,
        selection_result=selection_result,
        job_description=job_description,
        target_role=target_role,
        target_company=target_company,
        technical_areas=technical_areas,
        output_dir=output_dir,
    )
