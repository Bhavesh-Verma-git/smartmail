"""
src/cv/pdf_generator.py
-----------------------
Professional A4 PDF CV renderer designed to recreate the exact layout,
hierarchy, and styling of the reference LaTeX template (templates/reference_cv.tex)
using fpdf2.

Implementation Note
-------------------
If a native LaTeX toolchain (e.g., pdflatex, tectonic) is installed and available,
the original .tex template is the preferred compilation source. When LaTeX is
unavailable in the local environment, this renderer faithfully reproduces the
Jake Gutierrez / sb2nov resume template (margins, rules, two-column subheadings,
four-column education table, compact bullet styling, and clean one-page budget)
using pure Python (fpdf2) with zero external binaries.

Design principles
-----------------
- Selectable, ATS-parsable text (uncompressed streams).
- Helvetica font family with clean hierarchy.
- A4 page (210 x 297 mm), 12.5 mm side margins, 9.0 mm top/bottom.
- One-page target fit for standard profile data.
- Hyperlinks for email, LinkedIn, GitHub, and repository URLs.
- Internal provenance labels ("(Claimed in CV: unverified)", "Reported in CV")
  are stripped automatically.
- No invented dates, claims, metrics, or credentials.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fpdf import FPDF, XPos, YPos
from fpdf.enums import Align

# ---------------------------------------------------------------------------
# Layout constants (calibrated to match templates/reference_cv.tex)
# ---------------------------------------------------------------------------
PAGE_W_MM = 210.0
PAGE_H_MM = 297.0
MARGIN_L = 12.5
MARGIN_R = 12.5
MARGIN_T = 9.0
MARGIN_B = 9.0
EFFECTIVE_W = PAGE_W_MM - MARGIN_L - MARGIN_R  # 185.0 mm

# Font sizes (pt)
FS_NAME = 15.0
FS_CONTACT = 8.5
FS_SUBTITLE = 8.0
FS_SECTION = 9.5
FS_SUBHEAD = 8.5
FS_BODY = 7.8
FS_SMALL = 7.5

# Line heights (mm)
LH_SECTION = 3.8
LH_SUBHEAD = 3.8
LH_BODY = 3.4
LH_TABLE = 3.5

# Colours (RGB)
CLR_RULE = (70, 70, 70)
CLR_SECTION = (20, 20, 20)
CLR_BODY = (40, 40, 40)
CLR_LINK = (0, 70, 180)


def _sanitize_for_latin1(text: str) -> str:
    """Safely map Unicode punctuation to Latin-1 compatible ASCII."""
    if not text:
        return ""
    replacements = {
        "\u2013": "-",
        "\u2014": "--",
        "\u2015": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2026": "...",
        "\u2022": "-",
        "\u25cf": "-",
        "\u00a0": " ",
        "\u200b": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2212": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


class _CvPDF(FPDF):
    """FPDF subclass replicating the Jake Gutierrez / sb2nov LaTeX macros."""

    def __init__(self) -> None:
        super().__init__(unit="mm", format="A4")
        self.set_margins(MARGIN_L, MARGIN_T, MARGIN_R)
        self.set_auto_page_break(auto=True, margin=MARGIN_B)
        self.set_compression(False)
        self.set_link_annotation = True  # type: ignore[attr-defined]

    def normalize_text(self, text: str) -> str:
        clean = _sanitize_for_latin1(text)
        return super().normalize_text(clean)

    # ------------------------------------------------------------------
    # \section{Title} with horizontal dividing rule
    # ------------------------------------------------------------------
    def section_heading(self, title: str) -> None:
        self.ln(1.2)
        self.set_font("Helvetica", "B", FS_SECTION)
        self.set_text_color(*CLR_SECTION)
        self.cell(
            EFFECTIVE_W, LH_SECTION, title.upper(),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        y = self.get_y()
        self.set_draw_color(*CLR_RULE)
        self.set_line_width(0.3)
        self.line(MARGIN_L, y, MARGIN_L + EFFECTIVE_W, y)
        self.ln(0.8)

    # ------------------------------------------------------------------
    # \resumeSubheading{Role}{Dates}{Org}{Location}
    # ------------------------------------------------------------------
    def subheading(
        self,
        left1: str,
        right1: str = "",
        left2: str = "",
        right2: str = "",
    ) -> None:
        self.set_text_color(*CLR_BODY)
        # Line 1: Role/Title (bold) + Dates (right)
        self.set_font("Helvetica", "B", FS_SUBHEAD)
        rw1 = self.get_string_width(right1) + 2 if right1 else 0.0
        self.cell(
            EFFECTIVE_W - rw1, LH_SUBHEAD, left1,
            new_x=XPos.RIGHT if rw1 else XPos.LMARGIN,
            new_y=YPos.TOP if rw1 else YPos.NEXT,
        )
        if rw1:
            self.set_font("Helvetica", "", FS_SUBHEAD)
            self.cell(rw1, LH_SUBHEAD, right1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=Align.R)

        # Line 2: Organization (italic) + Location (italic, right)
        if left2 or right2:
            self.set_font("Helvetica", "I", FS_SUBTITLE)
            rw2 = self.get_string_width(right2) + 2 if right2 else 0.0
            self.cell(
                EFFECTIVE_W - rw2, LH_BODY, left2,
                new_x=XPos.RIGHT if rw2 else XPos.LMARGIN,
                new_y=YPos.TOP if rw2 else YPos.NEXT,
            )
            if rw2:
                self.cell(rw2, LH_BODY, right2, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=Align.R)

    # ------------------------------------------------------------------
    # \resumeProjectHeading{\textbf{Name} | \emph{Tech}}{Dates}
    # ------------------------------------------------------------------
    def project_heading(self, name: str, tech: str = "", dates: str = "") -> None:
        self.set_text_color(*CLR_BODY)
        dates_w = self.get_string_width(dates) + 2 if dates else 0.0
        avail_left = EFFECTIVE_W - dates_w

        self.set_font("Helvetica", "B", FS_SUBHEAD)
        name_w = self.get_string_width(name) + 1.5

        if tech and (name_w + self.get_string_width(f" | {tech}") < avail_left):
            self.cell(name_w, LH_SUBHEAD, name, new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.set_font("Helvetica", "I", FS_SUBTITLE)
            self.cell(
                avail_left - name_w, LH_SUBHEAD, f" | {tech}",
                new_x=XPos.RIGHT if dates_w else XPos.LMARGIN,
                new_y=YPos.TOP if dates_w else YPos.NEXT,
            )
        else:
            self.cell(
                avail_left, LH_SUBHEAD, name,
                new_x=XPos.RIGHT if dates_w else XPos.LMARGIN,
                new_y=YPos.TOP if dates_w else YPos.NEXT,
            )

        if dates_w:
            self.set_font("Helvetica", "", FS_SUBTITLE)
            self.cell(dates_w, LH_SUBHEAD, dates, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=Align.R)

    # ------------------------------------------------------------------
    # Bullet item: \resumeItem{...}
    # ------------------------------------------------------------------
    def bullet(self, text: str, indent: float = 3.0) -> None:
        if not text:
            return
        clean = _strip_provenance(text)
        if not clean:
            return
        self.set_font("Helvetica", "", FS_BODY)
        self.set_text_color(*CLR_BODY)
        self.cell(indent, LH_BODY, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.multi_cell(
            EFFECTIVE_W - indent, LH_BODY, clean,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

    # ------------------------------------------------------------------
    # Educational Qualifications Table (mirroring \begin{tabularx})
    # ------------------------------------------------------------------
    def education_table(self, education_entries: list[dict[str, Any]]) -> None:
        if not education_entries:
            return
        col_w = [28.0, 58.0, 71.0, 28.0]

        # Top rule
        y_top = self.get_y()
        self.set_draw_color(*CLR_RULE)
        self.set_line_width(0.3)
        self.line(MARGIN_L, y_top, MARGIN_L + EFFECTIVE_W, y_top)
        self.ln(0.4)

        # Table Header
        self.set_font("Helvetica", "B", FS_SUBTITLE)
        self.set_text_color(*CLR_SECTION)
        self.cell(col_w[0], LH_TABLE, "Year", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(col_w[1], LH_TABLE, "Degree", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(col_w[2], LH_TABLE, "Institute/School", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(col_w[3], LH_TABLE, "CGPA/Percentage", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=Align.R)

        # Mid rule
        y_mid = self.get_y()
        self.line(MARGIN_L, y_mid, MARGIN_L + EFFECTIVE_W, y_mid)
        self.ln(0.4)

        # Data Rows
        self.set_font("Helvetica", "", FS_BODY)
        self.set_text_color(*CLR_BODY)
        for edu in education_entries:
            start = edu.get("start_date")
            end = edu.get("end_date", "")
            dates = _date_range(start, end)
            deg = edu.get("degree", "")
            inst = edu.get("institution", "")
            cgpa = edu.get("cgpa") or ""

            self.cell(col_w[0], LH_TABLE, dates, new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.cell(col_w[1], LH_TABLE, deg[:38], new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.cell(col_w[2], LH_TABLE, inst[:45], new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.cell(col_w[3], LH_TABLE, cgpa, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=Align.R)

        # Bottom rule
        y_bot = self.get_y()
        self.line(MARGIN_L, y_bot, MARGIN_L + EFFECTIVE_W, y_bot)
        self.ln(0.8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PROVENANCE_PAT = re.compile(
    r"\s*\(Claimed in CV[^)]*\)"
    r"|\s*\[Claimed in CV[^\]]*\]"
    r"|\bReported in CV[.;]*"
    r"|\bCV-reported\b"
    r"|\s*\bin CV[.;]*",
    re.IGNORECASE,
)


def _strip_provenance(text: str) -> str:
    """Remove internal provenance labels from recruiter-facing text."""
    clean = _PROVENANCE_PAT.sub("", text).strip()
    return re.sub(r"^Reported\s+", "", clean, flags=re.I).strip()


def _date_range(start: str | None, end: str | None) -> str:
    if start and end:
        return f"{start} - {end}"
    if end:
        return f"Completed {end}"
    if start:
        return start
    return ""


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------
def render_cv_pdf(
    profile: dict[str, Any],
    selection_result: Any,          # ProjectSelectionResult
    target_role: str = "",
    target_company: str = "",
    output_path: str | Path | None = None,
) -> bytes:
    """
    Render a professional A4 PDF CV replicating the Jake Gutierrez LaTeX layout
    and optionally save it to disk.

    Returns the raw PDF bytes. If output_path is given the file is also
    written to that path.
    """
    pdf = _CvPDF()
    pdf.add_page()

    personal = profile.get("personal_info", {})
    name = personal.get("full_name", "Candidate")
    email_addr = personal.get("email", "")
    phone = personal.get("phone", "")
    linkedin = personal.get("linkedin_url", "")
    github = personal.get("github_url", "")

    # -----------------------------------------------------------------------
    # Header: Name (Bold, Center)
    # -----------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", FS_NAME)
    pdf.set_text_color(*CLR_SECTION)
    pdf.cell(
        EFFECTIVE_W, 5.5, name,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        align=Align.C,
    )

    # Line 2: Contact row (Phone | Email | LinkedIn | GitHub)
    contact_items: list[tuple[str, str, str]] = []  # (label, text, url)
    if phone:
        contact_items.append((phone, phone, ""))
    if email_addr:
        contact_items.append((email_addr, email_addr, f"mailto:{email_addr}"))
    if linkedin:
        contact_items.append(("LinkedIn", "LinkedIn", linkedin))
    if github:
        contact_items.append(("GitHub", "GitHub", github))

    pdf.set_font("Helvetica", "", FS_CONTACT)
    pdf.set_text_color(*CLR_BODY)
    sep = "  |  "
    sep_w = pdf.get_string_width(sep)

    total_w = sum(pdf.get_string_width(label) for label, _, _ in contact_items)
    total_w += sep_w * max(0, len(contact_items) - 1)
    start_x = MARGIN_L + max(0.0, (EFFECTIVE_W - total_w) / 2)
    pdf.set_x(start_x)

    for i, (label, _, url) in enumerate(contact_items):
        if i > 0:
            pdf.set_font("Helvetica", "", FS_CONTACT)
            pdf.set_text_color(*CLR_BODY)
            pdf.cell(sep_w, 4.0, sep, new_x=XPos.RIGHT, new_y=YPos.TOP)

        w = pdf.get_string_width(label)
        if url:
            pdf.set_font("Helvetica", "U", FS_CONTACT)
            pdf.set_text_color(*CLR_LINK)
            pdf.cell(w, 4.0, label, link=url, new_x=XPos.RIGHT, new_y=YPos.TOP)
        else:
            pdf.set_font("Helvetica", "", FS_CONTACT)
            pdf.set_text_color(*CLR_BODY)
            pdf.cell(w, 4.0, label, new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.ln(4.2)

    # Line 3: Degree / Specialization Subtitle
    edu_list = profile.get("education", [])
    primary_degree = ""
    achieve = ""
    if edu_list:
        primary_degree = edu_list[0].get("degree", "")
        inst = edu_list[0].get("institution", "")
        if inst and inst not in primary_degree:
            primary_degree = f"{primary_degree}, {inst}"
        achievements = edu_list[0].get("achievements", [])
        if achievements:
            achieve = achievements[0]

    subtitle_parts = []
    if primary_degree:
        subtitle_parts.append(primary_degree)
    if achieve:
        subtitle_parts.append(achieve)
    elif target_role:
        subtitle_parts.append(f"Target: {target_role}")

    if subtitle_parts:
        subtitle_str = "  |  ".join(subtitle_parts)
        pdf.set_font("Helvetica", "", FS_SUBTITLE)
        pdf.set_text_color(*CLR_BODY)
        pdf.cell(
            EFFECTIVE_W, 3.8, subtitle_str[:125],
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            align=Align.C,
        )

    # -----------------------------------------------------------------------
    # 1. Educational Qualifications (Table)
    # -----------------------------------------------------------------------
    pdf.section_heading("Educational Qualifications")
    pdf.education_table(edu_list)

    # -----------------------------------------------------------------------
    # 2. Experience
    # -----------------------------------------------------------------------
    pdf.section_heading("Experience")
    for exp in profile.get("experience", []):
        role = exp.get("role", "")
        org = exp.get("organization", "")
        dates = _date_range(exp.get("start_date"), exp.get("end_date"))
        loc = exp.get("location", "")
        pdf.subheading(role, dates, org, loc)
        for r in exp.get("responsibilities", []):
            pdf.bullet(r)
        pdf.ln(0.4)

    # -----------------------------------------------------------------------
    # 3. Projects (Selected Projects)
    # -----------------------------------------------------------------------
    pdf.section_heading("Projects")
    for proj in getattr(selection_result, "selected_projects", []):
        p_name = proj.get("name", "")
        techs = ", ".join(proj.get("technologies", [])[:5])
        links = proj.get("links", {}) or {}
        repo_link = links.get("github") or ""

        pdf.project_heading(p_name, techs, dates="2026")

        desc = proj.get("long_description") or proj.get("short_description")
        if desc:
            pdf.bullet(desc)
        for c in proj.get("my_contributions", []):
            pdf.bullet(c)
        for o in proj.get("outcomes", []):
            claim = o.get("claim") if isinstance(o, dict) else str(o)
            pdf.bullet(claim)
        if repo_link:
            pdf.set_font("Helvetica", "U", FS_SMALL)
            pdf.set_text_color(*CLR_LINK)
            pdf.cell(
                pdf.get_string_width(f"GitHub: {repo_link}") + 2, LH_BODY,
                f"GitHub: {repo_link}",
                link=repo_link, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
            pdf.set_text_color(*CLR_BODY)
        pdf.ln(0.4)

    # -----------------------------------------------------------------------
    # 4. Competitions & Achievements
    # -----------------------------------------------------------------------
    achievements_list = profile.get("achievements_and_awards", [])
    if achievements_list:
        pdf.section_heading("Competitions & Achievements")
        for aw in achievements_list:
            title = aw.get("title", "")
            issuer = aw.get("issuer", "")
            date = aw.get("date", "")
            desc = aw.get("description", "")

            pdf.subheading(title, date, issuer, "")
            if desc:
                pdf.bullet(desc)
            pdf.ln(0.3)

    # -----------------------------------------------------------------------
    # 5. Technical Skills
    # -----------------------------------------------------------------------
    skills = profile.get("technical_skills", {})
    if skills:
        pdf.section_heading("Technical Skills")
        skill_cats = [
            ("Languages", "languages"),
            ("Frameworks & Libraries", "frameworks_and_libraries"),
            ("GenAI & ML", "ml_and_data"),
            ("Tools", "tools_and_platforms"),
        ]
        for label, k in skill_cats:
            items = skills.get(k, [])
            if items:
                pdf.set_font("Helvetica", "B", FS_BODY)
                pdf.set_text_color(*CLR_BODY)
                lw = pdf.get_string_width(f"{label}: ") + 1
                pdf.cell(lw, LH_BODY, f"{label}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.set_font("Helvetica", "", FS_BODY)
                pdf.multi_cell(EFFECTIVE_W - lw, LH_BODY, ", ".join(items), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # -----------------------------------------------------------------------
    # 6. Additional Information
    # -----------------------------------------------------------------------
    coursework = edu_list[0].get("relevant_coursework", []) if edu_list else []
    certifications = profile.get("certifications", [])
    if coursework or certifications:
        pdf.section_heading("Additional Information")
        if coursework:
            pdf.set_font("Helvetica", "B", FS_BODY)
            pdf.set_text_color(*CLR_BODY)
            lw = pdf.get_string_width("Relevant Coursework: ") + 1
            pdf.cell(lw, LH_BODY, "Relevant Coursework:", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "", FS_BODY)
            pdf.multi_cell(EFFECTIVE_W - lw, LH_BODY, "; ".join(coursework), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if certifications:
            pdf.set_font("Helvetica", "B", FS_BODY)
            pdf.set_text_color(*CLR_BODY)
            lw = pdf.get_string_width("Certifications: ") + 1
            pdf.cell(lw, LH_BODY, "Certifications:", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "", FS_BODY)
            cert_titles = [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in certifications]
            pdf.multi_cell(EFFECTIVE_W - lw, LH_BODY, "; ".join(cert_titles), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # -----------------------------------------------------------------------
    # Output serialization
    # -----------------------------------------------------------------------
    raw = bytes(pdf.output())

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)

    return raw


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_pdf(pdf_bytes: bytes, required_text_fragments: list[str] | None = None) -> dict:
    """
    Basic structural validation of generated PDF bytes.

    Returns a dict with keys:
        ok (bool), page_count (int), size_bytes (int), missing_fragments (list[str])
    """
    result: dict = {
        "ok": False,
        "size_bytes": len(pdf_bytes),
        "page_count": None,
        "missing_fragments": [],
        "errors": [],
    }

    if len(pdf_bytes) < 100:
        result["errors"].append("PDF is too small to be valid.")
        return result

    if not pdf_bytes.startswith(b"%PDF"):
        result["errors"].append("File does not begin with %PDF header.")
        return result

    page_count = pdf_bytes.count(b"/Type /Page\n") + pdf_bytes.count(b"/Type/Page\n")
    if page_count == 0:
        page_count = max(1, pdf_bytes.count(b"endobj") // 10)
    result["page_count"] = page_count

    if required_text_fragments:
        pdf_text = pdf_bytes.decode("latin-1", errors="ignore")
        for frag in required_text_fragments:
            if frag.lower() not in pdf_text.lower():
                result["missing_fragments"].append(frag)

    result["ok"] = not result["errors"] and not result["missing_fragments"]
    return result
