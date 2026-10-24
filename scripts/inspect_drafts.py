"""
Inspect most recent drafts to diagnose personalization quality.
"""
import json
import os
from pathlib import Path

drafts_dir = Path("output/.drafts")
files = sorted(drafts_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:8]

for f in files:
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        company = data.get("company_name", "N/A")
        body = data.get("body", "")
        pr = data.get("personalization_record", None)
        finding = ""
        if isinstance(pr, dict):
            finding = pr.get("company_finding", "EMPTY")
        elif pr is None:
            finding = "NONE (personalization_record is null)"
        else:
            finding = f"TYPE={type(pr)}"

        lines = body.split("\n") if body else []
        alignment = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Given") or stripped.startswith("My background"):
                alignment = stripped[:200]
                break

        print(f"FILE: {f.name}")
        print(f"  COMPANY: {company}")
        print(f"  FINDING: {finding}")
        print(f"  ALIGN PARA: {alignment}")
        print()
    except Exception as e:
        print(f"{f.name}: ERROR - {e}")
